"""Private single-flight provider recorder; the existing OpenWebUI bridge is unchanged."""
from __future__ import annotations
from typing import Any
import asyncio
import copy
import hmac
import math
import os
import time
from pathlib import Path
from urllib.parse import urlsplit
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from . import SCHEMA
from .core import StopRun, require, safe_id, strict_json, canonical, write, text
from .budget import api_ledger
from .inference_limits import positive_int, teacher_limits, validate_teacher_limits

class SSE:
    def __init__(self) -> None:
        """Initialize local state without starting a provider request."""
        self.buffer=''; self.message={'role':'assistant','content':''}; self.calls={}; self.finish=None
        self.usage={};self.model=None;self.done=False;self.error=None
    def feed(self,chunk: str) -> None:
        """Assemble text, reasoning, tool calls and usage across provider SSE chunks."""
        self.buffer+=chunk
        require(len(self.buffer)<2_000_000,'SSE event too large')
        self.buffer=self.buffer.replace('\r\n','\n')
        while '\n\n' in self.buffer:
            event,self.buffer=self.buffer.split('\n\n',1)
            payload='\n'.join(x[5:].lstrip() for x in event.splitlines() if x.startswith('data:'))
            if not payload:continue
            if payload=='[DONE]':self.done=True;continue
            obj=strict_json(payload);require(isinstance(obj,dict),'Invalid SSE JSON')
            if obj.get('error'):self.error='upstream error'
            if obj.get('usage'):self.usage=obj['usage']
            self.model=obj.get('model',self.model)
            for c in obj.get('choices',[]):
                require(c.get('index',0)==0,'Multiple stream choices are unsupported')
                d=c.get('delta') or {}
                for k in ('content','reasoning','reasoning_content','thinking'):
                    if d.get(k) is not None:
                        require(isinstance(d[k],str),'Non-text delta');self.message[k]=self.message.get(k,'')+d[k]
                if d.get('reasoning_details'):self.message.setdefault('reasoning_details',[]).extend(d['reasoning_details'])
                for tc in d.get('tool_calls') or []:
                    item=self.calls.setdefault(tc['index'],{'id':'','type':'function','function':{'name':'','arguments':''}})
                    if tc.get('id'):item['id']+=tc['id']
                    for k in ('name','arguments'):
                        if (tc.get('function') or {}).get(k):item['function'][k]+=tc['function'][k]
                if c.get('finish_reason') is not None:self.finish=c['finish_reason']
    def response(self) -> dict[str, Any]:
        """Return the assembled provider response without abbreviating observations."""
        if self.calls:self.message['tool_calls']=[self.calls[k] for k in sorted(self.calls)]
        return {'model':self.model,'choices':[{'index':0,'message':self.message,'finish_reason':self.finish}],'usage':self.usage}

def app_factory() -> FastAPI:
    """Create a private single-worker recorder with authenticated session control."""
    defaults=teacher_limits()
    root=Path(os.environ['SA_CAPTURE_DIR']);root.mkdir(parents=True,exist_ok=True);root.chmod(0o700)
    # Prevent accidentally launching two worker processes on the same capture store.
    import fcntl
    process_lock=(root/'.worker.lock').open('a');fcntl.flock(process_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    url=os.environ.get('SA_UPSTREAM_URL','https://openrouter.ai/api/v1').rstrip('/')
    u=urlsplit(url);require(u.scheme=='https' or (u.scheme=='http' and u.hostname in {'localhost','127.0.0.1','host.docker.internal'}),'Unsafe upstream URL')
    require(not u.username and not u.password and not u.query and not u.fragment,'Invalid upstream URL')
    key=os.environ.get('SA_UPSTREAM_KEY') or os.environ.get('OPENROUTER_API_KEY');require(key,'Provider key missing')
    proxy=os.environ['SA_PROXY_TOKEN'];admin=os.environ['SA_ADMIN_TOKEN']
    require(len(proxy)>=24 and len(admin)>=24 and proxy!=admin,'Distinct strong proxy/admin tokens required')
    prices=strict_json(os.environ.get('SA_PRICES_JSON','{}'));require(prices,'Explicit provider price ceilings required')
    ledger=api_ledger();state={'session':None,'busy':False,'deadline':None};lock=asyncio.Lock();app=FastAPI()
    app.state.process_lock=process_lock
    def auth(req: Request,secret: str) -> None:
        """Verify the private bearer token using a constant-time comparison."""
        require(hmac.compare_digest(req.headers.get('authorization',''), 'Bearer '+secret),'Unauthorized')
    @app.exception_handler(StopRun)
    async def stop(req: Request,exc: Any) -> Any:"""Return a bounded data-contract error without exposing provider credentials."""; return JSONResponse({'error':str(exc)},status_code=422)
    @app.get('/healthz')
    async def health() -> Any:"""Report recorder health and the capture schema."""; return {'ok':True,'schema':SCHEMA}
    @app.get('/v1/models')
    async def models(req:Request) -> Any:
        """Expose only inventoried upstream models with configured price ceilings."""
        auth(req,proxy)
        async with httpx.AsyncClient(timeout=30,follow_redirects=False) as client:
            r=await client.get(url+'/models',headers={'Authorization':'Bearer '+key});r.raise_for_status()
            data=r.json()['data']
        return {'object':'list','data':[x for x in data if x.get('id') in prices]}
    @app.post('/admin/begin')
    async def begin(req:Request) -> Any:
        """Open a unique capture session after validating its model and budgets."""
        auth(req,admin);data=await req.json()
        limits=validate_teacher_limits(data['inference_limits']) if 'inference_limits' in data else copy.deepcopy(defaults)
        async with lock:
            require(state['session'] is None and not state['busy'],'Another capture session is active')
            bid=safe_id(data['benchmark_id']);require(not (root/(bid+'.json')).exists() and not (root/(bid+'.pending.json')).exists(),'Capture ID exists')
            require(data['model'] in prices and isinstance(data['question'],str) and len(data['question'])>=8,'Invalid session/model')
            state['session']={'schema':SCHEMA,'benchmark_id':bid,'question':data['question'],'model':data['model'],
                              'started':time.time(),'records':[],'complete':False,'inference_limits':limits}
            state['deadline']=time.monotonic()+limits['evaluation_limits']['max_seconds']
            write(root/(bid+'.pending.json'),state['session'])
        return {'ok':True,'inference_limits':limits}
    @app.post('/admin/end')
    async def end(req:Request) -> Any:
        """Persist the completed idle session without inventing missing responses."""
        auth(req,admin)
        async with lock:
            session=state['session'];require(session is not None and not state['busy'],'No idle session')
            session['complete']=bool(session['records']) and all(r['complete'] for r in session['records'])
            session['duration_seconds']=session['inference_limits']['evaluation_limits']['max_seconds']-(state['deadline']-time.monotonic())
            write(root/(session['benchmark_id']+'.json'),session)
            state['session']=None;state['deadline']=None
        return session
    @app.post('/v1/chat/completions')
    async def chat(req:Request) -> Any:
        """Reserve budget and forward one exact model request for the active session."""
        auth(req,proxy);raw=bytearray()
        async for chunk in req.stream():
            raw.extend(chunk);require(len(raw)<=2_000_000,'Request body too large')
        body=strict_json(raw.decode());require(isinstance(body,dict),'Invalid request')
        async with lock:
            session=state['session'];require(session is not None and not state['busy'],'Explicit single-flight session required')
            require(body.get('model')==session['model'],'Unexpected model/task call')
            require(body.get('n',1)==1 and isinstance(body.get('messages'),list),'One conversation/choice required')
            require(any(session['question'] in text(m.get('content')) for m in body['messages'] if m.get('role')=='user'),'Question absent; wrong session or task')
            limits=session['inference_limits'];episode=limits['evaluation_limits']
            require(all(r['complete'] for r in session['records']),'Previous response incomplete; no automatic continuation')
            require(len(session['records'])<episode['max_llm_calls'],'Episode call budget exhausted')
            used_output=0;used_tools=0
            for previous in session['records']:
                usage=previous['response'].get('usage') or {};value=usage.get('completion_tokens')
                require(type(value) is int and value>=0,'Unknown episode output usage; no further reservation')
                used_output+=value
                used_tools+=sum(len(c.get('message',{}).get('tool_calls') or []) for c in previous['response'].get('choices',[]))
            require(used_tools<=episode['max_tool_calls'],'Episode tool budget exhausted')
            remaining=state['deadline']-time.monotonic()
            require(remaining>0,'Episode elapsed budget exhausted')
            cap=limits['max_output_tokens']
            require(not ('max_completion_tokens' in body and 'max_tokens' in body),'Conflicting output-limit aliases')
            n=body.get('max_completion_tokens',body.get('max_tokens',cap))
            positive_int(n,'requested output tokens',cap)
            require(used_output+n<=episode['max_completion_tokens'],'Episode output reservation exceeds remaining token budget')
            if 'max_completion_tokens' not in body:body['max_tokens']=n
            if body.get('stream'):body['stream_options']={**(body.get('stream_options') or {}),'include_usage':True}
            bound=len(canonical(body).encode())+512
            require(bound<=limits['max_input_bound_bytes'],'Input byte bound exceeded; no silent truncation')
            rate=prices[body['model']]
            require(isinstance(rate,dict) and all(type(rate.get(k)) in (int,float) and math.isfinite(rate[k]) and rate[k]>0 for k in ('input','output')),'Positive conservative prices required')
            reservation={'tokens':bound+n,'usd':(bound*rate['input']+n*rate['output'])/1e6,'calls':1}
            ticket,_=ledger.reserve('api',reservation)
            state['busy']=True
        record={'schema':SCHEMA,'source':'upstream_wire','request':copy.deepcopy(body),'response':{},'error':None,'complete':False}
        def finish(response: Any,error: Any=None) -> None:
            """Persist the complete or interrupted response and reconcile its reservation."""
            record.update(response=response,error=error)
            choices=response.get('choices') or []
            record['complete']=error is None and len(choices)==1 and choices[0].get('finish_reason') in {'stop','tool_calls'}
            # Keep the conservative reservation unless *both* provider counters exist.
            usage=response.get('usage') or {};tokens=usage.get('total_tokens');cost=usage.get('cost')
            if type(tokens) is int and tokens>=0 and type(cost) in (int,float) and math.isfinite(cost) and cost>=0:
                ledger.settle(ticket,{'tokens':tokens,'usd':cost,'calls':1})
            else:ledger.settle(ticket)
            completion=usage.get('completion_tokens')
            if type(completion) is int and completion>=0 and (completion>n or used_output+completion>episode['max_completion_tokens']):
                record.update(complete=False,error='Observed episode output budget exceeded')
            if type(tokens) is int and tokens>reservation['tokens'] or type(cost) in (int,float) and math.isfinite(cost) and cost>reservation['usd']+1e-12:
                record.update(complete=False,error='Observed provider usage exceeded reservation')
            tool_count=sum(len(c.get('message',{}).get('tool_calls') or []) for c in choices)
            if used_tools+tool_count>episode['max_tool_calls']:
                record.update(complete=False,error='Observed episode tool budget exceeded')
            if time.monotonic()>state['deadline'] or time.time()-session['started']>episode['max_seconds']:
                record.update(complete=False,error=record['error'] or 'Observed episode time budget exceeded')
            session['records'].append(record)
            write(root/(session['benchmark_id']+'.pending.json'),session);state['busy']=False
        client=httpx.AsyncClient(timeout=httpx.Timeout(min(limits['request_timeout_seconds'],remaining),connect=20),follow_redirects=False)
        deadline=min(state['deadline'],time.monotonic()+limits['request_timeout_seconds'])
        try:
            require(time.monotonic()<deadline,'Episode deadline exhausted before provider request')
            outgoing=client.build_request('POST',url+'/chat/completions',json=body,headers={'Authorization':'Bearer '+key})
            async with asyncio.timeout(max(0,deadline-time.monotonic())):
                response=await client.send(outgoing,stream=bool(body.get('stream')))
            if response.status_code>=400:
                finish({},f'Provider HTTP {response.status_code}');await response.aclose();await client.aclose()
                return JSONResponse({'error':'Provider rejected request; no automatic retry'},status_code=response.status_code)
            if not body.get('stream'):
                obj=response.json();finish(obj);await client.aclose();return JSONResponse(obj)
        except BaseException as exc:
            if state['busy']:finish({},'Transport failure: '+type(exc).__name__)
            await client.aclose();raise
        async def relay() -> Any:
            """Stream provider events while capturing their complete decoded response."""
            assembly=SSE();size=0
            try:
                async with asyncio.timeout(max(0,deadline-time.monotonic())):
                    async for chunk in response.aiter_text():
                        size+=len(chunk);require(size<16_000_000,'Stream size limit exceeded');assembly.feed(chunk);yield chunk
                finish(assembly.response(),assembly.error or (None if assembly.done else 'Missing SSE DONE'))
            except BaseException as exc:
                if state['busy']:finish(assembly.response(),'Interrupted stream: '+type(exc).__name__)
                raise
            finally:await response.aclose();await client.aclose()
        return StreamingResponse(relay(),media_type='text/event-stream')
    return app
