"""Thin read-only adapters to upstream; isolated resource creation is explicit."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
import requests
from . import UPSTREAM, SCHEMA
from .core import StopRun, require, digest, canonical, read, rows, write, write_rows, safe_id, strip_reasoning, text, task_result_row
from .inference_limits import teacher_limits, validate_teacher_limits

BLOBS={
 'api/openwebui_bridge.py':'bbbbf9c1a7cc66bc57b493a9cc5d9820cdbb47b6',
 'lib/openwebui_client.py':'4f0bfa0857e9bf74515699d2fa1487dca531d344',
 'lib/bundle_common.py':'8cff1bab8b4f9c94bc4c076a55c61538d544778e',
 'scripts/generate_candidate_csv.py':'d285794255f02f1a3db2b529b188cc362c49f29a',
 'scripts/setup_openwebui.py':'9440a02495cae3ecc9ba2fe65ea73a5fdf241a3e'}

TASK_GENERATION_FIELDS = {
    'tool_parameters_json', 'summarizer_model_id', 'algorithm', 'target_length',
    'structure', 'generation_top_p', 'openwebui_extra_instructions',
}

def bundle() -> Path:"""Locate the original OpenWebUI evaluation bundle."""; return Path(__file__).resolve().parents[1]

def audit() -> dict[str, Any]:
    """Check the five original upstream files against the audited contract."""
    report={}
    for name,expected in BLOBS.items():
        p=bundle()/name;b=p.read_bytes() if p.is_file() else None
        actual=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest() if b is not None else None
        report[name]={'expected':expected,'actual':actual}
    return {'upstream_contract_commit':UPSTREAM,'ok':all(x['actual']==x['expected'] for x in report.values()),'files':report}

def upstream(relative: str) -> Any:
    """Load an existing bundle module without replacing its implementation."""
    p=bundle()/relative;require(p.is_file(),'Original repository file is missing: '+relative)
    if str(bundle()) not in sys.path:sys.path.insert(0,str(bundle()))
    if 'lib' in sys.modules:
        paths=list(getattr(sys.modules['lib'],'__path__',[]))
        require(not paths or str(bundle()/'lib') in paths,'Another lib module shadows upstream; start fresh process')
    name='_sa_upstream_'+relative.replace('/','_').replace('.','_')
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,p);mod=importlib.util.module_from_spec(spec)
        sys.modules[name]=mod;spec.loader.exec_module(mod)
    return sys.modules[name]

def load_env(path: str | Path) -> None:
    """Load the authorized environment file without overriding existing variables."""
    for k,v in upstream('scripts/setup_openwebui.py').read_env_file(Path(path)).items():os.environ.setdefault(k,v)

def clients(cache: Any=None) -> Any:
    """Create original clients after rejecting unrelated production defaults."""
    require(os.environ.get('GENERATION_BACKEND','openwebui')=='openwebui','Agentic run requires OpenWebUI')
    for k in ('SUMMARIZER_MODEL_ID','ALGORITHM','TARGET_LENGTH','STRUCTURE'):
        require(not os.environ.get('OPENWEBUI_DEFAULT_'+k),'Unrelated summarizer default: '+k)
    c=upstream('lib/openwebui_client.py').OpenWebUIClient(cache_path=cache)
    a=upstream('scripts/setup_openwebui.py').OpenWebUIAdmin(c.base_url);a.token=c.api_key
    return c,a

def inventory(admin: Any) -> dict[str, Any]:
    """Read the visible OpenWebUI model inventory."""
    data=admin._request('GET','/api/models').json();require(isinstance(data.get('data'),list),'Unknown model inventory schema')
    return {m['id']:m for m in data['data']}

def get_workspace_model(admin: Any, model_id: str) -> dict[str, Any]:
    """Read the full workspace policy; the visible inventory may mask model parameters."""
    require(isinstance(model_id,str) and model_id,'Workspace model ID required')
    response=admin._request('GET','/api/v1/models/model',params={'id':model_id},expected=(200,404))
    require(response.status_code==200,'Workspace model missing')
    info=response.json()
    require(isinstance(info,dict) and info.get('id')==model_id,'Workspace model response identity mismatch')
    require(isinstance(info.get('params'),dict) and isinstance(info.get('meta'),dict),
            'Workspace model policy or metadata missing')
    return info

def rag_fingerprint(admin: Any, tool_ids: Any, params: Any) -> str:
    """Hash tool code and Valves without writing their potentially secret contents."""
    state={'params':params,'tools':[]}
    for tid in tool_ids:
        safe_id(tid)
        info=admin._request('GET',f'/api/v1/tools/id/{tid}').json()
        valves=admin._request('GET',f'/api/v1/tools/id/{tid}/valves').json()
        # Hash only; source/valves may contain secrets and are not serialized to reports.
        state['tools'].append({'id':tid,'code':info.get('content'),'specs':info.get('specs'),'valves':valves})
    return digest(state)

def provision(cases: list[dict[str, Any]], output: Path, spec: dict[str, Any],
              apply: bool=False) -> dict[str, Any]:
    """Create new split KBs, optionally scoped to explicit cases and their full source paths."""
    require(not output.exists(),'Provisioning manifest exists; inspect partial resources rather than retry')
    namespace=safe_id(spec['namespace']);require(namespace.startswith('sa-'),'Use sa-* namespace')
    splits=spec.get('splits',['train','dev','test'])
    require(isinstance(splits,list) and splits,'splits must be a nonempty list')
    require(all(isinstance(split,str) and split in {'train','dev','test'} for split in splits),'Unknown provisioning split')
    require(len(set(splits))==len(splits),'splits must be unique')
    splits=[split for split in ('train','dev','test') if split in splits]
    selected=[case for case in cases if case['split'] in splits]
    explicit='splits' in spec or 'case_ids' in spec
    if explicit:
        require(all(isinstance(case.get('case_id'),str) and case['case_id'] for case in selected),'Selected source cases require nonempty case IDs')
        by_id={case['case_id']:case for case in selected}
        require(len(by_id)==len(selected),'Duplicate case IDs in the selected splits')
    if 'case_ids' in spec:
        case_ids=spec['case_ids']
        require(isinstance(case_ids,list) and case_ids,'case_ids must be a nonempty list')
        require(all(isinstance(case_id,str) and case_id for case_id in case_ids),'case_ids must contain nonempty strings')
        require(len(set(case_ids))==len(case_ids),'case_ids must be unique')
        require(set(case_ids)<=by_id.keys(),'case_ids must all belong to the selected splits')
        selected=[by_id[case_id] for case_id in sorted(case_ids)]
    paths_by_split={split:sorted({path for case in selected if case['split']==split for path in case['source_paths']}) for split in splits}
    require(all(paths_by_split.values()),'Empty split source corpus')
    c,a=clients(output.parent/'upload-cache.json');models=inventory(a)
    require(spec['blueprint'] in models and spec['base_model_id'] in models,'Blueprint/base model not visible')
    info=get_workspace_model(a,spec['blueprint']);meta=copy.deepcopy(info['meta'])
    tools=meta.get('toolIds') or []
    mode=spec.get('retrieval_mode','blueprint')
    require(mode in {'blueprint','split_bound'},'Unknown retrieval mode')
    bound=mode=='split_bound'
    require((tools or bound) and set(tools)==set(spec['allowed_tool_ids']),'Explicit exact blueprint tool allowlist required')
    if bound:
        require(spec.get('isolated_instance_reviewed') is True,'Split-bound tools require a dedicated reviewed test instance')
        require(spec['blueprint'].startswith('sa-'),'Split-bound retrieval requires a test blueprint')
    # Carry policy, NOT teacher-specific reasoning/sampling defaults.
    params={'system':(info.get('params') or {}).get('system',''),'function_calling':'native'}
    fp=None if bound else rag_fingerprint(a,tools,params)
    manifest={'schema':SCHEMA,'status':'planned','namespace':namespace,'base_model_id':spec['base_model_id'],
              'tool_ids':[] if bound else tools,'rag_params':params,'rag_fingerprint':fp,'splits':{}}
    if explicit:
        manifest['selection']={'scope':'explicit_case_selection' if 'case_ids' in spec else 'explicit_split_selection',
            'splits':splits,'case_ids':sorted(case['case_id'] for case in selected),
            'cases_hash':digest(sorted(selected,key=lambda case:case['case_id'])),
            'source_scope':'complete_case_source_paths'}
    if bound:
        manifest.update(retrieval_mode=mode,blueprint_hash=digest(info),created_tool_ids=[])
        existing=a._request('GET','/api/v1/tools/').json()
        require(isinstance(existing,list),'Unexpected tool inventory schema')
        existing_ids={item['id'] for item in existing}
    for split in splits:
        mid=namespace+'-'+split;require(mid not in models,'Refuse to replace workspace model')
        paths=paths_by_split[split]
        manifest['splits'][split]={'model_id':mid,'source_paths':paths,'file_ids':[]}
        if bound:
            from .split_retrieval import render_tool
            render_tool('sa-preflight',{f'file-{i}':Path(path).stem for i,path in enumerate(paths)},
                        count=spec.get('retrieval_count',3),max_output_chars=spec.get('max_retrieval_chars',6000))
            # Open WebUI requires Python identifiers and lowercases tool IDs.
            tid=safe_id(mid.replace('-','_')+'_retrieval')
            require(tid==tid.lower(),'Open WebUI retrieval tool IDs must be lowercase')
            require(tid not in existing_ids,'Refuse to replace retrieval tool')
            manifest['splits'][split]['tool_ids']=[tid]
    if not apply:return manifest
    require(spec.get('source_rights_reviewed') is True and spec.get('read_only_tools_reviewed') is True,'Review source rights and tool side effects before upload')
    manifest['status']='creating';write(output,manifest)
    for split,entry in manifest['splits'].items():
        entry['operation_pending']='create KB';write(output,manifest)
        kb=a._request('POST','/api/v1/knowledge/create',json={'name':namespace+'-'+split,'description':'Isolated Strategy A corpus; no QA references'}).json()
        require(kb.get('id'),'KB creation returned no id');entry['knowledge_id']=kb['id'];write(output,manifest)
        file_sources={}
        for path in entry['source_paths']:
            entry['operation_pending']='upload source';write(output,manifest)
            fid=c.upload_file(Path(path));entry['file_ids'].append(fid);write(output,manifest)
            if bound:file_sources[fid]=Path(path).stem
            a._request('POST',f"/api/v1/knowledge/{kb['id']}/file/add",json={'file_id':fid})
        m=copy.deepcopy(meta);m['knowledge']=[{'id':kb['id'],'name':namespace+'-'+split,'type':'collection'}]
        if bound:
            from .split_retrieval import render_tool
            tid=entry['tool_ids'][0]
            content=render_tool(kb['id'],file_sources,count=spec.get('retrieval_count',3),
                                max_output_chars=spec.get('max_retrieval_chars',6000))
            entry['operation_pending']='create split retrieval tool';write(output,manifest)
            a._request('POST','/api/v1/tools/create',json={'id':tid,'name':entry['model_id']+'-retrieval','content':content,
                'meta':{'description':'Read-only retrieval bound to one Strategy A split'},'access_grants':[]})
            manifest['created_tool_ids'].append(tid);manifest['tool_ids'].append(tid)
            entry['file_sources']=file_sources;write(output,manifest)
            m['toolIds']=[tid];m['builtinTools']={}
            m['capabilities']={**(m.get('capabilities') or {}),'builtin_tools':False,'file_context':False}
        entry['operation_pending']='create workspace model';write(output,manifest)
        a._request('POST','/api/v1/models/create',json={'id':entry['model_id'],'name':entry['model_id'],
            'base_model_id':spec['base_model_id'],'params':params,'meta':m,'access_grants':[],'is_active':True})
        entry['operation_pending']=None;entry['created']=True;write(output,manifest)
    if bound:manifest['rag_fingerprint']=rag_fingerprint(a,manifest['tool_ids'],params)
    manifest['status']='complete';write(output,manifest);return manifest

def verify_split_model(admin: Any, resources: dict[str, Any], split: str) -> list[str]:
    """Reject changed tool bindings or automatic retrieval on split-bound aliases."""
    entry=resources['splits'][split]
    tools=entry.get('tool_ids',resources['tool_ids'])
    if resources.get('retrieval_mode')=='split_bound':
        require(entry['model_id'] in inventory(admin),'Split workspace model not visible')
        info=get_workspace_model(admin,entry['model_id']);meta=info['meta']
        capabilities=meta.get('capabilities') or {}
        require(info.get('base_model_id')==resources['base_model_id'],'Split provider model changed')
        require(meta.get('toolIds')==tools,'Split tool binding changed')
        require(capabilities.get('builtin_tools') is False and capabilities.get('file_context') is False,
                'Automatic tools or file context are enabled on split-bound model')
        knowledge=meta.get('knowledge') or []
        require(len(knowledge)==1 and knowledge[0].get('id')==entry['knowledge_id'] and
                knowledge[0].get('type')=='collection','Split knowledge binding changed')
        require(info.get('params')==resources['rag_params'],'Split model policy changed')
    return tools

def bind(resources: dict[str, Any], output: Path, namespace: str, base_model_id: str,
         apply: bool=False, retrieval_spec: dict[str, Any] | None=None) -> dict[str, Any]:
    """Clone an isolated alias, optionally creating a new scoped search/read tool profile."""
    safe_id(namespace);require(namespace.startswith('sa-') and not output.exists(),'New sa-* namespace/manifest required')
    _,a=clients();models=inventory(a);require(base_model_id in models,'Student endpoint not visible')
    require(rag_fingerprint(a,resources['tool_ids'],resources['rag_params'])==resources['rag_fingerprint'],'RAG implementation drift')
    out=copy.deepcopy(resources);out['namespace']=namespace;out['base_model_id']=base_model_id;out['status']='planned'
    tool_bodies={}
    if retrieval_spec is not None:
        from .split_retrieval import render_tool
        require(resources.get('retrieval_mode')=='split_bound' and resources['namespace'].startswith('sa-'),
                'A source reader requires existing isolated split-bound resources')
        require(isinstance(retrieval_spec,dict) and set(retrieval_spec)=={
            'isolated_instance_reviewed','retrieval_count','max_retrieval_chars','max_source_chars'},
            'Source reader profile requires explicit review and all three limits')
        require(retrieval_spec['isolated_instance_reviewed'] is True and retrieval_spec['max_source_chars'] is not None,
                'Source reader requires a reviewed isolated instance and an explicit page budget')
        existing=a._request('GET','/api/v1/tools/').json()
        require(isinstance(existing,list),'Unexpected tool inventory schema')
        existing_ids={item['id'] for item in existing}
        out.update(tool_ids=[],created_tool_ids=[],retrieval_spec=copy.deepcopy(retrieval_spec),
                   reused_resource_namespace=resources['namespace'],rag_fingerprint=None)
        out['rag_params']['system']+='\n\nIf a useful search excerpt is incomplete, call read_source with its exact source_marker to read that source page before deciding. Read only relevant sources; the episode tool budget includes both search and read_source calls.'
    bodies={}
    for split,entry in out['splits'].items():
        mid=namespace+'-'+split;require(mid not in models,'Do not overwrite existing alias')
        require(resources['splits'][split]['model_id'] in models,'Source workspace model missing')
        info=get_workspace_model(a,resources['splits'][split]['model_id'])
        meta=copy.deepcopy(info['meta'])
        if retrieval_spec is not None:
            verify_split_model(a,resources,split)
            tid=safe_id(mid.replace('-','_')+'_retrieval')
            require(tid.isidentifier() and tid==tid.lower() and tid not in existing_ids,'New lowercase retrieval tool ID required')
            content=render_tool(entry['knowledge_id'],entry.get('file_sources'),
                                count=retrieval_spec['retrieval_count'],
                                max_output_chars=retrieval_spec['max_retrieval_chars'],
                                max_source_chars=retrieval_spec['max_source_chars'])
            tool_bodies[split]={'id':tid,'name':mid+'-retrieval','content':content,
                'meta':{'description':'Read-only search and source reading bound to one Strategy A split'},'access_grants':[]}
            entry['tool_ids']=[tid];out['tool_ids'].append(tid);meta['toolIds']=[tid]
        bodies[split]={'id':mid,'name':mid,'base_model_id':base_model_id,'meta':meta,
                       'params':out['rag_params'],'access_grants':[],'is_active':True}
        entry['model_id']=mid
    if apply:
        out['status']='creating';write(output,out)
        for split,body in bodies.items():
            if split in tool_bodies:
                tool=tool_bodies[split];out['operation_pending']=tool['id'];write(output,out)
                a._request('POST','/api/v1/tools/create',json=tool)
                out['created_tool_ids'].append(tool['id']);write(output,out)
            out['operation_pending']=body['id'];write(output,out);a._request('POST','/api/v1/models/create',json=body)
        if retrieval_spec is not None:out['rag_fingerprint']=rag_fingerprint(a,out['tool_ids'],out['rag_params'])
        out['status']='complete';out['operation_pending']=None;write(output,out)
    return out

def recorder_admin(url: str, op: str, payload: Any=None) -> dict[str, Any]:
    """Send one authenticated recorder operation without automatic retries."""
    r=requests.post(url.rstrip('/')+'/admin/'+op,headers={'Authorization':'Bearer '+os.environ['SA_ADMIN_TOKEN']},json=payload or {},timeout=30)
    require(r.ok,'Recorder admin failed; review pending session without blind retry');return r.json()

def begin_capture(url: str, benchmark_id: str, model: str, question: str,
                  limits: dict[str, Any]) -> None:
    """Require the recorder to acknowledge the exact resolved budget before any inference."""
    limits=validate_teacher_limits(limits)
    result=recorder_admin(url,'begin',{'benchmark_id':benchmark_id,'model':model,'question':question,
                                     'inference_limits':limits})
    require(result.get('inference_limits')==limits,'Recorder did not acknowledge inference limits; reconcile session before retry')


def teacher_extras(spec: dict[str, Any]) -> dict[str, Any]:
    """Preserve explicit model settings; inject reasoning only when the operator configures it."""
    require(spec.get('model_extra') is None or isinstance(spec['model_extra'],dict),'Invalid model extras')
    extras=copy.deepcopy(spec.get('model_extra') or {})
    require(isinstance(extras,dict) and not set(extras)&{'model','messages','tools','tool_ids','files','stream','max_tokens','max_completion_tokens'},
            'Model extras override protected controls')
    if 'reasoning' not in extras and os.environ.get('SA_TEACHER_REASONING_EFFORT'):
        effort=os.environ['SA_TEACHER_REASONING_EFFORT']
        require(effort in {'none','minimal','low','medium','high','xhigh','max'},'Unsupported teacher reasoning effort')
        extras['reasoning']={'effort':effort}
    return extras

def validate_native_bridge_urls(bridge_url: str, state_url: str) -> None:
    """Bind the state barrier to the exact generation origin before any client or artifact."""
    require(isinstance(bridge_url,str) and isinstance(state_url,str),'Invalid native bridge URLs')
    try:
        bridge=urlsplit(bridge_url)
        valid=bridge.scheme in {'http','https'} and bool(bridge.hostname) and not bridge.username and not bridge.password
        bridge.port  # Validate malformed and out-of-range ports.
    except ValueError:
        raise StopRun('Invalid native bridge URLs') from None
    origin=f'{bridge.scheme}://{bridge.netloc}'
    require(valid and bridge_url==origin+'/generate' and state_url==origin+'/strategy-a/native-state',
            'Native bridge state URL must match the exact generation origin')

def native_bridge_state(url: str) -> dict[str, Any]:
    """Read the private native task barrier before opening or closing a recorder session."""
    require(isinstance(url,str) and url.endswith('/strategy-a/native-state'),'Explicit native bridge state URL required')
    response=requests.get(url,headers={'Authorization':'Bearer '+os.environ['SA_ADMIN_TOKEN']},timeout=15,allow_redirects=False)
    try:
        require(response.status_code==200,'Native bridge state unavailable; keep recorder session open')
        state=response.json()
    finally:
        response.close()
    require(isinstance(state,dict) and state.get('transport')=='socketio-native-v1' and
            type(state.get('safe_to_end')) is bool and type(state.get('can_begin')) is bool,
            'Invalid native bridge state; keep recorder session open')
    return state

def collect(cases: list[dict[str, Any]], resources: dict[str, Any], run: Path,
            spec: dict[str, Any]) -> dict[str, Any]:
    """Collect one split, preserving explicit case_ids order when that selection is supplied."""
    require(not run.exists(),'Use a fresh run directory')
    require(resources['status']=='complete','Resource provisioning incomplete')
    inference=teacher_limits(spec);extras=teacher_extras(spec)
    split=spec['split'];require(split in {'train','dev','test'},'Unknown split');repeats=int(spec.get('repeats',1))
    require(0<repeats<=4 and (split=='train' or repeats==1),'Holdouts are pass@1')
    require(split!='test' or spec.get('unseal_test') is True,'Test is sealed')
    deployment=spec.get('deployment')
    if split!='train':
        require(isinstance(deployment,dict) and set(deployment)=={'model_id','revision','runtime_image','weights_hash','kv_cache_dtype'},'Holdout requires a frozen deployment identity (same base/quantization/runtime except adapter)')
        require(all(isinstance(v,str) and v for v in deployment.values()),'Incomplete deployment identity')
    selected=[c for c in cases if c['split']==split]
    if 'case_ids' in spec:
        case_ids=spec['case_ids']
        require(isinstance(case_ids,list) and case_ids,'case_ids must be a nonempty list')
        require(all(isinstance(case_id,str) and case_id for case_id in case_ids),'case_ids must contain nonempty strings')
        require(len(set(case_ids))==len(case_ids),'case_ids must be unique')
        require(spec.get('max_cases') is None,'case_ids and max_cases are mutually exclusive')
        by_id={case['case_id']:case for case in selected}
        require(len(by_id)==len(selected),'Duplicate case IDs in the selected split')
        require(set(case_ids)<=by_id.keys(),'case_ids must all belong to the selected split')
        selected=[by_id[case_id] for case_id in case_ids]
    elif spec.get('max_cases') is not None:
        require(type(spec['max_cases']) is int and spec['max_cases']>0,'max_cases must be a positive integer')
        selected=selected[:spec['max_cases']]
    require(selected,'No cases selected')
    if 'native_bridge_state_url' in spec:
        validate_native_bridge_urls(spec.get('bridge_url'),spec['native_bridge_state_url'])
    _,a=clients();require(rag_fingerprint(a,resources['tool_ids'],resources['rag_params'])==resources['rag_fingerprint'],'Tools/Valves/policy drift')
    split_tools=verify_split_model(a,resources,split)
    generate=upstream('scripts/generate_candidate_csv.py').generate_row
    csv=upstream('lib/bundle_common.py').write_csv_rows
    run.mkdir(parents=True);run.chmod(0o700);(run/'responses').mkdir();(run/'captures').mkdir()
    res=resources['splits'][split];limits=inference['evaluation_limits']
    protocol={'split':split,'repeats':repeats,'rag_fingerprint':resources['rag_fingerprint'],'knowledge_id':res['knowledge_id'],
              'tool_ids':split_tools,'model_extra':extras,'max_output_tokens':inference['max_output_tokens'],
              'temperature':float(spec.get('temperature',.6)),'deployment':deployment,'evaluation_policy':spec.get('evaluation_policy','qa-v3'),'evaluation_limits':limits}
    protocol['inference_limits']=inference
    if spec.get('native_bridge_state_url'):protocol['native_transport']='socketio-native-v1'
    write(run/'selection.json',{'cases_hash':digest(selected),'case_ids':[c['case_id'] for c in selected],
                              'protocol':protocol,'protocol_hash':digest(protocol)})
    write_rows(run/'cases.jsonl',selected)
    results=[];csv_rows=[]
    for case in selected:
        for rep in range(repeats):
            bid='sa-'+digest([str(run.resolve()),case['case_id'],rep])[:32]
            row={'task_id':case['case_id'],'benchmark_id':bid,'dataset':case['dataset'],
                 'query':case['question'],'request_prompt':case['question'],'openwebui_pipe_model':res['model_id'],
                 'kb_ids_json':canonical([res['knowledge_id']]),'openwebui_tool_ids_json':canonical(split_tools),
                 'source_paths_json':'[]','openwebui_include_trace':'true','generation_max_tokens':str(protocol['max_output_tokens']),
                 'generation_temperature':str(protocol['temperature']),
                 'openwebui_model_params_json':canonical({'model_extra_payload_json':extras})}
            fields = case.get('generation_fields', {})
            require(isinstance(fields, dict) and set(fields) <= TASK_GENERATION_FIELDS,
                    'Task generation fields override protected collection controls')
            row.update(fields)
            if spec.get('native_bridge_state_url'):
                native_state=native_bridge_state(spec['native_bridge_state_url'])
                require(native_state['can_begin'],
                        'Reconcile the prior native operation before another capture')
                require(type(native_state.get('episode_timeout_seconds')) is int
                        and native_state['episode_timeout_seconds']==limits['max_seconds'],
                        'Native episode timeout differs from the pinned spec; configure the private bridge before capture')
            begin_capture(spec['recorder_url'],bid,spec['upstream_model'],case['question'],inference)
            result=None
            try:
                result=generate(index=len(results)+1,row=row,bridge_url=spec['bridge_url'],timeout=inference['collection_timeout_seconds'],
                                overwrite=False,responses_dir=run/'responses',documents_dir=None)
            finally:
                if spec.get('native_bridge_state_url'):
                    try:
                        native_state=native_bridge_state(spec['native_bridge_state_url'])
                    except Exception as exc:
                        native_state={'safe_to_end':False,'error_type':type(exc).__name__}
                    if native_state.get('safe_to_end') is not True:
                        write(run/'RECONCILIATION_REQUIRED.json',{'benchmark_id':bid,'native_state':native_state,
                            'recorder_session_retained':True,'signal':'Native task inactivity unconfirmed; no automatic retry'})
                        raise StopRun('Native task inactivity unconfirmed; recorder session remains open')
                cap=recorder_admin(spec['recorder_url'],'end');write(run/'captures'/f'{bid}.json',cap)
            require(result is not None and cap['complete'],'Incomplete episode; inspect before retry')
            msg=cap['records'][-1]['response']['choices'][0]['message'];require(not msg.get('tool_calls'),'Missing final answer')
            answer=strip_reasoning(text(msg.get('content'))).strip();require(answer,'Empty answer')
            candidate={'schema':SCHEMA,'benchmark_id':bid,'case_id':case['case_id'],'candidate_answer':answer,
                       'ui_answer':result['candidate_answer'],'capture_path':f'captures/{bid}.json','capture_hash':digest(cap),
                       'model':spec['upstream_model'],'kind':'episode'}
            results.append(candidate)
            csv_rows.append(task_result_row(case, candidate, cap) if case.get('task_metadata')
                            else {'benchmark_id':bid,'candidate_answer':answer})
            write_rows(run/'candidates.jsonl',results);csv(run/'candidates.csv',csv_rows)
    write(run/'COMPLETE.json',{'count':len(results),'candidates_hash':digest(results)})
    return {'count':len(results),'run':str(run)}


def branch(cases: Any,source_runs: Any,run: Path,spec: Any) -> dict[str, Any]:
    """Terminal alternatives on identical observed train states, never an E2E score."""
    from .evaluation import loaded
    from .core import normalize
    require(not run.exists(),'Fresh branch directory required')
    inference=teacher_limits(spec);extras=teacher_extras(spec)
    repeats=int(spec.get('repeats',2));require(1<=repeats<=4,'Bounded branch count required')
    cm={c['case_id']:c for c in cases};run.mkdir(parents=True);run.chmod(0o700);(run/'captures').mkdir()
    cands=[];seen=set();selected={};csv=upstream('lib/bundle_common.py').write_csv_rows
    for source in source_runs:
        original,_,captures,_=loaded(source)
        for candidate in original:
            case=cm[candidate['case_id']]
            if case['split']!='train':continue
            cap=captures[candidate['benchmark_id']];request=cap['records'][-1]['request']
            state={'messages':normalize(request['messages']),'tools':request.get('tools',[])}
            key=digest(state)
            if key in seen:continue
            if len(seen)>=int(spec.get('max_states',30)):break
            seen.add(key);selected[case['case_id']]=case
            for i in range(repeats):
                bid='sa-'+digest([str(run.resolve()),key,i])[:32]
                body={**state,'model':spec['upstream_model'],'temperature':.7,'max_tokens':inference['max_output_tokens'],
                      'stream':False,**extras}
                begin_capture(spec['recorder_url'],bid,body['model'],case['question'],inference)
                response=None
                try:
                    response=requests.post(spec['recorder_url'].rstrip('/')+'/v1/chat/completions',json=body,
                        headers={'Authorization':'Bearer '+os.environ['SA_PROXY_TOKEN']},timeout=inference['request_timeout_seconds'])
                finally:
                    cap=recorder_admin(spec['recorder_url'],'end');write(run/'captures'/f'{bid}.json',cap)
                require(response is not None and response.ok and cap['complete'],'Incomplete/ambiguous branch; do not retry blindly')
                msg=cap['records'][-1]['response']['choices'][0]['message']
                require(not msg.get('tool_calls'),'Terminal branch requested another tool: not valid for this preference pilot')
                answer=strip_reasoning(text(msg.get('content'))).strip();require(answer,'Empty terminal candidate')
                cands.append({'schema':SCHEMA,'benchmark_id':bid,'case_id':case['case_id'],'candidate_answer':answer,
                    'capture_path':f'captures/{bid}.json','capture_hash':digest(cap),'model':spec['upstream_model'],'kind':'branch'})
                write_rows(run/'candidates.jsonl',cands);csv(run/'candidates.csv',[{'benchmark_id':c['benchmark_id'],'candidate_answer':c['candidate_answer']} for c in cands])
    require(cands,'No train state available')
    selected=list(selected.values());protocol={'split':'train','repeats':repeats,'kind':'terminal_state_branch',
        'model_extra':extras,'evaluation_limits':inference['evaluation_limits'],'inference_limits':inference}
    write_rows(run/'cases.jsonl',selected);write(run/'selection.json',{'cases_hash':digest(selected),'protocol':protocol,'protocol_hash':digest(protocol)})
    write(run/'COMPLETE.json',{'count':len(cands),'candidates_hash':digest(cands)})
    return {'branches':len(cands),'states':len(seen),'not_end_to_end':True}
