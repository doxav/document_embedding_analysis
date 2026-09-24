"""Offline tests. Providers/GPU frameworks are not claimed to be live-tested."""

from __future__ import annotations

import copy

import json

import sys

import time

import types

from pathlib import Path

from typing import Any

import pytest

ROOT=Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from scripts.promptfoo_openwebui_eval.posttrain import core as c, SCHEMA

from scripts.promptfoo_openwebui_eval.posttrain.budget import Ledger

from scripts.promptfoo_openwebui_eval.posttrain import evaluation, vidore, integration

from scripts.promptfoo_openwebui_eval.posttrain.recorder import SSE,app_factory

P=ROOT/'scripts/promptfoo_openwebui_eval/posttrain'

REV='2'*40

TOOLS=[{'type':'function','function':{'name':'search','parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query'],'additionalProperties':False}}}]

REASONING_TAGS = (
    ('<think>', '</think>'), ('<thinking>', '</thinking>'),
    ('<reason>', '</reason>'), ('<reasoning>', '</reasoning>'),
    ('<thought>', '</thought>'), ('<Thought>', '</Thought>'),
    ('<|begin_of_thought|>', '<|end_of_thought|>'), ('◁think▷', '◁/think▷'),
    ('[THINK]', '[/THINK]'),
)


def case(split='train',cid='q1',group='doc1'):
    return {'case_id':cid,'group_id':group,'split':split,'domain':'energy','language':'fr','dataset':'test','document_ids':[group],
            'question':'Quel est le revenu annuel ?','reference_answer':'42 EUR','relevant_source_markers':[c.source_marker('test','p1')],'requires_retrieval':True}


def record(answer='42 EUR'):
    return {'schema':SCHEMA,'source':'upstream_wire','complete':True,'error':None,
        'request':{'model':'teacher','tools':copy.deepcopy(TOOLS),'messages':[
            {'role':'user','content':case()['question']},
            {'role':'assistant','content':'','reasoning_content':'PRIVATE','tool_calls':[{'id':'call1','type':'function','function':{'name':'search','arguments':'{"query":"revenu"}'}}]},
            {'role':'tool','tool_call_id':'call1','content':'Source-ID: '+c.source_marker('test','p1')+'\nLe revenu est 42 EUR.'}]},
        'response':{'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':answer,'reasoning_content':'PRIVATE'}}],
                    'usage':{'total_tokens':100,'completion_tokens':10,'cost':.01}}}


def capture(answer='42 EUR'):
    return {'schema':SCHEMA,'complete':True,'duration_seconds':1,'records':[record(answer)]}


def candidate(cap,bid='b1',kind='episode'):
    return {'benchmark_id':bid,'case_id':'q1','candidate_answer':cap['records'][-1]['response']['choices'][0]['message']['content'],
            'capture_hash':c.digest(cap),'model':'teacher','kind':kind,'capture_path':'captures/'+bid+'.json'}


def grade(cand,passed=True,hard=True):
    return {'benchmark_id':cand['benchmark_id'],'pass':passed and hard,'hard_ok':hard,'score':1.0 if passed else .2,
            'answer_hash':c.digest(cand['candidate_answer']),'capture_hash':cand['capture_hash'],
            'metrics':{k:{'pass':hard if k in c.HARD else passed,'score':float(hard if k in c.HARD else passed),'reason':'fixture'} for k in c.METRICS}}


def export(cand,passed=True):
    g=grade(cand,passed)
    return {'testCase':{'vars':{'benchmark_id':cand['benchmark_id'],'candidate_answer':cand['candidate_answer']}},'success':passed,
            'gradingResult':{'pass':passed,'componentResults':[{**g['metrics'][k],'assertion':{'metric':k}} for k in c.METRICS]}}


def ledger(tmp,seconds=36000):
    return Ledger.initialize(tmp/'budget.sqlite',{'seconds':seconds,'usd':100,'tokens':1000000,'calls':100})


@pytest.mark.parametrize('bad',['{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}'])
def test_strict_json(bad):
    with pytest.raises(c.StopRun):c.strict_json(bad)


@pytest.mark.parametrize('path',['../escape','/etc/passwd'])
def test_path_traversal(tmp_path,path):
    with pytest.raises(c.StopRun):c.inside(tmp_path,path)


def test_symlink_refused(tmp_path):
    (tmp_path/'link').symlink_to('/etc/passwd')
    with pytest.raises(c.StopRun):c.inside(tmp_path,'link')


@pytest.mark.parametrize('opening,closing', REASONING_TAGS)
def test_reasoning_only_removed_from_assistant(opening: str, closing: str) -> None:
    """Strip every native reasoning format from targets and conditioning, preserving tools."""
    r=record();literal=f' literal {opening}quoted{closing}'
    r['request']['messages'][0]['content'] += literal
    r['request']['messages'][1]['content'] = f'{opening}PRIVATE CONTEXT{closing}'
    r['request']['messages'][-1]['content'] += literal
    r['response']['choices'][0]['message']['content'] = f'{opening}PRIVATE TARGET{closing}42 EUR'
    ex=c.turn_example(r,case())
    assert 'PRIVATE' not in c.canonical(ex)
    assert ex['messages'][0]['content'] == r['request']['messages'][0]['content']
    assert ex['messages'][-2]['content'] == r['request']['messages'][-1]['content']
    assert ex['messages'][-1]['content'] == '42 EUR'
    assert [m['train'] for m in ex['messages']]==[False,False,False,True]
    assert ex['messages'][1]['tool_calls'][0]['id']==ex['messages'][2]['tool_call_id']=='t00000001'
    assert isinstance(ex['tools'][0]['function']['parameters'],str)


@pytest.mark.parametrize('opening,closing', REASONING_TAGS)
@pytest.mark.parametrize('kind', ['unclosed', 'orphan', 'embedded', 'nested', 'attributes', 'mismatched'])
def test_bad_reasoning(opening: str, closing: str, kind: str) -> None:
    """Reject incomplete, nested, nonleading or unsupported reasoning markup explicitly."""
    value = {
        'unclosed': opening + 'unfinished',
        'orphan': closing + 'orphan',
        'embedded': 'answer ' + opening + 'quoted' + closing,
        'nested': opening + opening + 'nested' + closing + closing,
        'attributes': opening[:-1] + ' attr' + opening[-1] + 'hidden' + closing,
        'mismatched': opening + 'hidden</INVALID>',
    }[kind]
    with pytest.raises(c.StopRun):c.strip_reasoning(value)


@pytest.mark.parametrize('value', ['', 'Answer', 'Code <thinking_time>literal</thinking_time>.'])
def test_reasoning_leaves_unrelated_text(value: str) -> None:
    """Keep plain text and unrelated markup unchanged."""
    assert c.strip_reasoning(value) == value


def test_reasoning_multiple_prefixes() -> None:
    """Strip consecutive complete scratchpads and preserve their following answer."""
    assert c.strip_reasoning(' \n<thinking>first</thinking>\n[THINK]second[/THINK]\nAnswer') == 'Answer'


@pytest.mark.parametrize('value', [None, 7])
def test_reasoning_rejects_nontext(value: object) -> None:
    """Report invalid content types rather than leaking an incidental string error."""
    with pytest.raises(c.StopRun, match='must be text'):
        c.strip_reasoning(value)


@pytest.mark.parametrize('bad',['orphan','wrong_args','unknown_tool','truncated','debug','image','missing_observation'])
def test_invalid_trace(bad):
    r=record()
    if bad=='orphan':r['request']['messages'][-1]['tool_call_id']='orphan'
    if bad=='wrong_args':r['request']['messages'][1]['tool_calls'][0]['function']['arguments']='{"query":7}'
    if bad=='unknown_tool':r['request']['tools']=[]
    if bad=='truncated':r['response']['choices'][0]['finish_reason']='length'
    if bad=='debug':r['source']='bridge_summary'
    if bad=='image':r['request']['messages'][0]['content']=[{'type':'image_url'}]
    if bad=='missing_observation':r['request']['messages'].pop()
    with pytest.raises(c.StopRun):c.turn_example(r,case())


def test_reasoning_sft_preserves_selection_context_and_capture() -> None:
    """Restore only captured assistant reasoning, preserving tools, labels and source bytes."""
    cap = capture()
    cap['records'][0]['request']['messages'].insert(0, {'role': 'system', 'content': 'Policy'})
    cap['records'][0]['request']['messages'][-1]['content'] += '\n<think>source text</think>'
    target = cap['records'][0]['response']['choices'][0]['message']
    target['reasoning'] = target['reasoning_content']
    target['reasoning_details'] = [{'type': 'reasoning.text', 'text': 'PRI'},
                                   {'type': 'reasoning.text', 'text': 'VATE'}]
    original = copy.deepcopy(cap)
    base = [c.turn_example(cap['records'][0], case())]
    enriched, report = c.reasoning_sft(base, {'b1': cap})
    assert cap == original
    assert enriched[0]['capture_hash'] == base[0]['capture_hash']
    assert enriched[0]['tools'] == base[0]['tools']
    assert c.normalize(enriched[0]['messages']) == c.normalize(base[0]['messages'])
    assert [m['train'] for m in enriched[0]['messages']] == [m['train'] for m in base[0]['messages']]
    assert enriched[0]['messages'][-2] == base[0]['messages'][-2]
    assert enriched[0]['messages'][-1]['content'].count('PRIVATE') == 1
    assert report['assistant_messages_with_reasoning'] == 2
    assert report['target_indices_without_reasoning'] == []
    assert report['sft_hash'] == c.digest(base)
    assert report['sft_reasoning_hash'] == c.digest(enriched)


@pytest.mark.parametrize('message', [
    {'content': '<think>unclosed'},
    {'content': '<thinking>unreviewed</thinking>answer'},
    {'content': 'answer', 'reasoning': 12},
    {'content': 'answer', 'reasoning': 'a', 'reasoning_content': 'b'},
    {'content': '<think>a</think>answer', 'reasoning': 'b'},
    {'content': '<think>a</think><think>b</think>answer'},
    {'content': 'answer', 'reasoning_details': {}},
    {'content': 'answer', 'reasoning_details': [{'type': 'reasoning.encrypted', 'data': 'opaque'}]},
    {'content': 'answer', 'reasoning_details': [{'type': 'reasoning.text', 'text': 3}]},
    {'content': 'answer', 'reasoning': '<think>nested</think>'},
])
def test_reasoning_sft_rejects_ambiguous_text(message: dict[str, Any]) -> None:
    """Never invent text from opaque reasoning or silently choose conflicting representations."""
    with pytest.raises(c.StopRun):
        c.captured_reasoning_content(message)


def test_reasoning_sft_keeps_inline_and_reports_missing() -> None:
    """Keep historical inline text intact and make absent target reasoning explicit."""
    cap = capture()
    history = cap['records'][0]['request']['messages'][1]
    history['content'] = '<think>\nPRIVATE\n</think>\n\n'
    target = cap['records'][0]['response']['choices'][0]['message']
    target.pop('reasoning_content')
    base = [c.turn_example(cap['records'][0], case())]
    enriched, report = c.reasoning_sft(base, {'b1': cap})
    assert enriched[0]['messages'][1]['content'] == history['content']
    assert enriched[0]['messages'][-1] == base[0]['messages'][-1]
    assert report['target_indices_without_reasoning'] == [0]
    assert c.captured_reasoning_content({'content': 'answer', 'reasoning_details': [
        {'type': 'reasoning.text', 'text': 'captured'}]}) == '<think>\ncaptured\n</think>\n\nanswer'


@pytest.mark.parametrize('fault', ['capture', 'holdout', 'selection'])
def test_reasoning_sft_rejects_drift(fault: str) -> None:
    """The optional variant cannot bypass source identity or train-only selection."""
    cap = capture()
    base = [c.turn_example(cap['records'][0], case())]
    if fault == 'capture':
        cap['records'][0]['request']['messages'][0]['content'] += 'changed'
    elif fault == 'holdout':
        base[0]['split'] = 'test'
    else:
        base[0]['messages'][0]['content'] += 'changed'
    with pytest.raises(c.StopRun):
        c.reasoning_sft(base, {'b1': cap})


def test_build_reasoning_is_explicit_opt_in() -> None:
    """Existing builds keep their default; the extra export requires the named flag."""
    from scripts.promptfoo_openwebui_eval.posttrain.__main__ import parser
    args = ['build', '--dataset', 'dataset', '--runs', 'run', '--output', 'new-data']
    assert parser().parse_args(args).with_reasoning is False
    assert parser().parse_args(args + ['--with-reasoning']).with_reasoning is True


def test_parallel_tools():
    r=record();calls=r['request']['messages'][1]['tool_calls'];calls.append(copy.deepcopy(calls[0]));calls[1]['id']='call2'
    r['request']['messages'].append({'role':'tool','tool_call_id':'call2','content':'second'})
    assert c.turn_example(r,case())['messages'][-2]['tool_call_id']=='t00000002'


def test_grounding_never_sees_gold_or_oracle():
    q=case();q['reference_answer']='SECRET_GOLD';q['reference_context']='SECRET_ORACLE'
    p=evaluation.judge_payload('groundedness',q,'candidate',capture())
    assert 'SECRET_GOLD' not in c.canonical(p) and 'SECRET_ORACLE' not in c.canonical(p)
    assert 'retrieved_evidence' in p and 'reference_answer' not in p
    correctness=evaluation.judge_payload('correctness',q,'candidate',capture())
    assert 'retrieved_evidence' not in correctness and correctness['reference_answer']=='SECRET_GOLD'


def test_qrel_marker_must_be_in_final_received_tool_evidence():
    cap=capture();assert c.retrieval(cap,case())['pass']
    extra=copy.deepcopy(cap['records'][0]);extra['request']['messages'][-1]['content']='No relevant evidence'
    cap['records'].append(extra)
    assert not c.retrieval(cap,case())['pass'] # Earlier discarded evidence does not count.


def test_unknown_usage_and_no_judge_on_hard_failure(tmp_path,monkeypatch):
    cap=capture();cap['records'][0]['response'].pop('usage')
    assert not c.metrics(cap)['usage_known']
    hard=evaluation.hard_results(cap,case(),{})
    assert not hard['within_budget']['pass']
    cand=candidate(cap)
    monkeypatch.setattr(evaluation,'loaded',lambda run:([cand],{'q1':case()},{'b1':cap},{}))
    monkeypatch.setattr(evaluation,'judge',lambda *a,**k:pytest.fail('Paid judge should not execute'))
    result=evaluation.get_assert('42 EUR',{'config':{'run':str(tmp_path),'metric':'correctness'},'vars':{'benchmark_id':'b1'}})
    assert not result['pass'] and 'Withheld' in result['reason']


@pytest.mark.parametrize('change',['missing','duplicate','nan','string_pass','infra','stale','contradict'])
def test_import_rejects_bad_evaluation(change):
    cand=candidate(capture());row=export(cand);parts=row['gradingResult']['componentResults']
    if change=='missing':parts.pop()
    if change=='duplicate':parts.append(copy.deepcopy(parts[0]))
    if change=='nan':parts[0]['score']=float('nan')
    if change=='string_pass':parts[0]['pass']='true'
    if change=='infra':parts[0]['reason']='SA_INFRASTRUCTURE:timeout'
    if change=='stale':row['testCase']['vars']['candidate_answer']='other'
    if change=='contradict':row['success']=False
    with pytest.raises(c.StopRun):c.import_grades({'results':[row]},{'b1':cand})


def test_import_valid():
    cand=candidate(capture());assert c.import_grades({'results':{'results':[export(cand)]}},{'b1':cand})[0]['pass']


@pytest.mark.parametrize('passed',[True,False])
def test_import_promptfoo_explicit_failure_reason(passed: bool) -> None:
    """Promptfoo assertion failures remain valid grades, including semantic negatives."""
    cand=candidate(capture());row=export(cand,passed)
    row['failureReason']=0 if passed else 1
    row['gradingResult']['reason']='All assertions passed' if passed else 'Unsupported claim'
    row['error']=None if passed else row['gradingResult']['reason']
    result=c.import_grades({'results':{'results':[row]}},{'b1':cand})[0]
    assert result['pass'] is passed and result['hard_ok'] is True


@pytest.mark.parametrize('change',[
    'infrastructure','infrastructure_without_error','provider_error','legacy_error','different_error',
    'missing_reason','component_infrastructure','boolean_reason','string_reason','unknown_reason',
    'null_reason','none_for_failed','assert_for_passed','object_error','response_type',
    'aggregate_infrastructure','aggregate_reason_infrastructure','aggregate_error',
    'unknown_component_infrastructure','unknown_component_error',
])
def test_import_promptfoo_rejects_ambiguous_failure(change: str) -> None:
    """Explicit assertion status cannot conceal technical errors or inconsistent verdicts."""
    cand=candidate(capture());row=export(cand,False)
    row.update(failureReason=1,error='Unsupported claim')
    row['gradingResult']['reason']=row['error']
    if change=='infrastructure':row['failureReason']=2
    if change=='infrastructure_without_error':row.update(failureReason=2,error=None)
    if change=='provider_error':row['response']={'error':'Provider timeout'}
    if change=='legacy_error':row.pop('failureReason')
    if change=='different_error':row['error']='Network timeout'
    if change=='missing_reason':row['gradingResult'].pop('reason')
    if change=='component_infrastructure':row['gradingResult']['componentResults'][0]['reason']='SA_INFRASTRUCTURE:timeout'
    if change=='boolean_reason':row['failureReason']=True
    if change=='string_reason':row['failureReason']='1'
    if change=='unknown_reason':row['failureReason']=3
    if change=='null_reason':row['failureReason']=None
    if change=='none_for_failed':row.update(failureReason=0,error=None)
    if change=='assert_for_passed':row=export(cand,True);row['failureReason']=1
    if change=='object_error':row['error']={'message':'Unsupported claim'}
    if change=='response_type':row['response']=[]
    if change=='aggregate_infrastructure':row['error']=row['gradingResult']['reason']='SA_INFRASTRUCTURE:timeout'
    if change=='aggregate_reason_infrastructure':row['error']=None;row['gradingResult']['reason']='SA_INFRASTRUCTURE:timeout'
    if change=='aggregate_error':row['gradingResult']['error']='Timeout'
    if change=='unknown_component_infrastructure':row['gradingResult']['componentResults'].append({'metric':'other','reason':'SA_INFRASTRUCTURE:timeout'})
    if change=='unknown_component_error':row['gradingResult']['componentResults'].append({'metric':'other','error':'Timeout'})
    with pytest.raises(c.StopRun):c.import_grades({'results':[row]},{'b1':cand})


def test_import_promptfoo_hard_failure_cannot_become_training_data() -> None:
    """A recorded technical episode remains excluded even when its export imports successfully."""
    cap=capture();cand=candidate(cap);row=export(cand,False)
    row.update(failureReason=1,error='Episode budget exceeded')
    row['gradingResult']['reason']=row['error']
    part=next(p for p in row['gradingResult']['componentResults'] if p['assertion']['metric']=='within_budget')
    part['pass']=False;part['score']=0.0
    imported=c.import_grades({'results':[row]},{'b1':cand})
    assert imported[0]['hard_ok'] is False
    sft,pairs,_=c.build_data([case()],[cand],imported,{'b1':cap})
    assert sft==pairs==[]


def test_pairs_require_identical_observations_and_exclude_holdouts():
    good=capture();bad=capture('wrong');a=candidate(good);b=candidate(bad,'b2','branch')
    sft,pairs,report=c.build_data([case()],[a,b],[grade(a),grade(b,False)],{'b1':good,'b2':bad})
    assert len(sft)==len(pairs)==1 and not report['simpo_ready']
    bad['records'][0]['request']['messages'][-1]['content']='Different evidence';b=candidate(bad,'b2')
    assert not c.build_data([case()],[a,b],[grade(a),grade(b,False)],{'b1':good,'b2':bad})[1]
    assert not c.build_data([case('test')],[a],[],{})[0]


def test_error_does_not_become_negative():
    good=capture();bad=capture('wrong');a=candidate(good);b=candidate(bad,'b2')
    assert not c.build_data([case()],[a,b],[grade(a),grade(b,False,False)],{'b1':good,'b2':bad})[1]


def test_group_and_doc_isolation():
    with pytest.raises(c.StopRun):c.validate_cases([case(),case('test','q2')])
    other=case('test','q2','group2');other['document_ids']=['doc1']
    with pytest.raises(c.StopRun):c.validate_cases([case(),other])


def test_ledger_reserved_unknown_and_cumulative(tmp_path):
    l=ledger(tmp_path,100)
    t,d=l.reserve('local',{'seconds':60,'calls':1});assert d>time.time()
    with pytest.raises(c.StopRun):l.reserve('runpod',{'seconds':1})
    l.settle(t,{'seconds':40,'calls':1})
    t,_=l.reserve('local',{'seconds':60,'calls':1});l.settle(t)
    assert l.snapshot()['used']['seconds']==100
    with pytest.raises(c.StopRun):l.reserve('local',{'seconds':1})
    with pytest.raises(c.StopRun):Ledger.initialize(l.path,l.snapshot()['limits'])


def test_ledger_two_connections_and_ambiguous_api(tmp_path):
    a=ledger(tmp_path);b=Ledger(a.path)
    t,_=a.reserve('api',{'calls':1,'usd':5,'tokens':100})
    assert b.snapshot()['used']['usd']==5
    b.settle(t);assert a.snapshot()['used']['tokens']==100
    with pytest.raises(c.StopRun):a.settle(t)


def test_sse_delta_boundaries():
    s=SSE();events=[{'choices':[{'index':0,'delta':{'tool_calls':[{'index':0,'id':'abc','function':{'name':'search','arguments':'{"q'}}]}}]},
        {'choices':[{'index':0,'delta':{'tool_calls':[{'index':0,'function':{'arguments':'uery":"énergie"}'}}]},'finish_reason':'tool_calls'}]},
        {'usage':{'total_tokens':100},'choices':[]}]
    stream=''.join('data: '+json.dumps(e,ensure_ascii=False)+'\r\n\r\n' for e in events)+'data: [DONE]\r\n\r\n'
    for i in range(0,len(stream),3):s.feed(stream[i:i+3])
    assert s.done and c.strict_json(s.response()['choices'][0]['message']['tool_calls'][0]['function']['arguments'])=={'query':'énergie'}


def test_recorder_real_asgi_with_mocked_provider(tmp_path,monkeypatch):
    import httpx
    from fastapi.testclient import TestClient
    l=ledger(tmp_path)
    env={'SA_CAPTURE_DIR':str(tmp_path/'capture'),'SA_UPSTREAM_URL':'https://example.test/v1','SA_UPSTREAM_KEY':'SECRET',
         'SA_PROXY_TOKEN':'p'*30,'SA_ADMIN_TOKEN':'a'*30,'SA_PRICES_JSON':'{"teacher":{"input":1,"output":2}}','SA_API_BUDGET_DB':str(l.path)}
    for k,v in env.items():monkeypatch.setenv(k,v)
    real=httpx.AsyncClient
    async def provider(req):
        assert req.headers['authorization']=='Bearer SECRET'
        return httpx.Response(200,json=record()['response'])
    monkeypatch.setattr('scripts.promptfoo_openwebui_eval.posttrain.recorder.httpx.AsyncClient',lambda **kw:real(transport=httpx.MockTransport(provider),**kw))
    app=app_factory()
    try:
        with TestClient(app) as client:
            admin={'Authorization':'Bearer '+'a'*30};key={'Authorization':'Bearer '+'p'*30}
            assert client.post('/admin/begin',json={}).status_code==422
            assert client.post('/admin/begin',headers=admin,json={'benchmark_id':'r1','model':'teacher','question':case()['question']}).status_code==200
            body={'model':'teacher','messages':[{'role':'user','content':case()['question']}],'stream':False}
            assert client.post('/v1/chat/completions',headers=key,json=body).status_code==200
            cap=client.post('/admin/end',headers=admin).json()
            assert cap['complete'] and 'SECRET' not in c.canonical(cap)
            assert client.post('/admin/begin',headers=admin,json={'benchmark_id':'r1','model':'teacher','question':case()['question']}).status_code==422
    finally:app.state.process_lock.close()


@pytest.mark.parametrize('finish_reason', ['stop', 'length'])
def test_recorder_native_multiturn_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finish_reason: str
) -> None:
    """Exercise two provider calls with native tool messages, using a mock provider."""
    import httpx
    from fastapi.testclient import TestClient

    budget = ledger(tmp_path)
    env = {
        'SA_CAPTURE_DIR': str(tmp_path / 'capture'),
        'SA_UPSTREAM_URL': 'https://example.test/v1',
        'SA_UPSTREAM_KEY': 'TEST_PROVIDER_SECRET',
        'SA_PROXY_TOKEN': 'p' * 30,
        'SA_ADMIN_TOKEN': 'a' * 30,
        'SA_PRICES_JSON': '{"teacher":{"input":1,"output":2}}',
        'SA_API_BUDGET_DB': str(budget.path),
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    fixture = record()
    observation = fixture['request']['messages'][-1]
    observation['content'] += '\n' + 'Unabridged evidence. ' * 400 + '<think>quoted</think>'
    first_message = copy.deepcopy(fixture['request']['messages'][1])
    first_message['content'] = 'I will search the selected corpus.'
    # Open WebUI 0.8.6 convert_output_to_messages(raw=True) wraps reasoning
    # in the next request's assistant content while retaining native tool roles.
    conditioned_message = copy.deepcopy(first_message)
    conditioned_message.pop('reasoning_content')
    conditioned_message['content'] = '<think>PRIVATE</think>\n' + first_message['content']
    requests = [
        {'model': 'teacher', 'tools': copy.deepcopy(TOOLS), 'stream': True,
         'messages': [copy.deepcopy(fixture['request']['messages'][0])]},
        {'model': 'teacher', 'tools': copy.deepcopy(TOOLS), 'stream': True,
         'messages': [copy.deepcopy(fixture['request']['messages'][0]),
                      conditioned_message, copy.deepcopy(observation)]},
    ]
    usages = [
        {'prompt_tokens': 90, 'completion_tokens': 10, 'total_tokens': 100, 'cost': .001},
        {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120, 'cost': .002,
         'prompt_tokens_details': {'cached_tokens': 40}},
    ]
    messages = [first_message, fixture['response']['choices'][0]['message']]
    received = []
    real_client = httpx.AsyncClient

    async def provider(request: httpx.Request) -> httpx.Response:
        """Emit provider SSE for the expected tool turn and final answer only."""
        assert request.headers['authorization'] == 'Bearer TEST_PROVIDER_SECRET'
        index = len(received)
        assert index < len(requests)
        body = json.loads(request.content)
        assert body['messages'] == requests[index]['messages']
        assert body['tools'] == TOOLS
        assert body['stream_options']['include_usage'] is True
        received.append(body)
        delta = copy.deepcopy(messages[index])
        for call_index, call in enumerate(delta.get('tool_calls', [])):
            call['index'] = call_index
        events = [
            {'model': 'teacher', 'choices': [{'index': 0, 'delta': delta}]},
            {'model': 'teacher', 'choices': [{'index': 0, 'delta': {},
              'finish_reason': 'tool_calls' if index == 0 else finish_reason}]},
            {'model': 'teacher', 'choices': [], 'usage': usages[index]},
        ]
        stream = ''.join('data: ' + json.dumps(event) + '\n\n' for event in events)
        return httpx.Response(200, content=stream + 'data: [DONE]\n\n',
                              headers={'content-type': 'text/event-stream'})

    monkeypatch.setattr(
        'scripts.promptfoo_openwebui_eval.posttrain.recorder.httpx.AsyncClient',
        lambda **kwargs: real_client(transport=httpx.MockTransport(provider), **kwargs),
    )
    app = app_factory()
    try:
        with TestClient(app) as client:
            admin = {'Authorization': 'Bearer ' + 'a' * 30}
            proxy = {'Authorization': 'Bearer ' + 'p' * 30}
            begin = client.post('/admin/begin', headers=admin, json={
                'benchmark_id': 'native-episode', 'model': 'teacher',
                'question': case()['question'],
            })
            assert begin.status_code == 200
            for body in requests:
                response = client.post('/v1/chat/completions', headers=proxy, json=body)
                assert response.status_code == 200 and 'data: [DONE]' in response.text
            capture_response = client.post('/admin/end', headers=admin)
            assert capture_response.status_code == 200
            captured = capture_response.json()
    finally:
        app.state.process_lock.close()

    assert len(received) == len(captured['records']) == 2
    assert captured['records'][0]['complete'] is True
    assert captured['complete'] is (finish_reason == 'stop')
    assert c.read(tmp_path / 'capture' / 'native-episode.json') == captured
    assert [item['response']['usage'] for item in captured['records']] == usages
    assert c.evidence(captured) == [observation]
    assert 'TEST_PROVIDER_SECRET' not in c.canonical(captured)
    assert budget.snapshot()['used']['calls'] == 2
    assert budget.snapshot()['used']['tokens'] == 220
    assert budget.snapshot()['used']['usd'] == pytest.approx(.003)

    first = c.turn_example(captured['records'][0], case())
    assert [message['train'] for message in first['messages']] == [False, True]
    assert first['messages'][-1]['tool_calls'][0]['function']['name'] == 'search'
    if finish_reason == 'length':
        with pytest.raises(c.StopRun, match='Incomplete capture'):
            c.turn_example(captured['records'][1], case())
        return

    final = c.turn_example(captured['records'][1], case())
    assert [message['train'] for message in final['messages']] == [False, False, False, True]
    assert final['messages'][1]['content'] == first_message['content']
    assert final['messages'][2]['content'] == observation['content']
    assert final['messages'][2]['tool_call_id'] == final['messages'][1]['tool_calls'][0]['id']
    assert final['messages'][-1]['content'] == '42 EUR'
    assert 'PRIVATE' not in c.canonical([first, final])
    assert c.metrics(captured)['llm_calls'] == 2
    assert c.metrics(captured)['completion_tokens'] == 30


def snapshots():
    out=[]
    for dataset in vidore.CORPORA:
        pages=[];qs=[];rels=[]
        for i in range(6):
            pages.append({'corpus_id':str(i),'doc_id':'doc'+str(i),'markdown':'Page '+str(i),'page_number_in_doc':i})
            for lang in ('fr','en'):
                qid=str(i)+'-'+lang;qs.append({'query_id':qid,'query':'Question '+qid,'language':lang,'answer':str(i),'raw_answers':[str(i)],'content_type':['paragraph']})
                rels.append({'query_id':qid,'corpus_id':str(i),'score':1})
        out.append({'dataset':dataset,'revision':REV,'corpus':pages,'queries':qs,'qrels':rels})
    return out


def test_vidore_50_grouped_and_portable(tmp_path):
    a=vidore.build(snapshots(),tmp_path/'a');b=vidore.build(snapshots(),tmp_path/'b')
    assert a['cases_hash']==b['cases_hash'] and a['splits']=={'train':30,'dev':10,'test':10} and a['languages']=={'fr':25,'en':25}
    cases=vidore.load(tmp_path/'a');assert len(cases)==50
    assert all(q['reference_answer'] in q['raw_answers'] for q in cases)
    source=Path(cases[0]['source_paths'][0]);source.write_text('tampered')
    with pytest.raises(c.StopRun):vidore.load(tmp_path/'a')


def test_missing_bilingual_data_does_not_synthesize(tmp_path):
    data=snapshots();data[0]['queries']=[q for q in data[0]['queries'] if q['language']=='en']
    with pytest.raises(c.StopRun):vidore.build(data,tmp_path/'fail')
    assert not (tmp_path/'fail').exists()


def test_original_bundle_contract_on_real_checkout():
    assert (integration.bundle()/'scripts/generate_candidate_csv.py').is_file()
    assert integration.audit()['ok']
    import inspect
    fn=integration.upstream('scripts/generate_candidate_csv.py').generate_row
    assert set(inspect.signature(fn).parameters)=={'index','row','bridge_url','timeout','overwrite','responses_dir','documents_dir'}


def test_holdout_requires_identity_before_live_calls(tmp_path,monkeypatch):
    with pytest.raises(c.StopRun,match='deployment identity'):
        integration.collect([case('dev')],{'status':'complete'},tmp_path/'run',{'split':'dev','repeats':1})


def test_legacy_no_changes_paths():
    assert 'tinker' not in (P/'__main__.py').read_text().lower()
    assert 'tinker' not in (P/'requirements.txt').read_text().lower()
    assert not (P/'tinker_train.py').exists()


def test_integration_reuses_actual_generator_contract_without_gold(tmp_path,monkeypatch):
    q=case();resources={'status':'complete','tool_ids':['rag'],'rag_params':{},'rag_fingerprint':'fp',
            'splits':{'train':{'knowledge_id':'kb','model_id':'sa-model'}}}
    expected=[];context={}
    def generate(**kwargs):
        row=kwargs['row'];expected.append(row)
        assert 'reference_answer' not in row and 'gold_summary' not in row and 'context' not in row
        return {'candidate_answer':'UI presentation'}
    def original(path):
        return types.SimpleNamespace(generate_row=generate) if 'generate_candidate' in path else types.SimpleNamespace(write_csv_rows=lambda *a:None)
    def admin(url,operation,payload=None):
        if operation=='begin':context.update(payload);return {'ok':True,'inference_limits':payload['inference_limits']}
        return capture()
    monkeypatch.setattr(integration,'clients',lambda *a:(None,None));monkeypatch.setattr(integration,'rag_fingerprint',lambda *a:'fp')
    monkeypatch.setattr(integration,'upstream',original);monkeypatch.setattr(integration,'recorder_admin',admin)
    result=integration.collect([q],resources,tmp_path/'run',{'split':'train','bridge_url':'bridge','recorder_url':'recorder','upstream_model':'teacher'})
    assert result['count']==1 and len(expected)==1 and context['question']==q['question']
    generated=c.rows(tmp_path/'run/candidates.jsonl')[0]
    assert generated['candidate_answer']=='42 EUR' and generated['ui_answer']=='UI presentation'


def evaluation_fixture(path,passed=True):
    cap=capture();cand=candidate(cap);q=case('dev')
    c.write(path/cand['capture_path'],cap);c.write_rows(path/'candidates.jsonl',[cand]);c.write_rows(path/'cases.jsonl',[q])
    proto={'split':'dev','repeats':1,'evaluation_limits':{'max_completion_tokens':100,'max_llm_calls':4,'max_tool_calls':4,'max_seconds':30}}
    c.write(path/'selection.json',{'cases_hash':c.digest([q]),'protocol':proto,'protocol_hash':c.digest(proto)})
    c.write(path/'COMPLETE.json',{'count':1,'candidates_hash':c.digest([cand])})
    c.write(path/'promptfoo.json',{'results':[export(cand,passed)]})
    return cand


def test_scoring_identity_not_relabelled_by_import_environment(tmp_path,monkeypatch):
    path=tmp_path/'run';evaluation_fixture(path)
    monkeypatch.setenv('SA_PRICES_JSON',c.canonical({m:{'input':1,'output':5} for m in ('judge-A','judge-B','judge-C')}))
    monkeypatch.setenv('SA_CORRECTNESS_JUDGE','judge-A');monkeypatch.setenv('SA_GROUNDING_JUDGE','judge-B')
    evaluation.score(path)
    monkeypatch.setenv('SA_CORRECTNESS_JUDGE','judge-C')
    receipt=evaluation.import_run(path)
    assert receipt['judge_models']['correctness']=='judge-A'
    with pytest.raises(c.StopRun,match='identity changed'):evaluation.score(path)
    frozen=c.read(path/'scoring.json');frozen['judge_url']='https://other.example/v1';c.write(path/'scoring.json',frozen)
    with pytest.raises(c.StopRun,match='Scoring configuration'):evaluation.validated_grades(path)


def test_frozen_evaluation_limits_applied_before_paid_judge(tmp_path,monkeypatch):
    path=tmp_path/'run';evaluation_fixture(path)
    selection=c.read(path/'selection.json');selection['protocol']['evaluation_limits']['max_completion_tokens']=1
    selection['protocol_hash']=c.digest(selection['protocol']);c.write(path/'selection.json',selection)
    monkeypatch.setattr(evaluation,'judge',lambda *a,**k:pytest.fail('No paid call after hard gate failure'))
    result=evaluation.get_assert('42 EUR',{'config':{'run':str(path),'metric':'correctness'},'vars':{'benchmark_id':'b1'}})
    assert not result['pass'] and result['reason'].startswith('Withheld')


def test_compare_rejects_different_judge_protocol(tmp_path,monkeypatch):
    monkeypatch.setenv('SA_PRICES_JSON',c.canonical({m:{'input':1,'output':5} for m in ('judge-A','judge-B')}))
    monkeypatch.setenv('SA_CORRECTNESS_JUDGE','judge-A');monkeypatch.setenv('SA_GROUNDING_JUDGE','judge-B')
    a=tmp_path/'base';b=tmp_path/'adapted';evaluation_fixture(a,False);evaluation_fixture(b,True)
    for p in (a,b):evaluation.score(p);evaluation.import_run(p)
    assert evaluation.compare(a,b)['decision']=='PROMISING_PILOT_REVIEW_REQUIRED'
    scoring=c.read(b/'scoring.json');scoring['judge_models']['groundedness']='judge-C';c.write(b/'scoring.json',scoring)
    evaluation.import_run(b)  # even a deliberate reimport cannot make the two judge protocols equal
    with pytest.raises(c.StopRun,match='protocol differs'):evaluation.compare(a,b)


def test_multiturn_metrics_include_repeated_inputs_and_optional_subsets() -> None:
    """Provider billing counts each repeated prefix; cache/reasoning are subsets."""
    records = [{'response': {'usage': {'prompt_tokens': p, 'completion_tokens': c, 'total_tokens': p+c,
                 'prompt_tokens_details': {'cached_tokens': cache},
                 'completion_tokens_details': {'reasoning_tokens': reasoning}},
                'choices': [{'message': {'tool_calls': [{}] * tools}}]}}
               for p, c, cache, reasoning, tools in [(100, 20, 0, 5, 2), (150, 30, 90, 10, 0)]]
    result = c.metrics({'records': records, 'duration_seconds': 12})
    assert result['prompt_tokens'] == 250
    assert result['completion_tokens'] == 50
    assert result['total_tokens'] == 300
    assert result['cached_input_tokens'] == 90
    assert result['reasoning_tokens'] == 15
    assert result['llm_calls'] == result['tool_calls'] == 2
    assert result['total_usage_known'] is True


@pytest.mark.parametrize('usage', [{}, {'completion_tokens': 20},
    {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 999},
    {'prompt_tokens': True, 'completion_tokens': 20, 'total_tokens': 21}])
def test_multiturn_metrics_never_fabricate_unknown_totals(usage: dict[str, Any]) -> None:
    """Missing or inconsistent usage cannot be reported as zero-cost success."""
    result = c.metrics({'records': [{'response': {'usage': usage, 'choices': [{'message': {}}]}}]})
    assert result['total_tokens'] is None
    assert result['total_usage_known'] is False
    assert result['cached_input_tokens'] is None
    assert result['reasoning_tokens'] is None


@pytest.mark.parametrize('details', [None, [], 'invalid', {}, {'cached_tokens': 101}, {'cached_tokens': -1}])
def test_multiturn_metrics_missing_or_invalid_cache_is_unknown(details: Any) -> None:
    """Omitted, malformed or out-of-range optional details do not count as zero."""
    usage = {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120, 'prompt_tokens_details': details}
    result = c.metrics({'records': [{'response': {'usage': usage, 'choices': [{'message': {}}]}}]})
    assert result['total_tokens'] == 120
    assert result['cached_input_tokens'] is None
