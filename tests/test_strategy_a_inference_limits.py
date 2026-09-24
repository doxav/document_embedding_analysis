"""Inference limits use real local accounting and simulated HTTP, never paid providers."""
from __future__ import annotations

import copy
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from test_strategy_a_collection_selection import collection
from test_strategy_a_judge_pricing import judge_case, scoring_context
from test_strategy_a_native_collection import native_collection
from scripts.promptfoo_openwebui_eval.posttrain import evaluation, inference_limits as limits, integration, recorder
from scripts.promptfoo_openwebui_eval.posttrain.budget import Ledger
from scripts.promptfoo_openwebui_eval.posttrain.core import StopRun, canonical, read


@pytest.fixture(autouse=True)
def clear_inference_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep default checks independent from an operator's private runtime configuration."""
    names = ('SA_OUTPUT_TOKEN_CEILING', 'SA_MAX_OUTPUT_TOKENS', 'SA_JUDGE_MAX_OUTPUT_TOKENS',
             'SA_MAX_INPUT_BOUND', 'SA_JUDGE_MAX_INPUT_BOUND', 'SA_TEACHER_TIMEOUT_SECONDS',
             'SA_JUDGE_TIMEOUT_SECONDS', 'SA_COLLECTION_TIMEOUT_SECONDS', 'SA_MAX_EPISODE_SECONDS',
             'SA_NATIVE_MAX_SECONDS', 'SA_MAX_EPISODE_COMPLETION_TOKENS', 'SA_MAX_LLM_CALLS',
             'SA_MAX_TOOL_CALLS', 'SA_TEACHER_REASONING_EFFORT', 'SA_JUDGE_REASONING_EFFORT')
    for name in names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv('OPENWEBUI_TIMEOUT_SECONDS', raising=False)


def test_documented_defaults_have_distinct_units_and_budgets() -> None:
    """Per-call, cumulative, byte and transport limits remain separate values."""
    teacher = limits.teacher_limits()
    assert teacher == {'max_output_tokens': 8192, 'max_input_bound_bytes': 50000,
                       'request_timeout_seconds': 600, 'collection_timeout_seconds': 900,
                       'evaluation_limits': {'max_completion_tokens': 50000, 'max_llm_calls': 8,
                                             'max_tool_calls': 8, 'max_seconds': 600}}
    assert limits.judge_limits(True) == {'max_output_tokens': 4096, 'max_input_bound_bytes': 131072,
                                       'request_timeout_seconds': 300, 'reasoning_effort': 'low'}
    assert limits.judge_limits(False)['reasoning_effort'] is None


def test_explicit_old_spec_precedes_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """An existing explicit 4096/16384 protocol is retained without coercion or upgrading."""
    monkeypatch.setenv('SA_MAX_OUTPUT_TOKENS', '8192')
    monkeypatch.setenv('SA_MAX_EPISODE_COMPLETION_TOKENS', '50000')
    spec = {'max_output_tokens': 4096, 'max_input_bound_bytes': 16000,
            'evaluation_limits': copy.deepcopy(limits.HISTORICAL_EPISODE_LIMITS)}
    resolved = limits.teacher_limits(spec)
    assert resolved['max_output_tokens'] == 4096 and resolved['max_input_bound_bytes'] == 16000
    assert resolved['evaluation_limits'] == limits.HISTORICAL_EPISODE_LIMITS


@pytest.mark.parametrize('bad', [0, -1, True, '4096', 4.0, None, 50001])
def test_invalid_spec_output_is_rejected(bad: Any) -> None:
    """Spec outputs must be strict positive integers within the absolute ceiling."""
    with pytest.raises(StopRun):
        limits.teacher_limits({'max_output_tokens': bad})


@pytest.mark.parametrize('name,bad', [
    ('SA_OUTPUT_TOKEN_CEILING', '50001'), ('SA_OUTPUT_TOKEN_CEILING', 'true'),
    ('SA_MAX_OUTPUT_TOKENS', '0'), ('SA_MAX_INPUT_BOUND', '-1'),
    ('SA_JUDGE_MAX_OUTPUT_TOKENS', '50001'), ('SA_JUDGE_TIMEOUT_SECONDS', '0'),
    ('SA_JUDGE_MAX_INPUT_BOUND', '1.5'), ('SA_MAX_LLM_CALLS', '０８'),
    ('SA_MAX_TOOL_CALLS', ''), ('SA_MAX_EPISODE_SECONDS', '36001'),
    ('SA_MAX_EPISODE_COMPLETION_TOKENS', '50001')])
def test_invalid_environment_fails_before_use(monkeypatch: pytest.MonkeyPatch, name: str, bad: str) -> None:
    """Reject malformed environment values rather than falling back silently."""
    monkeypatch.setenv(name, bad)
    with pytest.raises(StopRun):
        if name.startswith('SA_JUDGE'):
            limits.judge_limits(True)
        else:
            limits.teacher_limits()


def test_output_cannot_exceed_episode_without_clamping() -> None:
    """An incompatible pair of explicit limits is rejected rather than shortened."""
    episode = {**limits.HISTORICAL_EPISODE_LIMITS, 'max_completion_tokens': 4096}
    with pytest.raises(StopRun, match='Per-call output exceeds'):
        limits.teacher_limits({'max_output_tokens': 8192, 'evaluation_limits': episode})


def test_teacher_reasoning_requires_operator_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generic endpoints receive no invented reasoning argument, and explicit specs win."""
    assert integration.teacher_extras({}) == {}
    monkeypatch.setenv('SA_TEACHER_REASONING_EFFORT', 'high')
    assert integration.teacher_extras({}) == {'reasoning': {'effort': 'high'}}
    explicit = {'reasoning': {'effort': 'low'}}
    assert integration.teacher_extras({'model_extra': explicit}) == explicit


def test_historical_hard_gate_does_not_read_new_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Old runs with no pinned limits keep their historical budget, independent of the shell."""
    from test_strategy_a_v3 import capture, case
    monkeypatch.setenv('SA_MAX_EPISODE_COMPLETION_TOKENS', '1')
    monkeypatch.setenv('SA_MAX_LLM_CALLS', '1')
    assert evaluation.hard_results(capture(), case(), {})['within_budget']['pass'] is True


@pytest.mark.parametrize('setting,value', [('SA_JUDGE_TIMEOUT_SECONDS', '301'),
                                        ('SA_JUDGE_MAX_OUTPUT_TOKENS', '8192'),
                                        ('SA_JUDGE_MAX_INPUT_BOUND', '131073'),
                                        ('SA_JUDGE_REASONING_EFFORT', 'medium')])
def test_judge_cache_cannot_charge_again_after_settings_drift(
    judge_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, setting: str, value: str,
) -> None:
    """Changing any effective setting requires a fresh run even with a prior valid verdict."""
    evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    monkeypatch.setenv(setting, value)
    with pytest.raises(StopRun, match='settings drift'):
        evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    assert len(judge_case['calls']) == 1 and len(judge_case['ledger'].snapshot()['tickets']) == 1


@pytest.mark.parametrize('setting,value', [('SA_JUDGE_TIMEOUT_SECONDS', '301'),
                                        ('SA_JUDGE_MAX_OUTPUT_TOKENS', '8192'),
                                        ('SA_JUDGE_REASONING_EFFORT', 'medium'),
                                        ('SA_PRICES_JSON', '{"judge/model":{"input":1,"output":6}}')])
def test_scoring_settings_are_frozen_before_any_paid_call(
    judge_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, setting: str, value: str,
) -> None:
    """A prepared scoring protocol rejects changed prices, reasoning, time or output limits."""
    context = scoring_context(judge_case, monkeypatch)
    evaluation.score(judge_case['run'])
    frozen = read(judge_case['run'] / 'scoring.json')
    assert frozen['judge_limits']['max_output_tokens'] == 4096
    monkeypatch.setenv(setting, value)
    result = evaluation.get_assert('Answer.', context)
    assert result['reason'].startswith('SA_INFRASTRUCTURE:') and 'settings drift' in result['reason']
    assert not judge_case['calls'] and not judge_case['ledger'].snapshot()['tickets']


def test_judge_can_use_50000_but_not_50001(judge_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """The actual request and pre-call reservation use the approved larger ceiling exactly."""
    monkeypatch.setenv('SA_JUDGE_MAX_OUTPUT_TOKENS', '50000')
    evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    call = judge_case['calls'][0];bound = len(canonical(call['body']).encode()) + 512
    assert call['body']['max_tokens'] == 50000
    assert call['ledger']['tickets'][0]['reserved']['tokens'] == bound + 50000
    monkeypatch.setenv('SA_JUDGE_MAX_OUTPUT_TOKENS', '50001')
    with pytest.raises(StopRun):
        evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    assert len(judge_case['calls']) == 1


@pytest.mark.parametrize('bad', ['none', 'max', '', 'invented'])
def test_unadvertised_judge_reasoning_is_rejected_before_reservation(
    judge_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, bad: str,
) -> None:
    """The reviewed Gemini effort contract excludes unsupported modes including none."""
    monkeypatch.setenv('SA_JUDGE_REASONING_EFFORT', bad)
    with pytest.raises(StopRun):
        evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    assert not judge_case['calls'] and not judge_case['ledger'].snapshot()['tickets']


@pytest.fixture
def recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """Run the real recorder ASGI routes and local ledger against one synthetic provider."""
    ledger = Ledger.initialize(tmp_path / 'api.sqlite', {'seconds': 3600, 'usd': 10, 'tokens': 500000, 'calls': 20})
    for name,value in {'SA_CAPTURE_DIR': str(tmp_path/'capture'), 'SA_UPSTREAM_URL':'https://example.test/v1',
                       'SA_UPSTREAM_KEY':'TEST_PROVIDER', 'SA_PROXY_TOKEN':'p'*30, 'SA_ADMIN_TOKEN':'a'*30,
                       'SA_PRICES_JSON':'{"teacher":{"input":1,"output":2}}', 'SA_API_BUDGET_DB':str(ledger.path)}.items():
        monkeypatch.setenv(name,value)
    state: dict[str, Any] = {'ledger': ledger, 'calls': [], 'completion': 8, 'tools': []}
    real = httpx.AsyncClient
    async def provider(request: httpx.Request) -> httpx.Response:
        """Record the literal body and return bounded accounting, without outbound network."""
        body = json.loads(request.content);state['calls'].append(body)
        if state.get('on_call'):
            state['on_call']()
        if state.get('continuous_stream'):
            class ContinuousStream(httpx.AsyncByteStream):
                """Keep producing valid fragments past the requested absolute deadline."""

                async def __aiter__(self) -> AsyncIterator[bytes]:
                    """Simulate activity that defeats an inactivity-only read timeout."""
                    try:
                        for _ in range(100):
                            yield b'data: {"model":"teacher","choices":[{"index":0,"delta":{"content":"x"}}]}\n\n'
                            await asyncio.sleep(.1)
                    finally:
                        state['stream_closed'] = True

            return httpx.Response(200,stream=ContinuousStream(),headers={'content-type':'text/event-stream'})
        message: dict[str, Any] = {'role':'assistant','content':'Synthetic output'}
        if state['tools']:
            message['tool_calls'] = state['tools']
        return httpx.Response(200,json={'model':'teacher','choices':[{'finish_reason':'tool_calls' if state['tools'] else 'stop','message':message}],
            'usage':{'prompt_tokens':20,'completion_tokens':state['completion'],'total_tokens':20+state['completion'],'cost':.00002}})
    monkeypatch.setattr(recorder.httpx,'AsyncClient',lambda **kwargs: real(transport=httpx.MockTransport(provider),**kwargs))
    app = recorder.app_factory()
    try:
        with TestClient(app) as client:
            state.update(client=client,admin={'Authorization':'Bearer '+'a'*30},proxy={'Authorization':'Bearer '+'p'*30})
            yield state
    finally:
        app.state.process_lock.close()


def begin(state: dict[str, Any], config: dict[str, Any] | None = None) -> Any:
    """Start one synthetic recorder session with optional current or legacy admin fields."""
    body: dict[str, Any] = {'benchmark_id':'limits-test','model':'teacher','question':'Synthetic question?'}
    if config is not None:
        body['inference_limits'] = config
    return state['client'].post('/admin/begin',headers=state['admin'],json=body)


def chat(state: dict[str, Any], **extra: Any) -> Any:
    """Submit an unchanged synthetic user message to the local recorder."""
    return state['client'].post('/v1/chat/completions',headers=state['proxy'],json={
        'model':'teacher','messages':[{'role':'user','content':'Synthetic question?'}],'stream':False,**extra})


def test_legacy_begin_uses_pinned_defaults(recorded: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Old callers omit limits; changing the process environment later does not alter a session."""
    reply = begin(recorded)
    assert reply.status_code == 200 and reply.json()['inference_limits']['max_output_tokens'] == 8192
    monkeypatch.setenv('SA_MAX_OUTPUT_TOKENS','1')
    assert chat(recorded).status_code == 200
    assert recorded['calls'][0]['max_tokens'] == 8192


def test_cumulative_limit_rejects_next_full_reservation(recorded: dict[str, Any]) -> None:
    """A next 50k request is refused after any output; it is never clamped or paid."""
    config = limits.teacher_limits({'max_output_tokens':50000})
    assert begin(recorded,config).json()['inference_limits'] == config
    assert chat(recorded,max_tokens=50000).status_code == 200
    reply = chat(recorded,max_tokens=50000)
    assert reply.status_code == 422 and 'remaining token budget' in reply.text
    assert len(recorded['calls']) == 1 and len(recorded['ledger'].snapshot()['tickets']) == 1


def test_input_bound_is_bytes_and_never_truncates(recorded: dict[str, Any]) -> None:
    """An oversized UTF-8 request is rejected before a provider call or budget reservation."""
    config = limits.teacher_limits({'max_input_bound_bytes':100})
    assert begin(recorded,config).status_code == 200
    assert chat(recorded).status_code == 422
    assert not recorded['calls'] and not recorded['ledger'].snapshot()['tickets']


def test_invalid_begin_limits_never_create_a_session(recorded: dict[str, Any]) -> None:
    """Authenticated admin inputs still obey the absolute cap before session allocation."""
    config = limits.teacher_limits();config['max_output_tokens'] = 50001
    assert begin(recorded,config).status_code == 422
    assert chat(recorded).status_code == 422
    assert not recorded['ledger'].snapshot()['tickets']


def test_output_alias_conflict_is_rejected(recorded: dict[str, Any]) -> None:
    """Two output aliases cannot make the recorder reserve a different provider limit."""
    begin(recorded)
    assert chat(recorded,max_tokens=10,max_completion_tokens=20).status_code == 422
    assert not recorded['calls'] and not recorded['ledger'].snapshot()['tickets']


def test_observed_output_overrun_is_settled_and_blocks_continuation(recorded: dict[str, Any]) -> None:
    """Truthful usage is settled while an excessive response is excluded without slicing it."""
    begin(recorded);recorded['completion'] = 9
    assert chat(recorded,max_tokens=8).status_code == 200
    assert chat(recorded,max_tokens=8).status_code == 422
    cap = recorded['client'].post('/admin/end',headers=recorded['admin']).json()
    assert cap['complete'] is False and cap['records'][0]['response']['usage']['completion_tokens'] == 9
    assert recorded['ledger'].snapshot()['tickets'][0]['actual']['tokens'] == 29


def test_tool_count_overrun_blocks_another_call(recorded: dict[str, Any]) -> None:
    """A response exceeding the tool budget is preserved and cannot trigger another reservation."""
    config = limits.teacher_limits();config['evaluation_limits']['max_tool_calls'] = 1
    begin(recorded,config)
    recorded['tools'] = [{'id':str(i),'type':'function','function':{'name':'search','arguments':'{}'}} for i in range(2)]
    assert chat(recorded).status_code == 200 and chat(recorded).status_code == 422
    assert len(recorded['calls']) == 1


def test_continuous_stream_cannot_outlive_absolute_deadline(recorded: dict[str, Any]) -> None:
    """Continuous activity is interrupted at the deadline, preserving the ambiguous reservation."""
    config = limits.teacher_limits({'request_timeout_seconds':1,
        'evaluation_limits':{**limits.HISTORICAL_EPISODE_LIMITS,'max_seconds':1}})
    begin(recorded,config);recorded['continuous_stream'] = True
    with pytest.raises((TimeoutError,ExceptionGroup)):
        chat(recorded,stream=True)
    cap = recorded['client'].post('/admin/end',headers=recorded['admin']).json()
    assert cap['complete'] is False and recorded['stream_closed'] is True
    assert len(recorded['calls']) == 1 and 'Interrupted stream' in cap['records'][0]['error']
    ticket = recorded['ledger'].snapshot()['tickets'][0]
    assert ticket['actual'] == ticket['reserved']


def test_late_nonstream_response_is_not_an_accepted_episode(
    recorded: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A response crossing the wall-clock episode deadline remains billed and is excluded."""
    clock = [100.0]
    monkeypatch.setattr(recorder,'time',SimpleNamespace(time=lambda:clock[0],monotonic=recorder.time.monotonic))
    config = limits.teacher_limits({'evaluation_limits':{**limits.HISTORICAL_EPISODE_LIMITS,'max_seconds':1}})
    begin(recorded,config);recorded['on_call'] = lambda:clock.__setitem__(0,102.0)
    assert chat(recorded).status_code == 200
    cap = recorded['client'].post('/admin/end',headers=recorded['admin']).json()
    assert cap['complete'] is False and cap['records'][0]['error'] == 'Observed episode time budget exceeded'
    assert recorded['ledger'].snapshot()['tickets'][0]['actual']['tokens'] == 28


def test_clock_rollback_does_not_extend_episode(recorded: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """The in-memory monotonic deadline still expires when the wall clock moves backward."""
    wall = [100.0];monotonic = [100.0]
    monkeypatch.setattr(recorder,'time',SimpleNamespace(time=lambda:wall[0],monotonic=lambda:monotonic[0]))
    config = limits.teacher_limits({'evaluation_limits':{**limits.HISTORICAL_EPISODE_LIMITS,'max_seconds':1}})
    begin(recorded,config);wall[0] = 90.0;monotonic[0] = 102.0
    assert chat(recorded).status_code == 422
    assert not recorded['calls'] and not recorded['ledger'].snapshot()['tickets']


def test_ledger_wait_does_not_extend_the_provider_deadline(
    recorded: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Time spent obtaining the ledger reservation cannot grant another provider time window."""
    monotonic = [100.0]
    monkeypatch.setattr(recorder,'time',SimpleNamespace(time=recorder.time.time,monotonic=lambda:monotonic[0]))
    original = Ledger.reserve
    def reserve(self: Ledger, *args: Any, **kwargs: Any) -> Any:
        """Obtain the real reservation, then simulate a wait past the original episode deadline."""
        result = original(self,*args,**kwargs);monotonic[0] = 102.0
        return result
    monkeypatch.setattr(Ledger,'reserve',reserve)
    config = limits.teacher_limits({'evaluation_limits':{**limits.HISTORICAL_EPISODE_LIMITS,'max_seconds':1}})
    begin(recorded,config)
    assert chat(recorded).status_code == 422
    assert not recorded['calls']
    cap = recorded['client'].post('/admin/end',headers=recorded['admin']).json()
    assert cap['complete'] is False


def test_native_effective_timeout_includes_the_historical_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lower upstream-client bound is exposed instead of being silently called the spec limit."""
    monkeypatch.setenv('SA_MAX_EPISODE_SECONDS','900')
    monkeypatch.setenv('SA_NATIVE_MAX_SECONDS','800')
    monkeypatch.setenv('OPENWEBUI_TIMEOUT_SECONDS','700')
    assert limits.native_episode_seconds() == 700


@pytest.mark.parametrize('reported', [None,600.0,True,599,601])
def test_native_timeout_mismatch_prevents_recorder_begin(
    native_collection: dict[str, Any], tmp_path: Path, reported: Any,
) -> None:
    """A missing or divergent bridge deadline is rejected before the recorder is opened."""
    native_collection['states'][0]['episode_timeout_seconds'] = reported
    with pytest.raises(StopRun,match='Native episode timeout differs'):
        integration.collect(native_collection['cases'],native_collection['resources'],tmp_path/'run',native_collection['spec'])
    assert 'recorder-begin' not in native_collection['external'] and not native_collection['generated']


def test_invalid_collection_limits_precede_clients(collection: dict[str, Any], tmp_path: Path) -> None:
    """A bad spec fails before read-only clients or the recorder can be contacted."""
    spec = {**collection['spec'],'max_output_tokens':50001}
    with pytest.raises(StopRun):
        integration.collect(collection['cases'],collection['resources'],tmp_path/'run',spec)
    assert not collection['external']


def test_recorder_must_echo_validated_limits_before_generation(
    collection: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A legacy recorder that ignores the new admin field cannot silently run another budget."""
    monkeypatch.setattr(integration,'recorder_admin',lambda *args,**kwargs:{'ok':True})
    with pytest.raises(StopRun,match='did not acknowledge'):
        integration.collect(collection['cases'],collection['resources'],tmp_path/'run',collection['spec'])
    assert not collection['generated']
