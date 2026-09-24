"""Verify real judge reservations and receipts with a simulated HTTP transport."""
from __future__ import annotations

import copy
import base64
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.promptfoo_openwebui_eval.posttrain import evaluation
from scripts.promptfoo_openwebui_eval.posttrain.budget import Ledger
from scripts.promptfoo_openwebui_eval.posttrain.core import StopRun, canonical, digest, read, write


@pytest.fixture
def judge_case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Use the actual SQLite ledger and file receipts while preventing every HTTP request."""
    run = tmp_path / 'run'
    run.mkdir()
    ledger = Ledger.initialize(tmp_path / 'api.sqlite',
                               {'seconds': 3600, 'usd': 1, 'tokens': 100000, 'calls': 10})
    verdict = {'pass': True, 'score': .95, 'reason': 'The cited evidence supports the answer.'}
    state: dict[str, Any] = {
        'run': run, 'ledger': ledger, 'model': 'judge/model',
        'payload': {'question': 'Question?', 'candidate_answer': 'Answer.'}, 'calls': [],
        'status': 200, 'closed': [], 'response': {
            'id': 'gen-TEST-001', 'model': 'judge/model', 'provider': 'Test Provider',
            'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': canonical(verdict)}}],
            'usage': {'prompt_tokens': 70, 'completion_tokens': 30, 'total_tokens': 100,
                      'cost': .0004},
        },
    }
    monkeypatch.delenv('SA_JUDGE_URL', raising=False)
    monkeypatch.setenv('SA_JUDGE_KEY', 'TEST_ONLY_JUDGE_SECRET')
    monkeypatch.setenv('SA_API_BUDGET_DB', str(ledger.path))
    monkeypatch.setenv('SA_PRICES_JSON', canonical({'judge/model': {'input': 1, 'output': 5}}))
    for name,value in {'SA_OUTPUT_TOKEN_CEILING':'50000','SA_JUDGE_MAX_OUTPUT_TOKENS':'4096',
                       'SA_JUDGE_MAX_INPUT_BOUND':'131072','SA_JUDGE_TIMEOUT_SECONDS':'300',
                       'SA_JUDGE_REASONING_EFFORT':'low'}.items():
        monkeypatch.setenv(name,value)

    def post(url: str, *, json: dict[str, Any], headers: dict[str, str],
             timeout: int, allow_redirects: bool) -> requests.Response:
        """Capture the outgoing body and its preexisting reservation without live inference."""
        assert headers == {'Authorization': 'Bearer TEST_ONLY_JUDGE_SECRET'}
        assert timeout == int(os.environ['SA_JUDGE_TIMEOUT_SECONDS'])
        assert allow_redirects is False
        state['calls'].append({'url': url, 'body': copy.deepcopy(json),
                               'ledger': ledger.snapshot()})
        attempt = read(next((run / 'judges').glob('*.attempt.json')))
        assert attempt['state'] == 'POST_POTENTIALLY_SENT'
        assert attempt['ticket'] == ledger.snapshot()['tickets'][0]['id']
        assert attempt['identity']['body'] == json
        if state.get('exception'):
            raise state['exception']
        response = requests.Response()
        response.status_code = state['status']
        response._content = state['raw'] if 'raw' in state else canonical(state['response']).encode()
        response._content_consumed = True

        def close() -> None:
            """Track explicit cleanup of the simulated HTTP response."""
            state['closed'].append(True)

        response.close = close
        return response

    monkeypatch.setattr(evaluation.requests, 'post', post)
    return state


@pytest.mark.parametrize('base', [None, 'https://openrouter.ai/api/v1',
                                  'https://openrouter.ai/api/v1/'])
@pytest.mark.parametrize('kind', ['correctness', 'groundedness'])
def test_openrouter_prices_bind_request_reservation_and_receipt(
    judge_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
    base: str | None, kind: str,
) -> None:
    """Route under the ceiling, reserve the final body, then settle observed usage once."""
    if base is not None:
        monkeypatch.setenv('SA_JUDGE_URL', base)
    if kind == 'groundedness':
        verdict = json.loads(judge_case['response']['choices'][0]['message']['content'])
        verdict.update(unsupported_claims=0, citation_errors=0)
        judge_case['response']['choices'][0]['message']['content'] = canonical(verdict)
    result = evaluation.judge(kind, judge_case['payload'], judge_case['model'], judge_case['run'])
    body = {
        'model': 'judge/model',
        'messages': [
            {'role': 'system', 'content': evaluation.CORRECTNESS if kind == 'correctness'
             else evaluation.GROUNDING},
            {'role': 'user', 'content': canonical(judge_case['payload'])},
        ],
        'temperature': 0, 'max_tokens': 4096, 'stream': False,
        'provider': {'max_price': {'prompt': 1, 'completion': 5},
                     'allow_fallbacks': False, 'require_parameters': True},
        'response_format': evaluation.judge_schema(kind),
        'reasoning': {'effort': 'low'},
    }
    assert len(judge_case['calls']) == 1
    call = judge_case['calls'][0]
    assert call['url'] == 'https://openrouter.ai/api/v1/chat/completions'
    assert call['body'] == body
    bound = len(canonical(body).encode()) + 512
    ticket = call['ledger']['tickets'][0]
    assert ticket['status'] == 'reserved' and ticket['actual'] is None
    assert ticket['reserved'] == {'calls': 1, 'tokens': bound + 4096,
                                  'usd': (bound + 4096 * 5) / 1e6}
    assert ticket['metadata'] == {'kind': kind, 'model': 'judge/model',
                                  'request_hash': digest(body), 'provider': body['provider'],
                                  'request_contract': 5}
    settled = judge_case['ledger'].snapshot()['tickets'][0]
    assert settled['status'] == 'settled'
    assert settled['actual'] == {'calls': 1, 'tokens': 100, 'usd': .0004}
    receipts = verdict_paths(judge_case)
    assert len(receipts) == 1
    assert receipts[0].stem == digest({'base': 'https://openrouter.ai/api/v1', 'kind': kind,
                                       'body': body, 'rubric_version': 3, 'request_contract': 5,
                                       'limits': evaluation.judge_limits(True), 'price': {'input': 1, 'output': 5}})
    wire = read(receipts[0].with_suffix('.wire.json'))
    assert read(receipts[0]) == {'result': result, 'model': 'judge/model', 'actual_model': 'judge/model',
                                'generation_id': 'gen-TEST-001', 'upstream_provider': 'Test Provider',
                                'usage': judge_case['response']['usage'],
                                'request_hash': digest(body), 'provider': body['provider'],
                                'ticket': ticket['id'], 'wire_hash': digest(wire)}
    assert json.loads(base64.b64decode(wire['response_body_b64'])) == judge_case['response']
    assert judge_case['closed'] == [True]
    assert evaluation.judge(kind, judge_case['payload'], judge_case['model'],
                            judge_case['run']) == result
    assert len(judge_case['calls']) == 1
    assert len(judge_case['ledger'].snapshot()['tickets']) == 1


@pytest.mark.parametrize('base', [
    'http://127.0.0.1:9000/v1', 'https://judge.example/v1',
    'https://openrouter.ai/api/v1/other', 'https://openrouter.ai.example/api/v1',
])
def test_other_endpoints_keep_the_existing_request_contract(
    judge_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, base: str,
) -> None:
    """Inject OpenRouter routing only for its exact normalized API base."""
    monkeypatch.setenv('SA_JUDGE_URL', base)
    evaluation.judge('correctness', judge_case['payload'], judge_case['model'],
                     judge_case['run'], temperature=.2)
    call = judge_case['calls'][0]
    assert call['url'] == base + '/chat/completions'
    assert call['body'] == {
        'model': 'judge/model',
        'messages': [{'role': 'system', 'content': evaluation.CORRECTNESS},
                     {'role': 'user', 'content': canonical(judge_case['payload'])}],
        'temperature': .2, 'max_tokens': 4096, 'stream': False,
    }
    receipt = read(verdict_paths(judge_case)[0])
    assert receipt['provider'] is None and receipt['request_hash'] == digest(call['body'])


@pytest.mark.parametrize('prices', [
    None, [], True, {}, {'other/model': {'input': 1, 'output': 5}},
    {'judge/model': None}, {'judge/model': []}, {'judge/model': '1/5'},
    {'judge/model': {'input': 1}}, {'judge/model': {'output': 5}},
    *({'judge/model': {'input': bad, 'output': 5}} for bad in [0, -1, True, '1', None]),
    *({'judge/model': {'input': 1, 'output': bad}} for bad in [0, -1, True, '5', None]),
])
def test_invalid_price_ceilings_stop_before_any_side_effect(
    judge_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, prices: Any,
) -> None:
    """Reject missing or nonpositive numeric ceilings before calls, files or reservations."""
    monkeypatch.setenv('SA_PRICES_JSON', canonical(prices))
    with pytest.raises(StopRun, match='Explicit positive price ceilings required'):
        evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    assert not judge_case['calls'] and not judge_case['ledger'].snapshot()['tickets']
    assert not list(judge_case['run'].iterdir())


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('field', ['input', 'output'])
def test_nonfinite_price_ceilings_never_reserve_or_call(
    judge_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, bad: float, field: str,
) -> None:
    """Strict JSON rejection keeps nonfinite prices away from the provider and ledger."""
    prices = {'judge/model': {'input': 1, 'output': 5, field: bad}}
    monkeypatch.setenv('SA_PRICES_JSON', json.dumps(prices))
    with pytest.raises(StopRun, match='Non-finite JSON'):
        evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    assert not judge_case['calls'] and not judge_case['ledger'].snapshot()['tickets']
    assert not list(judge_case['run'].iterdir())


@pytest.mark.parametrize('status', [302, 307, 308, 429, 503])
def test_http_error_has_no_retry_and_keeps_the_full_reservation(
    judge_case: dict[str, Any], status: int,
) -> None:
    """An unsuccessful, potentially charged POST remains reserved without a judge verdict."""
    judge_case['status'] = status
    with pytest.raises(StopRun, match=f'Judge HTTP {status}; no automatic charged retry'):
        evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    assert len(judge_case['calls']) == 1
    before = judge_case['calls'][0]['ledger']
    after = judge_case['ledger'].snapshot()
    assert after == before and len(after['tickets']) == 1
    assert after['tickets'][0]['status'] == 'reserved'
    assert after['tickets'][0]['actual'] is None
    assert after['used']['usd'] > 0
    assert not verdict_paths(judge_case)
    wire = read(next((judge_case['run'] / 'judges').glob('*.wire.json')))
    assert wire['status_code'] == status and wire['ticket'] == after['tickets'][0]['id']
    assert judge_case['closed'] == [True]
    with pytest.raises(StopRun, match='requires reconciliation'):
        evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    assert len(judge_case['calls']) == 1 and judge_case['ledger'].snapshot() == after


def test_unknown_usage_retains_the_conservative_amount(judge_case: dict[str, Any]) -> None:
    """A valid verdict with no accounting usage does not release its reserved cost."""
    judge_case['response'].pop('usage')
    result = evaluation.judge('correctness', judge_case['payload'], judge_case['model'],
                              judge_case['run'])
    ticket = judge_case['ledger'].snapshot()['tickets'][0]
    assert result['pass'] is True and ticket['status'] == 'reserved'
    assert ticket['actual'] is None


def verdict_paths(state: dict[str, Any]) -> list[Path]:
    """Select accepted verdict receipts separately from durable attempts and raw HTTP files."""
    return [p for p in (state['run'] / 'judges').glob('*.json') if p.stem.count('.') == 0]


@pytest.mark.parametrize('kind', ['correctness', 'groundedness'])
def test_schema_is_strict_and_preserves_the_existing_verdict_fields(kind: str) -> None:
    """Require all rubric fields and reject unrequested fields without a healing plugin."""
    expected = {'pass': {'type': 'boolean'}, 'score': {'type': 'number', 'minimum': 0, 'maximum': 1},
                'reason': {'type': 'string'}}
    if kind == 'groundedness':
        expected.update(unsupported_claims={'type': 'integer', 'minimum': 0},
                        citation_errors={'type': 'integer', 'minimum': 0})
    assert evaluation.judge_schema(kind) == {
        'type': 'json_schema', 'json_schema': {
            'name': 'strategy_a_' + kind, 'strict': True,
            'schema': {'type': 'object', 'properties': expected,
                       'required': list(expected), 'additionalProperties': False},
        },
    }


def test_wire_is_private_and_durable_before_decode_and_usage_before_verdict(
    judge_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observe the on-disk wire before either parser and actual accounting before verdict parsing."""
    original = evaluation.strict_json
    decoded: list[str] = []
    raw = canonical(judge_case['response']).encode()
    content = judge_case['response']['choices'][0]['message']['content']

    def parse(text: str) -> Any:
        """Check persistence and accounting at the actual JSON parsing boundary."""
        if text in (raw.decode(), content):
            path = next((judge_case['run'] / 'judges').glob('*.wire.json'))
            wire = read(path)
            assert path.stat().st_mode & 0o777 == 0o600
            assert base64.b64decode(wire['response_body_b64'], validate=True) == raw
            assert wire['response_body_sha256'] == hashlib.sha256(raw).hexdigest()
            ticket = judge_case['ledger'].snapshot()['tickets'][0]
            assert ticket['metadata']['wire_hash'] == digest(wire)
            if text == content:
                assert ticket['actual'] == {'calls': 1, 'tokens': 100, 'usd': .0004}
                decoded.append('verdict')
            else:
                assert ticket['status'] == 'reserved'
                decoded.append('response')
        return original(text)

    monkeypatch.setattr(evaluation, 'strict_json', parse)
    evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    assert decoded == ['response', 'verdict']
    for path in (judge_case['run'] / 'judges').glob('*.json'):
        assert path.stat().st_mode & 0o777 == 0o600
        assert 'TEST_ONLY_JUDGE_SECRET' not in path.read_text()
        assert 'Authorization' not in path.read_text()


@pytest.mark.parametrize('content', ['', 'not JSON', '```json\n{"pass":true,"score":1,"reason":"ok"}\n```',
                                  '{"pass":true,"score":1,"score":0,"reason":"duplicate"}'])
def test_invalid_verdict_preserves_generation_and_settles_known_cost_without_retry(
    judge_case: dict[str, Any], content: str,
) -> None:
    """A paid malformed judgment is an infrastructure error, never a semantic negative."""
    judge_case['response']['choices'][0]['message']['content'] = content
    with pytest.raises(ValueError, match='Invalid judge verdict JSON'):
        evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    assert not verdict_paths(judge_case)
    wire = read(next((judge_case['run'] / 'judges').glob('*.wire.json')))
    restored = json.loads(base64.b64decode(wire['response_body_b64']))
    assert restored == judge_case['response'] and restored['id'] == 'gen-TEST-001'
    ticket = judge_case['ledger'].snapshot()['tickets'][0]
    assert ticket['status'] == 'settled' and ticket['actual']['usd'] == .0004
    with pytest.raises(StopRun, match='requires reconciliation'):
        evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    assert len(judge_case['calls']) == 1 and judge_case['closed'] == [True]
    assert len(judge_case['ledger'].snapshot()['tickets']) == 1


@pytest.mark.parametrize('raw', [b'', b'<html>provider failed</html>', b'\xff', b'null', b'[]'])
def test_malformed_http_body_keeps_exact_bytes_and_reservation(
    judge_case: dict[str, Any], raw: bytes,
) -> None:
    """Unknown usage stays reserved and undecodable responses remain recoverable without another POST."""
    judge_case['raw'] = raw
    with pytest.raises((ValueError, StopRun), match='Invalid judge'):
        evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    wire = read(next((judge_case['run'] / 'judges').glob('*.wire.json')))
    assert base64.b64decode(wire['response_body_b64'], validate=True) == raw
    assert wire['response_body_sha256'] == hashlib.sha256(raw).hexdigest()
    ticket = judge_case['ledger'].snapshot()['tickets'][0]
    assert ticket['status'] == 'reserved' and ticket['actual'] is None
    with pytest.raises(StopRun, match='requires reconciliation'):
        evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    assert len(judge_case['calls']) == 1 and not verdict_paths(judge_case)
    assert judge_case['closed'] == [True]


def test_timeout_has_a_durable_ambiguous_receipt_and_never_reissues(
    judge_case: dict[str, Any],
) -> None:
    """A missing HTTP response is an unresolved paid attempt with its own ticket and request."""
    judge_case['exception'] = requests.Timeout('simulated response timeout')
    with pytest.raises(requests.Timeout):
        evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    wire = read(next((judge_case['run'] / 'judges').glob('*.wire.json')))
    ticket = judge_case['ledger'].snapshot()['tickets'][0]
    assert wire['transport_error'] == 'Timeout' and wire['ticket'] == ticket['id']
    assert wire['identity']['body'] == judge_case['calls'][0]['body']
    assert 'response_body_b64' not in wire and ticket['actual'] is None
    with pytest.raises(StopRun, match='requires reconciliation'):
        evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    assert len(judge_case['calls']) == 1 and len(judge_case['ledger'].snapshot()['tickets']) == 1


@pytest.mark.parametrize('field,value', [
    ('model', 'judge/model-resembling'), ('model', None), ('id', None), ('id', ''),
])
def test_openrouter_response_identity_must_match_the_resolved_judge(
    judge_case: dict[str, Any], field: str, value: Any,
) -> None:
    """A successful transport from an unidentified model cannot become an accepted verdict."""
    judge_case['response'][field] = value
    with pytest.raises(StopRun, match='identity mismatch|generation ID missing'):
        evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    assert not verdict_paths(judge_case)
    assert judge_case['ledger'].snapshot()['tickets'][0]['actual']['usd'] == .0004


@pytest.mark.parametrize('change', ['truncated', 'tool', 'refusal', 'wrong_role', 'error', 'extra_field',
                                    'missing_field', 'wrong_pass', 'negative_score'])
def test_invalid_final_judgment_remains_an_error_with_known_billing(
    judge_case: dict[str, Any], change: str,
) -> None:
    """Only an identified terminal assistant message satisfying the rubric schema is a judgment."""
    response = judge_case['response']
    choice = response['choices'][0]
    if change == 'truncated':
        choice['finish_reason'] = 'length'
    elif change == 'tool':
        choice['message']['tool_calls'] = [{'id': 'test-tool-call'}]
    elif change == 'refusal':
        choice['message']['refusal'] = 'Declined'
    elif change == 'wrong_role':
        choice['message']['role'] = 'tool'
    elif change == 'error':
        response['error'] = {'message': 'simulated provider error'}
    else:
        verdict = json.loads(choice['message']['content'])
        if change == 'extra_field':
            verdict['unsolicited'] = 'value'
        elif change == 'missing_field':
            verdict.pop('reason')
        elif change == 'wrong_pass':
            verdict['pass'] = 'true'
        else:
            verdict['score'] = -1
        choice['message']['content'] = canonical(verdict)
    with pytest.raises(StopRun, match='Judge|Invalid judge'):
        evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    assert not verdict_paths(judge_case)
    assert judge_case['ledger'].snapshot()['tickets'][0]['actual']['usd'] == .0004


@pytest.mark.parametrize('usage', [None, {}, {'total_tokens': 100, 'cost': .0004},
    {'prompt_tokens': 70, 'completion_tokens': 30, 'total_tokens': 99, 'cost': .0004},
    {'prompt_tokens': 70, 'completion_tokens': 30, 'total_tokens': 100, 'cost': -1},
    {'prompt_tokens': True, 'completion_tokens': 30, 'total_tokens': 31, 'cost': .0004},
    {'prompt_tokens': 70, 'completion_tokens': 30, 'total_tokens': 100, 'cost': '0.0004'}])
def test_unverified_usage_never_releases_reserved_budget(
    judge_case: dict[str, Any], usage: Any,
) -> None:
    """Missing, inconsistent or wrongly typed accounting fields keep the conservative reservation."""
    judge_case['response']['usage'] = usage
    result = evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    ticket = judge_case['ledger'].snapshot()['tickets'][0]
    assert result['pass'] is True and ticket['status'] == 'reserved' and ticket['actual'] is None


@pytest.mark.parametrize('part', ['wire', 'wire_and_receipt', 'attempt', 'verdict', 'actual_model',
                                  'generation_id', 'provider', 'usage'])
def test_cached_receipt_drift_is_rejected_without_another_post(
    judge_case: dict[str, Any], part: str,
) -> None:
    """Cached judgments remain bound to independent ledger and raw-response identity evidence."""
    evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    path = verdict_paths(judge_case)[0]
    cached = read(path)
    if part in {'wire', 'wire_and_receipt'}:
        wire_path = path.with_suffix('.wire.json')
        wire = read(wire_path)
        wire['status_code'] = 201
        write(wire_path, wire)
        if part == 'wire_and_receipt':
            cached['wire_hash'] = digest(wire)
            write(path, cached)
    elif part == 'attempt':
        attempt_path = path.with_suffix('.attempt.json')
        attempt = read(attempt_path)
        attempt['ticket'] = 'other-ticket'
        write(attempt_path, attempt)
    else:
        if part == 'verdict':
            cached['result']['pass'] = False
        else:
            cached[part] = 'changed'
        write(path, cached)
    with pytest.raises(StopRun, match='drift'):
        evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    assert len(judge_case['calls']) == 1


def test_failed_wire_write_keeps_the_attempt_and_closes_http(
    judge_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persistence failure after a possibly charged POST never permits its replay."""
    original = evaluation.durable_judge_write

    def write_receipt(path: Path, value: dict[str, Any]) -> None:
        """Simulate an unavailable disk only at the post-response archival boundary."""
        if path.name.endswith('.wire.json'):
            raise OSError('simulated disk failure')
        original(path, value)

    monkeypatch.setattr(evaluation, 'durable_judge_write', write_receipt)
    with pytest.raises(OSError):
        evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    assert judge_case['closed'] == [True]
    assert judge_case['ledger'].snapshot()['tickets'][0]['status'] == 'reserved'
    with pytest.raises(StopRun, match='requires reconciliation'):
        evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    assert len(judge_case['calls']) == 1


def scoring_context(state: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Isolate the scoring-contract boundary with synthetic candidates and no external retrieval."""
    candidate = {'benchmark_id': 'b1', 'case_id': 'q1', 'candidate_answer': 'Answer.', 'model': 'teacher/model'}
    case = {'question': 'Question?', 'reference_answer': 'Reference.'}
    monkeypatch.setattr(evaluation, 'loaded', lambda run: ([candidate], {'q1': case}, {'b1': {}}, {'protocol': {}}))
    monkeypatch.setattr(evaluation, 'hard_results', lambda *args: {'capture_integrity': {'pass': True}})
    monkeypatch.setenv('SA_CORRECTNESS_JUDGE', 'judge/model')
    monkeypatch.setenv('SA_GROUNDING_JUDGE', 'judge/model')
    return {'config': {'run': str(state['run']), 'metric': 'correctness'}, 'vars': {'benchmark_id': 'b1'}}


def test_score_freezes_request_contract_v5_and_rejects_prior_scoring(
    judge_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preparing current scoring is free; a prior contract cannot be silently rewritten or charged."""
    scoring_context(judge_case, monkeypatch)
    result = evaluation.score(judge_case['run'])
    path = judge_case['run'] / 'scoring.json'
    scoring = read(path)
    assert scoring['judge_request_contract'] == 5 and result['paid_actions'] is False
    scoring.pop('judge_request_contract')
    write(path, scoring)
    with pytest.raises(StopRun, match='Scoring identity changed'):
        evaluation.score(judge_case['run'], execute=True)
    assert not judge_case['calls'] and not judge_case['ledger'].snapshot()['tickets']


@pytest.mark.parametrize('contract', [None, 3, 4, '5', True, 5.0])
def test_assertion_rejects_wrong_request_contract_before_any_charge(
    judge_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch, contract: Any,
) -> None:
    """An obsolete or malformed scoring contract produces an infrastructure error, never a negative."""
    context = scoring_context(judge_case, monkeypatch)
    evaluation.score(judge_case['run'])
    path = judge_case['run'] / 'scoring.json'
    scoring = read(path)
    scoring['judge_request_contract'] = contract
    write(path, scoring)
    result = evaluation.get_assert('Answer.', context)
    assert result['pass'] is False and result['reason'].startswith('SA_INFRASTRUCTURE:')
    assert 'request contract drift' in result['reason']
    assert not judge_case['calls'] and not judge_case['ledger'].snapshot()['tickets']


def test_invalid_paid_verdict_is_reported_as_infrastructure_by_promptfoo(
    judge_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Promptfoo receives a technical failure while the paid response remains in a private receipt."""
    context = scoring_context(judge_case, monkeypatch)
    evaluation.score(judge_case['run'])
    judge_case['response']['choices'][0]['message']['content'] = 'invalid verdict'
    result = evaluation.get_assert('Answer.', context)
    assert result['pass'] is False and result['reason'].startswith('SA_INFRASTRUCTURE:ValueError:')
    assert not verdict_paths(judge_case) and len(judge_case['calls']) == 1
    assert judge_case['ledger'].snapshot()['tickets'][0]['actual']['usd'] == .0004


@pytest.mark.parametrize('kind,score,unsupported', [('correctness', .84, None),
                                                 ('groundedness', .89, 0),
                                                 ('groundedness', .95, 1)])
def test_valid_semantic_failure_preserves_the_original_thresholds(
    judge_case: dict[str, Any], kind: str, score: float, unsupported: int | None,
) -> None:
    """A structurally valid low score or unsupported claim remains a semantic failure."""
    verdict: dict[str, Any] = {'pass': True, 'score': score, 'reason': 'A supported rubric finding.'}
    if kind == 'groundedness':
        verdict.update(unsupported_claims=unsupported, citation_errors=0)
    judge_case['response']['choices'][0]['message']['content'] = canonical(verdict)
    result = evaluation.judge(kind, judge_case['payload'], judge_case['model'], judge_case['run'])
    assert result['pass'] is False and result['score'] == score
    assert result['reason'] == verdict['reason'] and len(verdict_paths(judge_case)) == 1


@pytest.mark.parametrize('field,value', [('prompt_tokens', 1000000), ('completion_tokens', 4097),
                                       ('cost', 2.0)])
def test_observed_budget_breach_is_settled_truthfully_but_cannot_accept_a_verdict(
    judge_case: dict[str, Any], field: str, value: int | float,
) -> None:
    """Real usage above the request allowance must be recorded and stop, not disappear from billing."""
    usage = judge_case['response']['usage']
    usage[field] = value
    usage['total_tokens'] = usage['prompt_tokens'] + usage['completion_tokens']
    with pytest.raises(StopRun, match='exceeded reservation'):
        evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    ticket = judge_case['ledger'].snapshot()['tickets'][0]
    assert ticket['status'] == 'settled' and ticket['actual']['usd'] == usage['cost']
    assert ticket['actual']['tokens'] == usage['total_tokens'] and not verdict_paths(judge_case)
    with pytest.raises(StopRun, match='requires reconciliation'):
        evaluation.judge('correctness', judge_case['payload'], judge_case['model'], judge_case['run'])
    assert len(judge_case['calls']) == 1
