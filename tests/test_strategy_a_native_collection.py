"""Keep recorder sessions open while the isolated native bridge cannot confirm task exit."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from test_strategy_a_collection_selection import collection, integration
from scripts.promptfoo_openwebui_eval.posttrain.core import StopRun, read


@pytest.fixture
def native_collection(collection: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Reuse real collection while substituting only the native task-state HTTP boundary."""
    collection['spec'].update(bridge_url='http://127.0.0.1:8014/generate', max_cases=1,
        native_bridge_state_url='http://127.0.0.1:8014/strategy-a/native-state')
    collection['states'] = [{'can_begin': True, 'safe_to_end': True, 'episode_timeout_seconds': 600},
                            {'can_begin': True, 'safe_to_end': True}]

    def state(url: str) -> dict[str, Any]:
        """Return the pre-generation and post-generation barrier observations in order."""
        assert url == collection['spec']['native_bridge_state_url']
        collection['external'].append('native-state')
        result = collection['states'].pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(integration, 'native_bridge_state', state)
    return collection


def fail_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make only the historical generator fail after its recorder session has opened."""
    original = integration.upstream

    def generate(**kwargs: Any) -> dict[str, Any]:
        """Represent a failed or timed-out native transport, without making a request."""
        raise RuntimeError('Simulated generation transport error')

    def upstream(path: str) -> Any:
        """Preserve the CSV writer and other existing adapter modules."""
        if path == 'scripts/generate_candidate_csv.py':
            return SimpleNamespace(generate_row=generate)
        return original(path)

    monkeypatch.setattr(integration, 'upstream', upstream)


def test_success_closes_capture_after_native_barrier(
    native_collection: dict[str, Any], tmp_path: Path,
) -> None:
    """A confirmed inactive native task allows the existing capture completion path."""
    case = native_collection
    run = tmp_path / 'run'
    assert integration.collect(case['cases'], case['resources'], run, case['spec'])['count'] == 1
    external = case['external']
    assert external.index('recorder-begin') < external.index('generate') < external.index('recorder-end')
    assert external[-2:] == ['native-state', 'recorder-end']
    assert read(run / 'selection.json')['protocol']['native_transport'] == 'socketio-native-v1'


def test_pending_previous_operation_prevents_recorder_begin(
    native_collection: dict[str, Any], tmp_path: Path,
) -> None:
    """An unresolved prior task cannot start another capture or candidate generation."""
    case = native_collection
    case['states'] = [{'can_begin': False, 'safe_to_end': False}]
    with pytest.raises(StopRun, match='Reconcile the prior native operation'):
        integration.collect(case['cases'], case['resources'], tmp_path / 'run', case['spec'])
    assert 'recorder-begin' not in case['external'] and not case['generated']


@pytest.mark.parametrize('after', [
    {'can_begin': False, 'safe_to_end': False, 'chat_id': 'chat-1', 'task_id': 'task-1'},
    RuntimeError('State endpoint unavailable'),
])
@pytest.mark.parametrize('generation_fails', [False, True])
def test_unconfirmed_exit_keeps_recorder_open(
    native_collection: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path, after: Any, generation_fails: bool,
) -> None:
    """Never close a recorder between native turns after failure or uncertain task status."""
    case = native_collection
    case['states'][1] = after
    if generation_fails:
        fail_generation(monkeypatch)
    run = tmp_path / 'run'
    with pytest.raises(StopRun, match='recorder session remains open'):
        integration.collect(case['cases'], case['resources'], run, case['spec'])
    assert 'recorder-begin' in case['external'] and 'recorder-end' not in case['external']
    receipt = read(run / 'RECONCILIATION_REQUIRED.json')
    assert receipt['recorder_session_retained'] is True and receipt['benchmark_id'].startswith('sa-')
    assert not list((run / 'captures').glob('*.json')) and not (run / 'COMPLETE.json').exists()


def test_failed_generation_with_confirmed_exit_can_close_capture(
    native_collection: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A technical failure with a proven inactive task still preserves its completed capture."""
    case = native_collection
    fail_generation(monkeypatch)
    run = tmp_path / 'run'
    with pytest.raises(RuntimeError, match='Simulated generation transport error'):
        integration.collect(case['cases'], case['resources'], run, case['spec'])
    assert case['external'][-2:] == ['native-state', 'recorder-end']
    assert len(list((run / 'captures').glob('*.json'))) == 1
    assert not (run / 'RECONCILIATION_REQUIRED.json').exists()


@pytest.mark.parametrize('bridge_url,state_url', [
    ('http://127.0.0.1:8014/generate', 'http://127.0.0.1:8015/strategy-a/native-state'),
    ('http://127.0.0.1:8014/generate', 'http://localhost:8014/strategy-a/native-state'),
    ('http://127.0.0.1:8014/generate', 'https://127.0.0.1:8014/strategy-a/native-state'),
    ('http://user@127.0.0.1:8014/generate', 'http://user@127.0.0.1:8014/strategy-a/native-state'),
    ('http://127.0.0.1:bad/generate', 'http://127.0.0.1:bad/strategy-a/native-state'),
    ('http://[bad/generate', 'http://[bad/strategy-a/native-state'),
    ('file:///generate', 'file:///strategy-a/native-state'),
    ('http://127.0.0.1:8014/generate?x=1', 'http://127.0.0.1:8014/strategy-a/native-state'),
    ('http://127.0.0.1:8014/generate', 'http://127.0.0.1:8014/strategy-a/native-state?x=1'),
    ('http://127.0.0.1:8014/generate', 'http://127.0.0.1:8014/strategy-a/native-state#fragment'),
    ('http://127.0.0.1:8014/generate', None),
])
def test_native_state_origin_must_match_before_clients_or_artifacts(
    native_collection: dict[str, Any], tmp_path: Path, bridge_url: str, state_url: Any,
) -> None:
    """A different idle server or malformed URL must not authorize recorder closure."""
    case = native_collection
    case['spec'].update(bridge_url=bridge_url, native_bridge_state_url=state_url)
    run = tmp_path / 'run'
    with pytest.raises(StopRun, match='[Nn]ative bridge'):
        integration.collect(case['cases'], case['resources'], run, case['spec'])
    assert not case['external'] and not case['generated'] and not run.exists()
