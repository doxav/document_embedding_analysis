"""Verify the real collection selection path with simulated transports and no live services."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.promptfoo_openwebui_eval.posttrain import integration
from scripts.promptfoo_openwebui_eval.posttrain.core import StopRun, digest, read, rows


@pytest.fixture
def collection(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Keep collection and artifact writing real while recording every external interaction."""
    cases = [
        {'case_id': case_id, 'split': split, 'dataset': 'fixture', 'question': f'Question {case_id}'}
        for case_id, split in [('train-a', 'train'), ('dev-a', 'dev'), ('train-b', 'train'),
                               ('test-a', 'test'), ('train-c', 'train')]
    ]
    resources = {
        'status': 'complete', 'tool_ids': ['rag'], 'rag_params': {}, 'rag_fingerprint': 'fp',
        'splits': {split: {'knowledge_id': f'kb-{split}', 'model_id': f'sa-{split}'}
                   for split in ('train', 'dev', 'test')},
    }
    spec = {'split': 'train', 'bridge_url': 'bridge', 'recorder_url': 'recorder',
            'upstream_model': 'teacher'}
    external: list[str] = []
    generated: list[str] = []
    csv_writer = integration.upstream('lib/bundle_common.py').write_csv_rows

    def clients(cache: Path | None = None) -> tuple[None, None]:
        """Record that selection has completed before the client boundary is reached."""
        external.append('clients')
        return None, None

    def fingerprint(admin: Any, tool_ids: list[str], params: dict[str, Any]) -> str:
        """Accept unchanged fixture tools without contacting Open WebUI."""
        external.append('fingerprint')
        return 'fp'

    def generate(**kwargs: Any) -> dict[str, str]:
        """Record the chosen case passed to the unchanged generator contract."""
        external.append('generate')
        generated.append(kwargs['row']['task_id'])
        return {'candidate_answer': 'UI answer'}

    def upstream(path: str) -> SimpleNamespace:
        """Reuse the actual CSV writer and replace only the live generator transport."""
        if path == 'scripts/generate_candidate_csv.py':
            return SimpleNamespace(generate_row=generate)
        assert path == 'lib/bundle_common.py'
        return SimpleNamespace(write_csv_rows=csv_writer)

    def recorder(url: str, operation: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return one fixture final answer per completed recorder session."""
        external.append('recorder-' + operation)
        if operation == 'begin':
            assert payload is not None
            return {'ok': True, 'inference_limits': payload['inference_limits']}
        assert operation == 'end'
        return {'complete': True, 'records': [
            {'response': {'choices': [{'message': {'content': 'Captured answer'}}]}}
        ]}

    monkeypatch.setattr(integration, 'clients', clients)
    monkeypatch.setattr(integration, 'rag_fingerprint', fingerprint)
    monkeypatch.setattr(integration, 'upstream', upstream)
    monkeypatch.setattr(integration, 'recorder_admin', recorder)
    return {'cases': cases, 'resources': resources, 'spec': spec,
            'external': external, 'generated': generated}


@pytest.mark.parametrize('repeats', [1, 2])
def test_explicit_selection_preserves_order_and_omits_other_cases(
    collection: dict[str, Any], tmp_path: Path, repeats: int,
) -> None:
    """An explicit allowlist controls generation order and every saved selection artifact."""
    requested = ['train-c', 'train-a']
    spec = {**collection['spec'], 'case_ids': requested, 'repeats': repeats}
    run = tmp_path / 'run'
    original = copy.deepcopy(collection['cases'])
    result = integration.collect(collection['cases'], collection['resources'], run, spec)
    expected = ['train-c'] * repeats + ['train-a'] * repeats
    assert result['count'] == len(expected)
    assert collection['generated'] == expected
    assert [item['case_id'] for item in rows(run / 'candidates.jsonl')] == expected
    selected = [next(item for item in original if item['case_id'] == case_id) for case_id in requested]
    assert rows(run / 'cases.jsonl') == selected
    assert read(run / 'selection.json')['case_ids'] == requested
    assert read(run / 'selection.json')['cases_hash'] == digest(selected)
    assert collection['cases'] == original and spec['case_ids'] == requested


@pytest.mark.parametrize('options,expected', [
    ({}, ['train-a', 'train-b', 'train-c']),
    ({'max_cases': 1}, ['train-a']),
    ({'max_cases': None}, ['train-a', 'train-b', 'train-c']),
    ({'case_ids': ['train-c'], 'max_cases': None}, ['train-c']),
])
def test_default_and_prefix_selection_remain_compatible(
    collection: dict[str, Any], tmp_path: Path, options: dict[str, Any], expected: list[str],
) -> None:
    """Keep the existing full-split and max_cases paths, including the first-live prefix."""
    run = tmp_path / 'run'
    result = integration.collect(collection['cases'], collection['resources'], run,
                                 {**collection['spec'], **options})
    assert result['count'] == len(expected) and collection['generated'] == expected
    assert read(run / 'selection.json')['case_ids'] == expected


@pytest.mark.parametrize('options,error', [
    ({'case_ids': []}, 'nonempty list'),
    ({'case_ids': None}, 'nonempty list'),
    ({'case_ids': 'train-a'}, 'nonempty list'),
    ({'case_ids': ('train-a',)}, 'nonempty list'),
    ({'case_ids': {'train-a': True}}, 'nonempty list'),
    ({'case_ids': True}, 'nonempty list'),
    ({'case_ids': ['']}, 'nonempty strings'),
    ({'case_ids': ['train-a', 7]}, 'nonempty strings'),
    ({'case_ids': ['train-a', 'train-a']}, 'unique'),
    ({'case_ids': ['unknown']}, 'selected split'),
    ({'case_ids': ['dev-a']}, 'selected split'),
    ({'case_ids': ['test-a']}, 'selected split'),
    ({'case_ids': ['train-a'], 'max_cases': 1}, 'mutually exclusive'),
    ({'case_ids': ['train-a'], 'max_cases': 0}, 'mutually exclusive'),
])
def test_invalid_selection_has_no_client_calls_or_artifacts(
    collection: dict[str, Any], tmp_path: Path, options: dict[str, Any], error: str,
) -> None:
    """Reject malformed, foreign-split or conflicting selectors before any external action."""
    run = tmp_path / 'run'
    with pytest.raises(StopRun, match=error):
        integration.collect(collection['cases'], collection['resources'], run,
                            {**collection['spec'], **options})
    assert not collection['external'] and not collection['generated'] and not run.exists()


@pytest.mark.parametrize('split', ['dev', 'test'])
def test_holdout_selection_rejects_training_id_before_clients(
    collection: dict[str, Any], tmp_path: Path, split: str,
) -> None:
    """An explicit selector cannot import a training case into either holdout split."""
    spec = {**collection['spec'], 'split': split, 'case_ids': ['train-a'], 'unseal_test': True,
            'deployment': {'model_id': 'base', 'revision': 'revision', 'runtime_image': 'image',
                           'weights_hash': 'hash', 'kv_cache_dtype': 'dtype'}}
    run = tmp_path / 'run'
    with pytest.raises(StopRun, match='selected split'):
        integration.collect(collection['cases'], collection['resources'], run, spec)
    assert not collection['external'] and not run.exists()


def test_ambiguous_source_case_ids_rejected_before_clients(
    collection: dict[str, Any], tmp_path: Path,
) -> None:
    """Do not silently choose one record when the source split contains duplicate IDs."""
    cases = collection['cases'] + [copy.deepcopy(collection['cases'][0])]
    run = tmp_path / 'run'
    with pytest.raises(StopRun, match='Duplicate case IDs'):
        integration.collect(cases, collection['resources'], run,
                            {**collection['spec'], 'case_ids': ['train-a']})
    assert not collection['external'] and not run.exists()
