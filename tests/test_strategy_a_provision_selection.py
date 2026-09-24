"""Verify explicit provisioning scope without any live Open WebUI requests."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from test_strategy_a_provision import Admin, Client, constants, integration, setup
from scripts.promptfoo_openwebui_eval.posttrain.core import StopRun, digest


@pytest.fixture
def selection(
    setup: tuple[Admin, Client, list[dict[str, Any]], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> dict[str, Any]:
    """Reuse the provision fake and supply one full four-page train document."""
    admin, client, cases, spec = setup
    for case in cases:
        case['case_id'] = case['split'] + '-a'
        case['document_ids'] = [case['split'] + '-document-a']
        case['reference_answer'] = 'Fixture reference; never uploaded.'
    pages = [cases[0]['source_paths'][0]]
    for index in range(4, 8):
        marker = 'vsrc_' + f'{index:020x}'
        path = tmp_path / f'{marker}.md'
        path.write_text(f'Source-ID: {marker}\nOriginal page {index}.', encoding='utf-8')
        pages.append(str(path))
    cases[0]['source_paths'] = [pages[3], pages[0], pages[2], pages[1], pages[0]]
    cases[0]['relevant_source_markers'] = [Path(pages[0]).stem]
    cases.append({'case_id': 'train-b', 'split': 'train', 'source_paths': [pages[4]],
                  'document_ids': ['train-document-b'], 'reference_answer': 'Another reference.'})
    client_calls: list[Path | None] = []

    def clients(cache: Path | None = None) -> tuple[Client, Admin]:
        """Expose the client boundary so invalid selectors must fail before reaching it."""
        client_calls.append(cache)
        return client, admin

    monkeypatch.setattr(integration, 'clients', clients)
    return {'admin': admin, 'client': client, 'cases': cases,
            'spec': {**spec, 'namespace': 'sa-wire-smoke'},
            'pages': pages, 'client_calls': client_calls}


@pytest.mark.parametrize('apply', [False, True])
def test_one_case_keeps_all_document_pages(
    selection: dict[str, Any], tmp_path: Path, apply: bool,
) -> None:
    """A train smoke scope retains non-qrel pages and excludes every unselected document."""
    cases = selection['cases']
    spec = {**selection['spec'], 'splits': ['train'], 'case_ids': ['train-a']}
    original_cases, original_spec = copy.deepcopy(cases), copy.deepcopy(spec)
    output = tmp_path / 'resources.json'
    result = integration.provision(cases, output, spec, apply=apply)
    expected_paths = sorted(selection['pages'][:4])
    assert list(result['splits']) == ['train']
    entry = result['splits']['train']
    assert entry['source_paths'] == expected_paths and entry['model_id'] == 'sa-wire-smoke-train'
    assert result['selection'] == {
        'scope': 'explicit_case_selection', 'splits': ['train'], 'case_ids': ['train-a'],
        'cases_hash': digest([cases[0]]), 'source_scope': 'complete_case_source_paths',
    }
    assert cases == original_cases and spec == original_spec
    admin, client = selection['admin'], selection['client']
    if apply:
        assert result['status'] == 'complete' and output.is_file()
        assert len(admin.knowledge) == len(admin.tools) == 1
        assert list(client.uploads.values()) == [Path(path) for path in expected_paths]
        tool_id, = entry['tool_ids']
        scope = constants(admin.tools[tool_id]['content'])
        assert scope['_FILE_SOURCES'] == {fid: path.stem for fid, path in client.uploads.items()}
        assert entry['file_sources'] == scope['_FILE_SOURCES']
        assert set(scope['_FILE_SOURCES'].values()) > set(cases[0]['relevant_source_markers'])
        assert integration.verify_split_model(admin, result, 'train') == [tool_id]
        assert set(admin.models) == {'sa-blueprint', 'recorded/teacher', 'sa-wire-smoke-train'}
    else:
        assert result['status'] == 'planned' and not output.exists()
        assert all(method == 'GET' for method, _, _ in admin.calls)
        assert not client.uploads and not admin.knowledge and not admin.tools


def test_splits_only_selects_every_case_in_requested_splits(
    selection: dict[str, Any], tmp_path: Path,
) -> None:
    """Omitting case_ids retains all complete source documents within selected splits."""
    result = integration.provision(selection['cases'], tmp_path / 'resources.json',
                                   {**selection['spec'], 'splits': ['train']})
    assert list(result['splits']) == ['train']
    assert result['splits']['train']['source_paths'] == sorted(selection['pages'])
    assert result['selection']['scope'] == 'explicit_split_selection'
    assert result['selection']['case_ids'] == ['train-a', 'train-b']


def test_selection_order_does_not_change_plan(
    selection: dict[str, Any], tmp_path: Path,
) -> None:
    """Canonical split, case and path order makes equivalent source selections identical."""
    plans = []
    for index, (splits, case_ids, cases) in enumerate([
        (['train', 'dev'], ['train-a', 'dev-a'], selection['cases']),
        (['dev', 'train'], ['dev-a', 'train-a'], list(reversed(selection['cases']))),
    ]):
        plans.append(integration.provision(cases, tmp_path / f'plan-{index}.json',
            {**selection['spec'], 'splits': splits, 'case_ids': case_ids}))
    assert plans[0] == plans[1]
    assert list(plans[0]['splits']) == ['train', 'dev']
    assert plans[0]['selection']['case_ids'] == ['dev-a', 'train-a']


def test_omitted_selectors_preserve_default_manifest(
    selection: dict[str, Any], tmp_path: Path,
) -> None:
    """Existing callers still plan every split and receive no new selection metadata."""
    result = integration.provision(selection['cases'], tmp_path / 'resources.json', selection['spec'])
    assert list(result['splits']) == ['train', 'dev', 'test']
    assert 'selection' not in result
    assert result['splits']['train']['source_paths'] == sorted(selection['pages'])
    assert len(result['splits']['dev']['source_paths']) == len(result['splits']['test']['source_paths']) == 1


def test_case_ids_without_splits_keep_all_three_required(
    selection: dict[str, Any], tmp_path: Path,
) -> None:
    """The default split set remains train/dev/test even when only case_ids is explicit."""
    result = integration.provision(selection['cases'], tmp_path / 'resources.json',
        {**selection['spec'], 'case_ids': ['test-a', 'train-a', 'dev-a']})
    assert list(result['splits']) == ['train', 'dev', 'test']
    assert result['selection']['splits'] == ['train', 'dev', 'test']
    assert result['selection']['case_ids'] == ['dev-a', 'test-a', 'train-a']
    assert result['splits']['train']['source_paths'] == sorted(selection['pages'][:4])


@pytest.mark.parametrize('options,error', [
    ({'splits': []}, 'nonempty list'),
    ({'splits': None}, 'nonempty list'),
    ({'splits': 'train'}, 'nonempty list'),
    ({'splits': ('train',)}, 'nonempty list'),
    ({'splits': True}, 'nonempty list'),
    ({'splits': ['']}, 'Unknown provisioning split'),
    ({'splits': ['train', 'other']}, 'Unknown provisioning split'),
    ({'splits': ['train', []]}, 'Unknown provisioning split'),
    ({'splits': ['train', 'train']}, 'unique'),
    ({'splits': ['train'], 'case_ids': []}, 'nonempty list'),
    ({'splits': ['train'], 'case_ids': None}, 'nonempty list'),
    ({'splits': ['train'], 'case_ids': 'train-a'}, 'nonempty list'),
    ({'splits': ['train'], 'case_ids': ('train-a',)}, 'nonempty list'),
    ({'splits': ['train'], 'case_ids': True}, 'nonempty list'),
    ({'splits': ['train'], 'case_ids': ['']}, 'nonempty strings'),
    ({'splits': ['train'], 'case_ids': ['train-a', []]}, 'nonempty strings'),
    ({'splits': ['train'], 'case_ids': ['train-a', 'train-a']}, 'unique'),
    ({'splits': ['train'], 'case_ids': ['unknown']}, 'selected splits'),
    ({'splits': ['train'], 'case_ids': ['dev-a']}, 'selected splits'),
    ({'splits': ['train'], 'case_ids': ['test-a']}, 'selected splits'),
    ({'splits': ['train', 'dev'], 'case_ids': ['train-a']}, 'Empty split'),
    ({'case_ids': ['train-a']}, 'Empty split'),
])
@pytest.mark.parametrize('apply', [False, True])
def test_invalid_selectors_precede_clients_and_artifacts(
    selection: dict[str, Any], tmp_path: Path, options: dict[str, Any], error: str, apply: bool,
) -> None:
    """Malformed, ambiguous and cross-split selectors cannot start any external operation."""
    output = tmp_path / 'resources.json'
    with pytest.raises(StopRun, match=error):
        integration.provision(selection['cases'], output, {**selection['spec'], **options}, apply=apply)
    assert not selection['client_calls'] and not selection['admin'].calls
    assert not selection['client'].uploads and not output.exists()


@pytest.mark.parametrize('fault', ['missing_id', 'invalid_id', 'duplicate_id', 'empty_cases', 'empty_paths'])
def test_ambiguous_or_empty_source_scope_fails_before_clients(
    selection: dict[str, Any], tmp_path: Path, fault: str,
) -> None:
    """An explicitly selected split must provide unambiguous cases and source paths."""
    cases = copy.deepcopy(selection['cases'])
    if fault == 'missing_id':
        del cases[0]['case_id']
    elif fault == 'invalid_id':
        cases[0]['case_id'] = []
    elif fault == 'duplicate_id':
        cases.append(copy.deepcopy(cases[0]))
    elif fault == 'empty_cases':
        cases = [case for case in cases if case['split'] != 'train']
    else:
        for case in cases:
            if case['split'] == 'train':
                case['source_paths'] = []
    output = tmp_path / 'resources.json'
    with pytest.raises(StopRun):
        integration.provision(cases, output, {**selection['spec'], 'splits': ['train']}, apply=True)
    assert not selection['client_calls'] and not selection['admin'].calls and not output.exists()


def test_unselected_split_cases_do_not_affect_explicit_scope(
    selection: dict[str, Any], tmp_path: Path,
) -> None:
    """Case selection checks only the requested split without pulling in holdout records."""
    cases = copy.deepcopy(selection['cases'])
    cases.append(copy.deepcopy(next(case for case in cases if case['split'] == 'test')))
    plan = integration.provision(cases, tmp_path / 'resources.json',
                                {**selection['spec'], 'splits': ['train'], 'case_ids': ['train-a']})
    assert list(plan['splits']) == ['train'] and plan['selection']['case_ids'] == ['train-a']
