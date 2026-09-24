"""Reviewed exclusions preserve source gold and fail closed before writing data."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from test_strategy_a_v3 import REV, snapshots
from scripts.promptfoo_openwebui_eval.posttrain import core, vidore


def exclusion(query_id: str = '0-fr') -> dict[str, str]:
    """Return a review record for a real query in the shared source fixture."""
    return {'dataset': next(iter(vidore.CORPORA)), 'revision': REV,
            'query_id': query_id, 'reason': 'Relevant table lost its visual indicators.'}


def test_review_excludes_before_selection_and_preserves_sources(tmp_path: Path) -> None:
    """An excluded language query removes its document group without synthesizing gold."""
    source = snapshots()
    original = copy.deepcopy(source)
    record = exclusion()
    manifest = vidore.build(source, tmp_path / 'reviewed', exclusions=[record])
    cases = vidore.load(tmp_path / 'reviewed')
    affected = [case for case in cases if case['dataset'] == record['dataset']]
    assert len(cases) == 50
    assert all(case['document_ids'] != ['doc0'] for case in affected)
    assert all(case['reference_answer'] in case['raw_answers'] for case in cases)
    assert manifest['review_exclusions'] == [record]
    assert manifest['review_exclusions_hash'] == core.digest([record])
    assert manifest['excluded']['review_exclusion'] == 1
    assert source == original


def test_default_and_empty_exclusions_preserve_manifest(tmp_path: Path) -> None:
    """Opting out keeps existing manifests and deterministic selections identical."""
    default = vidore.build(snapshots(), tmp_path / 'default')
    empty = vidore.build(snapshots(), tmp_path / 'empty', exclusions=[])
    assert default == empty
    assert 'review_exclusions' not in default


@pytest.mark.parametrize('records', [
    {}, [None], [dict(exclusion(), extra='unexpected')],
    [{key: value for key, value in exclusion().items() if key != 'reason'}],
    [dict(exclusion(), reason=' ')], [dict(exclusion(), query_id=0)],
    [dict(exclusion(), dataset='vidore/unknown')],
    [dict(exclusion(), revision='3' * 40)], [dict(exclusion(), revision='invalid')],
    [dict(exclusion(), query_id='absent')], [exclusion(), exclusion()],
])
def test_invalid_review_exclusions_fail_before_output(tmp_path: Path, records: Any) -> None:
    """Malformed, stale, unknown and duplicate exclusions must never be ignored."""
    with pytest.raises(core.StopRun, match='[Rr]eview exclusion'):
        vidore.build(snapshots(), tmp_path / 'invalid', exclusions=records)
    assert not (tmp_path / 'invalid').exists()


def test_exclusions_cannot_synthesize_missing_coverage(tmp_path: Path) -> None:
    """Removing too many bilingual groups fails with no partially valid dataset."""
    with pytest.raises(core.StopRun, match='fewer than five disjoint bilingual'):
        vidore.build(snapshots(), tmp_path / 'short', exclusions=[exclusion(), exclusion('1-fr')])
    assert not (tmp_path / 'short').exists()


def test_exclusion_order_is_deterministic(tmp_path: Path) -> None:
    """Review record ordering cannot change the dataset or its audit identity."""
    first = exclusion()
    second = dict(exclusion(), dataset=list(vidore.CORPORA)[1])
    left = vidore.build(snapshots(), tmp_path / 'left', exclusions=[first, second])
    right = vidore.build(snapshots(), tmp_path / 'right', exclusions=[second, first])
    assert left == right


def test_duplicate_source_query_is_not_an_unambiguous_exclusion(tmp_path: Path) -> None:
    """A reused source identifier cannot silently exclude multiple source questions."""
    source = snapshots()
    source[0]['queries'].append(copy.deepcopy(source[0]['queries'][0]))
    with pytest.raises(core.StopRun, match='exactly one source query'):
        vidore.build(source, tmp_path / 'ambiguous', exclusions=[exclusion()])
    assert not (tmp_path / 'ambiguous').exists()


def test_exclusion_audit_drift_is_rejected(tmp_path: Path) -> None:
    """Changing an archived review reason invalidates the exclusion audit hash."""
    directory = tmp_path / 'reviewed'
    vidore.build(snapshots(), directory, exclusions=[exclusion()])
    manifest = core.read(directory / 'manifest.json')
    manifest['review_exclusions'][0]['reason'] = 'Changed after review'
    core.write(directory / 'manifest.json', manifest)
    with pytest.raises(core.StopRun, match='Review exclusions manifest drift'):
        vidore.load(directory)


def fixed_plan(tmp_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create a verified baseline and an explicit document assignment fixture."""
    source = snapshots()
    baseline = tmp_path / 'baseline'
    vidore.build(source, baseline)
    documents = core.read(baseline / 'documents.json')
    unique = {(d['dataset'], d['doc_id']): {k: d[k] for k in ('dataset', 'revision', 'doc_id', 'split')}
              for d in documents}
    return source, {'baseline': str(baseline), 'baseline_manifest_sha256': core.sha(baseline / 'manifest.json'),
                    'documents': list(unique.values()), 'reserved_documents': []}


def test_fixed_split_replacement_preserves_all_holdout_cases(tmp_path: Path) -> None:
    """A reviewed train replacement cannot rerank any original dev/test question."""
    source, plan = fixed_plan(tmp_path)
    name = source[0]['dataset']
    item = next(d for d in plan['documents'] if d['dataset'] == name and d['split'] == 'train')
    old_doc = item['doc_id']
    used = {d['doc_id'] for d in plan['documents'] if d['dataset'] == name}
    item['doc_id'] = next(p['doc_id'] for p in source[0]['corpus'] if p['doc_id'] not in used)
    record = exclusion(old_doc.removeprefix('doc') + '-fr')
    target = tmp_path / 'fixed'
    manifest = vidore.build(source, target, exclusions=[record], split_plan=plan)
    old = core.rows(Path(plan['baseline']) / 'cases.jsonl')
    new = core.rows(target / 'cases.jsonl')
    assert sorted((c for c in old if c['split'] != 'train'), key=lambda c: c['case_id']) == sorted(
        (c for c in new if c['split'] != 'train'), key=lambda c: c['case_id'])
    assert manifest['splits'] == {'train': 30, 'dev': 10, 'test': 10}
    assert len(vidore.load(target)) == 50
    assert all(c['document_ids'] != [old_doc] for c in new if c['dataset'] == name)


@pytest.mark.parametrize('fault', ['move_holdout', 'promote_train', 'duplicate', 'reserved_collision',
                                   'bad_revision', 'missing_doc', 'bad_count', 'bad_split', 'baseline_hash'])
def test_split_plan_rejects_collisions_and_drift(tmp_path: Path, fault: str) -> None:
    """Invalid plans fail before any partially valid dataset is written."""
    source, plan = fixed_plan(tmp_path)
    first = plan['documents'][0]
    if fault == 'move_holdout':
        train = next(d for d in plan['documents'] if d['split'] == 'train')
        dev = next(d for d in plan['documents'] if d['dataset'] == train['dataset'] and d['split'] == 'dev')
        train['split'], dev['split'] = dev['split'], train['split']
    elif fault == 'promote_train':
        # The balanced swap is rejected even when checked from the train side first.
        train = next(d for d in plan['documents'] if d['split'] == 'train')
        test = next(d for d in plan['documents'] if d['dataset'] == train['dataset'] and d['split'] == 'test')
        train['split'], test['split'] = test['split'], train['split']
    elif fault == 'duplicate':
        plan['documents'].append(dict(first))
    elif fault == 'reserved_collision':
        plan['reserved_documents'].append(dict(first, split='test'))
    elif fault == 'bad_revision':
        first['revision'] = '0' * 40
    elif fault == 'missing_doc':
        first['doc_id'] = 'absent'
    elif fault == 'bad_count':
        plan['documents'].pop()
    elif fault == 'bad_split':
        first['split'] = 'validation'
    else:
        plan['baseline_manifest_sha256'] = '0' * 64
    with pytest.raises(core.StopRun, match='Split plan'):
        vidore.build(source, tmp_path / 'invalid', split_plan=plan)
    assert not (tmp_path / 'invalid').exists()


def test_fixed_ineligible_document_does_not_resplit(tmp_path: Path) -> None:
    """A lost train language must request a reviewed replacement instead of reranking."""
    source, plan = fixed_plan(tmp_path)
    item = next(d for d in plan['documents'] if d['dataset'] == source[0]['dataset'] and d['split'] == 'train')
    record = exclusion(item['doc_id'].removeprefix('doc') + '-fr')
    with pytest.raises(core.StopRun, match='fixed document lacks bilingual'):
        vidore.build(source, tmp_path / 'invalid', exclusions=[record], split_plan=plan)
    assert not (tmp_path / 'invalid').exists()


def test_split_plan_cannot_exclude_sealed_query(tmp_path: Path) -> None:
    """Exclusions cannot tune a baseline holdout even if another source query exists."""
    source, plan = fixed_plan(tmp_path)
    case = next(c for c in core.rows(Path(plan['baseline']) / 'cases.jsonl')
                if c['dataset'] == source[0]['dataset'] and c['split'] == 'test')
    with pytest.raises(core.StopRun, match='cannot exclude a sealed holdout'):
        vidore.build(source, tmp_path / 'invalid', exclusions=[exclusion(case['query_id'])], split_plan=plan)
    assert not (tmp_path / 'invalid').exists()


def test_split_plan_audit_drift_is_rejected(tmp_path: Path) -> None:
    """Changing the stored plan invalidates its independent audit hash."""
    source, plan = fixed_plan(tmp_path)
    directory = tmp_path / 'fixed'
    vidore.build(source, directory, split_plan=plan)
    manifest = core.read(directory / 'manifest.json')
    manifest['split_plan']['documents'][0]['split'] = 'test'
    core.write(directory / 'manifest.json', manifest)
    with pytest.raises(core.StopRun, match='Split plan manifest drift'):
        vidore.load(directory)


def test_cli_accepts_explicit_split_plan(tmp_path: Path) -> None:
    """The public command exposes the reviewed plan instead of a private builder bypass."""
    from scripts.promptfoo_openwebui_eval.posttrain.__main__ import parser
    args = parser().parse_args(['vidore', '--snapshot', 'snapshot.json', '--output', str(tmp_path),
                               '--split-plan', 'plan.json'])
    assert args.split_plan == Path('plan.json')


def test_split_plan_rejects_changed_sealed_reference(tmp_path: Path) -> None:
    """Reusing a revision string cannot authorize changed holdout question or gold."""
    source, plan = fixed_plan(tmp_path)
    case = next(c for c in core.rows(Path(plan['baseline']) / 'cases.jsonl')
                if c['dataset'] == source[0]['dataset'] and c['split'] == 'test')
    query = next(q for q in source[0]['queries'] if str(q['query_id']) == case['query_id'])
    query['answer'] = 'Changed gold'
    with pytest.raises(core.StopRun, match='sealed holdout reference drift'):
        vidore.build(source, tmp_path / 'invalid', split_plan=plan)
    assert not (tmp_path / 'invalid').exists()


def test_split_plan_preserves_reservations_across_rebuilds(tmp_path: Path) -> None:
    """An extra document sealed by an earlier experiment never becomes a train reserve."""
    source, plan = fixed_plan(tmp_path)
    name = source[0]['dataset']
    used = {d['doc_id'] for d in plan['documents'] if d['dataset'] == name}
    extra = next(p['doc_id'] for p in source[0]['corpus'] if p['doc_id'] not in used)
    plan['reserved_documents'] = [{'dataset': name, 'revision': REV, 'doc_id': extra, 'split': 'test'}]
    first = tmp_path / 'first'
    vidore.build(source, first, split_plan=plan)
    next_plan = dict(plan, baseline=str(first), baseline_manifest_sha256=core.sha(first / 'manifest.json'))
    vidore.build(source, tmp_path / 'good', split_plan=next_plan)
    next_plan['reserved_documents'] = []
    with pytest.raises(core.StopRun, match='cannot forget previously reserved'):
        vidore.build(source, tmp_path / 'invalid', split_plan=next_plan)
    assert not (tmp_path / 'invalid').exists()



def test_explicit_dev_source_repair_preserves_documents_and_sealed_test(tmp_path: Path) -> None:
    """Only an explicitly excluded dev annotation can change before evaluation."""
    source, plan = fixed_plan(tmp_path)
    old = core.rows(Path(plan['baseline']) / 'cases.jsonl')
    case = next(c for c in old if c['dataset'] == source[0]['dataset'] and c['split'] == 'dev' and c['language'] == 'fr')
    query = next(q for q in source[0]['queries'] if str(q['query_id']) == case['query_id'])
    replacement = dict(query, query_id='reviewed-dev-alternative', query='Different genuine fixture question')
    source[0]['queries'].append(replacement)
    source[0]['qrels'] += [dict(r, query_id=replacement['query_id']) for r in list(source[0]['qrels'])
                          if str(r['query_id']) == case['query_id']]
    record = exclusion(case['query_id'])
    with pytest.raises(core.StopRun, match='cannot exclude a sealed holdout'):
        vidore.build(source, tmp_path / 'default-rejects', exclusions=[record], split_plan=plan)
    plan['replace_excluded_dev_queries'] = True
    target = tmp_path / 'repaired'
    vidore.build(source, target, exclusions=[record], split_plan=plan)
    new = core.rows(target / 'cases.jsonl')
    assert next(c for c in new if c['case_id'] == case['case_id'])['query_id'] == replacement['query_id']
    assert sorted((c for c in new if c['case_id'] != case['case_id']), key=lambda c: c['case_id']) == sorted(
        (c for c in old if c['case_id'] != case['case_id']), key=lambda c: c['case_id'])
    assert sorted(core.read(target / 'documents.json'), key=lambda d: d['path']) == sorted(
        core.read(Path(plan['baseline']) / 'documents.json'), key=lambda d: d['path'])
    test = next(c for c in old if c['dataset'] == source[0]['dataset'] and c['split'] == 'test')
    with pytest.raises(core.StopRun, match='cannot exclude a sealed holdout'):
        vidore.build(source, tmp_path / 'test-rejects', exclusions=[exclusion(test['query_id'])], split_plan=plan)


@pytest.mark.parametrize('flag', [None, 1, 'true', []])
def test_dev_repair_flag_requires_explicit_boolean(tmp_path: Path, flag: Any) -> None:
    """Truthy strings cannot silently opt into changing development annotations."""
    source, plan = fixed_plan(tmp_path)
    plan['replace_excluded_dev_queries'] = flag
    with pytest.raises(core.StopRun, match='option must be boolean'):
        vidore.build(source, tmp_path / 'invalid', split_plan=plan)
    assert not (tmp_path / 'invalid').exists()
