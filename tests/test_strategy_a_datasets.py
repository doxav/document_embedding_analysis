"""Portable task import and reviewed wire-to-training exports, without paid providers."""
from __future__ import annotations

import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.promptfoo_openwebui_eval.posttrain import core as c, datasets, evaluation, vidore
from test_strategy_a_v3 import capture, candidate, export, record
from test_strategy_a_collection_selection import collection


def task_csv(root: Path) -> tuple[Path, dict[str, Any]]:
    """Write a source-backed synthesis task with a separate original reference."""
    (root / 'paper.md').write_text('The original paper reports a revenue of 42 EUR.', encoding='utf-8')
    row = {'dataset': 'bigsurvey', 'task_id': 'paper', 'request_prompt': 'Summarize the provided revenue evidence.',
           'gold_summary': 'Reference synthesis reserved for evaluation only.',
           'source_paths_json': '["paper.md"]', 'expected_path': 'reference.md',
           'algorithm': 'reviewed-algorithm', 'tool_parameters_json': '{"language":"en"}'}
    path = root / 'tasks.csv'
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    return path, {'bigsurvey:paper': {'split': 'train', 'group_id': 'paper', 'language': 'en'}}


def test_csv_round_trip_and_separate_reference(tmp_path: Path) -> None:
    """Reuse task prompts/parameters, hash sources and leave reference text out of documents."""
    csv_path, assignments = task_csv(tmp_path)
    out = tmp_path / 'dataset'
    assert datasets.import_csv(csv_path, tmp_path, assignments, out)['count'] == 1
    assert not out.exists()
    datasets.import_csv(csv_path, tmp_path, assignments, out, execute=True)
    case = vidore.load(out)[0]
    assert case['question'] == 'Summarize the provided revenue evidence.'
    assert case['generation_fields'] == {'algorithm': 'reviewed-algorithm', 'tool_parameters_json': '{"language":"en"}'}
    source = Path(case['source_paths'][0]).read_text()
    assert 'Source-ID: vsrc_' in source and '42 EUR' in source
    assert case['reference_answer'] not in source and case['reference_answer'] not in case['question']
    with pytest.raises(c.StopRun, match='already exists'):
        datasets.import_csv(csv_path, tmp_path, assignments, out, execute=True)
    Path(case['source_paths'][0]).write_text('changed')
    with pytest.raises(c.StopRun, match='bytes changed'):
        vidore.load(out)


@pytest.mark.parametrize('fault', ['missing_assignment', 'unused_assignment', 'holdout_leak',
                                  'reference_in_source', 'missing_source', 'absolute_source', 'duplicate_task'])
def test_csv_rejects_ambiguous_inputs(tmp_path: Path, fault: str) -> None:
    """Reject leakage, missing data and task duplication before writing the dataset."""
    csv_path, assignments = task_csv(tmp_path)
    with csv_path.open() as handle:
        records = list(csv.DictReader(handle))
    if fault == 'missing_assignment': assignments = {}
    elif fault == 'unused_assignment': assignments['bigsurvey:unused'] = assignments['bigsurvey:paper']
    elif fault == 'holdout_leak':
        records.append({**records[0], 'task_id': 'other'})
        assignments['bigsurvey:other'] = {'split': 'test', 'group_id': 'other'}
    elif fault == 'reference_in_source': (tmp_path / 'paper.md').write_text(records[0]['gold_summary'])
    elif fault == 'missing_source': (tmp_path / 'paper.md').unlink()
    elif fault == 'absolute_source': records[0]['source_paths_json'] = json.dumps([str(tmp_path / 'paper.md')])
    else: records.append(records[0])
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    with pytest.raises(c.StopRun):
        datasets.import_csv(csv_path, tmp_path, assignments, tmp_path / 'dataset', execute=True)
    assert not (tmp_path / 'dataset').exists()


def graded_run(root: Path, case: dict[str, Any], accepted: bool, suffix: str) -> tuple[Path, dict[str, Any]]:
    """Create an actual two-call wire artifact and Promptfoo receipt with synthetic responses."""
    run = root / suffix
    cap = capture('42 EUR' if accepted else '99 EUR')
    final = cap['records'][0]
    final['request']['messages'][0]['content'] = case['question']
    final['request']['messages'].insert(0, {'role': 'system', 'content': 'Use evidence and the available search tool.'})
    final['request']['messages'][-1]['content'] = 'Source-ID: ' + case['relevant_source_markers'][0] + '\nRevenue: 42 EUR.'
    first = copy.deepcopy(final)
    first['request']['messages'] = copy.deepcopy(final['request']['messages'][:2])
    first['response']['choices'][0] = {'index': 0, 'finish_reason': 'tool_calls',
                                     'message': copy.deepcopy(final['request']['messages'][2])}
    for rec in (first, final):
        rec['response']['usage'] = {'prompt_tokens': 90, 'completion_tokens': 10, 'total_tokens': 100,
                                    'prompt_tokens_details': {'cached_tokens': 30},
                                    'completion_tokens_details': {'reasoning_tokens': 5}, 'cost': .001}
    cap['records'] = [first, final]
    cand = candidate(cap, bid=suffix)
    cand['candidate_answer'] = '42 EUR' if accepted else '99 EUR'
    cand['case_id'] = case['case_id']
    c.write(run / cand['capture_path'], cap)
    c.write_rows(run / 'candidates.jsonl', [cand]); c.write_rows(run / 'cases.jsonl', [case])
    protocol = {'split': 'train', 'repeats': 1, 'task': 'synthesis'}
    c.write(run / 'selection.json', {'cases_hash': c.digest([case]), 'protocol': protocol, 'protocol_hash': c.digest(protocol)})
    c.write(run / 'COMPLETE.json', {'count': 1, 'candidates_hash': c.digest([cand])})
    c.write(run / 'scoring.json', {'judge_models': {'correctness': 'judge-a', 'groundedness': 'judge-b'}})
    c.write(run / 'promptfoo.json', {'results': [export(cand, accepted)]})
    evaluation.import_run(run)
    review = {'benchmark_id': suffix, 'accepted': accepted, 'finding': 'Evidence and response reviewed.',
              'grades_sha256': c.sha(run / 'grades.jsonl')}
    return run, review


def fixture_dataset(root: Path) -> tuple[Path, list[Path], Path]:
    """Build a reusable complete source dataset with accepted and rejected multi-turn runs."""
    csv_path, assignments = task_csv(root)
    dataset = root / 'dataset'
    datasets.import_csv(csv_path, root, assignments, dataset, execute=True)
    case = vidore.load(dataset)[0]
    entries = [graded_run(root, case, passed, suffix) for passed, suffix in ((True, 'good'), (False, 'bad'))]
    runs, reviews = [x[0] for x in entries], [x[1] for x in entries]
    catalog = root / 'episodes.jsonl'
    c.write_rows(catalog, datasets.catalog_runs(runs, reviews))
    return dataset, runs, catalog


def test_reviewed_multiturn_sft_reasoning_failures_and_pairs(tmp_path: Path) -> None:
    """Export both SFT views and complete labelled trajectories with measured cumulative usage."""
    dataset, runs, catalog = fixture_dataset(tmp_path)
    output = tmp_path / 'build'
    preview = datasets.build_reviewed(dataset, runs, catalog, output, with_reasoning=True)
    assert preview['sft_examples'] == 2 and not output.exists()
    report = datasets.build_reviewed(dataset, runs, catalog, output, with_reasoning=True, execute=True)
    plain, thinking = c.rows(output / 'sft.jsonl'), c.rows(output / 'sft-reasoning.jsonl')
    trajectories = c.rows(output / 'trajectories.jsonl')
    assert [row['label'] for row in trajectories] == ['success', 'failure']
    assert trajectories[0]['usage']['total_tokens'] == 200
    assert trajectories[0]['usage']['llm_calls'] == 2
    assert len(trajectories[1]['capture']['records']) == 2
    assert len(plain) == len(thinking) == 2 and 'PRIVATE' not in c.canonical(plain)
    assert 'PRIVATE' in c.canonical(thinking)
    assert all(row['tools'] and row['messages'][0]['role'] == 'system' for row in plain)
    assert all(sum(m['train'] for m in row['messages']) == 1 for row in plain)
    assert plain[-1]['messages'][-2] == thinking[-1]['messages'][-2]
    assert report['pairs'] == 1 and report['ppo_ready'] is False
    assert c.rows(output / 'pairs.jsonl')[0]['chosen']['content'] == '42 EUR'
    with (output / 'task-results.csv').open() as handle:
        task_results = list(csv.DictReader(handle))
    assert task_results[0]['candidate_answer'] == '42 EUR'
    assert task_results[0]['generation_total_tokens'] == '200'
    assert task_results[0]['generation_cost'] == '0.002'
    assert task_results[0]['openwebui_pipe_model'] == 'teacher'
    assert task_results[0]['algorithm'] == 'reviewed-algorithm'


@pytest.mark.parametrize('fault', ['usage', 'review_hash', 'grade', 'missing_run', 'capture'])
def test_build_rejects_drift(tmp_path: Path, fault: str) -> None:
    """Refuse stale metrics, reviews, captures and incomplete run coverage."""
    dataset, runs, catalog = fixture_dataset(tmp_path)
    episodes = c.rows(catalog)
    if fault == 'usage': episodes[0]['usage']['total_tokens'] += 1
    elif fault == 'review_hash': episodes[0]['manual_review']['grades_sha256'] = 'changed'
    elif fault == 'grade': episodes[0]['grade']['metrics']['correctness']['score'] = .95
    elif fault == 'missing_run': runs = runs[:1]
    else:
        path = runs[0] / 'captures/good.json'
        cap = c.read(path); cap['records'][0]['request']['messages'][0]['content'] = 'changed'; c.write(path, cap)
    c.write_rows(catalog, episodes)
    with pytest.raises(c.StopRun):
        datasets.build_reviewed(dataset, runs, catalog, tmp_path / 'bad-build', execute=True)
    assert not (tmp_path / 'bad-build').exists()


def test_thinking_alias_and_conflict() -> None:
    """Retain captured thinking once and reject conflicting representations."""
    rec = record()
    rec['response']['choices'][0]['message'].pop('reasoning_content')
    rec['response']['choices'][0]['message']['thinking'] = 'CAPTURED'
    case = {'case_id': 'q1', 'group_id': 'g1', 'split': 'train'}
    examples, _ = c.reasoning_sft([c.turn_example(rec, case)], {'x': {'records': [rec]}})
    assert 'CAPTURED' in examples[0]['messages'][-1]['content']
    rec['response']['choices'][0]['message']['reasoning'] = 'DIFFERENT'
    with pytest.raises(c.StopRun, match='disagree'):
        c.captured_reasoning_content(rec['response']['choices'][0]['message'])


def test_manual_veto_stays_a_failure_outside_positive_sft(tmp_path: Path) -> None:
    """Preserve a reviewer veto without treating its passing automatic score as a negative margin."""
    dataset, runs, catalog = fixture_dataset(tmp_path)
    case = vidore.load(dataset)[0]
    run, review = graded_run(tmp_path, case, True, 'veto')
    review['accepted'] = False
    veto = datasets.catalog_runs([run], [review])[0]
    c.write_rows(catalog, c.rows(catalog) + [veto])
    output = tmp_path / 'reviewed-build'
    report = datasets.build_reviewed(dataset, runs + [run], catalog, output, execute=True)
    assert report['trajectories'] == 3 and report['sft_examples'] == 2 and report['pairs'] == 1
    assert c.rows(output / 'trajectories.jsonl')[-1]['label'] == 'failure'


def test_catalog_refuses_promotion_and_unreviewed_attempts(tmp_path: Path) -> None:
    """An automatic failure or an unreviewed run cannot become an accepted solution."""
    csv_path, assignments = task_csv(tmp_path)
    dataset = tmp_path / 'dataset'
    datasets.import_csv(csv_path, tmp_path, assignments, dataset, execute=True)
    run, review = graded_run(tmp_path, vidore.load(dataset)[0], False, 'bad')
    with pytest.raises(c.StopRun, match='missing review'):
        datasets.catalog_runs([run], [])
    review['accepted'] = True
    with pytest.raises(c.StopRun, match='cannot promote'):
        datasets.catalog_runs([run], [review])


def test_cli_build_preview_and_execute(tmp_path: Path) -> None:
    """Exercise the documented entry point without external worktrees or GPU code."""
    dataset, runs, catalog = fixture_dataset(tmp_path)
    output = tmp_path / 'cli-build'
    command = [sys.executable, '-m', 'scripts.promptfoo_openwebui_eval.posttrain', 'build',
               '--dataset', str(dataset), '--runs', *map(str, runs), '--reviewed-catalog', str(catalog),
               '--output', str(output), '--with-reasoning']
    subprocess.run(command, check=True, capture_output=True)
    assert not output.exists()
    subprocess.run(command + ['--execute'], check=True, capture_output=True)
    assert c.read(output / 'build.json')['sft_examples'] == 2


def test_collection_preserves_task_parameters_without_gold(
    collection: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the collector with the actual task CSV adapter and a simulated transport."""
    from types import SimpleNamespace
    from scripts.promptfoo_openwebui_eval.posttrain import integration
    csv_path, assignments = task_csv(tmp_path)
    original_upstream = datasets.upstream
    dataset = tmp_path / 'dataset'
    datasets.import_csv(csv_path, tmp_path, assignments, dataset, execute=True)
    cases = vidore.load(dataset)
    cap = capture()
    cap['records'][0]['response']['usage']['prompt_tokens'] = 90
    captured_rows: list[dict[str, Any]] = []

    def generate(**kwargs: Any) -> dict[str, str]:
        """Record the exact outbound row while leaving collection/artifact handling real."""
        captured_rows.append(kwargs['row'])
        return {'candidate_answer': 'UI answer'}

    def upstream(path: str) -> Any:
        """Substitute only generation, preserving the repository CSV writer."""
        return SimpleNamespace(generate_row=generate) if 'generate_candidate' in path else original_upstream(path)

    def recorder(url: str, operation: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Supply a complete recorded response without charging a provider."""
        return {'inference_limits': payload['inference_limits']} if payload else cap

    monkeypatch.setattr(integration, 'upstream', upstream)
    monkeypatch.setattr(integration, 'recorder_admin', recorder)
    run = tmp_path / 'run'
    integration.collect(cases, collection['resources'], run, collection['spec'])
    assert captured_rows[0]['algorithm'] == 'reviewed-algorithm'
    assert captured_rows[0]['tool_parameters_json'] == '{"language":"en"}'
    assert cases[0]['reference_answer'] not in c.canonical(captured_rows)
    with (run / 'candidates.csv').open() as handle:
        result = next(csv.DictReader(handle))
    assert result['gold_summary'] == cases[0]['reference_answer']
    assert result['generation_total_tokens'] == '100'
    cases[0]['generation_fields']['source_paths_json'] = '["reference.md"]'
    with pytest.raises(c.StopRun, match='protected collection controls'):
        integration.collect(cases, collection['resources'], tmp_path / 'bad-run', collection['spec'])
    assert len(captured_rows) == 1
