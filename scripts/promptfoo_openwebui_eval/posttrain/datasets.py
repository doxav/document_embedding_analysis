"""Task adapters and reviewed wire datasets, independent of a GPU training backend."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

from . import SCHEMA
from .core import (build_data, digest, inside, metrics, reasoning_sft, require,
                   rows, sha, source_marker, strict_json, task_result_row, turn_example, validate_cases, write, write_rows, write_text)
from .evaluation import loaded, validated_grades
from .integration import TASK_GENERATION_FIELDS, upstream
from .vidore import load


def import_csv(csv_path: Path, source_root: Path, assignments: dict[str, Any],
               output: Path, execute: bool = False) -> dict[str, Any]:
    """Adapt existing task CSVs using explicit group/split assignments and original sources."""
    require(csv_path.is_file() and source_root.is_dir(), 'CSV and source root must exist')
    require(not output.exists(), 'Dataset already exists')
    records = upstream('lib/bundle_common.py').read_csv_rows(csv_path)
    require(records and isinstance(assignments, dict), 'Task rows and explicit split assignments required')
    cases: list[dict[str, Any]] = []
    documents: dict[str, dict[str, Any]] = {}
    bodies: dict[str, str] = {}
    seen: set[str] = set()
    for row in records:
        dataset, task_id = row.get('dataset'), row.get('task_id')
        require(dataset and task_id, 'Task dataset and task_id required')
        key = dataset + ':' + task_id
        require(key not in seen and key in assignments, 'Duplicate task or missing explicit split assignment')
        seen.add(key)
        assignment = assignments[key]
        require(isinstance(assignment, dict) and assignment.get('split') in {'train', 'dev', 'test'}
                and isinstance(assignment.get('group_id'), str) and assignment['group_id'],
                'Each task needs a split and source group_id')
        question = row.get('request_prompt') or row.get('query')
        reference = row.get('gold_summary')
        require(question and reference, 'Original request_prompt/query and gold_summary required')
        require(reference.strip() not in question, 'Reference answer occurs in teacher prompt')
        sources = strict_json(row.get('source_paths_json') or '[]')
        require(isinstance(sources, list) and sources and all(isinstance(p, str) for p in sources),
                'Source-backed task required; open retrieval without declared sources needs a separate adapter')
        paths: list[str] = []
        markers: list[str] = []
        document_ids: list[str] = []
        for relative in sources:
            source = inside(source_root, relative)
            require(source.is_file(), 'Task source file missing')
            require(relative not in {row.get('expected_path'), row.get('dea_solution_path')},
                    'Reference documents must not be uploaded as sources')
            source_bytes = source.read_bytes()
            original = source_bytes.decode('utf-8')
            require(original.strip() and reference.strip() not in original, 'Empty source or reference leakage')
            content_hash = hashlib.sha256(source_bytes).hexdigest()
            marker = source_marker(dataset, content_hash)
            path = 'documents/' + marker + '.md'
            body = 'Source-ID: ' + marker + '\n\n' + original
            doc = {'dataset': dataset, 'doc_id': content_hash, 'split': assignment['split'],
                   'source_marker': marker, 'path': path, 'source_sha256': content_hash,
                   'original_path': relative, 'sha256': hashlib.sha256(body.encode()).hexdigest()}
            if path in documents:
                require(documents[path]['split'] == doc['split'], 'Source document crosses splits')
            documents[path] = doc
            bodies[path] = body
            paths.append(path); markers.append(marker); document_ids.append(content_hash)
        cases.append({'schema': SCHEMA, 'case_id': 'task-' + digest(key)[:24],
                      'dataset': dataset, 'task_id': task_id, 'group_id': assignment['group_id'],
                      'split': assignment['split'], 'language': assignment.get('language', 'unknown'),
                      'domain': assignment.get('domain', dataset), 'question': question,
                      'reference_answer': reference, 'reference_kind': 'original task CSV gold_summary',
                      'document_ids': document_ids, 'source_paths': paths, 'requires_retrieval': True,
                      'relevant_source_markers': markers, 'retrieval_annotation': 'any_declared_source_not_official_qrels',
                      'generation_fields': {k: row[k] for k in TASK_GENERATION_FIELDS if row.get(k)},
                      'task_metadata': {k: v for k, v in row.items() if v}})
    require(seen == set(assignments), 'Unused split assignment; supply exactly the selected tasks')
    validate_cases(cases)
    # Content identity prevents leakage even when two dataset names alias the same source.
    splits: dict[str, str] = {}
    for doc in documents.values():
        require(splits.setdefault(doc['source_sha256'], doc['split']) == doc['split'],
                'Source content crosses splits under different dataset names')
    docs = list(documents.values())
    manifest = {'schema': SCHEMA, 'adapter': 'task-csv-v1', 'cases_hash': digest(cases),
                'documents_hash': digest(docs), 'input_csv_sha256': sha(csv_path),
                'assignments_hash': digest(assignments), 'count': len(cases),
                'evaluation_policy': 'QA/grounding gates plus separately reviewed task-specific DEA metrics'}
    if execute:
        output.mkdir(parents=True, mode=0o700)
        for path, body in bodies.items():
            write_text(output / path, body)
        write_rows(output / 'cases.jsonl', cases)
        write(output / 'documents.json', docs)
        write(output / 'manifest.json', manifest)
    return manifest


def catalog_runs(runs: list[Path], reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join complete graded runs to explicit manual reviews; incomplete runs remain in the archive."""
    review_map = {review['benchmark_id']: review for review in reviews}
    require(len(review_map) == len(reviews), 'Duplicate manual review')
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for run in runs:
        candidates, cases, captures, selection = loaded(run)
        grades = {grade['benchmark_id']: grade for grade in validated_grades(run)}
        for candidate in candidates:
            bid = candidate['benchmark_id']
            require(bid not in seen and bid in review_map and bid in grades, 'Duplicate episode or missing review/grade')
            seen.add(bid)
            case, cap, grade = cases[candidate['case_id']], captures[bid], grades[bid]
            review = review_map[bid]
            require(type(review.get('accepted')) is bool and isinstance(review.get('finding'), str)
                    and review['finding'].strip(), 'Explicit manual verdict and finding required')
            require(review.get('grades_sha256') == sha(run / 'grades.jsonl'), 'Manual review grading receipt changed')
            require(not review['accepted'] or grade['pass'], 'Manual review cannot promote a failed automatic grade')
            require(grade['capture_hash'] == digest(cap) and grade['answer_hash'] == digest(candidate['candidate_answer']),
                    'Grade does not match wire candidate')
            result.append({'case_id': case['case_id'], 'benchmark_id': bid, 'question': case['question'],
                           'language': case.get('language'), 'domain': case.get('domain'),
                           'accepted': review['accepted'], 'status': 'ACCEPTED' if review['accepted'] else 'SEMANTIC_REJECTED',
                           'model': candidate['model'], 'answer': candidate['candidate_answer'],
                           'capture_hash': digest(cap), 'capture_path': str((run / candidate['capture_path']).resolve()),
                           'original_run': str(run.resolve()), 'protocol_hash': selection['protocol_hash'],
                           'usage': metrics(cap), 'grade': grade, 'manual_review': review})
    require(seen == set(review_map), 'Manual review includes unknown episodes')
    return result


def build_reviewed(dataset: Path, runs: list[Path], catalog: Path, output: Path,
                   with_reasoning: bool = False, execute: bool = False) -> dict[str, Any]:
    """Build SFT and labelled trajectories from reviewed wire captures without losing failures."""
    require(not output.exists(), 'Output dataset already exists')
    cases = load(dataset)
    case_map = {case['case_id']: case for case in cases}
    selection = upstream('scripts/export_reviewed_episodes.py').select_covered
    selected, questions, coverage = selection(rows(catalog), cases)
    selected_map = {row['benchmark_id']: row for row in selected}
    candidates: list[dict[str, Any]] = []
    grades: list[dict[str, Any]] = []
    captures: dict[str, dict[str, Any]] = {}
    trajectories: list[dict[str, Any]] = []
    task_results: list[dict[str, Any]] = []
    for run in runs:
        run_candidates, run_cases, caps, protocol = loaded(run)
        require(all(cid in case_map and digest(case) == digest(case_map[cid]) for cid, case in run_cases.items()),
                'Run cases/references differ from selected dataset')
        run_grades = {grade['benchmark_id']: grade for grade in validated_grades(run)}
        for candidate in run_candidates:
            bid = candidate['benchmark_id']
            if bid not in selected_map:
                continue
            require(bid not in captures, 'Duplicate run/capture')
            reviewed, cap, grade = selected_map[bid], caps[bid], run_grades[bid]
            require(reviewed['case_id'] == candidate['case_id'] and reviewed['capture_hash'] == digest(cap)
                    and reviewed['grade'] == grade and reviewed['protocol_hash'] == protocol['protocol_hash']
                    and reviewed['usage'] == metrics(cap) and reviewed['answer'] == candidate['candidate_answer']
                    and reviewed['model'] == candidate['model'], 'Reviewed episode and original run differ')
            require(reviewed['manual_review'].get('grades_sha256') == sha(run / 'grades.jsonl'),
                    'Manual review grading receipt changed')
            require(grade['answer_hash'] == digest(candidate['candidate_answer']), 'Candidate answer changed')
            require(cap.get('complete') is True and cap.get('records'), 'Incomplete trajectory')
            for record in cap['records']:
                turn_example(record, case_map[candidate['case_id']])
            captures[bid] = cap
            # A manual veto remains a labelled failure, but cannot create a preference
            # pair with the original passing score. Only automatic semantic failures
            # qualify for the existing margin-based pair builder.
            if reviewed['accepted'] or not grade['pass']:
                candidates.append(candidate); grades.append(grade)
            trajectories.append({**reviewed, 'capture': copy.deepcopy(cap),
                                 'protocol': protocol['protocol'], 'training_use': 'reviewed_episode_not_ppo_rollout'})
            case = case_map[candidate['case_id']]
            if case.get('task_metadata'):
                task_results.append(task_result_row(case, candidate, cap))
    require(set(captures) == set(selected_map), 'Missing source runs for reviewed episodes')
    sft, pairs, report = build_data(cases, candidates, grades, captures)
    require(sft, 'No verified positive trajectories')
    reasoning, reasoning_report = reasoning_sft(sft, captures) if with_reasoning else (None, None)
    report.update(coverage=coverage, trajectories=len(trajectories), catalog_hash=sha(catalog),
                  trajectories_hash=digest(trajectories), ppo_ready=False)
    if execute:
        output.mkdir(parents=True, mode=0o700)
        write_rows(output / 'sft.jsonl', sft)
        write_rows(output / 'pairs.jsonl', pairs)
        write_rows(output / 'episodes.jsonl', selected)
        write_rows(output / 'trajectories.jsonl', trajectories)
        write_rows(output / 'questions.jsonl', questions)
        if task_results:
            upstream('lib/bundle_common.py').write_csv_rows(output / 'task-results.csv', task_results)
        if reasoning is not None:
            write_rows(output / 'sft-reasoning.jsonl', reasoning)
            write(output / 'reasoning-build.json', reasoning_report)
        write(output / 'build.json', report)
    return report
