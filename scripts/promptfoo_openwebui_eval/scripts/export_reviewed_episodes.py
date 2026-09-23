#!/usr/bin/env python3
"""Export reviewed TRAIN episodes for questions with at least one accepted answer."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))

from lib.bundle_common import file_sha256, read_jsonl

COMPLETE_STATUSES = {"ACCEPTED", "SEMANTIC_REJECTED"}
INCIDENT_STATUSES = {"TECHNICAL_FAILURE", "BUDGET_INCOMPLETE"}
USAGE_FIELDS = (
    "llm_calls", "tool_calls", "prompt_tokens", "completion_tokens", "total_tokens",
    "cached_input_tokens", "reasoning_tokens", "seconds",
)


def _require(condition: bool, message: str) -> None:
    """Reject inconsistent metadata without echoing private episode contents."""
    if not condition:
        raise ValueError(message)


def _identifier(row: dict[str, Any], field: str) -> str:
    """Read a nonempty identifier without using it as a filesystem path."""
    value = row.get(field)
    _require(isinstance(value, str) and bool(value.strip()), f"Missing {field}")
    return value


def select_covered(
    episodes: list[dict[str, Any]], cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Keep reviewed outcomes with a solution; never turn incidents into negatives."""
    case_map: dict[str, dict[str, Any]] = {}
    for case in cases:
        _require(isinstance(case, dict), "Case must be an object")
        case_id = _identifier(case, "case_id")
        _require(case_id not in case_map, "Duplicate case_id")
        case_map[case_id] = case

    solutions: dict[str, str] = {}
    complete: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempted: set[str] = set()
    incidents = 0
    for row in episodes:
        _require(isinstance(row, dict), "Episode must be an object")
        case_id = _identifier(row, "case_id")
        _require(case_id in case_map and case_map[case_id].get("split") == "train",
                 "Every episode must refer to a known TRAIN case")
        attempted.add(case_id)
        status = row.get("status")
        _require(isinstance(status, str) and status in COMPLETE_STATUSES | INCIDENT_STATUSES,
                 "Unknown episode status")
        _require(type(row.get("accepted")) is bool, "accepted must be a boolean")
        _require(row["accepted"] == (status == "ACCEPTED"), "Status contradicts acceptance")
        if status in INCIDENT_STATUSES:
            incidents += 1
            continue

        benchmark_id = _identifier(row, "benchmark_id")
        _require(benchmark_id not in seen, "Duplicate benchmark_id")
        seen.add(benchmark_id)
        grade, review, usage = (row.get(key) for key in ("grade", "manual_review", "usage"))
        _require(all(isinstance(value, dict) for value in (grade, review, usage)),
                 "Complete episodes require grade, manual_review and usage objects")
        _require(type(grade.get("pass")) is bool and type(grade.get("hard_ok")) is bool,
                 "Grade pass and hard_ok must be booleans")
        _require(type(review.get("accepted")) is bool and review["accepted"] == row["accepted"],
                 "Manual review contradicts acceptance")
        _require(grade.get("benchmark_id") == benchmark_id and
                 grade.get("capture_hash") == _identifier(row, "capture_hash"),
                 "Grade does not identify the same episode capture")
        _require(not row["accepted"] or (grade["pass"] and grade["hard_ok"]),
                 "Accepted episode lacks a passing grade and hard gates")
        for field in USAGE_FIELDS:
            value = usage.get(field)
            valid_number = (type(value) is int and value >= 0) or (
                field == "seconds" and type(value) is float and math.isfinite(value) and value >= 0)
            _require(value is None or valid_number,
                     f"Invalid usage field: {field}")
        scores: dict[str, float | None] = {}
        for field in ("correctness", "groundedness"):
            score = None
            if grade["hard_ok"]:
                metrics = grade.get("metrics")
                _require(isinstance(metrics, dict) and isinstance(metrics.get(field), dict),
                         "Missing executed judge metric")
                score = metrics[field].get("score")
                _require(type(score) in (int, float) and 0 <= score <= 1,
                         "Invalid executed judge score")
            scores[field] = score
        complete.append({**row, "label": "success" if row["accepted"] else "failure",
                         "quality_judges_executed": grade["hard_ok"], "quality_scores": scores})
        if row["accepted"]:
            solutions.setdefault(case_id, benchmark_id)

    selected = [{**row, "accepted_solution_benchmark_id": solutions[row["case_id"]]}
                for row in complete if row["case_id"] in solutions]
    questions = [case for case in cases if case["case_id"] in solutions]
    successes = sum(row["accepted"] for row in selected)
    report = {
        "schema": "dea.reviewed-episodes.v1", "questions": len(questions),
        "episodes": len(selected), "successes": successes, "failures": len(selected) - successes,
        "source_episodes": len(episodes), "technical_incidents_excluded": incidents,
        "excluded_question_ids": sorted(attempted - solutions.keys()),
        "every_question_has_accepted_solution": True,
        "interpretation": "TRAIN selection after retries, not pass@1 or same-state preference pairs",
    }
    return selected, questions, report


def export_dataset(catalog: Path, cases: Path, output: Path, execute: bool = False) -> dict[str, Any]:
    """Validate an offline catalog and optionally write a new private dataset view."""
    _require(catalog.is_file() and cases.is_file(), "Catalog and cases must be existing JSONL files")
    _require(not output.exists() and not output.is_symlink(), "Output already exists; choose a new directory")
    hashes = {"catalog": file_sha256(catalog), "cases": file_sha256(cases)}
    try:
        selected, questions, report = select_covered(read_jsonl(catalog), read_jsonl(cases))
    except json.JSONDecodeError:
        raise ValueError("Input contains invalid JSON") from None
    _require(hashes == {"catalog": file_sha256(catalog), "cases": file_sha256(cases)},
             "Input changed during export")
    report["input_sha256"] = hashes
    if not execute:
        return report
    _require(bool(selected), "No accepted solution; no training dataset was written")
    output.mkdir(parents=True, mode=0o700)
    # COMPLETE is represented by manifest.json, written last. A failed write leaves
    # an incomplete directory that cannot accidentally be reused by this command.
    for name, rows in (("episodes.jsonl", selected), ("questions.jsonl", questions)):
        with (output / name).open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    report["output_sha256"] = {name: file_sha256(output / name)
                              for name in ("episodes.jsonl", "questions.jsonl")}
    report["source_catalog"] = str(catalog.resolve())
    report["source_cases"] = str(cases.resolve())
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
    return report


def main() -> int:
    """Preview selection by default; require --execute for filesystem writes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        report = export_dataset(args.catalog, args.cases, args.output, args.execute)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Export failed: {str(exc) if isinstance(exc, ValueError) else 'filesystem error'}\n")
    # Do not print private input paths or excluded question identifiers.
    print(json.dumps({key: report[key] for key in ("questions", "episodes", "successes", "failures")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
