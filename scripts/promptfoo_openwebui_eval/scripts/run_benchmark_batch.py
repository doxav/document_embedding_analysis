#!/usr/bin/env python3
"""Run one safely isolated OpenWebUI tool-parameter batch and score it offline."""

from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys

BUNDLE_ROOT = _Path(__file__).resolve().parents[1]
if str(BUNDLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(BUNDLE_ROOT))
if str(BUNDLE_ROOT / "scripts") not in _sys.path:
    _sys.path.insert(0, str(BUNDLE_ROOT / "scripts"))

import argparse
import csv
import json
import os
import re
import shlex
import statistics
import subprocess
from pathlib import Path
from typing import Any

from lib.bundle_common import read_csv_rows, write_csv_rows
from setup_openwebui import OpenWebUIAdmin, read_env_file


TOOL_ID = "large_thematic_summarizer"
ALGORITHMS = {"raptor", "dtcrs", "kohaku"}


def parse_json_object(value: str, option_name: str) -> dict[str, Any]:
    """Decode a CLI JSON object with a descriptive validation error."""
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option_name} must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{option_name} must decode to a JSON object")
    return parsed


def safe_label(value: str) -> str:
    """Validate and return a portable benchmark configuration label."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("--label may contain only letters, digits, dot, underscore, and hyphen")
    return value


def build_benchmark_rows(
    rows: list[dict[str, str]],
    *,
    label: str,
    repetitions: int,
    valves: dict[str, Any],
    request_overrides: dict[str, Any],
) -> list[dict[str, Any]]:
    """Expand source rows into uniquely identified, short-target benchmark repetitions."""
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    requested_target = str(request_overrides.get("target_length") or "short").strip().lower()
    if requested_target != "short":
        raise ValueError("target_length is fixed to short for this benchmark")
    algorithm = str(valves.get("algorithm") or "").strip().lower()
    if algorithm not in ALGORITHMS:
        raise ValueError(f"valves.algorithm must be one of: {', '.join(sorted(ALGORITHMS))}")
    parameters = json.dumps(
        {"valves": valves, "request": {**request_overrides, "target_length": "short"}},
        ensure_ascii=False,
        sort_keys=True,
    )
    expanded: list[dict[str, Any]] = []
    for repeat in range(1, repetitions + 1):
        for row in rows:
            task_id = str(row.get("task_id") or "").strip()
            if not task_id:
                raise ValueError("every input row must have a task_id")
            current: dict[str, Any] = dict(row)
            current.update(request_overrides)
            benchmark_id = f"{label}__{task_id}__r{repeat}"
            current.update(
                {
                    "benchmark_id": benchmark_id,
                    "benchmark_source_task_id": task_id,
                    "benchmark_repeat": repeat,
                    "benchmark_config_id": label,
                    "benchmark_parameters_json": parameters,
                    "algorithm": algorithm,
                    "target_length": "short",
                    "candidate_answer": "",
                    "__description": f"{row.get('__description') or task_id}::{label}::repeat-{repeat}",
                }
            )
            expanded.append(current)
    return expanded


def _number(value: Any) -> float:
    """Convert an optional numeric value to float, using zero only when absent."""
    if value in (None, ""):
        return 0.0
    return float(value)


def collect_records(candidate_csv: Path, promptfoo_json: Path) -> list[dict[str, Any]]:
    """Join generation metrics with Promptfoo quality scores by benchmark id."""
    candidates = {row["benchmark_id"]: row for row in read_csv_rows(candidate_csv)}
    payload = json.loads(promptfoo_json.read_text(encoding="utf-8"))
    results = ((payload.get("results") or {}).get("results")) if isinstance(payload, dict) else None
    if not isinstance(results, list):
        raise ValueError("Promptfoo output must contain results.results as a list")
    records: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        variables = result.get("vars") if isinstance(result.get("vars"), dict) else {}
        benchmark_id = str(variables.get("benchmark_id") or "")
        candidate = candidates.get(benchmark_id)
        if candidate is None:
            raise ValueError(f"Promptfoo result has unknown benchmark_id: {benchmark_id!r}")
        named = result.get("namedScores") if isinstance(result.get("namedScores"), dict) else {}
        answer = str(candidate.get("candidate_answer") or "")
        records.append(
            {
                "benchmark_id": benchmark_id,
                "config_id": candidate.get("benchmark_config_id"),
                "task_id": candidate.get("benchmark_source_task_id") or candidate.get("task_id"),
                "repeat": int(candidate.get("benchmark_repeat") or 1),
                "algorithm": candidate.get("algorithm"),
                "parameters_json": candidate.get("benchmark_parameters_json"),
                "output_chars": len(answer),
                "generation_seconds": _number(candidate.get("generation_duration_seconds")),
                "prompt_tokens": int(_number(candidate.get("generation_prompt_tokens"))),
                "completion_tokens": int(_number(candidate.get("generation_completion_tokens"))),
                "total_tokens": int(_number(candidate.get("generation_total_tokens"))),
                "cost": _number(candidate.get("generation_cost")),
                "tool_calls": int(_number(candidate.get("generation_tool_calls"))),
                "promptfoo_score": _number(result.get("score")),
                "dea_score": _number(named.get("dea_score")),
                "plan": _number(named.get("plan")),
                "content": _number(named.get("content")),
                "resources": _number(named.get("resources")),
                "length_alignment": _number(named.get("length_alignment")),
                "rouge_l_f": _number(named.get("rouge_l_f")),
                "judge_ran": int(_number(named.get("dea_judge_ran"))),
                "judge_problem_count": int(_number(named.get("dea_judge_problem_count"))),
                "passed": bool(result.get("success")),
                "error": str(result.get("error") or result.get("failureReason") or ""),
            }
        )
    return sorted(records, key=lambda item: (str(item["task_id"]), int(item["repeat"])))


def merge_promptfoo_results(primary: dict[str, Any], recovery: dict[str, Any]) -> dict[str, Any]:
    """Replace primary Promptfoo rows with successfully recovered benchmark rows."""
    primary_rows = ((primary.get("results") or {}).get("results"))
    recovery_rows = ((recovery.get("results") or {}).get("results"))
    if not isinstance(primary_rows, list) or not isinstance(recovery_rows, list):
        raise ValueError("both Promptfoo payloads must contain results.results lists")
    replacements: dict[str, dict[str, Any]] = {}
    for row in recovery_rows:
        variables = row.get("vars") if isinstance(row, dict) else None
        benchmark_id = str((variables or {}).get("benchmark_id") or "")
        if not benchmark_id:
            raise ValueError("every recovery row must contain vars.benchmark_id")
        if benchmark_id in replacements:
            raise ValueError(f"recovery contains duplicate benchmark_id: {benchmark_id}")
        replacements[benchmark_id] = row

    merged = json.loads(json.dumps(primary))
    merged_rows = ((merged.get("results") or {}).get("results"))
    primary_ids = {
        str(((row.get("vars") or {}) if isinstance(row, dict) else {}).get("benchmark_id") or "")
        for row in merged_rows
    }
    unknown = sorted(set(replacements) - primary_ids)
    if unknown:
        raise ValueError(f"recovery contains unknown benchmark ids: {', '.join(unknown)}")
    for index, row in enumerate(merged_rows):
        benchmark_id = str(((row.get("vars") or {}) if isinstance(row, dict) else {}).get("benchmark_id") or "")
        if benchmark_id in replacements:
            merged_rows[index] = replacements[benchmark_id]
    return merged


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate batch means and sample deviations for quality and cost metrics."""
    if not records:
        raise ValueError("cannot summarize an empty benchmark batch")
    metric_names = (
        "generation_seconds",
        "total_tokens",
        "cost",
        "output_chars",
        "dea_score",
        "plan",
        "content",
        "resources",
        "length_alignment",
        "rouge_l_f",
    )
    summary: dict[str, Any] = {
        "config_id": records[0]["config_id"],
        "algorithm": records[0]["algorithm"],
        "parameters_json": records[0]["parameters_json"],
        "runs": len(records),
        "tasks": len({record["task_id"] for record in records}),
        "passes": sum(bool(record["passed"]) for record in records),
        "judge_successes": sum(int(record["judge_ran"]) for record in records),
    }
    for metric in metric_names:
        values = [float(record[metric]) for record in records]
        summary[f"mean_{metric}"] = statistics.fmean(values)
        summary[f"stdev_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return summary


def write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Write detailed benchmark records with a stable header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("cannot write an empty benchmark record CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _relative_to_bundle(path: Path) -> str:
    """Return a bundle-relative path required by the mounted Promptfoo container."""
    try:
        return str(path.resolve().relative_to(BUNDLE_ROOT.resolve()))
    except ValueError as exc:
        raise ValueError(f"benchmark artifacts must be below {BUNDLE_ROOT}") from exc


def _run_command(command: list[str], log_path: Path, *, accepted_codes: set[int]) -> None:
    """Run a command, capture its complete log, and validate its exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=BUNDLE_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode not in accepted_codes:
        raise RuntimeError(f"Command failed with exit {completed.returncode}; see {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Base CSV containing the selected dataset tasks once.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--valves-json", required=True)
    parser.add_argument("--request-overrides-json", default="{}")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument("--bridge-timeout-seconds", type=int, default=2400)
    parser.add_argument("--env-file", type=Path, default=BUNDLE_ROOT / ".env")
    parser.add_argument("--promptfoo-config", default="multilexsum.step2.dea.yaml")
    parser.add_argument("--tool-id", default=TOOL_ID)
    args = parser.parse_args()
    label = safe_label(args.label)
    if args.max_concurrency <= 0 or args.bridge_timeout_seconds <= 0:
        raise ValueError("concurrency and timeout must be positive")
    valves = parse_json_object(args.valves_json, "--valves-json")
    request_overrides = parse_json_object(args.request_overrides_json, "--request-overrides-json")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_benchmark_rows(
        read_csv_rows(args.input),
        label=label,
        repetitions=args.repetitions,
        valves=valves,
        request_overrides=request_overrides,
    )
    input_csv = output_dir / "input.csv"
    candidate_csv = output_dir / "candidates.csv"
    promptfoo_json = output_dir / "promptfoo.json"
    write_csv_rows(input_csv, rows)

    env = read_env_file(args.env_file)
    base_url = str(env.get("OPENWEBUI_BASE_URL") or "").rstrip("/")
    api_key = str(env.get("OPENWEBUI_API_KEY") or "")
    if not base_url or not api_key:
        raise RuntimeError("OPENWEBUI_BASE_URL and OPENWEBUI_API_KEY are required in the benchmark env file")
    admin = OpenWebUIAdmin(base_url, timeout_seconds=60)
    admin.authenticate(api_key, "", "")
    original_response = admin._request("GET", f"/api/v1/tools/id/{args.tool_id}/valves")
    original_valves = original_response.json()
    if not isinstance(original_valves, dict):
        raise RuntimeError(f"OpenWebUI returned invalid current valves for {args.tool_id!r}")

    generate_command = [
        str(BUNDLE_ROOT / ".venv/bin/python"),
        str(BUNDLE_ROOT / "scripts/generate_candidate_csv.py"),
        "--input",
        str(input_csv),
        "--output",
        str(candidate_csv),
        "--bridge-url",
        "http://127.0.0.1:8001/generate",
        "--resume",
        "--max-concurrency",
        str(args.max_concurrency),
        "--responses-dir",
        str(output_dir / "responses"),
        "--documents-dir",
        str(output_dir / "documents"),
    ]
    promptfoo_command = [
        "docker",
        "compose",
        "run",
        "--rm",
        "promptfoo",
        "bash",
        "-lc",
        " ".join(
            shlex.quote(part)
            for part in [
                "promptfoo",
                "eval",
                "-c",
                f"./{args.promptfoo_config}",
                "-t",
                f"./{_relative_to_bundle(candidate_csv)}",
                "--output",
                f"./{_relative_to_bundle(promptfoo_json)}",
                "--no-progress-bar",
                "--no-cache",
                "--max-concurrency",
                str(args.max_concurrency),
            ]
        ),
    ]
    commands_path = output_dir / "commands.sh"
    reproduce_command = [
        str(BUNDLE_ROOT / ".venv/bin/python"),
        str(Path(__file__).resolve()),
        *_sys.argv[1:],
    ]
    commands_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "cd "
        + shlex.quote(str(BUNDLE_ROOT))
        + "\n"
        + shlex.join(reproduce_command)
        + "\n",
        encoding="utf-8",
    )

    try:
        applied = admin.set_tool_valves(args.tool_id, valves)
        if applied != valves:
            raise RuntimeError(f"OpenWebUI applied unexpected valves: {applied}")
        command_env = os.environ.copy()
        command_env["BRIDGE_REQUEST_TIMEOUT_SECONDS"] = str(args.bridge_timeout_seconds)
        with (output_dir / "generation.log").open("w", encoding="utf-8") as log:
            generated = subprocess.run(
                generate_command,
                cwd=BUNDLE_ROOT,
                env=command_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if generated.returncode != 0:
            raise RuntimeError(f"Generation failed with exit {generated.returncode}; see {output_dir / 'generation.log'}")
        _run_command(promptfoo_command, output_dir / "promptfoo.log", accepted_codes={0, 100})
    finally:
        restored = admin.set_tool_valves(args.tool_id, original_valves)
        if restored != original_valves:
            raise RuntimeError(f"Failed to restore original tool valves: {restored}")

    records = collect_records(candidate_csv, promptfoo_json)
    summary = summarize_records(records)
    write_records_csv(output_dir / "records.csv", records)
    (output_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
