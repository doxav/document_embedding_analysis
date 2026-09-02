#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys

BUNDLE_ROOT = _Path(__file__).resolve().parents[1]
if str(BUNDLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(BUNDLE_ROOT))

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

from lib.bundle_common import read_csv_rows, trim_text, write_csv_rows


def _request_timeout_seconds() -> int:
    """Return the bridge request timeout used by long live generations."""
    raw = os.environ.get("BRIDGE_REQUEST_TIMEOUT_SECONDS", "1800").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("BRIDGE_REQUEST_TIMEOUT_SECONDS must be an integer") from exc
    if value <= 0:
        raise ValueError("BRIDGE_REQUEST_TIMEOUT_SECONDS must be positive")
    return value


def _raise_for_status(response: requests.Response) -> None:
    """Raise HTTP errors with the bridge response body preserved."""
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = trim_text(response.text, 1200)
        raise requests.HTTPError(f"{exc}; response body: {detail}", response=response) from exc


def extract_generation_metrics(payload: dict[str, Any], duration_seconds: float) -> dict[str, str]:
    """Extract stable timing, token, cost, request, and tool-call CSV metrics."""
    trace = payload.get("trace") if isinstance(payload.get("trace"), dict) else {}
    usage = trace.get("usage") if isinstance(trace.get("usage"), dict) else {}
    rounds = trace.get("rounds") if isinstance(trace.get("rounds"), list) else []
    tool_calls = trace.get("tool_calls") if isinstance(trace.get("tool_calls"), list) else []
    return {
        "generation_duration_seconds": f"{duration_seconds:.6f}",
        "generation_prompt_tokens": str(int(usage.get("prompt_tokens") or 0)),
        "generation_completion_tokens": str(int(usage.get("completion_tokens") or 0)),
        "generation_total_tokens": str(int(usage.get("total_tokens") or 0)),
        "generation_cost": str(float(usage.get("cost") or 0.0)),
        "generation_llm_requests": str(len(rounds)),
        "generation_tool_calls": str(len(tool_calls)),
    }


def write_generated_document(documents_dir: Path, task_id: Any, index: int, content: str) -> Path:
    """Write one generated Markdown document using a filesystem-safe task id."""
    safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id or index)).strip("._") or str(index)
    documents_dir.mkdir(parents=True, exist_ok=True)
    document_path = documents_dir / f"{safe_task_id}.md"
    document_path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return document_path


def _artifact_stem(row: dict[str, Any], index: int) -> str:
    """Return the unique safe stem for one generated benchmark artifact."""
    identifier = row.get("benchmark_id") or row.get("task_id") or index
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(identifier)).strip("._") or str(index)


def merge_resume_rows(
    input_rows: list[dict[str, str]],
    completed_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Reuse completed candidates while preserving the current input order."""
    completed_by_id: dict[str, dict[str, str]] = {}
    for row in completed_rows:
        identifier = str(row.get("benchmark_id") or row.get("task_id") or "").strip()
        if not identifier:
            raise ValueError("resume output rows must have a benchmark_id or task_id")
        if identifier in completed_by_id:
            raise ValueError(f"resume output contains duplicate row id: {identifier}")
        completed_by_id[identifier] = row

    merged: list[dict[str, str]] = []
    for row in input_rows:
        identifier = str(row.get("benchmark_id") or row.get("task_id") or "").strip()
        completed = completed_by_id.get(identifier)
        merged.append(completed if completed and completed.get("candidate_answer") else row)
    return merged


def generate_row(
    *,
    index: int,
    row: dict[str, str],
    bridge_url: str,
    timeout: int,
    overwrite: bool,
    responses_dir: Path | None,
    documents_dir: Path | None,
) -> dict[str, Any]:
    """Generate one candidate and persist its optional raw/Markdown artifacts."""
    current: dict[str, Any] = dict(row)
    if current.get("candidate_answer") and not overwrite:
        return current
    payload = {key: value for key, value in current.items() if not key.startswith("__metadata:")}
    started_at = time.monotonic()
    response = requests.post(bridge_url, json=payload, timeout=timeout)
    _raise_for_status(response)
    data = response.json()
    duration_seconds = time.monotonic() - started_at
    current["candidate_answer"] = str(data.get("output") or "").strip()
    current.update(extract_generation_metrics(data, duration_seconds))
    artifact_stem = _artifact_stem(current, index)
    if documents_dir is not None:
        write_generated_document(documents_dir, artifact_stem, index, current["candidate_answer"])
    if responses_dir is not None:
        response_path = responses_dir / f"{artifact_stem}.json"
        response_path.write_text(
            json.dumps(
                {
                    "task_id": current.get("task_id"),
                    "benchmark_id": current.get("benchmark_id"),
                    "duration_seconds": duration_seconds,
                    "response": data,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return current


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate candidate answers through the local OpenWebUI bridge and write them back into a CSV.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bridge-url", default="http://127.0.0.1:8001/generate")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed rows already present in --output.")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--responses-dir", type=Path, help="Optional directory for the complete bridge JSON response per row.")
    parser.add_argument("--documents-dir", type=Path, help="Optional directory for the generated Markdown document per row.")
    args = parser.parse_args()
    if args.max_concurrency <= 0:
        raise ValueError("--max-concurrency must be positive")
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")

    output_path = Path(args.output)
    if args.responses_dir is not None:
        args.responses_dir.mkdir(parents=True, exist_ok=True)
    timeout = _request_timeout_seconds()
    rows = read_csv_rows(Path(args.input))
    selected_rows = rows[: args.limit] if args.limit else rows
    if args.resume and output_path.exists():
        selected_rows = merge_resume_rows(selected_rows, read_csv_rows(output_path))
    total = len(selected_rows)
    completed: dict[int, dict[str, Any]] = {}
    futures: dict[Future[dict[str, Any]], int] = {}
    with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        for index, row in enumerate(selected_rows, start=1):
            future = executor.submit(
                generate_row,
                index=index,
                row=row,
                bridge_url=args.bridge_url,
                timeout=timeout,
                overwrite=args.overwrite,
                responses_dir=args.responses_dir,
                documents_dir=args.documents_dir,
            )
            futures[future] = index
        for future in as_completed(futures):
            index = futures[future]
            current = future.result()
            completed[index] = current
            ordered_rows = [completed[key] for key in sorted(completed)]
            write_csv_rows(output_path, ordered_rows)
            print(
                f"[{len(completed)}/{total}] generated {current.get('dataset')}::{current.get('task_id')} "
                f"({current.get('benchmark_id') or index})",
                flush=True,
            )

    out_rows = [completed[index] for index in sorted(completed)]
    write_csv_rows(output_path, out_rows)
    print(f"Wrote {len(out_rows)} rows to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
