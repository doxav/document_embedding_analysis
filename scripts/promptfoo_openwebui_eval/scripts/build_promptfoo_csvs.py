#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys

BUNDLE_ROOT = _Path(__file__).resolve().parents[1]
if str(BUNDLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(BUNDLE_ROOT))

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lib.bundle_common import (
    CSV_FIELDS,
    build_request_prompt,
    dea_to_markdown,
    derive_facts_from_sections,
    first_sentence,
    infer_source_document_mode_from_resources,
    load_json,
    load_text,
    normalize_mqs_relpath,
    parse_jsonish,
    repo_root,
    resolve_repo_path,
    source_titles_from_manifest,
    trim_text,
    write_csv_rows,
)


def _description(dataset: str, task_id: str, title: str) -> str:
    clean_title = trim_text(title, 80).replace("\n", " ")
    return f"{dataset}::{task_id}::{clean_title}"


def _estimate_length_bounds(expected_summary: str) -> tuple[int, int]:
    n = len(expected_summary or "")
    min_chars = max(200, int(n * 0.35))
    max_chars = max(1200, int(n * 1.35))
    return min_chars, max_chars


def _basic_answer_from_sources(
    *,
    title: str,
    abstract: str,
    instruction: str,
    source_paths: list[str],
    source_titles: list[str],
    dataset: str,
) -> str:
    snippets: list[str] = []
    for i, rel_path in enumerate(source_paths[:8]):
        path = resolve_repo_path(rel_path)
        if not path or not path.exists():
            continue
        text = load_text(path)
        sentence = first_sentence(text)
        label = source_titles[i] if i < len(source_titles) else path.stem
        if sentence:
            snippets.append(f"- **{label}**: {sentence}")

    lines = [f"# {title}".strip(), ""]
    if abstract:
        lines.extend(["## Overview", "", abstract.strip(), ""])

    lines.extend(["## Task", "", instruction.strip(), ""])
    if snippets:
        heading = "## Source-grounded notes" if dataset in {"bigsurvey", "multilexsum"} else "## Available evidence"
        lines.extend([heading, "", *snippets, ""])

    if dataset == "multilexsum":
        lines.extend(
            [
                "## Preliminary long summary",
                "",
                "This draft summarizes the available case documents at a coarse level and should be replaced by a fuller synthesis in later steps.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Preliminary synthesis",
                "",
                "This baseline draft condenses the most visible recurring points from the available source documents and is intended only as a low-difficulty sanity check for the evaluation pipeline.",
                "",
            ]
        )

    return "\n".join(lines).strip()


def _mqs_rows_from_items(dataset: str, dataset_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for item_dir in sorted(dataset_dir.glob("item_*")):
        task_path = item_dir / "task.json"
        if not task_path.exists():
            continue
        task = load_json(task_path)
        source_paths = [
            f"output/{dataset}/{(item_dir.name + '/sources/' + p.name)}"
            for p in sorted((item_dir / "sources").glob("*.md"))
        ]
        rows.append(
            {
                "id": task["task_id"],
                "input": {
                    "title": task.get("title", ""),
                    "abstract": task.get("abstract", ""),
                    "instruction_path": f"output/{dataset}/{item_dir.name}/instruction.md",
                    "source_paths": source_paths,
                },
                "expected_path": f"output/{dataset}/{item_dir.name}/full_text.md",
                "sections_path": f"output/{dataset}/{item_dir.name}/sections.json",
                "source_manifest_path": f"output/{dataset}/{item_dir.name}/source_manifest.json",
                "expected": load_text(item_dir / "full_text.md"),
            }
        )
    return rows


def _load_mqs_like_records(dataset: str) -> list[dict[str, Any]]:
    root = repo_root()
    dataset_dir = root / "output" / dataset
    mqs_path = dataset_dir / "MQS_evaluation_dataset.jsonl"
    if mqs_path.exists():
        rows = []
        with mqs_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    return _mqs_rows_from_items(dataset, dataset_dir)


def _promptfoo_row_from_mqs_record(record: dict[str, Any], dataset: str, args) -> dict[str, Any]:
    root = repo_root()
    expected_path = normalize_mqs_relpath(dataset, record["expected_path"])
    sections_path = normalize_mqs_relpath(dataset, record["sections_path"])
    source_manifest_path = normalize_mqs_relpath(dataset, record["source_manifest_path"])
    instruction_path = normalize_mqs_relpath(dataset, record["input"]["instruction_path"])
    source_paths = [normalize_mqs_relpath(dataset, p) for p in record["input"].get("source_paths", [])]

    expected_abs = (root / expected_path).resolve()
    item_dir = expected_abs.parent
    instruction_abs = (root / instruction_path).resolve()
    sections_abs = (root / sections_path).resolve()
    source_manifest_abs = (root / source_manifest_path).resolve()
    dea_solution_path = item_dir / "dea_solution.json"

    title = str(record["input"].get("title", "")).strip()
    abstract = str(record["input"].get("abstract", "")).strip()
    instruction = load_text(instruction_abs).strip()
    expected_summary = str(record.get("expected") or load_text(expected_abs)).strip()
    sections = load_json(sections_abs)
    source_manifest = load_json(source_manifest_abs)
    source_titles = source_titles_from_manifest(source_manifest)
    facts = derive_facts_from_sections(sections, limit=args.facts_limit)
    request_prompt = build_request_prompt(dataset, title, abstract, instruction)
    min_chars, max_chars = _estimate_length_bounds(expected_summary)

    row = {
        "__description": _description(dataset, str(record["id"]), title),
        "__metadata:dataset": dataset,
        "__metadata:stage": args.stage,
        "__metadata:task_type": "source_provided_mds",
        "__metadata:source_document_mode": "provided_files",
        "task_id": str(record["id"]),
        "dataset": dataset,
        "task_type": "source_provided_mds",
        "title": title,
        "abstract": abstract,
        "instruction": instruction,
        "request_prompt": request_prompt,
        "query": request_prompt,
        "context": expected_summary,
        "gold_summary": expected_summary,
        "expected_path": expected_path,
        "sections_path": sections_path,
        "source_manifest_path": source_manifest_path,
        "dea_solution_path": str(dea_solution_path.relative_to(root)),
        "instruction_path": instruction_path,
        "source_paths_json": json.dumps(source_paths, ensure_ascii=False),
        "kb_ids_json": json.dumps(args.kb_ids_value, ensure_ascii=False),
        "source_titles_json": json.dumps(source_titles, ensure_ascii=False),
        "source_count": len(source_paths),
        "facts_to_check": json.dumps(facts, ensure_ascii=False),
        "required_terms": "",
        "forbidden_terms": "",
        "candidate_content_type": "markdown",
        "openwebui_pipe_model": args.pipe_model,
        "tool_parameters_json": json.dumps(args.tool_parameters_value, ensure_ascii=False),
        "summarizer_model_id": args.summarizer_model_id or "",
        "algorithm": args.algorithm or "",
        "target_length": args.target_length or "",
        "structure": args.structure or "",
        "generation_temperature": str(args.temperature),
        "generation_top_p": str(args.top_p),
        "generation_max_tokens": str(args.max_tokens),
        "min_chars": min_chars,
        "max_chars": max_chars,
        "source_document_mode": "provided_files",
        "openwebui_extra_instructions": args.extra_instructions or "",
        "candidate_answer": "",
    }

    if args.stage == "step1":
        row["candidate_answer"] = _basic_answer_from_sources(
            title=title,
            abstract=abstract,
            instruction=instruction,
            source_paths=source_paths,
            source_titles=source_titles,
            dataset=dataset,
        )
    return row


def _local_source_paths_from_dea_solution(solution: dict[str, Any]) -> tuple[list[str], list[str]]:
    root = repo_root()
    rel_paths: list[str] = []
    titles: list[str] = []
    for resource in solution.get("resources", []) or []:
        title = str(resource.get("resource_description") or resource.get("resource") or "").strip()
        if title:
            titles.append(title)
        for key in ("path", "local_path", "reference", "resource"):
            raw = str(resource.get(key) or "").strip()
            if not raw or raw.startswith(("http://", "https://", "doi:")):
                continue
            path = Path(raw)
            if not path.is_absolute():
                path = (root / path).resolve()
            if path.exists() and path.is_file():
                rel_paths.append(str(path.relative_to(root)))
                break
    return rel_paths, titles


def _promptfoo_row_from_dea_solution(solution_path: Path, dataset: str, args) -> dict[str, Any]:
    root = repo_root()
    solution = load_json(solution_path)
    title = str(solution.get("title", "")).strip()
    abstract = str(solution.get("abstract") or solution.get("context") or "").strip()
    if dataset == "patent":
        instruction = "Write a long-form patent-style technical document using the provided title, context, and any available references."
    else:
        instruction = "Write a long-form scientific or technical document using the provided title, context, and any available references."

    expected_summary = dea_to_markdown(solution)
    source_paths, source_titles = _local_source_paths_from_dea_solution(solution)
    sections = [
        {
            "section": str(step.get("section", "")).strip(),
            "text": str(step.get("content", "")).strip(),
        }
        for step in solution.get("plan", []) or []
    ]
    facts = derive_facts_from_sections(sections, limit=args.facts_limit)
    request_prompt = build_request_prompt(dataset, title, abstract, instruction)
    min_chars, max_chars = _estimate_length_bounds(expected_summary)
    source_document_mode = infer_source_document_mode_from_resources(solution.get("resources", []) or [])

    row = {
        "__description": _description(dataset, str(solution.get("id") or solution_path.stem), title),
        "__metadata:dataset": dataset,
        "__metadata:stage": args.stage,
        "__metadata:task_type": "native_dea",
        "__metadata:source_document_mode": source_document_mode,
        "task_id": str(solution.get("id") or solution_path.stem),
        "dataset": dataset,
        "task_type": "native_dea",
        "title": title,
        "abstract": abstract,
        "instruction": instruction,
        "request_prompt": request_prompt,
        "query": request_prompt,
        "context": expected_summary,
        "gold_summary": expected_summary,
        "expected_path": str(solution_path.relative_to(root)),
        "sections_path": "",
        "source_manifest_path": "",
        "dea_solution_path": str(solution_path.relative_to(root)),
        "instruction_path": "",
        "source_paths_json": json.dumps(source_paths, ensure_ascii=False),
        "kb_ids_json": json.dumps(args.kb_ids_value, ensure_ascii=False),
        "source_titles_json": json.dumps(source_titles, ensure_ascii=False),
        "source_count": len(source_paths),
        "facts_to_check": json.dumps(facts, ensure_ascii=False),
        "required_terms": "",
        "forbidden_terms": "",
        "candidate_content_type": "markdown",
        "openwebui_pipe_model": args.pipe_model,
        "tool_parameters_json": json.dumps(args.tool_parameters_value, ensure_ascii=False),
        "summarizer_model_id": args.summarizer_model_id or "",
        "algorithm": args.algorithm or "",
        "target_length": args.target_length or "",
        "structure": args.structure or "",
        "generation_temperature": str(args.temperature),
        "generation_top_p": str(args.top_p),
        "generation_max_tokens": str(args.max_tokens),
        "min_chars": min_chars,
        "max_chars": max_chars,
        "source_document_mode": source_document_mode,
        "openwebui_extra_instructions": args.extra_instructions or "",
        "candidate_answer": "",
    }

    if args.stage == "step1":
        row["candidate_answer"] = _basic_answer_from_sources(
            title=title,
            abstract=abstract,
            instruction=instruction,
            source_paths=source_paths,
            source_titles=source_titles,
            dataset=dataset,
        )
    return row


def _collect_rows(args) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if args.mode == "mds":
        for dataset in args.dataset:
            records = _load_mqs_like_records(dataset)
            for record in records[: args.limit or None]:
                rows.append(_promptfoo_row_from_mqs_record(record, dataset, args))
    else:
        root = repo_root()
        for dataset in args.dataset:
            dataset_dir = root / "output" / dataset
            for solution_path in sorted(dataset_dir.glob("*.json"))[: args.limit or None]:
                rows.append(_promptfoo_row_from_dea_solution(solution_path, dataset, args))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Promptfoo CSV datasets from document_embedding_analysis outputs.")
    parser.add_argument("--mode", choices=["mds", "native"], required=True)
    parser.add_argument("--dataset", action="append", required=True, help="Repeatable dataset name: bigsurvey, multilexsum, latex, arxiv, patent")
    parser.add_argument("--stage", choices=["step1", "step2-input", "step3"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--facts-limit", type=int, default=12)
    parser.add_argument("--pipe-model", default=os.environ.get("OPENWEBUI_PIPE_MODEL", "summarizer---kohaku"))
    parser.add_argument("--tool-parameters-json", default="", help="JSON object copied into the <tool_parameters> block.")
    parser.add_argument("--kb-id", action="append", default=[], help="Repeatable OpenWebUI knowledge-base id/name copied into <kb_list>.")
    parser.add_argument("--kb-ids-json", default="", help="JSON list/object copied into the <kb_list> block.")
    parser.add_argument("--summarizer-model-id", default="")
    parser.add_argument("--algorithm", default="")
    parser.add_argument("--target-length", default="long")
    parser.add_argument("--structure", default="")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--extra-instructions", default="")
    args = parser.parse_args()
    tool_parameters = parse_jsonish(args.tool_parameters_json, default={}) if args.tool_parameters_json else {}
    if not isinstance(tool_parameters, dict):
        raise SystemExit("--tool-parameters-json must decode to a JSON object")
    kb_ids = parse_jsonish(args.kb_ids_json, default=[]) if args.kb_ids_json else args.kb_id
    if isinstance(kb_ids, str):
        kb_ids = [kb_ids]
    if not isinstance(kb_ids, (list, dict)):
        raise SystemExit("--kb-ids-json must decode to a JSON list or object")
    args.tool_parameters_value = tool_parameters
    args.kb_ids_value = kb_ids

    rows = _collect_rows(args)
    write_csv_rows(Path(args.output), rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
