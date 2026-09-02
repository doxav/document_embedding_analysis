#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _safe_name(value: Any, fallback: str) -> str:
    """Return a filesystem-safe non-empty artifact name."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or fallback)).strip("._")
    return safe or fallback


def export_results(input_path: Path, documents_dir: Path) -> list[dict[str, Any]]:
    """Export Promptfoo response outputs to Markdown and return compact result records."""
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    results = (((payload.get("results") or {}).get("results")) if isinstance(payload, dict) else None)
    if not isinstance(results, list):
        raise ValueError("Promptfoo result JSON must contain results.results as a list")
    documents_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    for index, result in enumerate(results, start=1):
        if not isinstance(result, dict):
            raise ValueError(f"Promptfoo result #{index} is not an object")
        variables = result.get("vars") if isinstance(result.get("vars"), dict) else {}
        response = result.get("response") if isinstance(result.get("response"), dict) else {}
        output = str(response.get("output") or "")
        dataset = _safe_name(variables.get("dataset"), "dataset")
        task_id = _safe_name(variables.get("task_id"), str(index))
        document_path = documents_dir / f"{dataset}_{task_id}.md"
        document_path.write_text(output.rstrip() + "\n", encoding="utf-8")
        exported.append(
            {
                "dataset": variables.get("dataset"),
                "task_id": variables.get("task_id"),
                "model": variables.get("openwebui_pipe_model"),
                "tool_ids": variables.get("openwebui_tool_ids_json"),
                "algorithm": variables.get("algorithm"),
                "target_length": variables.get("target_length"),
                "structure": variables.get("structure"),
                "success": bool(result.get("success")),
                "score": result.get("score"),
                "named_scores": result.get("namedScores") or {},
                "error": result.get("error") or result.get("failureReason") or "",
                "latency_ms": response.get("latencyMs"),
                "token_usage": response.get("tokenUsage") or {},
                "cost": response.get("cost"),
                "metadata": response.get("metadata") or {},
                "output_chars": len(output),
                "document_path": str(document_path),
            }
        )
    return exported


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Promptfoo outputs as Markdown plus a compact JSON summary.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--documents-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    exported = export_results(args.input, args.documents_dir)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(exported, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(exported)} documents to {args.documents_dir}")
    print(f"Wrote summary to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
