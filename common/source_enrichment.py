from __future__ import annotations

import json
import re
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from common.task_dataset import TaskBundle, TaskSection, TaskSource, write_task_bundle


def _normalize_locator(key: str, value: Any) -> str:
    raw = str(value).strip()
    if key == "url":
        doi = raw.removeprefix("doi:").strip()
        if re.match(r"^10\.\d{4,9}/\S+$", doi, flags=re.IGNORECASE):
            return f"https://doi.org/{doi}"
    return raw


def _candidate_values(resource: dict[str, Any]) -> list[str]:
    values = []
    for key in ("path", "local_path", "reference", "resource", "url"):
        value = resource.get(key)
        if value:
            values.append(_normalize_locator(key, value))
    return values


def source_document_mode(solution: dict[str, Any]) -> str:
    resources = solution.get("resources", [])
    if not resources:
        return "no_resources"
    if not any(_candidate_values(r) for r in resources):
        return "reference_only_open_retrieval"
    return "source_locator_best_effort"


def enrich_dea_solution(solution: dict[str, Any], output_dir: Path, *, fetch_remote: bool = False, copy_local: bool = True, strict: bool = False, overwrite: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    src_dir = output_dir / "sources"
    src_dir.mkdir(exist_ok=True)
    manifest = []
    for r in solution.get("resources", []):
        rid = r.get("resource_id")
        title = r.get("resource_description") or r.get("resource") or f"resource_{rid}"
        out_file = src_dir / f"source_{int(rid or 0):04d}.md"
        status, err, resolved_from = "failed", "no resolvable source", None
        candidates = _candidate_values(r)

        if out_file.exists() and not overwrite:
            status, err = "skipped", None
        elif not candidates:
            status = "reference_only"
            err = "no source document path or URL; use open retrieval from resource_description"
        else:
            for raw in candidates:
                if raw.startswith(("http://", "https://")):
                    if not fetch_remote:
                        err = "remote fetch disabled"
                        continue
                    try:
                        with urllib.request.urlopen(raw, timeout=30) as resp:
                            out_file.write_bytes(resp.read())
                        status, err, resolved_from = "fetched", None, raw
                        break
                    except Exception as exc:
                        err = str(exc)
                        continue
                if copy_local:
                    p = Path(raw)
                    if p.exists() and p.is_file():
                        shutil.copyfile(p, out_file)
                        status, err, resolved_from = "copied", None, raw
                        break
                    err = f"local path not found: {raw}"

        if status == "failed" and strict:
            raise FileNotFoundError(err)

        manifest.append({
            "resource_id": rid,
            "status": status,
            "source_path": str(out_file) if status in {"copied", "skipped", "fetched"} else None,
            "error": err,
            "title": title,
            "resolved_from": resolved_from,
            "locator": candidates[0] if candidates else None,
            "url": resolved_from if isinstance(resolved_from, str) and resolved_from.startswith("http") else None,
        })

    enriched = dict(solution)
    mode = source_document_mode(solution)
    metadata = enriched.setdefault("metadata", {})
    metadata["source_document_mode"] = mode
    metadata["source_enrichment"] = manifest
    enriched["source_manifest"] = manifest
    enriched["source_document_mode"] = mode
    enriched["resolved_sources"] = sum(1 for m in manifest if m["status"] in {"copied", "skipped", "fetched"})
    return enriched


def write_enriched_task_bundle(enriched_solution: dict[str, Any], output_dir: Path) -> Path:
    sections = [TaskSection(section=p.get("section", "Section"), text=p.get("content", ""), source_ids=[str(x) for x in p.get("resources_used", [])], citations=p.get("citations", [])) for p in enriched_solution.get("plan", [])]
    sources = []
    for m in enriched_solution.get("source_manifest", []):
        if m.get("status") not in {"copied", "skipped", "fetched"}:
            continue
        p = Path(m["source_path"])
        sources.append(TaskSource(source_id=str(m["resource_id"]), title=m.get("title", ""), text=p.read_text() if p.exists() else "", path=str(p), metadata={"status": m["status"]}))

    bundle = TaskBundle(task_id=output_dir.name, dataset="enriched_dea", title=enriched_solution.get("title", ""), abstract=enriched_solution.get("abstract", ""), instruction="Synthesize final document using provided sources.", sections=sections, sources=sources, metadata=enriched_solution.get("metadata", {}))
    item = write_task_bundle(bundle, output_dir.parent)
    (item / "source_manifest.json").write_text(json.dumps(enriched_solution.get("source_manifest", []), indent=2), encoding="utf-8")
    (item / "dea_solution.json").write_text(json.dumps(enriched_solution, indent=2), encoding="utf-8")
    return item
