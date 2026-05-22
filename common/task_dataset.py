from __future__ import annotations

import json
import hashlib
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MQS_EVALUATION_DATASET = "MQS_evaluation_dataset.jsonl"


def _slug(v: str, max_length: int = 120) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (v or "").strip()).strip("_").lower()
    if len(s) > max_length:
        digest = hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]
        s = f"{s[: max_length - len(digest) - 1].rstrip('_')}_{digest}"
    return s or "item"


@dataclass
class TaskSource:
    source_id: str
    title: str = ""
    text: str = ""
    path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSection:
    section: str
    text: str
    source_ids: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TaskBundle:
    task_id: str
    dataset: str
    title: str
    abstract: str
    instruction: str
    sections: list[TaskSection]
    sources: list[TaskSource]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(f"## {s.section}\n\n{s.text}" for s in self.sections)

    def to_dea_solution(self, source_path_by_id: dict[str, str] | None = None) -> dict[str, Any]:
        source_path_by_id = source_path_by_id or {}
        source_index = {s.source_id: i + 1 for i, s in enumerate(self.sources)}
        return {
            "id": self.task_id,
            "title": self.title,
            "abstract": self.abstract,
            "context": self.abstract,
            "plan": [
                {
                    "section_id": i + 1,
                    "section": s.section,
                    "content": s.text,
                    "resources_used": [source_index[x] for x in s.source_ids if x in source_index],
                    "citations": s.citations,
                }
                for i, s in enumerate(self.sections)
            ],
            "resources": [
                {
                    "resource_id": i + 1,
                    "source_id": src.source_id,
                    "resource_description": src.title or src.source_id,
                    "resource": source_path_by_id.get(src.source_id) or src.path or src.title or src.source_id,
                    "reference": source_path_by_id.get(src.source_id) or src.path,
                    "text": src.text,
                    "metadata": src.metadata,
                }
                for i, src in enumerate(self.sources)
            ],
            "metadata": {"dataset": self.dataset, **self.metadata},
            "target_file_path": "full_text.md",
        }


def _validate_bundle(bundle: TaskBundle) -> None:
    if not bundle.task_id:
        raise ValueError("TaskBundle.task_id must be non-empty")
    source_ids = [s.source_id for s in bundle.sources]
    duplicates = sorted({sid for sid in source_ids if source_ids.count(sid) > 1})
    if duplicates:
        raise ValueError(f"Duplicate TaskSource.source_id values: {duplicates}")


def write_task_bundle(bundle: TaskBundle, root: Path) -> Path:
    _validate_bundle(bundle)
    item_dir = root / f"item_{_slug(bundle.task_id)}"
    item_dir.mkdir(parents=True, exist_ok=True)
    sec_dir = item_dir / "sections"
    src_dir = item_dir / "sources"
    sec_dir.mkdir(exist_ok=True)
    src_dir.mkdir(exist_ok=True)

    (item_dir / "title.txt").write_text(bundle.title, encoding="utf-8")
    (item_dir / "abstract.txt").write_text(bundle.abstract, encoding="utf-8")
    (item_dir / "instruction.md").write_text(bundle.instruction, encoding="utf-8")
    (item_dir / "full_text.md").write_text(bundle.full_text, encoding="utf-8")
    (item_dir / "sections.json").write_text(json.dumps([s.__dict__ for s in bundle.sections], indent=2, ensure_ascii=False), encoding="utf-8")
    (item_dir / "metadata.json").write_text(json.dumps(bundle.metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    source_manifest = []
    source_path_by_id: dict[str, str] = {}
    for i, s in enumerate(bundle.sources, start=1):
        fname = f"source_{i:04d}_{_slug(s.title or s.source_id)}.md"
        rel_path = f"sources/{fname}"
        (src_dir / fname).write_text(s.text or "", encoding="utf-8")
        source_path_by_id[s.source_id] = rel_path
        source_manifest.append({"source_id": s.source_id, "path": rel_path, "file": rel_path, "title": s.title})

    for i, s in enumerate(bundle.sections, start=1):
        (sec_dir / f"section_{i:04d}_{_slug(s.section)}.md").write_text(s.text, encoding="utf-8")

    sources_payload = [{**src.__dict__, "path": source_path_by_id.get(src.source_id, src.path)} for src in bundle.sources]
    task_payload = {
        "task_id": bundle.task_id,
        "dataset": bundle.dataset,
        "title": bundle.title,
        "abstract": bundle.abstract,
        "instruction": bundle.instruction,
        "sections": [s.__dict__ for s in bundle.sections],
        "sources": sources_payload,
        "metadata": bundle.metadata,
    }
    (item_dir / "task.json").write_text(json.dumps(task_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    dea = bundle.to_dea_solution(source_path_by_id)
    (item_dir / "dea_solution.json").write_text(json.dumps(dea, indent=2, ensure_ascii=False), encoding="utf-8")
    (item_dir / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return item_dir


def _MQS_evaluation_record_from_item(root: Path, item: Path) -> dict[str, Any]:
    task = json.loads((item / "task.json").read_text(encoding="utf-8"))
    source_paths = [
        str((item / "sources" / p.name).relative_to(root))
        for p in sorted((item / "sources").glob("*.md"))
    ]
    return {
        "id": task["task_id"],
        "input": {
            "title": task["title"],
            "abstract": task["abstract"],
            "instruction_path": str((item / "instruction.md").relative_to(root)),
            "source_paths": source_paths,
        },
        "expected_path": str((item / "full_text.md").relative_to(root)),
        "sections_path": str((item / "sections.json").relative_to(root)),
        "source_manifest_path": str((item / "source_manifest.json").relative_to(root)),
        "expected": (item / "full_text.md").read_text(encoding="utf-8"),
    }


def write_MQS_evaluation_dataset_from_items(root: Path, items: list[Path]) -> None:
    with (root / MQS_EVALUATION_DATASET).open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(_MQS_evaluation_record_from_item(root, item), ensure_ascii=False) + "\n")


def _clean_generated_dataset(root: Path) -> None:
    for path in root.glob("item_*"):
        if path.is_dir():
            shutil.rmtree(path)
    for name in ("dataset_manifest.json", MQS_EVALUATION_DATASET):
        path = root / name
        if path.exists():
            path.unlink()


def write_dataset(bundles: list[TaskBundle], root: Path, *, clean: bool = True) -> None:
    seen_task_ids: set[str] = set()
    for bundle in bundles:
        _validate_bundle(bundle)
        if bundle.task_id in seen_task_ids:
            raise ValueError(f"Duplicate TaskBundle.task_id: {bundle.task_id}")
        seen_task_ids.add(bundle.task_id)
    root.mkdir(parents=True, exist_ok=True)
    if clean:
        _clean_generated_dataset(root)
    items = [write_task_bundle(b, root) for b in bundles]
    (root / "dataset_manifest.json").write_text(
        json.dumps({"count": len(items)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_MQS_evaluation_dataset_from_items(root, items)
