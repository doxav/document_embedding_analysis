from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

RE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

CSV_FIELDS = [
    "__description",
    "__metadata:dataset",
    "__metadata:stage",
    "__metadata:task_type",
    "__metadata:source_document_mode",
    "task_id",
    "dataset",
    "task_type",
    "title",
    "abstract",
    "instruction",
    "request_prompt",
    "query",
    "context",
    "gold_summary",
    "expected_path",
    "sections_path",
    "source_manifest_path",
    "dea_solution_path",
    "instruction_path",
    "source_paths_json",
    "kb_ids_json",
    "source_titles_json",
    "source_count",
    "facts_to_check",
    "required_terms",
    "forbidden_terms",
    "candidate_content_type",
    "openwebui_pipe_model",
    "tool_parameters_json",
    "summarizer_model_id",
    "algorithm",
    "target_length",
    "structure",
    "generation_temperature",
    "generation_top_p",
    "generation_max_tokens",
    "min_chars",
    "max_chars",
    "source_document_mode",
    "openwebui_extra_instructions",
    "candidate_answer",
]


def repo_root() -> Path:
    root = os.environ.get("DEA_REPO_ROOT", "").strip()
    if not root:
        raise RuntimeError("DEA_REPO_ROOT is not set")
    return Path(root).resolve()


def resolve_repo_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def normalize_mqs_relpath(dataset: str, rel_path: str | None) -> str:
    rel = str(rel_path or "").strip().lstrip("./")
    if not rel:
        return ""
    if rel.startswith("output/"):
        return rel
    return f"output/{dataset}/{rel}"


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(load_text(path))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def raise_csv_field_size_limit() -> None:
    """Allow generated Promptfoo CSV rows to contain embedded source documents."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    raise_csv_field_size_limit()
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ensure_all_fields(row: dict[str, Any]) -> dict[str, Any]:
    out = {field: row.get(field, "") for field in CSV_FIELDS}
    for key, value in row.items():
        if key not in out:
            out[key] = value
    return out


def write_csv_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [ensure_all_fields(dict(r)) for r in rows]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


def parse_jsonish(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list, int, float, bool)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def first_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return ""
    parts = RE_SENTENCE_SPLIT.split(text, maxsplit=1)
    return parts[0].strip()


def first_sentences(text: str, limit: int = 5) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    parts = [p.strip() for p in RE_SENTENCE_SPLIT.split(text) if p.strip()]
    return parts[:limit]


def trim_text(text: str, limit: int = 1200) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def xml_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _xml_tag(name: Any) -> str | None:
    tag = str(name or "").strip()
    if re.fullmatch(r"[A-Za-z_][\w.:-]*", tag):
        return tag
    return None


def _xml_value(value: Any) -> str:
    if isinstance(value, (dict, list, bool, int, float)) or value is None:
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _tool_param_lines(params: dict[str, Any], indent: str = "  ") -> list[str]:
    lines: list[str] = []
    for raw_key, value in params.items():
        key = _xml_tag(raw_key)
        if not key:
            continue
        if isinstance(value, dict):
            child_lines = _tool_param_lines(value, indent + "  ")
            if child_lines:
                lines.append(f"{indent}<{key}>")
                lines.extend(child_lines)
                lines.append(f"{indent}</{key}>")
            continue
        lines.append(f"{indent}<{key}>{xml_escape(_xml_value(value))}</{key}>")
    return lines


def build_request_prompt(dataset: str, title: str, abstract: str, instruction: str) -> str:
    dataset = (dataset or "").strip().lower()
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    instruction = (instruction or "").strip()

    intro_lines = [instruction] if instruction else []
    if title:
        intro_lines.extend(["", f"Title: {title}"])
    if abstract:
        intro_lines.extend(["", "Context / abstract:", abstract])
    intro_lines.extend([
        "",
        "Return markdown.",
        "Stay within the task scope.",
        "Do not invent claims, numbers, dates, or citations.",
    ])
    if dataset in {"bigsurvey", "multilexsum"}:
        intro_lines.append("Use only the provided source documents.")
    else:
        intro_lines.append("Use the supplied title, context, and any available references.")
    return "\n".join(intro_lines).strip()


def build_openwebui_user_prompt(vars_dict: dict[str, Any], file_ids: list[str] | None = None) -> str:
    request_prompt = (
        vars_dict.get("request_prompt")
        or build_request_prompt(
            vars_dict.get("dataset", ""),
            vars_dict.get("title", ""),
            vars_dict.get("abstract", ""),
            vars_dict.get("instruction", ""),
        )
    )
    blocks: list[str] = [request_prompt]
    overrides = parse_jsonish(vars_dict.get("tool_parameters_json"), default={})
    if not isinstance(overrides, dict):
        overrides = {}
    for key, value in {
        "summarizer_model_id": vars_dict.get("summarizer_model_id") or os.environ.get("OPENWEBUI_DEFAULT_SUMMARIZER_MODEL_ID", ""),
        "algorithm": vars_dict.get("algorithm") or os.environ.get("OPENWEBUI_DEFAULT_ALGORITHM", ""),
        "target_length": vars_dict.get("target_length") or os.environ.get("OPENWEBUI_DEFAULT_TARGET_LENGTH", ""),
        "structure": vars_dict.get("structure") or os.environ.get("OPENWEBUI_DEFAULT_STRUCTURE", ""),
    }.items():
        if str(value).strip():
            overrides[key] = value

    tool_lines = _tool_param_lines(overrides)
    if tool_lines:
        blocks.append("<tool_parameters>\n" + "\n".join(tool_lines) + "\n</tool_parameters>")
    if file_ids:
        blocks.append("<files_list>\n" + json.dumps(file_ids, ensure_ascii=False) + "\n</files_list>")
    kb_ids = parse_jsonish(vars_dict.get("kb_ids_json"), default=[]) or []
    if kb_ids:
        blocks.append("<kb_list>\n" + json.dumps(kb_ids, ensure_ascii=False) + "\n</kb_list>")
    extra = str(vars_dict.get("openwebui_extra_instructions", "") or "").strip()
    if extra:
        blocks.append(extra)
    return "\n\n".join(blocks).strip()


def source_titles_from_manifest(source_manifest: list[dict[str, Any]]) -> list[str]:
    titles = []
    for item in source_manifest:
        title = (item.get("title") or item.get("source_id") or "").strip()
        if title:
            titles.append(title)
    return titles


def derive_facts_from_sections(sections: list[dict[str, Any]], limit: int = 12) -> list[str]:
    facts: list[str] = []
    if not sections:
        return facts
    if len(sections) == 1:
        facts.extend(first_sentences(sections[0].get("text", ""), min(limit, 8)))
        return facts[:limit]
    for section in sections:
        heading = str(section.get("section", "")).strip()
        sentence = first_sentence(section.get("text", ""))
        if heading and sentence:
            facts.append(f"{heading}: {sentence}")
        elif heading:
            facts.append(heading)
        elif sentence:
            facts.append(sentence)
        if len(facts) >= limit:
            break
    return facts[:limit]


def dea_to_markdown(solution: dict[str, Any]) -> str:
    lines = [f"# {solution.get('title', '')}".strip(), ""]
    abstract = str(solution.get("abstract") or solution.get("context") or "").strip()
    if abstract:
        lines.extend(["## Abstract", "", abstract, ""])
    for step in solution.get("plan", []) or []:
        section = str(step.get("section") or "Section").strip()
        content = str(step.get("content") or "").strip()
        lines.extend([f"## {section}", "", content, ""])
    resources = solution.get("resources") or []
    if resources:
        lines.extend(["## References", ""])
        for resource in resources:
            rid = resource.get("resource_id", "")
            desc = resource.get("resource_description") or resource.get("resource") or ""
            if desc:
                lines.append(f"{rid}. {desc}".strip())
        lines.append("")
    return "\n".join(lines).strip()


def infer_source_document_mode_from_resources(resources: list[dict[str, Any]]) -> str:
    if not resources:
        return "no_resources"
    for resource in resources:
        for key in ("path", "local_path", "reference", "resource"):
            raw = str(resource.get(key) or "").strip()
            if raw and not raw.startswith(("http://", "https://", "doi:")):
                candidate = resolve_repo_path(raw) if not Path(raw).is_absolute() else Path(raw)
                if candidate.exists():
                    return "source_locator_best_effort"
    return "reference_only_open_retrieval"


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default
