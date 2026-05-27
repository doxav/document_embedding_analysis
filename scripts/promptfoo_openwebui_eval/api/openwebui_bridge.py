from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys

BUNDLE_ROOT = _Path(__file__).resolve().parents[1]
if str(BUNDLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(BUNDLE_ROOT))

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from lib.bundle_common import as_bool, build_openwebui_user_prompt, load_text, parse_jsonish, resolve_repo_path, trim_text
from lib.openwebui_client import OllamaClient, OpenWebUIClient

app = FastAPI(title="OpenWebUI generation bridge")


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "backend": os.environ.get("GENERATION_BACKEND", "openwebui"),
        "base_url": os.environ.get("OPENWEBUI_BASE_URL", ""),
        "ollama_base_url": os.environ.get("OLLAMA_BASE_URL", ""),
    }


def _source_context(local_paths: list[Path]) -> str:
    if not local_paths:
        return ""
    max_total = int(os.environ.get("OLLAMA_MAX_SOURCE_CHARS", "60000"))
    max_each = int(os.environ.get("OLLAMA_MAX_SOURCE_FILE_CHARS", "6000"))
    parts: list[str] = []
    used = 0
    for path in local_paths:
        if used >= max_total:
            break
        text = trim_text(load_text(path), min(max_each, max_total - used))
        if not text:
            continue
        block = f"### {path.name}\n{text}"
        parts.append(block)
        used += len(block)
    if not parts:
        return ""
    return "Source documents:\n\n" + "\n\n".join(parts)


@app.post("/generate")
def generate(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        vars_dict = {k: ("" if v is None else v) for k, v in payload.items()}

        source_paths = parse_jsonish(vars_dict.get("source_paths_json"), default=[]) or []
        local_paths: list[Path] = []
        missing_paths: list[str] = []
        for path_str in source_paths:
            resolved = resolve_repo_path(str(path_str))
            if resolved and resolved.exists() and resolved.is_file():
                local_paths.append(resolved)
            else:
                missing_paths.append(str(path_str))

        extra_payload: dict[str, Any] = {}
        for src_key, dst_key, caster in (
            ("generation_temperature", "temperature", float),
            ("generation_top_p", "top_p", float),
            ("generation_max_tokens", "max_tokens", lambda x: int(float(x))),
        ):
            raw = str(vars_dict.get(src_key) or "").strip()
            if raw:
                try:
                    extra_payload[dst_key] = caster(raw)
                except Exception:
                    pass

        backend = os.environ.get("GENERATION_BACKEND", "openwebui").strip().lower()
        if backend == "ollama":
            user_prompt = build_openwebui_user_prompt(vars_dict, file_ids=None)
            context = _source_context(local_paths)
            if context:
                user_prompt = f"{user_prompt}\n\n{context}"
            model = (
                os.environ.get("OLLAMA_MODEL", "").strip()
                or str(vars_dict.get("openwebui_pipe_model") or "").strip()
                or "qwen-laptop:latest"
            )
            result = OllamaClient().chat(model=model, user_prompt=user_prompt, extra_payload=extra_payload or None)
            return {
                "output": result["output"],
                "file_ids": [],
                "missing_paths": missing_paths,
                "backend": "ollama",
            }

        client = OpenWebUIClient()
        file_ids = client.ensure_file_ids(local_paths) if local_paths else []
        include_files_payload = as_bool(os.environ.get("OPENWEBUI_INCLUDE_FILES_PAYLOAD"), True)
        files_payload = [{"type": "file", "id": file_id} for file_id in file_ids] if include_files_payload and file_ids else None
        user_prompt = build_openwebui_user_prompt(vars_dict, file_ids=file_ids)
        model = (
            str(vars_dict.get("openwebui_pipe_model") or "").strip()
            or os.environ.get("OPENWEBUI_PIPE_MODEL", "").strip()
            or "summarizer---kohaku"
        )
        result = client.chat(
            model=model,
            user_prompt=user_prompt,
            files_payload=files_payload,
            extra_payload=extra_payload or None,
            trigger_outlet=as_bool(os.environ.get("OPENWEBUI_TRIGGER_OUTLET"), False),
        )
        return {
            "output": result["output"],
            "file_ids": file_ids,
            "missing_paths": missing_paths,
            "backend": "openwebui",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
