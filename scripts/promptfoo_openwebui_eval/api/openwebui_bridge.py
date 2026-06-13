from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys

BUNDLE_ROOT = _Path(__file__).resolve().parents[1]
if str(BUNDLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(BUNDLE_ROOT))

import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException

from lib.bundle_common import as_bool, build_openwebui_user_prompt, load_text, parse_jsonish, resolve_repo_path, trim_text
from lib.openwebui_client import OpenAIEndpointClient, OpenWebUIClient

app = FastAPI(title="OpenWebUI generation bridge")

FORWARDED_BRIDGE_KEYS = (
    "source_paths_json",
    "kb_ids_json",
    "tool_parameters_json",
    "summarizer_model_id",
    "algorithm",
    "target_length",
    "structure",
    "openwebui_extra_instructions",
    "openwebui_system_prompt",
    "openwebui_include_trace",
    "include_trace",
)

GENERATION_OPTIONS: tuple[tuple[str, str, Callable[[Any], int | float]], ...] = (
    ("generation_temperature", "temperature", float),
    ("generation_top_p", "top_p", float),
    ("generation_max_tokens", "max_tokens", lambda value: int(float(value))),
)

OPENAI_OPTION_ALIASES: tuple[tuple[str, str, Callable[[Any], int | float]], ...] = (
    ("temperature", "generation_temperature", float),
    ("top_p", "generation_top_p", float),
    ("max_tokens", "generation_max_tokens", lambda value: int(float(value))),
    ("max_completion_tokens", "generation_max_tokens", lambda value: int(float(value))),
)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "backend": os.environ.get("GENERATION_BACKEND", "openwebui"),
        "base_url": os.environ.get("OPENWEBUI_BASE_URL", ""),
        "openai_endpoint_base_url": os.environ.get("OPENAI_ENDPOINT_BASE_URL") or os.environ.get("OPENAI_BASE_URL", ""),
    }


def _source_context(local_paths: list[Path]) -> str:
    if not local_paths:
        return ""
    max_total = int(os.environ.get("OPENAI_ENDPOINT_MAX_SOURCE_CHARS", "60000"))
    max_each = int(os.environ.get("OPENAI_ENDPOINT_MAX_SOURCE_FILE_CHARS", "6000"))
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


def _message_content_to_text(content: Any) -> str:
    """Convert OpenAI chat message content into plain text for the bridge."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
                continue
            nested_content = item.get("content")
            if isinstance(nested_content, str):
                parts.append(nested_content)
        return "\n".join(part for part in parts if part.strip())
    raise HTTPException(
        status_code=400,
        detail="message content must be a string or a list of text parts",
    )


def _openai_messages_to_bridge_parts(messages: Any) -> tuple[str, str]:
    """Split OpenAI chat messages into native system and request prompts."""
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")

    system_blocks: list[str] = []
    prompt_blocks: list[tuple[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise HTTPException(status_code=400, detail=f"messages[{index}] must be an object")
        role = str(message.get("role") or "").strip()
        if not role:
            raise HTTPException(status_code=400, detail=f"messages[{index}].role is required")
        text = _message_content_to_text(message.get("content")).strip()
        if not text:
            continue
        if role == "system":
            system_blocks.append(text)
        else:
            prompt_blocks.append((role, text))

    if not prompt_blocks:
        raise HTTPException(status_code=400, detail="messages must contain at least one text content")
    if len(prompt_blocks) == 1 and prompt_blocks[0][0] == "user":
        request_prompt = prompt_blocks[0][1]
    else:
        request_prompt = "\n\n".join(f"{role}: {text}" for role, text in prompt_blocks)
    return "\n\n".join(system_blocks), request_prompt


def _openai_messages_to_prompt(messages: Any) -> str:
    """Build a single bridge prompt from OpenAI chat messages."""
    _system_prompt, request_prompt = _openai_messages_to_bridge_parts(messages)
    return request_prompt


def _copy_numeric_option(source: dict[str, Any], destination: dict[str, Any], source_key: str, destination_key: str, caster: Callable[[Any], int | float]) -> None:
    """Validate and copy a supported generation option into bridge payload fields."""
    if source_key not in source or source[source_key] is None:
        return
    try:
        value = caster(source[source_key])
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"{source_key} must be numeric")
    destination[destination_key] = str(value)


def _bridge_payload_from_openai(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate a non-streaming OpenAI chat completion request to bridge payload."""
    model = str(payload.get("model") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    if as_bool(payload.get("stream"), False):
        raise HTTPException(status_code=400, detail="stream=true is not supported by this bridge")

    system_prompt, request_prompt = _openai_messages_to_bridge_parts(payload.get("messages"))
    bridge_payload: dict[str, Any] = {
        "openwebui_pipe_model": model,
        "request_prompt": request_prompt,
    }
    if system_prompt:
        bridge_payload["openwebui_system_prompt"] = system_prompt
    for key in FORWARDED_BRIDGE_KEYS:
        if key in payload:
            bridge_payload[key] = payload[key]

    for source_key, destination_key, caster in OPENAI_OPTION_ALIASES:
        _copy_numeric_option(payload, bridge_payload, source_key, destination_key, caster)
    return bridge_payload


def _extra_payload(vars_dict: dict[str, Any]) -> dict[str, Any]:
    extra_payload: dict[str, Any] = {}
    for src_key, dst_key, caster in GENERATION_OPTIONS:
        raw = str(vars_dict.get(src_key) or "").strip()
        if raw:
            try:
                extra_payload[dst_key] = caster(raw)
            except (TypeError, ValueError):
                pass
    raw_model_params = parse_jsonish(vars_dict.get("openwebui_model_params_json"), default={}) or {}
    if not isinstance(raw_model_params, dict):
        raise HTTPException(status_code=400, detail="openwebui_model_params_json must be a JSON object")
    raw_extra = raw_model_params.get("model_extra_payload_json", {})
    model_extra = parse_jsonish(raw_extra, default=raw_extra) if isinstance(raw_extra, str) else raw_extra
    if model_extra:
        if not isinstance(model_extra, dict):
            raise HTTPException(status_code=400, detail="model_extra_payload_json must be a JSON object")
        # User-configured provider/model-specific payload keys are applied last.
        extra_payload.update(model_extra)
    return extra_payload


@app.post("/v1/chat/completions")
def openai_chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a minimal OpenAI-compatible chat completion through the bridge."""
    bridge_payload = _bridge_payload_from_openai(payload)
    result = generate(bridge_payload)
    model = str(bridge_payload["openwebui_pipe_model"])
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": str(result.get("output") or "")},
                "finish_reason": "stop",
            }
        ],
    }


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

        extra_payload = _extra_payload(vars_dict)

        backend = os.environ.get("GENERATION_BACKEND", "openwebui").strip().lower()
        include_trace = as_bool(vars_dict.get("openwebui_include_trace") or vars_dict.get("include_trace"), False)

        if backend == "openai_endpoint":
            user_prompt = build_openwebui_user_prompt(vars_dict, file_ids=None)
            context = _source_context(local_paths)
            if context:
                user_prompt = f"{user_prompt}\n\n{context}"
            model = (
                str(vars_dict.get("openwebui_pipe_model") or "").strip()
                or os.environ.get("OPENAI_ENDPOINT_MODEL", "").strip()
                or os.environ.get("OPENAI_MODEL", "").strip()
            )
            if not model:
                raise RuntimeError("OpenAI endpoint model is not set")
            result = OpenAIEndpointClient().chat(model=model, user_prompt=user_prompt, extra_payload=extra_payload or None)
            return {
                "output": result["output"],
                "file_ids": [],
                "missing_paths": missing_paths,
                "backend": "openai_endpoint",
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
            system_prompt=str(vars_dict.get("openwebui_system_prompt") or "").strip(),
            files_payload=files_payload,
            extra_payload=extra_payload or None,
            trigger_outlet=as_bool(os.environ.get("OPENWEBUI_TRIGGER_OUTLET"), False),
            include_trace=include_trace,
        )
        response = {
            "output": result["output"],
            "file_ids": file_ids,
            "missing_paths": missing_paths,
            "backend": "openwebui",
        }
        if include_trace:
            response["trace"] = result.get("trace", {})
        return response
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
