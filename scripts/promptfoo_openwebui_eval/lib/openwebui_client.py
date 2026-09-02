from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import requests

from lib.bundle_common import file_sha256, trim_text


_RESERVED_CHAT_PAYLOAD_KEYS = {
    "chat_id",
    "files",
    "id",
    "messages",
    "model",
    "stream",
    "tool_ids",
    "user_message",
}


def _env_int(name: str, default: int) -> int:
    """Return a positive integer environment setting."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


def _extract_chat_text(payload: dict[str, Any], source_name: str) -> str:
    """Extract assistant text from OpenAI-compatible and OpenWebUI responses."""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(
                    str(item.get("text") or item.get("content") or "")
                    for item in content
                    if isinstance(item, dict) and (item.get("text") or item.get("content"))
                ).strip()
        if first.get("text"):
            return str(first["text"])
    for key in ("response", "output", "content"):
        if payload.get(key):
            return str(payload[key])
    raise RuntimeError(f"Could not extract completion text from {source_name} response")


def _first_choice_message(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    message = choices[0].get("message")
    return message if isinstance(message, dict) else {}


def _message_trace(message: dict[str, Any]) -> dict[str, Any]:
    """Return the assistant fields useful for debugging tools and reasoning."""
    trace: dict[str, Any] = {}
    for key in ("role", "content", "reasoning", "reasoning_content", "thinking", "tool_calls"):
        value = message.get(key)
        if value not in (None, "", []):
            trace[key] = value
    return trace


def _parse_tool_ids(value: Any) -> list[str]:
    """Normalize a JSON array, comma-separated string, or list of tool ids."""
    if value is None or value == "":
        return []
    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RuntimeError("OpenWebUI tool ids must be valid JSON or comma-separated ids") from exc
        else:
            parsed = [item.strip() for item in stripped.split(",") if item.strip()]
    if not isinstance(parsed, list) or not all(isinstance(item, str) and item.strip() for item in parsed):
        raise RuntimeError("OpenWebUI tool ids must be a list of non-empty strings")
    return list(dict.fromkeys(item.strip() for item in parsed))


def _merge_usage(total: dict[str, Any], usage: dict[str, Any]) -> None:
    """Accumulate numeric usage fields, including nested token detail objects."""
    for key, value in usage.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value
        elif isinstance(value, dict):
            nested = total.setdefault(key, {})
            if isinstance(nested, dict):
                _merge_usage(nested, value)


def _stored_output_text(output: Any) -> str:
    """Extract the final assistant message, falling back to the latest tool result."""
    if not isinstance(output, list):
        raise RuntimeError("OpenWebUI stored assistant output is not a list")
    assistant_texts: list[str] = []
    tool_texts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        parts = item.get("content") if item_type == "message" else item.get("output")
        if not isinstance(parts, list):
            continue
        text = "\n".join(
            str(part.get("text") or "")
            for part in parts
            if isinstance(part, dict) and part.get("type") in {"output_text", "input_text"} and part.get("text")
        ).strip()
        if not text:
            continue
        if item_type == "message":
            assistant_texts.append(text)
        elif item_type == "function_call_output":
            tool_texts.append(text)
    if assistant_texts:
        return assistant_texts[-1]
    if tool_texts:
        return tool_texts[-1]
    raise RuntimeError("OpenWebUI stored assistant output contains no text")


def _validate_extra_payload(extra_payload: dict[str, Any] | None) -> None:
    """Reject model extras that could replace bridge-controlled chat fields."""
    conflicts = sorted(_RESERVED_CHAT_PAYLOAD_KEYS.intersection(extra_payload or {}))
    if conflicts:
        raise RuntimeError(f"Model extra payload cannot override reserved fields: {', '.join(conflicts)}")


class OpenWebUIClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, cache_path: str | Path | None = None, timeout: int | None = None) -> None:
        self.base_url = (base_url or os.environ.get("OPENWEBUI_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENWEBUI_API_KEY", "")
        if not self.base_url:
            raise RuntimeError("OPENWEBUI_BASE_URL is not set")
        self.timeout = timeout if timeout is not None else _env_int("OPENWEBUI_TIMEOUT_SECONDS", 600)
        cache_env = cache_path or os.environ.get("OPENWEBUI_FILE_CACHE_PATH", "/tmp/openwebui_file_cache.json")
        self.cache_path = Path(cache_env)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache()
        self._knowledge_collections_cache: dict[str, list[str]] = {}
        self._model_tool_ids_cache: dict[str, list[str]] = {}

    def _headers(self, json_mode: bool = True) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if json_mode:
            headers["Content-Type"] = "application/json"
        return headers

    def _load_cache(self) -> dict[str, Any]:
        if self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        self.cache_path.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False), encoding="utf-8")

    def _cached_file_exists(self, file_id: str) -> bool:
        """Return whether a cached file id still exists in the configured OpenWebUI."""
        response = requests.get(
            f"{self.base_url}/api/v1/files/{file_id}",
            headers=self._headers(json_mode=False),
            timeout=self.timeout,
        )
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return True

    def upload_file(self, file_path: Path, wait: bool = True) -> str:
        file_path = file_path.resolve()
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        digest = file_sha256(file_path)
        cached = self.cache.get(digest)
        cached_file_id = str((cached or {}).get("file_id") or "")
        if cached_file_id and self._cached_file_exists(cached_file_id):
            return cached_file_id
        with file_path.open("rb") as f:
            response = requests.post(
                f"{self.base_url}/api/v1/files/?process=true&process_in_background=false",
                headers=self._headers(json_mode=False),
                files={"file": f},
                timeout=self.timeout,
            )
        response.raise_for_status()
        file_data = response.json()
        file_id = file_data["id"]
        if wait:
            self.wait_for_processing(file_id)
        self.cache[digest] = {"file_id": file_id, "path": str(file_path), "name": file_path.name, "size": file_path.stat().st_size}
        self._save_cache()
        return str(file_id)

    def wait_for_processing(self, file_id: str, timeout: int | None = None, poll_seconds: float = 2.0) -> None:
        processing_timeout = timeout if timeout is not None else _env_int("OPENWEBUI_FILE_PROCESS_TIMEOUT_SECONDS", 300)
        start = time.time()
        while time.time() - start < processing_timeout:
            response = requests.get(f"{self.base_url}/api/v1/files/{file_id}/process/status", headers=self._headers(json_mode=False), timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            status = payload.get("status")
            if status == "completed":
                return
            if status == "failed":
                raise RuntimeError(f"OpenWebUI file processing failed for {file_id}: {payload}")
            time.sleep(poll_seconds)
        raise TimeoutError(f"Timed out waiting for OpenWebUI file processing: {file_id}")

    def ensure_file_ids(self, file_paths: list[Path]) -> list[str]:
        return [self.upload_file(path) for path in file_paths]

    @staticmethod
    def _extract_text_from_response(payload: dict[str, Any]) -> str:
        return _extract_chat_text(payload, "OpenWebUI")

    def _post_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/chat/completions",
            headers=self._headers(json_mode=True),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _chat_with_server_tools(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        user_prompt: str,
        files_payload: list[dict[str, str]] | None,
        tool_ids: list[str],
        extra_payload: dict[str, Any] | None,
        include_trace: bool,
    ) -> dict[str, Any]:
        """Run OpenWebUI server tools through its streaming, persisted-chat path."""
        now_seconds = int(time.time())
        chat_response = requests.post(
            f"{self.base_url}/api/v1/chats/new",
            headers=self._headers(json_mode=True),
            json={
                "chat": {
                    "title": "Promptfoo OpenWebUI evaluation",
                    "models": [model],
                    "params": {},
                    "history": {"messages": {}, "currentId": None},
                    "messages": [],
                    "tags": ["promptfoo-eval"],
                    "timestamp": int(time.time() * 1000),
                },
                "folder_id": None,
            },
            timeout=self.timeout,
        )
        chat_response.raise_for_status()
        chat_id = str(chat_response.json().get("id") or "")
        if not chat_id:
            raise RuntimeError("OpenWebUI chat creation returned no thread id")

        user_message_id = str(uuid.uuid4())
        assistant_message_id = str(uuid.uuid4())
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "chat_id": chat_id,
            "id": assistant_message_id,
            "user_message": {
                "id": user_message_id,
                "parentId": None,
                "childrenIds": [assistant_message_id],
                "role": "user",
                "content": user_prompt,
                "files": files_payload or [],
                "models": [model],
                "timestamp": now_seconds,
            },
            "tool_ids": tool_ids,
        }
        if files_payload:
            payload["files"] = files_payload
        _validate_extra_payload(extra_payload)
        if extra_payload:
            payload.update(extra_payload)

        started_at = time.monotonic()
        streamed_usage: dict[str, Any] = {}
        with requests.post(
            f"{self.base_url}/api/chat/completions",
            headers=self._headers(json_mode=True),
            json=payload,
            timeout=self.timeout,
            stream=True,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data: ") or raw_line == "data: [DONE]":
                    continue
                try:
                    event = json.loads(raw_line[6:])
                except json.JSONDecodeError:
                    continue
                usage = event.get("usage") if isinstance(event, dict) else None
                if isinstance(usage, dict):
                    streamed_usage = usage

        stored_response = requests.get(
            f"{self.base_url}/api/v1/chats/{chat_id}",
            headers=self._headers(json_mode=False),
            timeout=self.timeout,
        )
        stored_response.raise_for_status()
        stored_chat = stored_response.json()
        stored_messages = (((stored_chat.get("chat") or {}).get("history") or {}).get("messages") or {})
        assistant_message = stored_messages.get(assistant_message_id)
        if not isinstance(assistant_message, dict):
            raise RuntimeError(f"OpenWebUI chat {chat_id} contains no persisted assistant response")
        if assistant_message.get("error"):
            raise RuntimeError(f"OpenWebUI tool execution failed: {assistant_message['error']}")
        output = assistant_message.get("output") or []
        text = _stored_output_text(output)
        stored_usage = assistant_message.get("usage")
        usage = stored_usage if isinstance(stored_usage, dict) else streamed_usage
        tool_output_by_call_id: dict[str, str] = {}
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "function_call_output":
                continue
            try:
                tool_output_by_call_id[str(item.get("call_id") or "")] = _stored_output_text([item])
            except RuntimeError:
                tool_output_by_call_id[str(item.get("call_id") or "")] = ""
        tool_calls = [
            {
                "id": str(item.get("call_id") or item.get("id") or ""),
                "name": str(item.get("name") or ""),
                "arguments": str(item.get("arguments") or ""),
                "result": trim_text(tool_output_by_call_id.get(str(item.get("call_id") or ""), ""), 4000),
            }
            for item in output
            if isinstance(item, dict) and item.get("type") == "function_call"
        ]
        result: dict[str, Any] = {
            "output": text,
            "raw": {"chat_id": chat_id, "assistant_message": assistant_message},
        }
        if include_trace:
            total_duration = round(time.monotonic() - started_at, 6)
            result["trace"] = {
                "rounds": [
                    {
                        "round": 0,
                        "duration_seconds": total_duration,
                        "message": {"role": "assistant", "content": text},
                        **({"usage": usage} if usage else {}),
                    }
                ],
                "tool_calls": tool_calls,
                **({"usage": usage} if usage else {}),
                "total_duration_seconds": total_duration,
                "message_count": len(messages),
                "chat_id": chat_id,
            }
        return result

    def _model_tool_ids(self, model: str) -> list[str]:
        """Return tool ids attached to a workspace model's metadata."""
        if model in self._model_tool_ids_cache:
            return self._model_tool_ids_cache[model]
        response = requests.get(
            f"{self.base_url}/api/models",
            headers=self._headers(json_mode=False),
            timeout=self.timeout,
        )
        response.raise_for_status()
        tool_ids: list[str] = []
        for item in response.json().get("data", []):
            if isinstance(item, dict) and item.get("id") == model:
                raw_ids = (((item.get("info") or {}).get("meta") or {}).get("toolIds") or [])
                tool_ids = _parse_tool_ids(raw_ids)
                break
        self._model_tool_ids_cache[model] = tool_ids
        return tool_ids

    def resolve_tool_ids(self, model: str, explicit_tool_ids: Any = None) -> list[str]:
        """Resolve per-request ids, then env ids, then workspace-model defaults."""
        if explicit_tool_ids is not None:
            return _parse_tool_ids(explicit_tool_ids)
        configured = os.environ.get("OPENWEBUI_TOOL_IDS", "").strip()
        if configured:
            return _parse_tool_ids(configured)
        return self._model_tool_ids(model)

    def _model_knowledge_collections(self, model: str) -> list[str]:
        configured = [item.strip() for item in os.environ.get("OPENWEBUI_KNOWLEDGE_COLLECTIONS", "").split(",") if item.strip()]
        if configured:
            return configured
        if model in self._knowledge_collections_cache:
            return self._knowledge_collections_cache[model]

        collections: list[str] = []
        try:
            response = requests.get(f"{self.base_url}/api/models", headers=self._headers(json_mode=False), timeout=self.timeout)
            response.raise_for_status()
            items = response.json().get("data", [])
            for item in items:
                if isinstance(item, dict) and item.get("id") == model:
                    knowledge = ((item.get("info") or {}).get("meta") or {}).get("knowledge") or []
                    collections = [str(entry["id"]) for entry in knowledge if isinstance(entry, dict) and entry.get("id")]
                    break
        except Exception:
            collections = []
        if not collections:
            response = requests.get(f"{self.base_url}/api/v1/knowledge/", headers=self._headers(json_mode=False), timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            items = payload if isinstance(payload, list) else payload.get("items", [])
            collections = [str(item["id"]) for item in items if isinstance(item, dict) and item.get("id")]

        self._knowledge_collections_cache[model] = collections
        return collections

    def _query_knowledge_files(self, model: str, arguments: str) -> str:
        params = json.loads(arguments or "{}")
        query = str(params.get("query") or "").strip()
        if not query:
            raise ValueError("query_knowledge_files requires a non-empty query")
        count = max(1, min(int(params.get("count") or params.get("k") or 5), 20))
        collections = self._model_knowledge_collections(model)
        if not collections:
            return json.dumps({"results": [], "error": "No knowledge collections available."}, ensure_ascii=False)

        response = requests.post(
            f"{self.base_url}/api/v1/retrieval/query/collection",
            headers=self._headers(json_mode=True),
            json={"collection_names": collections, "query": query, "k": count},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return json.dumps(self._format_knowledge_results(response.json(), count), ensure_ascii=False)

    @staticmethod
    def _format_knowledge_results(payload: dict[str, Any], count: int) -> dict[str, Any]:
        max_chars = int(os.environ.get("OPENWEBUI_TOOL_RESULT_MAX_CHARS", "1200"))
        documents = payload.get("documents") or []
        metadatas = payload.get("metadatas") or []
        distances = payload.get("distances") or []
        results: list[dict[str, Any]] = []
        doc_rows = documents if documents and isinstance(documents[0], list) else [documents]
        meta_rows = metadatas if metadatas and isinstance(metadatas[0], list) else [metadatas]
        distance_rows = distances if distances and isinstance(distances[0], list) else [distances]
        for row_index, row in enumerate(doc_rows):
            for item_index, document in enumerate(row or []):
                meta = (meta_rows[row_index][item_index] if row_index < len(meta_rows) and item_index < len(meta_rows[row_index]) else {}) or {}
                distance = distance_rows[row_index][item_index] if row_index < len(distance_rows) and item_index < len(distance_rows[row_index]) else None
                results.append({"content": trim_text(str(document), max_chars), "metadata": meta, "distance": distance})
                if len(results) >= count:
                    return {"results": results}
        return {"results": results}

    def _execute_tool_call(self, model: str, tool_call: dict[str, Any]) -> str:
        function = tool_call.get("function") or {}
        name = function.get("name")
        try:
            if name == "query_knowledge_files":
                return self._query_knowledge_files(model, str(function.get("arguments") or "{}"))
            return json.dumps({"error": f"Unsupported tool: {name}"}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": f"{name} failed: {exc}"}, ensure_ascii=False)

    @staticmethod
    def _assistant_tool_message(message: dict[str, Any]) -> dict[str, Any]:
        return {"role": "assistant", "content": message.get("content"), "tool_calls": message.get("tool_calls") or []}

    def chat(
        self,
        *,
        model: str,
        user_prompt: str,
        system_prompt: str | None = None,
        files_payload: list[dict[str, str]] | None = None,
        tool_ids: Any = None,
        extra_payload: dict[str, Any] | None = None,
        trigger_outlet: bool = False,
        include_trace: bool = False,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_prompt})
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if files_payload:
            payload["files"] = files_payload
        resolved_tool_ids = self.resolve_tool_ids(model, tool_ids)
        if resolved_tool_ids:
            payload["tool_ids"] = resolved_tool_ids
        _validate_extra_payload(extra_payload)
        if extra_payload:
            payload.update(extra_payload)

        if resolved_tool_ids:
            return self._chat_with_server_tools(
                model=model,
                messages=messages,
                user_prompt=user_prompt,
                files_payload=files_payload,
                tool_ids=resolved_tool_ids,
                extra_payload=extra_payload,
                include_trace=include_trace,
            )

        max_tool_rounds = int(os.environ.get("OPENWEBUI_MAX_TOOL_CALL_ROUNDS", "8"))
        started_at = time.monotonic()
        trace: dict[str, Any] = {"rounds": [], "tool_calls": []}
        data: dict[str, Any] = {}
        for round_index in range(max_tool_rounds + 1):
            payload["messages"] = messages
            round_started_at = time.monotonic()
            data = self._post_chat_completion(payload)
            round_duration = time.monotonic() - round_started_at
            message = _first_choice_message(data)
            if include_trace:
                usage = data.get("usage")
                if isinstance(usage, dict):
                    _merge_usage(trace.setdefault("usage", {}), usage)
                trace["rounds"].append(
                    {
                        "round": round_index,
                        "duration_seconds": round(round_duration, 6),
                        "message": _message_trace(message),
                        **({"usage": usage} if isinstance(usage, dict) else {}),
                    }
                )
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                break
            messages.append(self._assistant_tool_message(message))
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                tool_started_at = time.monotonic()
                tool_result = self._execute_tool_call(model, tool_call)
                tool_duration = time.monotonic() - tool_started_at
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(tool_call.get("id") or ""),
                        "content": tool_result,
                    }
                )
                if include_trace:
                    function = tool_call.get("function") or {}
                    trace["tool_calls"].append(
                        {
                            "round": round_index,
                            "id": str(tool_call.get("id") or ""),
                            "name": str(function.get("name") or ""),
                            "arguments": str(function.get("arguments") or ""),
                            "duration_seconds": round(tool_duration, 6),
                            "result": trim_text(tool_result, _env_int("OPENWEBUI_TRACE_TOOL_RESULT_MAX_CHARS", 4000)),
                        }
                    )
        else:
            raise RuntimeError(f"OpenWebUI exceeded {max_tool_rounds} tool-call rounds")

        text = self._extract_text_from_response(data)
        if trigger_outlet:
            completed_payload = {"model": model, "messages": [{"role": "user", "content": user_prompt}, {"role": "assistant", "content": text}]}
            try:
                requests.post(f"{self.base_url}/api/chat/completed", headers=self._headers(json_mode=True), json=completed_payload, timeout=self.timeout)
            except Exception:
                pass
        result = {"output": text, "raw": data}
        if include_trace:
            trace["total_duration_seconds"] = round(time.monotonic() - started_at, 6)
            trace["message_count"] = len(messages)
            result["trace"] = trace
        return result


class OpenAIEndpointClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: int | None = None) -> None:
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_ENDPOINT_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "http://127.0.0.1:11434/v1"
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_ENDPOINT_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout if timeout is not None else _env_int("OPENAI_ENDPOINT_TIMEOUT_SECONDS", 600)

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _extract_text_from_response(payload: dict[str, Any]) -> str:
        return _extract_chat_text(payload, "OpenAI endpoint")

    def chat(self, *, model: str, user_prompt: str, extra_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": user_prompt}],
            "stream": False,
        }
        if extra_payload:
            payload.update(extra_payload)
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return {"output": self._extract_text_from_response(data), "raw": data}
