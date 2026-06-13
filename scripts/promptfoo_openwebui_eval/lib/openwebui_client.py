from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from lib.bundle_common import file_sha256, trim_text


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

    def upload_file(self, file_path: Path, wait: bool = True) -> str:
        file_path = file_path.resolve()
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        digest = file_sha256(file_path)
        cached = self.cache.get(digest)
        if cached and cached.get("file_id"):
            return str(cached["file_id"])
        with file_path.open("rb") as f:
            response = requests.post(f"{self.base_url}/api/v1/files/", headers=self._headers(json_mode=False), files={"file": f}, timeout=self.timeout)
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

    def chat(self, *, model: str, user_prompt: str, files_payload: list[dict[str, str]] | None = None, extra_payload: dict[str, Any] | None = None, trigger_outlet: bool = False) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if files_payload:
            payload["files"] = files_payload
        if extra_payload:
            payload.update(extra_payload)

        max_tool_rounds = int(os.environ.get("OPENWEBUI_MAX_TOOL_CALL_ROUNDS", "8"))
        data: dict[str, Any] = {}
        for _ in range(max_tool_rounds + 1):
            payload["messages"] = messages
            data = self._post_chat_completion(payload)
            message = _first_choice_message(data)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                break
            messages.append(self._assistant_tool_message(message))
            messages.extend(
                {
                    "role": "tool",
                    "tool_call_id": str(tool_call.get("id") or ""),
                    "content": self._execute_tool_call(model, tool_call),
                }
                for tool_call in tool_calls
                if isinstance(tool_call, dict)
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
        return {"output": text, "raw": data}


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
