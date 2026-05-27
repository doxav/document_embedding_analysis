from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from lib.bundle_common import file_sha256


class OpenWebUIClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, cache_path: str | Path | None = None, timeout: int = 600) -> None:
        self.base_url = (base_url or os.environ.get("OPENWEBUI_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENWEBUI_API_KEY", "")
        if not self.base_url:
            raise RuntimeError("OPENWEBUI_BASE_URL is not set")
        self.timeout = timeout
        cache_env = cache_path or os.environ.get("OPENWEBUI_FILE_CACHE_PATH", "/tmp/openwebui_file_cache.json")
        self.cache_path = Path(cache_env)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache()

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

    def wait_for_processing(self, file_id: str, timeout: int = 300, poll_seconds: float = 2.0) -> None:
        start = time.time()
        while time.time() - start < timeout:
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
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts: list[str] = []
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get("text") or item.get("content") or ""
                            if text:
                                parts.append(str(text))
                        elif item:
                            parts.append(str(item))
                    return "\n".join(parts).strip()
            if first.get("text"):
                return str(first["text"])
        for key in ("response", "output", "content"):
            if payload.get(key):
                return str(payload[key])
        raise RuntimeError(f"Could not extract completion text from OpenWebUI response: {payload}")

    def chat(self, *, model: str, user_prompt: str, files_payload: list[dict[str, str]] | None = None, extra_payload: dict[str, Any] | None = None, trigger_outlet: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": user_prompt}], "stream": False}
        if files_payload:
            payload["files"] = files_payload
        if extra_payload:
            payload.update(extra_payload)
        response = requests.post(f"{self.base_url}/api/chat/completions", headers=self._headers(json_mode=True), json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        text = self._extract_text_from_response(data)
        if trigger_outlet:
            completed_payload = {"model": model, "messages": [{"role": "user", "content": user_prompt}, {"role": "assistant", "content": text}]}
            try:
                requests.post(f"{self.base_url}/api/chat/completed", headers=self._headers(json_mode=True), json=completed_payload, timeout=self.timeout)
            except Exception:
                pass
        return {"output": text, "raw": data}


class OllamaClient:
    def __init__(self, base_url: str | None = None, timeout: int = 600) -> None:
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout = timeout

    @staticmethod
    def _extract_text_from_response(payload: dict[str, Any]) -> str:
        message = payload.get("message")
        if isinstance(message, dict) and message.get("content"):
            return str(message["content"])
        if payload.get("response"):
            return str(payload["response"])
        raise RuntimeError(f"Could not extract completion text from Ollama response: {payload}")

    def chat(self, *, model: str, user_prompt: str, extra_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if extra_payload:
            for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("max_tokens", "num_predict")):
                if src in extra_payload:
                    options[dst] = extra_payload[src]
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": user_prompt}],
            "stream": False,
        }
        if options:
            payload["options"] = options
        response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return {"output": self._extract_text_from_response(data), "raw": data}
