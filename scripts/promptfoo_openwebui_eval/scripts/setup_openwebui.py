#!/usr/bin/env python3
"""Provision and smoke-test an OpenWebUI instance for the evaluation bundle."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import time
import uuid
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_MODEL_ID = "deepseek/deepseek-v4-flash-0731"
DEFAULT_CUSTOM_MODEL_ID = "dea-contradiction-auditor"
DEFAULT_REASONING = {"enabled": False, "effort": "low"}


def read_env_file(path: Path) -> dict[str, str]:
    """Read the simple KEY=VALUE subset used by this bundle's dotenv files."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def update_env_file(path: Path, updates: dict[str, str]) -> None:
    """Atomically update selected dotenv keys while preserving unrelated lines."""
    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    remaining = dict(updates)
    output: list[str] = []
    for line in existing_lines:
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=", line)
        if match and match.group(1) in remaining:
            key = match.group(1)
            output.append(f"{key}={remaining.pop(key)}")
        else:
            output.append(line)
    if remaining:
        if output and output[-1] != "":
            output.append("")
        output.extend(f"{key}={value}" for key, value in remaining.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def prepare_local_env(parent_env_path: Path, local_env_path: Path) -> None:
    """Create the ignored managed-OpenWebUI dotenv without printing secrets."""
    parent = read_env_file(parent_env_path)
    local = read_env_file(local_env_path)
    provider_key = (
        local.get("OPENROUTER_API_KEY")
        or parent.get("OPENROUTER_API_KEY")
        or parent.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    )
    if not provider_key:
        raise RuntimeError(
            f"No provider key found in {local_env_path}, {parent_env_path}, or the process environment"
        )
    updates = {
        "OPENROUTER_API_KEY": provider_key,
        "OWI_LLM_BASE_URL": local.get("OWI_LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        "OWI_LLM_MODEL_ID": local.get("OWI_LLM_MODEL_ID", DEFAULT_BASE_MODEL_ID),
        "OWI_ADMIN_EMAIL": local.get("OWI_ADMIN_EMAIL", "admin@localhost"),
        "OWI_ADMIN_PASSWORD": local.get("OWI_ADMIN_PASSWORD") or secrets.token_urlsafe(24),
        "WEBUI_SECRET_KEY": local.get("WEBUI_SECRET_KEY") or secrets.token_urlsafe(36),
        "OPENWEBUI_PORT": local.get("OPENWEBUI_PORT", "8080"),
    }
    update_env_file(local_env_path, updates)
    configured_base_url = parent.get("OPENWEBUI_BASE_URL", "").strip().lower()
    if configured_base_url in {"", "none", "null"}:
        update_env_file(
            parent_env_path,
            {"OPENWEBUI_BASE_URL": "http://127.0.0.1:8080", "OPENWEBUI_MANAGED": "true"},
        )
        print(f"Recorded the managed endpoint in {parent_env_path}.")
    print(f"Managed OpenWebUI configuration prepared in {local_env_path} (secrets redacted).")


def resolve_tool_path(raw_path: str, tools_root: Path | None, repo_root: Path) -> Path:
    """Resolve absolute, repository-relative, and tools/-prefixed tool paths."""
    if not raw_path.strip():
        raise ValueError("OWI_DEFAULT_TOOL is empty")
    raw = Path(raw_path).expanduser()
    candidates = [raw] if raw.is_absolute() else [repo_root / raw, Path.cwd() / raw]
    if tools_root is not None and not raw.is_absolute():
        candidates.append(tools_root / raw)
        if raw.parts and raw.parts[0] == "tools":
            candidates.append(tools_root.joinpath(*raw.parts[1:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"OWI_DEFAULT_TOOL was not found; checked: {rendered}")


def tool_identity(content: str, path: Path) -> tuple[str, str, str]:
    """Extract the OpenWebUI tool id, name, and description from its header."""
    header = content[:8000]

    def field(name: str, fallback: str) -> str:
        """Return one header field or its fallback."""
        match = re.search(rf"(?m)^\s*{re.escape(name)}:\s*(.+?)\s*$", header)
        return match.group(1).strip() if match else fallback

    fallback_id = re.sub(r"\W+", "_", path.stem).strip("_").lower()
    tool_id = field("id", fallback_id)
    if not tool_id.isidentifier():
        raise ValueError(f"Tool id {tool_id!r} from {path} is not a valid Python identifier")
    return tool_id.lower(), field("name", path.stem), field("description", "")


class OpenWebUIAdmin:
    """Small authenticated client for the stable v0.10.2 provisioning routes."""

    def __init__(self, base_url: str, timeout_seconds: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.token = ""

    def _request(
        self,
        method: str,
        path: str,
        *,
        expected: tuple[int, ...] = (200,),
        **kwargs: Any,
    ) -> requests.Response:
        """Send one authenticated request and raise a bounded descriptive error."""
        headers = dict(kwargs.pop("headers", {}))
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        response = requests.request(
            method,
            f"{self.base_url}{path}",
            headers=headers,
            timeout=self.timeout_seconds,
            **kwargs,
        )
        if response.status_code not in expected:
            detail = response.text.replace("\n", " ")[:500]
            raise RuntimeError(f"OpenWebUI {method} {path} returned HTTP {response.status_code}: {detail}")
        return response

    def wait_until_ready(self, timeout_seconds: int = 300) -> None:
        """Wait until the OpenWebUI HTTP health endpoint answers successfully."""
        deadline = time.monotonic() + timeout_seconds
        last_error = "no response"
        while time.monotonic() < deadline:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.ok:
                    return
                last_error = f"HTTP {response.status_code}"
            except requests.RequestException as exc:
                last_error = type(exc).__name__
            time.sleep(2)
        raise RuntimeError(f"OpenWebUI did not become ready at {self.base_url}: {last_error}")

    def authenticate(self, api_key: str, email: str, password: str) -> dict[str, Any]:
        """Authenticate with a valid API key first, then fall back to credentials."""
        if api_key:
            self.token = api_key
            response = self._request("GET", "/api/v1/auths/", expected=(200, 401, 403))
            if response.status_code == 200:
                return response.json()
            self.token = ""
        if not email or not password:
            raise RuntimeError("OpenWebUI credentials are required because OPENWEBUI_API_KEY is absent or invalid")
        response = self._request(
            "POST",
            "/api/v1/auths/signin",
            json={"email": email, "password": password},
        )
        payload = response.json()
        self.token = str(payload.get("token") or "")
        if not self.token:
            raise RuntimeError("OpenWebUI signin succeeded without returning a bearer token")
        return payload

    def ensure_api_key(self) -> str:
        """Return the current persistent API key, creating it when absent."""
        response = self._request("GET", "/api/v1/auths/api_key", expected=(200, 404))
        if response.status_code == 404:
            response = self._request("POST", "/api/v1/auths/api_key")
        api_key = str(response.json().get("api_key") or "")
        if not api_key:
            raise RuntimeError("OpenWebUI did not return an API key")
        return api_key

    def ensure_tool(self, tool_path: Path) -> str:
        """Create or synchronize the selected local Python tool by id."""
        content = tool_path.read_text(encoding="utf-8")
        tool_id, name, description = tool_identity(content, tool_path)
        response = self._request("GET", f"/api/v1/tools/id/{tool_id}", expected=(200, 404))
        if response.status_code == 404:
            self._request(
                "POST",
                "/api/v1/tools/create",
                json={
                    "id": tool_id,
                    "name": name,
                    "content": content,
                    "meta": {"description": description},
                    "access_grants": [],
                },
            )
            print(f"Created OpenWebUI tool {tool_id!r} from {tool_path}.")
        else:
            current = response.json()
            meta = dict(current.get("meta") or {})
            meta["description"] = description
            access_grants = current.get("access_grants") or []
            if (
                current.get("content") != content
                or current.get("name") != name
                or (current.get("meta") or {}).get("description") != description
            ):
                self._request(
                    "POST",
                    f"/api/v1/tools/id/{tool_id}/update",
                    json={
                        "id": tool_id,
                        "name": name,
                        "content": content,
                        "meta": meta,
                        "access_grants": access_grants,
                    },
                )
                print(f"Updated OpenWebUI tool {tool_id!r} from {tool_path}.")
            else:
                print(f"OpenWebUI tool {tool_id!r} already matches {tool_path}.")
        return tool_id

    def update_tool_valves(self, tool_id: str, overrides: dict[str, Any]) -> dict[str, Any]:
        """Merge validated admin-valve overrides into an installed tool."""
        response = self._request("GET", f"/api/v1/tools/id/{tool_id}/valves", expected=(200, 404))
        current = response.json() if response.status_code == 200 else {}
        if not isinstance(current, dict):
            raise RuntimeError(f"OpenWebUI returned invalid valves for tool {tool_id!r}")
        merged = {**current, **overrides}
        return self.set_tool_valves(tool_id, merged)

    def set_tool_valves(self, tool_id: str, valves: dict[str, Any]) -> dict[str, Any]:
        """Replace a tool's admin Valves with an exact validated configuration."""
        updated = self._request(
            "POST",
            f"/api/v1/tools/id/{tool_id}/valves/update",
            json=valves,
        ).json()
        if not isinstance(updated, dict):
            raise RuntimeError(f"OpenWebUI returned invalid updated valves for tool {tool_id!r}")
        return updated

    def ensure_model(self, model_id: str, base_model_id: str, tool_id: str) -> None:
        """Create or minimally update a workspace model with the requested tool and reasoning defaults."""
        system_prompt = (
            f"You have access to the OpenWebUI tool '{tool_id}'. When a user request matches this tool's "
            "purpose, you MUST call its relevant function exactly once before answering. Pass attached files "
            "to the tool, then base the final answer on the tool result. Do not claim to have used the tool "
            "unless a tool result is present."
        )
        response = self._request("GET", "/api/v1/models/model", params={"id": model_id}, expected=(200, 404))
        if response.status_code == 404:
            payload: dict[str, Any] = {
                "id": model_id,
                "base_model_id": base_model_id,
                "name": f"{base_model_id} + {tool_id}",
                "meta": {
                    "description": "Managed evaluation model with an attached OpenWebUI tool.",
                    "toolIds": [tool_id],
                },
                "params": {"reasoning": dict(DEFAULT_REASONING), "system": system_prompt},
                "access_grants": [],
                "is_active": True,
            }
            self._request("POST", "/api/v1/models/create", json=payload)
            self._request("GET", "/api/models")
            print(f"Created OpenWebUI model {model_id!r} over {base_model_id!r}.")
            return
        current = response.json()
        meta = dict(current.get("meta") or {})
        tool_ids = list(meta.get("toolIds") or [])
        changed = False
        if tool_id not in tool_ids:
            tool_ids.append(tool_id)
            meta["toolIds"] = tool_ids
            changed = True
        params = dict(current.get("params") or {})
        if params.get("reasoning") != DEFAULT_REASONING:
            params["reasoning"] = dict(DEFAULT_REASONING)
            changed = True
        if params.get("system") != system_prompt:
            params["system"] = system_prompt
            changed = True
        if changed:
            self._request(
                "POST",
                "/api/v1/models/model/update",
                json={
                    "id": model_id,
                    "base_model_id": current.get("base_model_id") or base_model_id,
                    "name": current.get("name") or model_id,
                    "meta": meta,
                    "params": params,
                    "access_grants": current.get("access_grants") or [],
                    "is_active": current.get("is_active", True),
                },
            )
            self._request("GET", "/api/models")
            print(f"Updated OpenWebUI model {model_id!r} with tool/reasoning defaults.")
        else:
            print(f"OpenWebUI model {model_id!r} already has the requested tool and parameters.")

    def test_inference(self, model_id: str) -> dict[str, Any]:
        """Run a bounded non-streaming inference and require the expected marker."""
        marker = f"OWI_INFERENCE_OK_{uuid.uuid4().hex[:8]}"
        response = self._request(
            "POST",
            "/api/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": f"Reply with exactly {marker}"}],
                "stream": False,
                "max_tokens": 32,
                "reasoning": dict(DEFAULT_REASONING),
            },
        )
        payload = response.json()
        content = str((((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""))
        if marker not in content:
            raise RuntimeError(f"OpenWebUI inference did not return the expected marker; got {content[:160]!r}")
        return payload

    def _upload_text(self, name: str, content: str) -> str:
        """Upload and synchronously process one small Markdown smoke fixture."""
        response = self._request(
            "POST",
            "/api/v1/files/?process=true&process_in_background=false",
            files={"file": (name, content.encode("utf-8"), "text/markdown")},
        )
        file_id = str(response.json().get("id") or "")
        if not file_id:
            raise RuntimeError(f"OpenWebUI file upload for {name} returned no id")
        return file_id

    def test_tool_in_thread(self, model_id: str, tool_id: str) -> dict[str, Any]:
        """Invoke the configured tool with two attachments inside a persisted chat thread."""
        first_id = self._upload_text("audit-target.md", "# Budget\nThe approved budget is EUR 10 million.\n")
        second_id = self._upload_text("audit-reference.md", "# Budget\nThe approved budget is EUR 12 million.\n")
        valves_response = self._request("GET", f"/api/v1/tools/id/{tool_id}/valves", expected=(200, 404))
        original_valves = valves_response.json() if valves_response.status_code == 200 else {}
        smoke_valves = dict(original_valves)
        smoke_valves.update({"strategy": "1_git_diff", "post_analysis": "off", "diagnostic_status": True})
        self.set_tool_valves(tool_id, smoke_valves)
        try:
            chat = self._request(
                "POST",
                "/api/v1/chats/new",
                json={
                    "chat": {
                        "title": "OpenWebUI tool smoke test",
                        "models": [model_id],
                        "params": {},
                        "history": {"messages": {}, "currentId": None},
                        "messages": [],
                        "tags": ["promptfoo-smoke"],
                        "timestamp": int(time.time() * 1000),
                    },
                    "folder_id": None,
                },
            ).json()
            chat_id = str(chat.get("id") or "")
            if not chat_id:
                raise RuntimeError("OpenWebUI chat creation returned no thread id")
            user_message_id = str(uuid.uuid4())
            assistant_message_id = str(uuid.uuid4())
            files = [{"type": "file", "id": first_id}, {"type": "file", "id": second_id}]
            prompt = (
                "Call audit_documents_contradictions exactly once on both attached files. "
                "Do not answer before calling it. After the tool result, return that report and do not call it again."
            )
            self._request(
                "POST",
                "/api/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                    "chat_id": chat_id,
                    "id": assistant_message_id,
                    "user_message": {
                        "id": user_message_id,
                        "parentId": None,
                        "childrenIds": [assistant_message_id],
                        "role": "user",
                        "content": prompt,
                        "files": files,
                        "models": [model_id],
                        "timestamp": int(time.time()),
                    },
                    "tool_ids": [tool_id],
                    "files": files,
                    "reasoning": dict(DEFAULT_REASONING),
                    "max_tokens": 1200,
                },
            )
            stored_chat = self._request("GET", f"/api/v1/chats/{chat_id}").json()
            messages = (((stored_chat.get("chat") or {}).get("history") or {}).get("messages") or {})
            assistant_message = messages.get(assistant_message_id) or {}
            serialized_output = json.dumps(assistant_message.get("output") or [], ensure_ascii=False)
            function_was_called = (
                '"type": "function_call"' in serialized_output
                and '"name": "audit_documents_contradictions"' in serialized_output
            )
            report_was_returned = "Contradiction audit" in serialized_output or "audit-target.md" in serialized_output
            if not function_was_called or not report_was_returned:
                raise RuntimeError(
                    f"Tool smoke test did not return an audit report in chat {chat_id}; "
                    f"assistant={json.dumps(assistant_message, ensure_ascii=False)[:1200]}"
                )
            return {"smoke_chat_id": chat_id, "assistant_message": assistant_message}
        finally:
            self.set_tool_valves(tool_id, original_valves)


def load_configuration(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, str]]:
    """Load parent and optional managed-instance dotenv values."""
    parent = read_env_file(args.env_file)
    local = read_env_file(args.local_env_file)
    return parent, local


def configure_instance(args: argparse.Namespace) -> None:
    """Authenticate, provision the selected tool/model, and run requested checks."""
    parent, local = load_configuration(args)
    base_url = os.environ.get("OPENWEBUI_BASE_URL") or parent.get("OPENWEBUI_BASE_URL") or "http://127.0.0.1:8080"
    admin = OpenWebUIAdmin(base_url, timeout_seconds=args.timeout)
    admin.wait_until_ready(args.wait_timeout)
    identity = admin.authenticate(
        os.environ.get("OPENWEBUI_API_KEY") or parent.get("OPENWEBUI_API_KEY", ""),
        os.environ.get("OWI_ADMIN_EMAIL") or parent.get("OWI_ADMIN_EMAIL") or local.get("OWI_ADMIN_EMAIL", ""),
        os.environ.get("OWI_ADMIN_PASSWORD")
        or parent.get("OWI_ADMIN_PASSWORD")
        or local.get("OWI_ADMIN_PASSWORD", ""),
    )
    print(f"OpenWebUI authentication OK for {identity.get('email', '<unknown>')} ({identity.get('role', '<unknown>')}).")
    api_key = admin.ensure_api_key()
    updates: dict[str, str] = {"OPENWEBUI_API_KEY": api_key}

    raw_tool = str(args.tool_path or "").strip() or os.environ.get("OWI_DEFAULT_TOOL") or parent.get(
        "OWI_DEFAULT_TOOL", ""
    )
    tools_root_raw = os.environ.get("OPENWEBUI_TOOLS_ROOT") or parent.get("OPENWEBUI_TOOLS_ROOT", "")
    tools_root = Path(tools_root_raw).expanduser() if tools_root_raw else None
    tool_path = resolve_tool_path(raw_tool, tools_root, args.repo_root)
    tool_id = admin.ensure_tool(tool_path)
    if args.tool_valves_json:
        valve_overrides = json.loads(args.tool_valves_json)
        if not isinstance(valve_overrides, dict):
            raise ValueError("--tool-valves-json must decode to a JSON object")
        admin.update_tool_valves(tool_id, valve_overrides)
        print(f"Updated OpenWebUI tool {tool_id!r} valves: {', '.join(sorted(valve_overrides))}.")
    if not args.preserve_defaults:
        updates["OPENWEBUI_TOOL_IDS"] = tool_id

    base_model_id = (
        args.base_model_id
        or os.environ.get("OWI_BASE_MODEL_ID")
        or parent.get("OWI_BASE_MODEL_ID")
        or local.get("OWI_LLM_MODEL_ID", DEFAULT_BASE_MODEL_ID)
    )
    model_id = (
        args.model_id
        or os.environ.get("OWI_DEFAULT_MODEL_ID")
        or parent.get("OWI_DEFAULT_MODEL_ID")
        or DEFAULT_CUSTOM_MODEL_ID
    )
    admin.ensure_model(model_id, base_model_id, tool_id)
    if not args.preserve_defaults:
        updates["OWI_DEFAULT_MODEL_ID"] = model_id
        if not parent.get("OPENWEBUI_PIPE_MODEL"):
            updates["OPENWEBUI_PIPE_MODEL"] = model_id
    update_env_file(args.env_file, updates)
    print(f"Stored the OpenWebUI API key and resolved ids in {args.env_file} (secret redacted).")

    if args.test_inference:
        inference = admin.test_inference(model_id)
        usage = inference.get("usage") or {}
        print(f"OpenWebUI inference OK (usage={json.dumps(usage, ensure_ascii=False)}).")
    if args.test_tool:
        tool_result = admin.test_tool_in_thread(model_id, tool_id)
        print(f"OpenWebUI tool/thread test OK (chat_id={tool_result['smoke_chat_id']}).")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    bundle_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=bundle_root / ".env")
    parser.add_argument("--local-env-file", type=Path, default=bundle_root / "openwebui_docker" / ".env")
    parser.add_argument("--repo-root", type=Path, default=bundle_root.parents[1])
    parser.add_argument("--prepare-local-env", action="store_true")
    parser.add_argument("--configure", action="store_true")
    parser.add_argument("--test-inference", action="store_true")
    parser.add_argument("--test-tool", action="store_true")
    parser.add_argument("--tool-path", type=Path, help="Provision this tool instead of OWI_DEFAULT_TOOL.")
    parser.add_argument("--model-id", help="Custom OpenWebUI model id to create or update.")
    parser.add_argument("--base-model-id", help="Base provider model id for --model-id.")
    parser.add_argument("--tool-valves-json", default="", help="JSON object merged into the selected tool's admin valves.")
    parser.add_argument(
        "--preserve-defaults",
        action="store_true",
        help="Provision an additional tool/model without changing the default tool/model keys in .env.",
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--wait-timeout", type=int, default=300)
    return parser


def main() -> None:
    """Run the selected provisioning actions."""
    args = build_parser().parse_args()
    if args.timeout <= 0 or args.wait_timeout <= 0:
        raise ValueError("--timeout and --wait-timeout must be positive")
    if args.prepare_local_env:
        prepare_local_env(args.env_file, args.local_env_file)
    if args.configure:
        configure_instance(args)
    if not args.prepare_local_env and not args.configure:
        raise ValueError("Select --prepare-local-env and/or --configure")


if __name__ == "__main__":
    main()
