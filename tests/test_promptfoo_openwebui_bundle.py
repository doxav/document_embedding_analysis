import importlib.util
import sys
from pathlib import Path

import pytest
import requests


BUNDLE_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "promptfoo_openwebui_eval"
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))

from lib.bundle_common import build_openwebui_user_prompt, read_csv_rows, write_csv_rows
from lib.openwebui_client import OpenAIEndpointClient, OpenWebUIClient


def _load_generate_candidate_csv_module():
    """Load the generation CLI module without requiring scripts/ to be a package."""
    module_path = BUNDLE_ROOT / "scripts" / "generate_candidate_csv.py"
    spec = importlib.util.spec_from_file_location("generate_candidate_csv", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_openwebui_prompt_includes_tool_file_and_kb_blocks():
    prompt = build_openwebui_user_prompt(
        {
            "request_prompt": "Summarize the supplied documents.",
            "tool_parameters_json": '{"emit_diagnostics": false}',
            "algorithm": "kohaku",
            "target_length": "long",
            "structure": "sectioned",
            "kb_ids_json": '["kb-lit-review"]',
        },
        file_ids=["file-1"],
    )

    assert "Summarize the supplied documents." in prompt
    assert "<tool_parameters>" in prompt
    assert "<algorithm>kohaku</algorithm>" in prompt
    assert "<target_length>long</target_length>" in prompt
    assert "<structure>sectioned</structure>" in prompt
    assert "<emit_diagnostics>false</emit_diagnostics>" in prompt
    assert "<files_list>" in prompt
    assert '"file-1"' in prompt
    assert "<kb_list>" in prompt
    assert '"kb-lit-review"' in prompt


def test_csv_reader_accepts_large_embedded_documents(tmp_path: Path):
    csv_path = tmp_path / "large.csv"
    large_text = "x" * 140_000

    write_csv_rows(csv_path, [{"task_id": "large", "request_prompt": large_text}])

    rows = read_csv_rows(csv_path)
    assert rows[0]["task_id"] == "large"
    assert rows[0]["request_prompt"] == large_text


def test_openai_chat_completions_reuses_bridge_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from api import openwebui_bridge

    captured_payload: dict[str, object] = {}

    def fake_generate(payload: dict[str, object]) -> dict[str, object]:
        captured_payload.update(payload)
        return {"output": "final answer"}

    monkeypatch.setattr(openwebui_bridge, "generate", fake_generate)

    data = openwebui_bridge.openai_chat_completions(
        {
            "model": "qwen3527b",
            "messages": [{"role": "user", "content": "Find facts in the knowledge base."}],
            "temperature": 0.2,
            "max_tokens": 4096,
            "kb_ids_json": '["energy-france"]',
        }
    )

    assert data["object"] == "chat.completion"
    assert data["model"] == "qwen3527b"
    assert data["choices"][0]["message"]["content"] == "final answer"
    assert captured_payload["openwebui_pipe_model"] == "qwen3527b"
    assert captured_payload["request_prompt"] == "Find facts in the knowledge base."
    assert captured_payload["generation_temperature"] == "0.2"
    assert captured_payload["generation_max_tokens"] == "4096"
    assert captured_payload["kb_ids_json"] == '["energy-france"]'


def test_openai_chat_completions_rejects_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    fastapi = pytest.importorskip("fastapi")
    from api import openwebui_bridge

    def fake_generate(payload: dict[str, object]) -> dict[str, object]:
        raise AssertionError("streaming requests must not call generate")

    monkeypatch.setattr(openwebui_bridge, "generate", fake_generate)

    with pytest.raises(fastapi.HTTPException) as exc_info:
        openwebui_bridge.openai_chat_completions(
            {
                "model": "qwen3527b",
                "stream": True,
                "messages": [{"role": "user", "content": "Find facts in the knowledge base."}],
            }
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "stream=true is not supported by this bridge"


def test_openai_endpoint_backend_uses_generic_client(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from api import openwebui_bridge

    captured_payload: dict[str, object] = {}

    class FakeOpenAIEndpointClient:
        def chat(
            self,
            *,
            model: str,
            user_prompt: str,
            extra_payload: dict[str, object] | None = None,
        ) -> dict[str, object]:
            captured_payload.update({"model": model, "user_prompt": user_prompt, "extra_payload": extra_payload})
            return {"output": "endpoint answer"}

    monkeypatch.setenv("GENERATION_BACKEND", "openai_endpoint")
    monkeypatch.setattr(openwebui_bridge, "OpenAIEndpointClient", FakeOpenAIEndpointClient)

    result = openwebui_bridge.generate(
        {
            "openwebui_pipe_model": "qwen3527b",
            "request_prompt": "Find facts in the knowledge base.",
            "generation_temperature": "0.2",
        }
    )

    assert result["backend"] == "openai_endpoint"
    assert result["output"] == "endpoint answer"
    assert captured_payload["model"] == "qwen3527b"
    assert captured_payload["user_prompt"] == "Find facts in the knowledge base."
    assert captured_payload["extra_payload"] == {"temperature": 0.2}


def test_clients_use_configured_request_timeouts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENWEBUI_TIMEOUT_SECONDS", "1800")
    monkeypatch.setenv("OPENAI_ENDPOINT_TIMEOUT_SECONDS", "1200")

    openwebui_client = OpenWebUIClient(
        base_url="http://openwebui.test",
        api_key="token",
        cache_path=tmp_path / "openwebui-file-cache.json",
    )
    endpoint_client = OpenAIEndpointClient(base_url="http://endpoint.test/v1", api_key="token")

    assert openwebui_client.timeout == 1800
    assert endpoint_client.timeout == 1200


def test_openwebui_client_completes_tool_call_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenWebUIClient(base_url="http://openwebui.test", api_key="token")
    posted_payloads: list[dict[str, object]] = []

    def fake_post_chat_completion(payload: dict[str, object]) -> dict[str, object]:
        posted_payloads.append(payload)
        if len(posted_payloads) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "query_knowledge_files", "arguments": '{"query":"energy"}'},
                                }
                            ],
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"role": "assistant", "content": "final facts"}}]}

    monkeypatch.setattr(client, "_post_chat_completion", fake_post_chat_completion)
    monkeypatch.setattr(client, "_execute_tool_call", lambda model, tool_call: '{"results":[{"content":"fact"}]}')

    result = client.chat(model="qwen3527b", user_prompt="Find energy facts.")

    assert result["output"] == "final facts"
    second_messages = posted_payloads[1]["messages"]
    assert second_messages[1]["tool_calls"][0]["function"]["name"] == "query_knowledge_files"
    assert second_messages[2] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": '{"results":[{"content":"fact"}]}',
    }


def test_generate_csv_http_errors_include_response_body() -> None:
    module = _load_generate_candidate_csv_module()
    response = requests.Response()
    response.status_code = 500
    response.url = "http://bridge.test/generate"
    response._content = b'{"detail":"OpenWebUI timed out"}'

    with pytest.raises(requests.HTTPError) as exc_info:
        module._raise_for_status(response)

    assert "500 Server Error" in str(exc_info.value)
    assert "OpenWebUI timed out" in str(exc_info.value)
