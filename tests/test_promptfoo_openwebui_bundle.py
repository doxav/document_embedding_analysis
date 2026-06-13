import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _load_dea_metrics_module():
    """Load the DEA assertion module without requiring assertions/ to be a package."""
    module_path = BUNDLE_ROOT / "assertions" / "dea_metrics.py"
    spec = importlib.util.spec_from_file_location("dea_metrics", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_trace_module():
    """Load the trace optimization skeleton when its optional dependencies exist."""
    pytest.importorskip("opto")
    module_path = BUNDLE_ROOT / "trace_openwebui_dea_skeleton.py"
    spec = importlib.util.spec_from_file_location("trace_openwebui_dea_skeleton", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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


def test_openai_chat_completions_forwards_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from api import openwebui_bridge

    captured_payload: dict[str, object] = {}

    def fake_generate(payload: dict[str, object]) -> dict[str, object]:
        captured_payload.update(payload)
        return {"output": "final answer"}

    monkeypatch.setattr(openwebui_bridge, "generate", fake_generate)

    openwebui_bridge.openai_chat_completions(
        {
            "model": "qwen3527b",
            "messages": [
                {"role": "system", "content": "Answer with a strict marker."},
                {"role": "user", "content": "Run the proof."},
            ],
        }
    )

    assert captured_payload["openwebui_system_prompt"] == "Answer with a strict marker."
    assert captured_payload["request_prompt"] == "Run the proof."


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


def test_openwebui_backend_passes_system_prompt_and_trace_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from api import openwebui_bridge

    captured_payload: dict[str, object] = {}

    class FakeOpenWebUIClient:
        def ensure_file_ids(self, file_paths: list[Path]) -> list[str]:
            return []

        def chat(
            self,
            *,
            model: str,
            user_prompt: str,
            system_prompt: str | None = None,
            files_payload: list[dict[str, str]] | None = None,
            extra_payload: dict[str, object] | None = None,
            trigger_outlet: bool = False,
            include_trace: bool = False,
        ) -> dict[str, object]:
            captured_payload.update(
                {
                    "model": model,
                    "user_prompt": user_prompt,
                    "system_prompt": system_prompt,
                    "extra_payload": extra_payload,
                    "include_trace": include_trace,
                }
            )
            return {"output": "openwebui answer", "trace": {"rounds": []}}

    monkeypatch.setenv("GENERATION_BACKEND", "openwebui")
    monkeypatch.setattr(openwebui_bridge, "OpenWebUIClient", FakeOpenWebUIClient)

    result = openwebui_bridge.generate(
        {
            "openwebui_pipe_model": "qwen3527b",
            "openwebui_system_prompt": "Answer with a strict marker.",
            "openwebui_include_trace": "true",
            "request_prompt": "Run the proof.",
            "generation_temperature": "0.2",
        }
    )

    assert result["backend"] == "openwebui"
    assert result["trace"] == {"rounds": []}
    assert captured_payload["model"] == "qwen3527b"
    assert captured_payload["system_prompt"] == "Answer with a strict marker."
    assert captured_payload["include_trace"] is True
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
    executed_tool_calls: list[dict[str, object]] = []

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

    def fake_execute_tool_call(model: str, tool_call: dict[str, object]) -> str:
        executed_tool_calls.append(tool_call)
        return '{"results":[{"content":"fact"}]}'

    monkeypatch.setattr(client, "_post_chat_completion", fake_post_chat_completion)
    monkeypatch.setattr(client, "_execute_tool_call", fake_execute_tool_call)

    result = client.chat(model="qwen3527b", user_prompt="Find energy facts.", include_trace=True)

    assert result["output"] == "final facts"
    assert len(executed_tool_calls) == 1
    second_messages = posted_payloads[1]["messages"]
    assert second_messages[1]["tool_calls"][0]["function"]["name"] == "query_knowledge_files"
    assert second_messages[2] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": '{"results":[{"content":"fact"}]}',
    }
    assert result["trace"]["tool_calls"][0]["name"] == "query_knowledge_files"
    assert result["trace"]["message_count"] == 3


def test_openwebui_client_sends_system_prompt_and_traces_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenWebUIClient(base_url="http://openwebui.test", api_key="token")
    posted_payloads: list[dict[str, object]] = []

    def fake_post_chat_completion(payload: dict[str, object]) -> dict[str, object]:
        posted_payloads.append(payload)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "SYSTEM_PROOF",
                        "reasoning_content": "short private reasoning trace",
                    }
                }
            ]
        }

    monkeypatch.setattr(client, "_post_chat_completion", fake_post_chat_completion)

    result = client.chat(
        model="qwen3527b",
        system_prompt="Always answer SYSTEM_PROOF.",
        user_prompt="What is 2+2?",
        include_trace=True,
    )

    assert posted_payloads[0]["messages"][0] == {"role": "system", "content": "Always answer SYSTEM_PROOF."}
    assert posted_payloads[0]["messages"][1] == {"role": "user", "content": "What is 2+2?"}
    assert result["output"] == "SYSTEM_PROOF"
    assert result["trace"]["rounds"][0]["message"]["reasoning_content"] == "short private reasoning trace"


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


def test_dea_metrics_resolves_solution_local_target_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_dea_metrics_module()
    solution_dir = tmp_path / "item"
    solution_dir.mkdir()
    target = solution_dir / "full_text.md"
    target.write_text("# Target", encoding="utf-8")
    solution = {
        "title": "Demo",
        "context": "Context",
        "target_file_path": "full_text.md",
        "plan": [],
        "resources": [],
    }
    solution_path = solution_dir / "dea_solution.json"
    solution_path.write_text(json.dumps(solution), encoding="utf-8")
    monkeypatch.setenv("DEA_REPO_ROOT", str(tmp_path))

    loaded = module._load_solution({"vars": {"dea_solution_path": str(solution_path)}})

    assert loaded["target_file_path"] == str(target.resolve())


def test_dea_metrics_length_alignment_uses_ratio_closeness() -> None:
    module = _load_dea_metrics_module()

    assert module._ratio_alignment(1.0) == 1.0
    assert module._ratio_alignment(0.25) == 0.25
    assert module._ratio_alignment(1.5) == 0.5


def test_trace_metrics_expose_duration_and_call_objectives() -> None:
    module = _load_trace_module()

    metrics = module.trace_metrics(
        [
            {
                "latency_s": 3.0,
                "raw_response": {
                    "trace": {
                        "rounds": [{"message": {"reasoning_content": "abc"}}, {"message": {}}],
                        "tool_calls": [{"name": "search"}],
                    }
                },
            },
            {
                "latency_s": 1.0,
                "raw_response": {
                    "trace": {
                        "rounds": [{"message": {"thinking": "xy"}}],
                        "tool_calls": [],
                    }
                },
            },
        ]
    )

    assert metrics["execution_duration_s"] == pytest.approx(2.0)
    assert metrics["duration_s"] == pytest.approx(2.0)
    assert metrics["latency_s"] == pytest.approx(2.0)
    assert metrics["execution_duration_score"] == pytest.approx(1 / 3)
    assert metrics["llm_calls"] == pytest.approx(1.5)
    assert metrics["llm_rounds"] == pytest.approx(1.5)
    assert metrics["tool_calls"] == pytest.approx(0.5)
    assert metrics["reasoning_chars"] == pytest.approx(2.5)


def test_trace_objective_config_keeps_quality_and_duration() -> None:
    module = _load_trace_module()
    args = SimpleNamespace(
        objective_mode="weighted",
        weights='{"length_alignment": 0.6}',
        minimize=[],
    )

    config = module.build_objective_config(args)

    assert config.weights["length_alignment"] == 0.6
    assert "score" not in config.weights
    assert config.weights["execution_duration_score"] > 0
    assert "execution_duration_s" in config.minimize
    assert "llm_calls" in config.minimize
    assert config.scalarize_dict == "weighted"


def test_trace_default_objective_uses_single_global_quality_score() -> None:
    module = _load_trace_module()
    args = SimpleNamespace(
        objective_mode="weighted",
        weights=json.dumps(module.DEFAULT_OBJECTIVE_WEIGHTS),
        minimize=[],
    )

    config = module.build_objective_config(args)

    assert config.weights["score"] == pytest.approx(1.0)
    assert "dea_composite" not in config.weights
    assert config.weights["execution_duration_score"] > 0
    assert config.weights["llm_calls_score"] > 0
    assert config.weights["tool_calls_score"] > 0


def test_trace_direct_dea_exposes_length_alignment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_trace_module()
    import common.doc_eval as doc_eval

    def fake_evaluate_document(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "dea_evaluation_status": {"status": "computed"},
            "dea_evaluation_scores": {
                "plan_total_similarity": 0.8,
                "content_total_similarity": 0.7,
                "resources_citation_coverage_score": 0.5,
                "content_length_ratio_to_target": 1.25,
            },
            "article_metrics": {
                "rouge_scores": {"rouge-l": {"f": 0.4}},
                "entity_recall": 0.3,
                "citation_count": 2,
            },
        }

    solution_path = tmp_path / "dea_solution.json"
    solution_path.write_text(json.dumps({"title": "Synthetic DEA"}), encoding="utf-8")
    monkeypatch.setenv("DEA_REPO_ROOT", str(Path.cwd()))
    monkeypatch.setattr(doc_eval, "evaluate_document", fake_evaluate_document)

    result = module.evaluate_candidate_with_dea({"dea_solution_path": str(solution_path)}, "candidate")
    named_scores = result["namedScores"]

    assert result["score"] == pytest.approx(0.67)
    assert named_scores["dea_composite"] == pytest.approx(0.67)
    assert named_scores["length_alignment"] == pytest.approx(0.75)
    assert "length=0.750" in result["reason"]


def test_trace_direct_dea_judge_forwards_judge_flag_and_trace_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_trace_module()
    judge_flags: list[bool] = []

    def fake_evaluate_candidate_with_dea(
        row: dict[str, object],
        output: str,
        *,
        use_dea_judge: bool = False,
    ) -> dict[str, object]:
        judge_flags.append(use_dea_judge)
        return {
            "score": 0.75,
            "reason": f"{row['task_id']} scored",
            "namedScores": {"dea_composite": 0.75, "plan": 0.8, "content": 0.7},
        }

    monkeypatch.setattr(module, "evaluate_candidate_with_dea", fake_evaluate_candidate_with_dea)
    outputs = [
        {
            "row": {"task_id": f"dea-row-{index}"},
            "candidate_answer": "Candidate.",
            "latency_s": 2.0,
            "raw_response": {
                "trace": {
                    "rounds": [{"message": {"reasoning_content": "abc"}}],
                    "tool_calls": [{"name": "query"}],
                }
            },
        }
        for index in range(1, 4)
    ]

    scores, feedback = module.aggregate_dea_batch(outputs, use_dea_judge=True)

    assert judge_flags == [True, True, True]
    assert scores["score"] == pytest.approx(0.75)
    assert scores["dea_composite"] == pytest.approx(0.75)
    assert scores["llm_calls"] == pytest.approx(1.0)
    assert scores["tool_calls"] == pytest.approx(1.0)
    assert "dea-row-1" in feedback


def test_trace_tool_params_are_forwarded_to_bridge_row() -> None:
    module = _load_trace_module()

    row, warnings = module.merge_tool_params_into_row(
        {
            "task_id": "tool-forwarding",
            "request_prompt": "Summarize the files.",
            "tool_parameters_json": '{"emit_diagnostics": false}',
        },
        {
            "algorithm": "raptor",
            "target_length": "short",
            "summary_structure": "sectioned",
            "summarizer_model_id": "summarizer---raptor",
            "instruction": "Use numbered evidence bullets.",
            "max_final_summary_chars": 1800,
        },
        target_model_id="summarizer---raptor",
    )

    tool_parameters = json.loads(row["tool_parameters_json"])
    assert warnings == []
    assert row["openwebui_pipe_model"] == "summarizer---raptor"
    assert row["algorithm"] == "raptor"
    assert row["target_length"] == "short"
    assert row["structure"] == "sectioned"
    assert row["summarizer_model_id"] == "summarizer---raptor"
    assert tool_parameters["emit_diagnostics"] is False
    assert tool_parameters["instruction"] == "Use numbered evidence bullets."
    assert tool_parameters["max_final_summary_chars"] == 1800

    prompt = build_openwebui_user_prompt(row, file_ids=["file-proof"])
    assert "<algorithm>raptor</algorithm>" in prompt
    assert "<target_length>short</target_length>" in prompt
    assert "<structure>sectioned</structure>" in prompt
    assert "<summarizer_model_id>summarizer---raptor</summarizer_model_id>" in prompt
    assert "<instruction>Use numbered evidence bullets.</instruction>" in prompt
    assert "<max_final_summary_chars>1800</max_final_summary_chars>" in prompt
    assert '"file-proof"' in prompt


def test_trace_tool_model_ids_cover_algorithm_specific_pipes() -> None:
    module = _load_trace_module()

    assert module.TOOL_MODEL_IDS_BY_ALGORITHM == {
        "dtcrs": "summarizer---dtcrs",
        "kohaku": "summarizer---kohaku",
        "kohaku_openrouter": "summarizer---kohaku-OR",
        "lightrag": "summarizer---lightrag",
        "raptor": "summarizer---raptor",
    }


def test_trace_method_mapping_and_batch_dataset() -> None:
    module = _load_trace_module()
    args = SimpleNamespace(
        optimization_method="step3_promptfoo",
        eval_backend="dea",
        use_dea_judge=False,
        generation_mode="bridge",
    )

    module.apply_optimization_method(args)
    dataset = module.build_training_dataset([{"task_id": str(i)} for i in range(7)], batch_size=3)

    assert args.eval_backend == "promptfoo_step3"
    assert args.generation_mode == "prepared_rows"
    assert [len(batch) for batch in dataset["inputs"]] == [3, 3, 1]
    assert dataset["inputs"] == dataset["infos"]


@pytest.mark.parametrize(
    ("method", "eval_backend", "use_dea_judge", "generation_mode"),
    [
        ("direct_dea", "dea", False, "bridge"),
        ("direct_dea_judge", "dea", True, "bridge"),
        ("direct_llm_judge", "llm_judge", False, "bridge"),
        ("step2_promptfoo", "promptfoo", False, "bridge"),
        ("step3_promptfoo", "promptfoo_step3", False, "prepared_rows"),
    ],
)
def test_trace_optimization_method_mapping(
    method: str,
    eval_backend: str,
    use_dea_judge: bool,
    generation_mode: str,
) -> None:
    module = _load_trace_module()
    args = SimpleNamespace(
        optimization_method=method,
        eval_backend="dea",
        use_dea_judge=False,
        generation_mode="bridge",
    )

    module.apply_optimization_method(args)

    assert args.eval_backend == eval_backend
    assert args.use_dea_judge is use_dea_judge
    assert args.generation_mode == generation_mode


def test_trace_noop_optimizer_adapter() -> None:
    module = _load_trace_module()
    optimizer = module.build_optimizer(SimpleNamespace(optimizer_mode="noop", optimizer_max_tokens=64), agent=object())

    optimizer.zero_feedback()
    optimizer.backward(target=object(), feedback="feedback")

    assert optimizer.step() == {}


def test_trace_promptfoo_step3_live_uses_prepared_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_trace_module()

    def fake_run_shell(command: str, *, cwd: Path | None = None, env: dict[str, str] | None = None) -> SimpleNamespace:
        assert "step3_live_rows.csv" in command
        (tmp_path / "promptfoo_result.json").write_text(json.dumps({"score": 0.72}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module, "run_shell", fake_run_shell)
    outputs = module.prepare_batch_rows(
        [{"task_id": "live-1", "request_prompt": "Summarize."}],
        row_transformer=lambda row: dict(row, openwebui_pipe_model="summarizer---kohaku"),
    )

    scores, feedback, artifacts = module.evaluate_batch_with_promptfoo_live(
        outputs,
        promptfoo_config=tmp_path / "dummy.step3.dea.yaml",
        promptfoo_eval_cmd="promptfoo eval -c {config} -t {csv} --output {result_json}",
        working_dir=tmp_path,
        artifact_dir=tmp_path,
    )

    assert scores["score"] == pytest.approx(0.72)
    assert scores["promptfoo_score"] == pytest.approx(0.72)
    assert scores["execution_duration_s"] >= 0
    assert Path(artifacts["csv"]).name == "step3_live_rows.csv"
    assert "Promptfoo summary score parsed" in feedback


def test_trace_batch_trainer_runs_step3_batch_and_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_trace_module()

    def fake_run_shell(command: str, *, cwd: Path | None = None, env: dict[str, str] | None = None) -> SimpleNamespace:
        assert "step3_live_rows.csv" in command
        (tmp_path / "promptfoo_result.json").write_text(
            json.dumps(
                {
                    "results": [
                        {"score": 0.8, "namedScores": {"promptfoo_score": 0.8}, "description": "row-1"},
                        {"score": 0.7, "namedScores": {"promptfoo_score": 0.7}, "description": "row-2"},
                        {"score": 0.9, "namedScores": {"promptfoo_score": 0.9}, "description": "row-3"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module, "run_shell", fake_run_shell)
    rows = [
        {"task_id": f"row-{index}", "request_prompt": "Summarize.", "gold_summary": "Reference."}
        for index in range(1, 4)
    ]
    agent = module.OpenWebUIModel(
        initial_system_prompt="Answer tersely.",
        initial_param_json=json.dumps({"generation_temperature": 0, "generation_max_tokens": 64}),
        bridge_url="http://bridge.invalid/generate",
        target_model_id="qwen-laptop:latest",
        generation_mode="prepared_rows",
        prompt_mode="bridge_field",
    )
    objective = module.build_objective_config(
        SimpleNamespace(objective_mode="weighted", weights=json.dumps(module.DEFAULT_OBJECTIVE_WEIGHTS), minimize=[])
    )
    guide = module.BatchEvaluationGuide(
        mode="promptfoo_step3",
        promptfoo_config=str(tmp_path / "dummy.step3.dea.yaml"),
        promptfoo_eval_cmd="promptfoo eval -c {config} -t {csv} --output {result_json}",
        promptfoo_workdir=str(tmp_path),
        artifact_dir=str(tmp_path),
    )
    trainer = module.TraceBatchTrainer(
        agent,
        module.NoOpOptimizer(),
        objective_config=objective,
        artifact_dir=tmp_path,
        optimize_target="model",
        eval_backend="promptfoo_step3",
        optimization_method="step3_promptfoo",
        optimizer_mode="noop",
    )

    payload = trainer.train(guide, module.build_training_dataset(rows, batch_size=3), num_epochs=1)

    assert len(payload["history"]) == 1
    snapshot = payload["history"][0]
    assert snapshot["batch_size"] == 3
    assert snapshot["batch_task_ids"] == ["row-1", "row-2", "row-3"]
    assert snapshot["optimizer_mode"] == "noop"
    assert snapshot["mean_scores"]["promptfoo_score"] == pytest.approx(0.8)
    assert snapshot["mean_scores"]["llm_calls_score"] == pytest.approx(1.0)
    assert snapshot["mean_scores"]["tool_calls_score"] == pytest.approx(1.0)
    assert snapshot["mean_scores"]["execution_duration_s"] >= 0
    assert snapshot["scalar_objective"] > 0
    assert json.loads(snapshot["current_state"]["param_json"])["generation_max_tokens"] == 64
    assert json.loads(snapshot["current_state"]["model_param_json"])["generation_max_tokens"] == 64
    assert snapshot["current_state"]["tool_param_json"] == ""
    assert payload["best_weighted"] == snapshot
    assert (tmp_path / "step_0000_epoch_000_batch_000.candidates.csv").exists()
    trace_path = tmp_path / "step_0000_epoch_000_batch_000.traces.json"
    summary_path = tmp_path / "step_0000_epoch_000_batch_000.summary.json"
    assert summary_path.exists()
    traces = json.loads(trace_path.read_text(encoding="utf-8"))
    assert len(traces) == 3
    assert traces[0]["trace"]["mode"] == "prepared_rows"


def test_trace_batch_trainer_runs_tool_target_and_forwards_params(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_trace_module()

    def fake_run_shell(command: str, *, cwd: Path | None = None, env: dict[str, str] | None = None) -> SimpleNamespace:
        assert "step3_live_rows.csv" in command
        (tmp_path / "promptfoo_result.json").write_text(json.dumps({"score": 0.76}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module, "run_shell", fake_run_shell)
    tool_params = {
        "algorithm": "raptor",
        "target_length": "short",
        "structure": "sectioned",
        "instruction": "Use source-grounded bullets.",
        "max_final_summary_chars": 1800,
    }
    agent = module.OpenWebUISummarizerAgent(
        initial_param_json=json.dumps(tool_params),
        bridge_url="http://bridge.invalid/generate",
        target_model_id="summarizer---raptor",
        generation_mode="prepared_rows",
    )
    objective = module.build_objective_config(
        SimpleNamespace(objective_mode="weighted", weights='{"promptfoo_score": 1}', minimize=[])
    )
    guide = module.BatchEvaluationGuide(
        mode="promptfoo_step3",
        promptfoo_config=str(tmp_path / "dummy.step3.dea.yaml"),
        promptfoo_eval_cmd="promptfoo eval -c {config} -t {csv} --output {result_json}",
        promptfoo_workdir=str(tmp_path),
        artifact_dir=str(tmp_path),
    )
    trainer = module.TraceBatchTrainer(
        agent,
        module.NoOpOptimizer(),
        objective_config=objective,
        artifact_dir=tmp_path,
        optimize_target="tool",
        eval_backend="promptfoo_step3",
        optimization_method="step3_promptfoo",
        optimizer_mode="noop",
    )

    rows = [{"task_id": f"tool-row-{index}", "request_prompt": "Summarize."} for index in range(1, 4)]
    payload = trainer.train(guide, module.build_training_dataset(rows, batch_size=3), num_epochs=1)

    snapshot = payload["history"][0]
    assert snapshot["optimize_target"] == "tool"
    assert snapshot["mean_scores"]["promptfoo_score"] == pytest.approx(0.76)
    assert json.loads(snapshot["current_state"]["param_json"])["algorithm"] == "raptor"
    assert json.loads(snapshot["current_state"]["tool_param_json"])["algorithm"] == "raptor"
    assert snapshot["current_state"]["model_param_json"] == ""
    candidates = read_csv_rows(tmp_path / "step_0000_epoch_000_batch_000.candidates.csv")
    assert len(candidates) == 3
    assert {row["openwebui_pipe_model"] for row in candidates} == {"summarizer---raptor"}
    assert {row["algorithm"] for row in candidates} == {"raptor"}
    assert {row["target_length"] for row in candidates} == {"short"}
    forwarded = json.loads(candidates[0]["tool_parameters_json"])
    assert forwarded["instruction"] == "Use source-grounded bullets."
    assert forwarded["max_final_summary_chars"] == 1800


def test_trace_batch_trainer_runs_step2_bridge_then_offline_promptfoo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_trace_module()
    bridge_rows: list[dict[str, object]] = []

    def fake_bridge_generate(
        row: dict[str, object],
        bridge_url: str,
        timeout_seconds: int = 1800,
    ) -> tuple[str, float, dict[str, object]]:
        bridge_rows.append(dict(row))
        return (
            f"generated answer for {row['task_id']}",
            1.25,
            {
                "trace": {
                    "rounds": [{"message": {"reasoning_content": "r"}}],
                    "tool_calls": [{"name": "large_thematic_summarizer"}],
                }
            },
        )

    def fake_run_shell(command: str, *, cwd: Path | None = None, env: dict[str, str] | None = None) -> SimpleNamespace:
        assert "candidate_rows.csv" in command
        (tmp_path / "promptfoo_result.json").write_text(
            json.dumps(
                {
                    "results": [
                        {"score": 0.6, "namedScores": {"promptfoo_score": 0.6}, "description": "row-1"},
                        {"score": 0.8, "namedScores": {"promptfoo_score": 0.8}, "description": "row-2"},
                        {"score": 1.0, "namedScores": {"promptfoo_score": 1.0}, "description": "row-3"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module, "bridge_generate", fake_bridge_generate)
    monkeypatch.setattr(module, "run_shell", fake_run_shell)
    agent = module.OpenWebUISummarizerAgent(
        initial_param_json=json.dumps({"algorithm": "kohaku", "target_length": "long", "structure": "thematic"}),
        bridge_url="http://bridge.invalid/generate",
        target_model_id="summarizer---kohaku",
        generation_mode="bridge",
        bridge_timeout_seconds=10,
    )
    objective = module.build_objective_config(
        SimpleNamespace(objective_mode="weighted", weights='{"promptfoo_score": 1}', minimize=[])
    )
    guide = module.BatchEvaluationGuide(
        mode="promptfoo",
        promptfoo_config=str(tmp_path / "dummy.step2.dea.yaml"),
        promptfoo_eval_cmd="promptfoo eval -c {config} -t {csv} --output {result_json}",
        promptfoo_workdir=str(tmp_path),
        artifact_dir=str(tmp_path),
    )
    trainer = module.TraceBatchTrainer(
        agent,
        module.NoOpOptimizer(),
        objective_config=objective,
        artifact_dir=tmp_path,
        optimize_target="tool",
        eval_backend="promptfoo",
        optimization_method="step2_promptfoo",
        optimizer_mode="noop",
    )

    rows = [{"task_id": f"step2-row-{index}", "request_prompt": "Summarize."} for index in range(1, 4)]
    payload = trainer.train(guide, module.build_training_dataset(rows, batch_size=3), num_epochs=1)

    assert len(bridge_rows) == 3
    assert {row["openwebui_pipe_model"] for row in bridge_rows} == {"summarizer---kohaku"}
    assert {row["algorithm"] for row in bridge_rows} == {"kohaku"}
    snapshot = payload["history"][0]
    assert snapshot["optimization_method"] == "step2_promptfoo"
    assert snapshot["mean_scores"]["promptfoo_score"] == pytest.approx(0.8)
    assert snapshot["mean_scores"]["llm_calls"] == pytest.approx(1.0)
    assert snapshot["mean_scores"]["tool_calls"] == pytest.approx(1.0)
    assert Path(snapshot["evaluation_artifacts"]["csv"]).name == "step2_candidate_rows.csv"
    offline_rows = read_csv_rows(tmp_path / "step2_candidate_rows.csv")
    assert [row["candidate_answer"] for row in offline_rows] == [
        "generated answer for step2-row-1",
        "generated answer for step2-row-2",
        "generated answer for step2-row-3",
    ]


def test_trace_default_llm_judge_prompt_formats_literal_json_schema() -> None:
    module = _load_trace_module()

    prompt = module.build_llm_judge_prompt(
        {"request_prompt": "What is 2+2?", "gold_summary": "4"},
        "The answer is 4.",
        module.DEFAULT_LLM_JUDGE_PROMPT,
    )

    assert '{"score": 0.0' in prompt
    assert '"metrics": {"correctness": 0.0' in prompt
    assert "What is 2+2?" in prompt
    assert "The answer is 4." in prompt
