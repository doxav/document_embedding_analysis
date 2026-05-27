import json
import time

import pytest

from common.dea_judge import (
    _build_judge_prompt,
    _extract_bibliography_context,
    _extract_candidate_context,
    _extract_gold_context,
    _select_weak_section_extracts,
    _format_score_context,
    _parse_judge_response,
    run_dea_judge,
)
from common.doc_eval import evaluate_document


class FakeJudgeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    class _Message:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = FakeJudgeClient._Message(content)

    class _Response:
        def __init__(self, content):
            self.choices = [FakeJudgeClient._Choice(content)]

    @property
    def chat(self):
        outer = self

        class Chat:
            @property
            def completions(self):
                class Completions:
                    def create(self, **kwargs):
                        outer.calls.append(kwargs)
                        return FakeJudgeClient._Response(outer.response)
                return Completions()
        return Chat()


def test_interpret_scores_basic():
    scores = {
        "plan_contents_embedding_similarity": 0.42,
        "plan_titles_embedding_similarity": 0.86,
        "content_length_ratio_to_target": 0.35,
    }

    text = _format_score_context(scores, {}, {}, {})

    assert "plan_contents_embedding_similarity = 0.42" in text
    assert "weak" in text
    assert "plan_titles_embedding_similarity = 0.86" in text
    assert "good" in text
    assert "content_length_ratio_to_target = 0.35" in text


def test_extract_gold_plan_basic():
    solution = {
        "title": "Gold",
        "context": "Context",
        "plan": [
            {"section": "Intro", "content": "Intro content"},
            {"section": "Methods", "content": "Methods content"},
        ],
    }

    out = _extract_gold_context(solution)

    assert "Gold" in out
    assert "Intro" in out
    assert "Methods" in out


def test_extract_candidate_markdown_plan():
    md = """
# Title

## Intro
Intro content.

## Methods
Methods content.

## References
1. Ref
"""

    out = _extract_candidate_context(md, content_type="markdown")

    assert "Intro" in out
    assert "Methods" in out
    assert "References" in out


def test_extract_candidate_context_ignores_markdown_title_as_empty_section():
    md = """
# Candidate Title

## Intro
Intro content.
"""

    out = _extract_candidate_context(md, content_type="markdown")

    assert "1. Intro" in out
    assert "Candidate Title" not in out


def test_extract_candidate_without_headings():
    out = _extract_candidate_context(
        "Plain document without headings.",
        content_type="markdown",
    )

    assert "no detected section headings" in out.lower()
    assert "Plain document" in out


def test_extract_gold_bibliography():
    solution = {
        "resources": [
            {"resource_id": 1, "resource_description": "Important paper"},
            {"resource_id": 2, "name": "Another source"},
        ]
    }

    out = _extract_bibliography_context(solution, "")

    assert "Important paper" in out
    assert "Another source" in out


def test_extract_candidate_bibliography_with_substring_heading():
    candidate = """
## Selected bibliography
1. Example reference
"""
    out = _extract_bibliography_context(None, candidate, content_type="markdown")
    assert "Selected bibliography" in out


def test_weak_section_extracts_ignore_references_and_gold_short_sections():
    solution = {
        "plan": [
            {"section": "Intro", "content": "Gold intro content " * 20},
            {"section": "Mechanism", "content": "Short."},
        ],
    }
    candidate = """
# Candidate

## Intro
Candidate intro content with enough detail to avoid the short-section rule.

## Mechanism
Short.

## References
1. Example reference
"""

    out = _select_weak_section_extracts(
        candidate,
        solution,
        content_type="markdown",
        article_metrics={"citation_count": 1},
    )

    assert "References" not in out
    assert "Mechanism (very short content)" not in out


def test_build_prompt_contains_required_schema():
    prompt = _build_judge_prompt(
        document_content="# Candidate",
        solution={"title": "Gold", "plan": []},
        content_type="markdown",
        dea_scores={},
        article_metrics={},
        prometheus_scores={},
        writehere_scores={},
        max_prompt_chars=20000,
    )

    assert "qualitative_assessment" in prompt
    assert "keep" in prompt
    assert "problems" in prompt
    assert "uncertainties" in prompt
    assert "Do not provide recommended fixes" in prompt


def test_prompt_respects_max_chars():
    long_doc = "# Title\n\n" + ("very long text " * 10000)

    prompt = _build_judge_prompt(
        document_content=long_doc,
        solution={"title": "Gold", "plan": []},
        content_type="markdown",
        dea_scores={},
        article_metrics={},
        prometheus_scores={},
        writehere_scores={},
        max_prompt_chars=4000,
    )

    assert len(prompt) <= 4000
    assert "qualitative_assessment" in prompt


def test_parse_valid_judge_json():
    raw = json.dumps({
        "qualitative_assessment": "Partial.",
        "keep": [],
        "problems": [],
        "uncertainties": [],
    })

    result = _parse_judge_response(raw)

    assert result["status"] == "ok"
    assert result["qualitative_assessment"] == "Partial."


def test_parse_fenced_json():
    raw = '''```json
{
  "qualitative_assessment": "Partial.",
  "keep": [],
  "problems": [],
  "uncertainties": []
}
```'''

    result = _parse_judge_response(raw)

    assert result["status"] == "ok"


def test_parse_strips_thinking_block_before_json():
    raw = '''<think>
I should reason about the schema before answering.
{"partial": "object inside thinking should be ignored"}
</think>
{
  "qualitative_assessment": "Partial.",
  "keep": [],
  "problems": [],
  "uncertainties": []
}
'''

    result = _parse_judge_response(raw)

    assert result["status"] == "ok"
    assert result["qualitative_assessment"] == "Partial."


def test_parse_extracts_result_json_from_surrounding_text():
    raw = '''Here is the evaluation:
{
  "qualitative_assessment": "Usable.",
  "keep": [],
  "problems": [],
  "uncertainties": []
}
Done.'''

    result = _parse_judge_response(raw)

    assert result["status"] == "ok"
    assert result["qualitative_assessment"] == "Usable."


def test_parse_invalid_json_returns_error():
    result = _parse_judge_response("not json")

    assert result["status"] == "error"
    assert result["qualitative_assessment"] == ""
    assert result["problems"] == []
    assert "invalid_judge_json:" in result["error"]


def test_parse_invalid_keep_shape_returns_error():
    raw = json.dumps({
        "qualitative_assessment": "Partial.",
        "keep": [{"point": "good only"}],
        "problems": [],
        "uncertainties": [],
    })

    result = _parse_judge_response(raw)
    assert result["status"] == "error"


def test_parse_invalid_uncertainty_shape_returns_error():
    raw = json.dumps({
        "qualitative_assessment": "Partial.",
        "keep": [],
        "problems": [],
        "uncertainties": [{"question": "unclear"}],
    })

    result = _parse_judge_response(raw)
    assert result["status"] == "error"


def test_parse_rejects_too_many_same_impact_problems():
    raw = json.dumps({
        "qualitative_assessment": "Weak.",
        "keep": [],
        "problems": [
            {
                "issue": f"Problem {i}",
                "main_impact": "plan",
                "priority": "P1 wrong",
                "impact": "Harms structure.",
                "confidence": "high",
                "evidence": "Missing requirement.",
            }
            for i in range(4)
        ],
        "uncertainties": [],
    })

    result = _parse_judge_response(raw)
    assert result["status"] == "error"


def test_parse_rejects_too_many_keep_items():
    raw = json.dumps({
        "qualitative_assessment": "Noisy.",
        "keep": [
            {"point": f"point {i}", "evidence": "e", "why_keep": "w"}
            for i in range(6)
        ],
        "problems": [],
        "uncertainties": [],
    })

    result = _parse_judge_response(raw)
    assert result["status"] == "error"


def test_parse_rejects_too_many_uncertainties():
    raw = json.dumps({
        "qualitative_assessment": "Noisy.",
        "keep": [],
        "problems": [],
        "uncertainties": [
            {"question": f"question {i}", "needed_evidence": "e"}
            for i in range(6)
        ],
    })

    result = _parse_judge_response(raw)
    assert result["status"] == "error"


def test_parse_normalizes_problem_fields_for_optimizer_stability():
    raw = json.dumps({
        "qualitative_assessment": "Partial.",
        "keep": [],
        "problems": [
            {
                "issue": "Weak support.",
                "main_impact": "references",
                "priority": "[P1] P1 wrong",
                "impact": "Hard to verify claims.",
                "confidence": "uncertain",
                "evidence": "x" * 2000,
            }
        ],
        "uncertainties": [],
    })

    result = _parse_judge_response(raw)

    assert result["status"] == "ok"
    problem = result["problems"][0]
    assert problem["main_impact"] == "bibliography"
    assert problem["priority"] == "P1 wrong"
    assert problem["confidence"] == "medium"
    assert len(problem["evidence"]) <= 800


def test_run_dea_judge_calls_llm_once():
    response = json.dumps({
        "qualitative_assessment": "The document has a useful plan but weak content.",
        "keep": [
            {
                "point": "The introduction is relevant.",
                "evidence": "Intro content",
                "why_keep": "It matches the expected opening."
            }
        ],
        "problems": [
            {
                "issue": "Methods section is missing.",
                "main_impact": "plan",
                "priority": "P1 wrong",
                "impact": "The document omits a required part of the topic.",
                "confidence": "high",
                "evidence": "Gold plan contains Methods; candidate plan does not."
            }
        ],
        "uncertainties": []
    })

    client = FakeJudgeClient(response)

    result = run_dea_judge(
        document_content="# Candidate\n\n## Intro\nIntro content",
        solution={
            "title": "Gold",
            "plan": [
                {"section": "Intro", "content": "Intro content"},
                {"section": "Methods", "content": "Methods content"},
            ],
        },
        content_type="markdown",
        dea_scores={"plan_contents_embedding_similarity": 0.4},
        article_metrics={"citation_count": 0},
        prometheus_scores={},
        writehere_scores={},
        model="fake",
        client=client,
    )

    assert result["status"] == "ok"
    assert len(client.calls) == 1
    assert result["problems"][0]["main_impact"] == "plan"


def test_run_dea_judge_passes_openai_extra_body_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEA_JUDGE_EXTRA_BODY_JSON", '{"reasoning": {"enabled": false}}')
    response = json.dumps({
        "qualitative_assessment": "The document is acceptable.",
        "keep": [],
        "problems": [],
        "uncertainties": [],
    })
    client = FakeJudgeClient(response)

    result = run_dea_judge(
        document_content="# Candidate",
        solution=None,
        content_type="markdown",
        dea_scores={},
        article_metrics={},
        prometheus_scores={},
        writehere_scores={},
        model="fake",
        client=client,
    )

    assert result["status"] == "ok"
    assert client.calls[0]["extra_body"] == {"reasoning": {"enabled": False}}


def test_run_dea_judge_rejects_invalid_extra_body_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEA_JUDGE_EXTRA_BODY_JSON", "[]")
    client = FakeJudgeClient("{}")

    result = run_dea_judge(
        document_content="# Candidate",
        solution=None,
        content_type="markdown",
        dea_scores={},
        article_metrics={},
        prometheus_scores={},
        writehere_scores={},
        model="fake",
        client=client,
    )

    assert result["status"] == "error"
    assert "DEA_JUDGE_EXTRA_BODY_JSON must contain a JSON object" in result["error"]


def test_run_dea_judge_skips_without_llm():
    result = run_dea_judge(
        document_content="# Candidate",
        solution=None,
        content_type="markdown",
        dea_scores={},
        article_metrics={},
        prometheus_scores={},
        writehere_scores={},
        model=None,
        client=None,
        lm=None,
    )

    assert result["status"] == "skipped"


def test_run_dea_judge_invalid_output_safe():
    client = FakeJudgeClient("not json")

    result = run_dea_judge(
        document_content="# Candidate",
        solution=None,
        content_type="markdown",
        dea_scores={},
        article_metrics={},
        prometheus_scores={},
        writehere_scores={},
        model="fake",
        client=client,
    )

    assert result["status"] == "error"
    assert result["problems"] == []
    assert result["raw_response"] == "not json"


def test_evaluate_document_judge_disabled():
    result = evaluate_document(
        document_content="# Candidate",
        solution={"title": "Gold", "context": "ctx", "plan": []},
        skip_dea=True,
        use_enhanced_metrics=False,
        use_dea_judge=False,
    )

    assert result["dea_judge"]["status"] == "skipped"


def test_evaluate_document_judge_enabled_fake_client():
    response = json.dumps({
        "qualitative_assessment": "Weak document.",
        "keep": [],
        "problems": [
            {
                "issue": "No bibliography.",
                "main_impact": "bibliography",
                "priority": "P1 wrong",
                "impact": "The document lacks evidence support.",
                "confidence": "high",
                "evidence": "Candidate has no References section."
            }
        ],
        "uncertainties": []
    })

    client = FakeJudgeClient(response)

    result = evaluate_document(
        document_content="# Candidate\n\n## Intro\nText",
        solution={"title": "Gold", "context": "ctx", "plan": []},
        skip_dea=True,
        use_enhanced_metrics=False,
        use_dea_judge=True,
        dea_judge_client=client,
        dea_judge_model="fake",
    )

    assert result["dea_judge"]["status"] == "ok"
    assert len(client.calls) == 1
    assert result["dea_judge"]["problems"][0]["main_impact"] == "bibliography"


def test_evaluate_document_judge_prompt_marks_skipped_dea():
    seen = {}

    def fake_lm(messages, temperature=0):
        seen["prompt"] = messages[0]["content"]
        return json.dumps({
            "qualitative_assessment": "Article metrics only.",
            "keep": [],
            "problems": [],
            "uncertainties": [],
        })

    result = evaluate_document(
        document_content="# Candidate\n\n## Intro\nText",
        solution={"title": "Gold", "context": "ctx", "plan": []},
        skip_dea=True,
        use_enhanced_metrics=False,
        use_dea_judge=True,
        dea_judge_lm=fake_lm,
    )

    assert result["dea_evaluation_scores"] == {}
    assert result["dea_evaluation_status"]["status"] == "skipped"
    assert "status: skipped (skip_dea=True)" in seen["prompt"]


@pytest.mark.performance
def test_judge_prompt_builder_performance():
    solution = {
        "title": "Gold",
        "context": "ctx",
        "plan": [
            {"section": f"Section {i}", "content": "gold content " * 100}
            for i in range(100)
        ],
        "resources": [
            {"resource_id": i, "resource_description": "resource " * 50}
            for i in range(30)
        ],
    }

    candidate = "# Candidate\n" + "\n".join(
        f"## Section {i}\n" + ("candidate content " * 100)
        for i in range(100)
    )

    start = time.perf_counter()
    prompt = _build_judge_prompt(
        document_content=candidate,
        solution=solution,
        content_type="markdown",
        dea_scores={"plan_contents_embedding_similarity": 0.5},
        article_metrics={"citation_count": 0},
        prometheus_scores={},
        writehere_scores={},
        max_prompt_chars=20000,
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 0.2
    assert len(prompt) <= 20000


@pytest.mark.performance
def test_run_dea_judge_fake_client_performance():
    client = FakeJudgeClient(json.dumps({
        "qualitative_assessment": "Partial.",
        "keep": [],
        "problems": [],
        "uncertainties": [],
    }))

    start = time.perf_counter()
    result = run_dea_judge(
        document_content="# Candidate\n\n## Intro\nText",
        solution={"title": "Gold", "context": "ctx", "plan": []},
        content_type="markdown",
        dea_scores={},
        article_metrics={},
        prometheus_scores={},
        writehere_scores={},
        model="fake",
        client=client,
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 0.1
    assert len(client.calls) == 1
    assert result["status"] == "ok"
