import json
import logging
from pathlib import Path

import pytest

from common.doc_eval import (
    evaluate_document,
    evaluate_document_content,
    temporary_transform_dea_into_markdown,
    evaluate_article_quality,
    _get_dea_filename,
    _load_existing_dea,
    get_solution_from_dea,
    DEA_evaluation,
)


def _sample_solution():
    return {
        "id": "demo-id",
        "title": "Sample Document",
        "abstract": "A short abstract about AI and ML.",
        "context": "A short abstract about AI and ML.",
        "target_file_path": "output/latex/sample.json",
        "plan": [
            {"section": "Introduction", "content": "Intro content."},
            {"section": "Methods", "content": "Methods content."},
        ],
        "resources": [
            {"resource_id": 1, "resource_description": "Some ref"},
        ],
    }


def test_evaluate_document_runs_without_dea():
    sol = _sample_solution()
    result = evaluate_document(
        document_content="An introduction about AI and ML.",
        solution=sol,
        skip_dea=True,
        use_enhanced_metrics=False,
    )
    # Basic shape
    assert "article_metrics" in result
    assert "dea_evaluation_scores" in result
    assert isinstance(result["dea_evaluation_scores"], dict)


def test_evaluate_document_skip_dea_marks_status_and_does_not_call_dea(monkeypatch):
    def fail_dea(*args, **kwargs):
        raise AssertionError("DEA_evaluation should not run when skip_dea=True")

    monkeypatch.setattr("common.doc_eval.DEA_evaluation", fail_dea)

    result = evaluate_document(
        document_content="Short candidate text.",
        solution=_sample_solution(),
        skip_dea=True,
        use_enhanced_metrics=False,
        use_dea_judge=False,
    )

    assert result["dea_evaluation_scores"] == {}
    assert result["dea_evaluation_status"] == {
        "status": "skipped",
        "reason": "skip_dea=True",
    }


def test_evaluate_document_dea_error_is_explicit_without_fallback_scores(monkeypatch):
    def fail_dea(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("common.doc_eval.DEA_evaluation", fail_dea)

    result = evaluate_document(
        document_content="Short candidate text.",
        solution=_sample_solution(),
        skip_dea=False,
        use_enhanced_metrics=False,
        use_dea_judge=False,
    )

    assert result["dea_evaluation_scores"] == {}
    assert result["dea_evaluation_status"]["status"] == "error"
    assert "boom" in result["dea_evaluation_status"]["error"]


def test_evaluate_document_similarity_ranks_better_candidate_higher():
    reference = "Artificial intelligence focuses on machine learning methods."
    better = "Artificial intelligence and machine learning methods are central topics."
    worse = "Gardening tips and cooking recipes are unrelated."

    better_scores = evaluate_document_content(
        document_content=better,
        reference_content=reference,
        use_enhanced_metrics=False,
    )
    worse_scores = evaluate_document_content(
        document_content=worse,
        reference_content=reference,
        use_enhanced_metrics=False,
    )

    better_f1 = better_scores["article_metrics"]["rouge_scores"]["rouge-l"]["f"]
    worse_f1 = worse_scores["article_metrics"]["rouge_scores"]["rouge-l"]["f"]
    assert better_f1 > worse_f1


def test_evaluate_document_accepts_latex_content():
    sol = _sample_solution()
    result = evaluate_document(
        document_content=r"\\section{Intro} This is latex.",
        solution=sol,
        content_type="latex",
        skip_dea=True,
        use_enhanced_metrics=False,
    )
    assert "article_metrics" in result


def test_temporary_transform_dea_into_markdown_produces_headings():
    sol = _sample_solution()
    md = temporary_transform_dea_into_markdown(sol)
    assert "# Sample Document" in md
    assert "## Introduction" in md


def test_entity_recall_prefers_overlap():
    reference = "Paris hosts the Eiffel Tower in France."
    better = "The Eiffel Tower stands in Paris, France."
    worse = "berlin has the brandenburg gate as a landmark."  # all lowercase -> no capitalized entities

    better_scores = evaluate_article_quality(better, reference)
    worse_scores = evaluate_article_quality(worse, reference)

    assert better_scores["entity_recall"] > worse_scores["entity_recall"]


def test_evaluate_document_content_without_reference_returns_empty():
    result = evaluate_document_content(
        document_content="Only content",
        reference_content=None,
        use_enhanced_metrics=False,
    )
    # No reference -> no article_metrics
    assert result == {}


def test_evaluate_document_content_with_reference_has_rouge():
    reference = "Neural networks learn representations."
    candidate = "Representations are learned by neural networks."
    result = evaluate_document_content(
        document_content=candidate,
        reference_content=reference,
        use_enhanced_metrics=False,
    )
    rouge = result["article_metrics"]["rouge_scores"]["rouge-l"]["f"]
    assert rouge > 0


def test_dea_evaluation_skip_env_returns_empty():
    sol = _sample_solution()
    scores = DEA_evaluation("content", sol, skip_env=True)
    assert scores == {}


def test_evaluate_document_with_solution_uses_markdown_reference():
    sol = _sample_solution()
    result = evaluate_document(
        document_content="Intro content about methods.",
        solution=sol,
        skip_dea=True,
        use_enhanced_metrics=False,
    )
    assert "article_metrics" in result
    # Should include the generated markdown reference structure
    assert "rouge_scores" in result["article_metrics"]


def test_get_dea_filename_sanitizes_topic(tmp_path):
    topic = "Quantum_computing/101"
    out = _get_dea_filename(topic, tmp_path)
    assert out.name == "Quantum computing101.json"


def test_load_existing_dea_reads_file(tmp_path):
    dea_file = tmp_path / "sample.json"
    dea_file.write_text(json.dumps({"title": "T", "id": "1"}), encoding="utf-8")
    intent, solution = _load_existing_dea("Sample", dea_file, logging)
    assert solution["target_file_path"] == str(dea_file.resolve())
    assert intent.startswith("Write a wikipedia like article")


def test_get_solution_from_dea_reads_latex(tmp_path):
    root = tmp_path
    data_dir = root / "output" / "latex"
    data_dir.mkdir(parents=True, exist_ok=True)
    sample = {"id": "x", "title": "Demo", "context": "ctx"}
    (data_dir / "demo.json").write_text(json.dumps(sample), encoding="utf-8")

    ctx, sol = get_solution_from_dea("demo", root)
    assert ctx == "ctx"
    assert sol["target_file_path"].endswith("demo.json")


def test_reference_influences_rouge():
    reference = "Deep learning models use backpropagation."
    aligned = "Backpropagation is used by deep learning models."
    off_topic = "Gardening tips for indoor plants."

    aligned_scores = evaluate_document_content(
        document_content=aligned,
        reference_content=reference,
        use_enhanced_metrics=False,
    )["article_metrics"]["rouge_scores"]["rouge-l"]["f"]

    off_scores = evaluate_document_content(
        document_content=off_topic,
        reference_content=reference,
        use_enhanced_metrics=False,
    )["article_metrics"]["rouge_scores"]["rouge-l"]["f"]

    assert aligned_scores > off_scores
