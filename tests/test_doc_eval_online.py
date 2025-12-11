import os
import pytest

from common.doc_eval import evaluate_document


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY for online evaluation"
)
def test_evaluate_document_online_prometheus_writehere():
    """Sanity check that enhanced metrics run when an OpenAI model is available."""
    model = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    solution = {
        "id": "online-demo",
        "title": "Online Eval",
        "abstract": "Short abstract",
        "context": "Short context",
        "target_file_path": "output/latex/sample.json",
        "plan": [],
        "resources": [],
    }

    result = evaluate_document(
        document_content="A brief paragraph about AI research progress.",
        solution=solution,
        content_type="markdown",
        use_enhanced_metrics=True,
        skip_dea=True,
        openai_model=model,
    )

    assert "prometheus_scores" in result
    assert "writehere_scores" in result
    # Ensure scores are not empty when model is provided
    assert result["prometheus_scores"]
    assert result["writehere_scores"]
