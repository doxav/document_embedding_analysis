from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys

BUNDLE_ROOT = _Path(__file__).resolve().parents[1]
if str(BUNDLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(BUNDLE_ROOT))

import json
import os
import sys
from pathlib import Path
from typing import Any


def _ensure_repo_on_path() -> None:
    repo_root = os.environ.get("DEA_REPO_ROOT", "").strip()
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _load_solution(context) -> dict[str, Any]:
    vars_dict = context["vars"]
    path_str = vars_dict.get("dea_solution_path") or vars_dict.get("solution_path")
    if not path_str:
        raise RuntimeError("Missing dea_solution_path in dataset vars")
    path = Path(path_str)
    if not path.is_absolute():
        repo_root = Path(os.environ.get("DEA_REPO_ROOT", "")).resolve()
        path = (repo_root / path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        preferred = []
        for key in (
            "embedding2_cosine_similarity",
            "embedding1_cosine_similarity",
            "cosine_similarity",
            "score",
            "similarity",
        ):
            v = value.get(key)
            if isinstance(v, (int, float)):
                preferred.append(float(v))
        if preferred:
            return sum(preferred) / len(preferred)
        vals = [float(v) for v in value.values() if isinstance(v, (int, float))]
        if vals:
            return sum(vals) / len(vals)
        return None
    return None


def _pick(scores: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        if key in scores:
            value = _numeric(scores.get(key))
            if value is not None:
                return value
    return None


def _clamp(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def get_assert(output: str, context):
    _ensure_repo_on_path()
    from common.doc_eval import evaluate_document

    solution = _load_solution(context)
    vars_dict = context["vars"]
    config = context.get("config", {}) or {}

    use_dea_judge = str(
        config.get("useDeaJudge", os.environ.get("DEA_USE_JUDGE", "false"))
    ).strip().lower() in {"1", "true", "yes", "on"}

    result = evaluate_document(
        document_content=output,
        solution=solution,
        content_type=vars_dict.get("candidate_content_type", "markdown"),
        use_enhanced_metrics=bool(config.get("useEnhancedMetrics", False)),
        skip_dea=False,
        openai_model=config.get("openaiModel") or os.environ.get("OPENAI_MODEL") or None,
        dea_embedding_backend=config.get("deaEmbeddingBackend") or os.environ.get("DEA_EMBEDDING_BACKEND") or None,
        dea_embedding_model=config.get("deaEmbeddingModel") or os.environ.get("DEA_EMBEDDING_MODEL") or None,
        skip_entity_recall=bool(config.get("skipEntityRecall", True)),
        use_dea_judge=use_dea_judge,
        dea_judge_model=config.get("deaJudgeModel") or os.environ.get("DEA_JUDGE_MODEL") or None,
    )

    dea_status = result.get("dea_evaluation_status", {}) or {}
    if dea_status.get("status") == "error":
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"DEA evaluation error: {dea_status.get('error', 'unknown error')}",
            "namedScores": {
                "dea_status_ok": 0.0,
            },
        }

    dea_scores = result.get("dea_evaluation_scores", {}) or {}
    article_metrics = result.get("article_metrics", {}) or {}
    rouge_scores = article_metrics.get("rouge_scores", {}) or {}
    rouge_l = (
        rouge_scores.get("rouge-l", {}) or rouge_scores.get("rougeL", {}) or {}
    )
    if isinstance(rouge_l, dict):
        rouge_l_f = _clamp(_numeric(rouge_l.get("f")))
    else:
        rouge_l_f = None
    if rouge_l_f is None:
        rouge_l_f = _clamp(
            _numeric(
                rouge_scores.get("ROUGEL_f1")
                or rouge_scores.get("rougeL_f1")
                or rouge_scores.get("rouge-l_f1")
                or rouge_scores.get("rouge_l_f1")
            )
        )
    entity_recall = _clamp(_numeric(article_metrics.get("entity_recall")))
    citation_count = _numeric(article_metrics.get("citation_count")) or 0.0

    plan = _clamp(
        _pick(
            dea_scores,
            [
                "plan_embedding_similarity",
                "section_total_similarity",
                "plan_total_similarity",
                "global_plan_embedding_similarity",
                "global_section_total_similarity",
                "global_plan_total_similarity",
            ],
        )
    )
    content = _clamp(
        _pick(
            dea_scores,
            [
                "plan_contents_embedding_similarity",
                "content_total_similarity",
                "global_plan_contents_embedding_similarity",
                "global_content_total_similarity",
            ],
        )
    )
    resources = _clamp(
        _pick(
            dea_scores,
            [
                "plan_resources_embedding_similarity",
                "resources_citation_coverage_score",
                "bibliography_coverage1",
                "bibliography_coverage2",
                "global_plan_resources_embedding_similarity",
            ],
        )
    )

    weights = {
        "plan": float(config.get("planWeight", 0.30)),
        "content": float(config.get("contentWeight", 0.45)),
        "resources": float(config.get("resourceWeight", 0.15)),
        "rouge_l": float(config.get("rougeWeight", 0.10)),
    }

    available = {
        "plan": plan,
        "content": content,
        "resources": resources,
        "rouge_l": rouge_l_f,
    }
    weighted_terms = {
        key: value for key, value in available.items() if value is not None and weights.get(key, 0) > 0
    }
    if not weighted_terms:
        return {
            "pass": False,
            "score": 0.0,
            "reason": "No DEA or article metrics were available for scoring.",
            "namedScores": {
                "dea_status_ok": 0.0,
            },
        }

    total_weight = sum(weights[k] for k in weighted_terms)
    composite = sum(weights[k] * weighted_terms[k] for k in weighted_terms) / total_weight

    plan_min = float(config.get("planMin", 0.0))
    content_min = float(config.get("contentMin", 0.0))
    resource_min = float(config.get("resourceMin", 0.0))
    overall_min = float(config.get("overallMin", 0.70))

    hard_fail_reasons = []
    if plan is not None and plan < plan_min:
        hard_fail_reasons.append(f"plan {plan:.3f} < {plan_min:.3f}")
    if content is not None and content < content_min:
        hard_fail_reasons.append(f"content {content:.3f} < {content_min:.3f}")
    if resources is not None and resources < resource_min:
        hard_fail_reasons.append(f"resources {resources:.3f} < {resource_min:.3f}")

    dea_judge = result.get("dea_judge", {}) or {}
    dea_judge_status = str(dea_judge.get("status", "skipped"))
    dea_judge_problem_count = float(len(dea_judge.get("problems", []) or []))

    named_scores = {
        "dea_status_ok": 1.0 if dea_status.get("status") not in {"error"} else 0.0,
        "plan": plan if plan is not None else 0.0,
        "content": content if content is not None else 0.0,
        "resources": resources if resources is not None else 0.0,
        "rouge_l_f": rouge_l_f if rouge_l_f is not None else 0.0,
        "entity_recall": entity_recall if entity_recall is not None else 0.0,
        "citation_count": citation_count,
        "dea_composite": composite,
        "dea_judge_ran": 0.0 if dea_judge_status in {"skipped", "error"} else 1.0,
        "dea_judge_problem_count": dea_judge_problem_count,
    }

    for key, value in dea_scores.items():
        numeric = _numeric(value)
        if numeric is not None:
            named_scores.setdefault(str(key), float(numeric))

    pass_result = (not hard_fail_reasons) and composite >= overall_min

    reason_bits = [
        f"plan={plan:.3f}" if plan is not None else "plan=n/a",
        f"content={content:.3f}" if content is not None else "content=n/a",
        f"resources={resources:.3f}" if resources is not None else "resources=n/a",
        f"rouge_l_f={rouge_l_f:.3f}" if rouge_l_f is not None else "rouge_l_f=n/a",
        f"score={composite:.3f}",
        f"dea_status={dea_status.get('status', 'unknown')}",
        f"judge_status={dea_judge_status}",
    ]
    if hard_fail_reasons:
        reason_bits.append("hard_fail=" + "; ".join(hard_fail_reasons))

    return {
        "pass": pass_result,
        "score": composite,
        "reason": " | ".join(reason_bits),
        "namedScores": named_scores,
    }
