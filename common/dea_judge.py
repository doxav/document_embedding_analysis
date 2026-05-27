from __future__ import annotations

import json
import os
import re
from typing import Any


SYSTEM_MESSAGE = """You are a strict document evaluation judge.

You compare a candidate document against a gold DEA solution.
Use the quantitative scores and text extracts provided.

Your task is to produce a qualitative assessment, list strong points to keep, list concrete problems, and list uncertainties.

Do not recommend fixes. The optimizer will decide fixes.
Do not invent evidence.
Use exact quotes or concrete observed behavior when possible.
Return valid JSON only.
"""

REQUIRED_KEYS = {
    "qualitative_assessment",
    "keep",
    "problems",
    "uncertainties",
}

REQUIRED_PROBLEM_KEYS = {
    "issue",
    "main_impact",
    "priority",
    "impact",
    "confidence",
    "evidence",
}

REQUIRED_KEEP_KEYS = {
    "point",
    "evidence",
    "why_keep",
}

REQUIRED_UNCERTAINTY_KEYS = {
    "question",
    "needed_evidence",
}

_BIBLIOGRAPHY_HEADINGS = {"references", "bibliography", "external links"}

_MAIN_IMPACT_CANON = {
    "plan": "plan",
    "section content": "sections content",
    "sections content": "sections content",
    "content": "sections content",
    "citations": "citations",
    "citation": "citations",
    "bibliography": "bibliography",
    "references": "bibliography",
    "reference": "bibliography",
    "format": "format",
}

_PRIORITY_CANON = {
    "p0": "P0 blocker",
    "p1": "P1 wrong",
    "p2": "P2 minor",
    "p3": "P3 polish",
}


def _empty_payload(status: str, reason: str | None = None, error: str | None = None, raw_response: str | None = None) -> dict:
    payload = {
        "status": status,
        "qualitative_assessment": "",
        "keep": [],
        "problems": [],
        "uncertainties": [],
    }
    if reason is not None:
        payload["reason"] = reason
    if error is not None:
        payload["error"] = error
    if raw_response is not None:
        payload["raw_response"] = raw_response
    return payload


def _truncate(text: Any, max_chars: int) -> str:
    text = "" if text is None else str(text)
    if len(text) <= max_chars:
        return text
    if max_chars <= 20:
        return text[:max_chars]
    return text[: max_chars - 15].rstrip() + " ...[truncated]"


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _normalize_heading_title(title: Any) -> str:
    text = str(title or "").strip().lower().rstrip(":")
    return re.sub(r"^h[1-6]\s+", "", text).strip()


def _is_bibliography_heading(title: Any) -> bool:
    text = _normalize_heading_title(title)
    return any(heading in text for heading in _BIBLIOGRAPHY_HEADINGS)


def _interpret_metric(name: str, value: Any) -> tuple[str, str]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "No standard range available.", "not interpreted"

    lowered = name.lower()
    if "ratio" in lowered:
        if 0.85 <= numeric <= 1.15:
            interp = "good"
        elif 0.65 <= numeric < 0.85 or 1.15 < numeric <= 1.30:
            interp = "partial"
        else:
            interp = "weak"
        return "Ideal is around 1.0.", interp

    if "similarity" in lowered or "coverage" in lowered or "score" in lowered:
        if numeric >= 0.80:
            interp = "good"
        elif numeric >= 0.60:
            interp = "partial"
        else:
            interp = "weak"
        return "Range: 0-1, higher is better.", interp

    return "No standard range available.", "not interpreted"


def _format_mapping(title: str, mapping: dict | None, max_chars: int = 3000) -> str:
    if not mapping:
        return f"{title}:\n- none available"
    lines = [f"{title}:"]
    for key, value in mapping.items():
        if isinstance(value, dict):
            flat = json.dumps(value, sort_keys=True, ensure_ascii=False)
            lines.append(f"- {key}: {_truncate(flat, 500)}")
        else:
            lines.append(f"- {key}: {_format_value(value)}")
    return _truncate("\n".join(lines), max_chars)


def _format_score_context(
    dea_scores: dict | None,
    article_metrics: dict | None,
    prometheus_scores: dict | None,
    writehere_scores: dict | None,
    dea_status: dict | None = None,
) -> str:
    lines = ["DEA scores:"]
    if dea_status:
        status = dea_status.get("status", "unknown")
        reason = dea_status.get("reason")
        suffix = f" ({reason})" if reason else ""
        lines.append(f"- status: {status}{suffix}")
    if dea_scores:
        for name, value in dea_scores.items():
            range_text, interpretation = _interpret_metric(name, value)
            lines.append(
                f"- {name} = {_format_value(value)}. {range_text} Interpretation: {interpretation}."
            )
    else:
        lines.append("- none available")

    article_lines = ["Article metrics:"]
    article_metrics = article_metrics or {}
    if "citation_count" in article_metrics:
        article_lines.append(f"- citation_count: {article_metrics.get('citation_count')}")
    else:
        article_lines.append("- citation_count: not available")
    if article_metrics.get("rouge_scores"):
        article_lines.append(f"- ROUGE scores: {json.dumps(article_metrics['rouge_scores'], sort_keys=True)}")
    else:
        article_lines.append("- ROUGE scores: none available")
    for key, value in article_metrics.items():
        if key not in {"citation_count", "rouge_scores"}:
            article_lines.append(f"- {key}: {_truncate(json.dumps(value, default=str), 500)}")

    return "\n\n".join(
        [
            "\n".join(lines),
            _truncate("\n".join(article_lines), 4000),
            _format_mapping("Prometheus scores", prometheus_scores),
            _format_mapping("WriteHere scores", writehere_scores),
        ]
    )


def _resource_to_text(resource: Any) -> str:
    if isinstance(resource, dict):
        pieces = []
        for key in ("resource_id", "name", "title", "resource_description", "description", "url"):
            if resource.get(key):
                pieces.append(str(resource[key]))
        if not pieces:
            pieces = [json.dumps(resource, ensure_ascii=False, sort_keys=True)]
        return " | ".join(pieces)
    return str(resource)


def _extract_gold_context(
    solution: dict | None,
    max_gold_section_excerpt_chars: int = 500,
    max_gold_sections: int = 40,
) -> str:
    if not solution:
        return "Gold context:\n- no gold solution available"

    lines = ["Gold context:"]
    lines.append(f"Gold title: {solution.get('title', '')}")
    context = solution.get("context") or solution.get("abstract") or ""
    lines.append(f"Gold context / abstract: {_truncate(context, 1500)}")
    lines.append("Gold plan:")
    plan = solution.get("plan") or []
    if not plan:
        lines.append("- none available")
    for idx, section in enumerate(plan, 1):
        if isinstance(section, dict):
            title = section.get("section") or section.get("title") or f"Section {idx}"
            content = section.get("content") or section.get("text") or ""
        else:
            title, content = str(section), ""
        lines.append(f"{idx}. {title}")
        if idx <= max_gold_sections:
            lines.append(f"   Content excerpt: {_truncate(content, max_gold_section_excerpt_chars)}")
        else:
            lines.append("   Content excerpt: omitted after section excerpt limit")
        lines.append(f"   Content length: {len(str(content))}")

    lines.append("Gold bibliography extract:")
    resources = solution.get("resources") or []
    if resources:
        for idx, resource in enumerate(resources[:10], 1):
            lines.append(f"{idx}. {_truncate(_resource_to_text(resource), 300)}")
    else:
        lines.append("- none available")
    return "\n".join(lines)


def _parse_candidate_sections(
    document_content: str,
    content_type: str,
    include_h1: bool = False,
) -> list[dict[str, Any]]:
    text = document_content or ""
    if content_type == "latex":
        matches = list(re.finditer(r"\\(?:section|subsection|subsubsection)\*?\{([^}]+)\}", text))
        return _sections_from_matches(text, matches, lambda m: m.group(1).strip())

    pattern = r"^(#{1,6})\s+(.+)$" if include_h1 else r"^(#{2,6})\s+(.+)$"
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    return _sections_from_matches(text, matches, lambda m: m.group(2).strip())


def _sections_from_matches(text: str, matches: list[re.Match], title_getter) -> list[dict[str, Any]]:
    sections = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        sections.append({"title": title_getter(match), "content": content})
    return sections


def _extract_candidate_context(
    document_content: str,
    content_type: str,
    max_section_excerpt_chars: int = 500,
    max_sections_with_excerpts: int = 40,
) -> str:
    sections = _parse_candidate_sections(document_content, content_type)
    if not sections:
        return (
            "Candidate has no detected section headings.\n"
            f"Full beginning excerpt: {_truncate(document_content, 2000)}"
        )

    lines = ["Candidate plan:"]
    for idx, section in enumerate(sections, 1):
        content = section["content"]
        lines.append(f"{idx}. {section['title']}")
        if idx <= max_sections_with_excerpts:
            lines.append(f"   Content excerpt: {_truncate(content, max_section_excerpt_chars)}")
        else:
            lines.append("   Content excerpt: omitted after section excerpt limit")
        lines.append(f"   Content length: {len(content)}")
    return "\n".join(lines)


def _extract_candidate_bibliography(document_content: str, content_type: str) -> str:
    sections = _parse_candidate_sections(document_content, content_type)
    bib_sections = [
        section for section in sections
        if _is_bibliography_heading(section["title"])
    ]
    if not bib_sections:
        return "Candidate bibliography extract:\n- none detected"
    lines = ["Candidate bibliography extract:"]
    for section in bib_sections[:3]:
        lines.append(f"{section['title']}: {_truncate(section['content'], 1200)}")
    return "\n".join(lines)


def _extract_bibliography_context(
    solution: dict | None,
    document_content: str,
    content_type: str = "markdown",
    max_bibliography_items: int = 10,
    max_bibliography_item_chars: int = 300,
) -> str:
    lines = ["Gold bibliography extract:"]
    resources = (solution or {}).get("resources") or []
    if resources:
        for idx, resource in enumerate(resources[:max_bibliography_items], 1):
            lines.append(f"{idx}. {_truncate(_resource_to_text(resource), max_bibliography_item_chars)}")
    else:
        lines.append("- none available")
    lines.append(_extract_candidate_bibliography(document_content, content_type))
    return "\n".join(lines)


def _gold_titles(solution: dict | None) -> set[str]:
    titles = set()
    for idx, section in enumerate((solution or {}).get("plan") or [], 1):
        if isinstance(section, dict):
            title = section.get("section") or section.get("title") or f"Section {idx}"
        else:
            title = str(section)
        titles.add(_normalize_heading_title(title))
    return titles


def _gold_content_lengths(solution: dict | None) -> dict[str, int]:
    lengths = {}
    for idx, section in enumerate((solution or {}).get("plan") or [], 1):
        if not isinstance(section, dict):
            continue
        title = section.get("section") or section.get("title") or f"Section {idx}"
        lengths[_normalize_heading_title(title)] = len(str(section.get("content") or section.get("text") or ""))
    return lengths


def _select_weak_section_extracts(
    document_content: str,
    solution: dict | None,
    content_type: str,
    article_metrics: dict | None,
    max_worst_sections: int = 4,
    max_worst_section_chars: int = 800,
) -> str:
    sections = [
        section for section in _parse_candidate_sections(document_content, content_type)
        if not _is_bibliography_heading(section["title"])
    ]
    if not sections:
        return "Weak-looking candidate section extracts:\n- no sections detected"

    gold_titles = _gold_titles(solution)
    gold_lengths = _gold_content_lengths(solution)
    citation_count = (article_metrics or {}).get("citation_count", 0)
    selected: list[tuple[str, dict[str, Any]]] = []
    seen: set[int] = set()

    def add(reason: str, section: dict[str, Any]) -> None:
        if len(selected) >= max_worst_sections:
            return
        marker = id(section)
        if marker not in seen and section["content"].strip():
            selected.append((reason, section))
            seen.add(marker)

    for section in sorted(sections, key=lambda s: len(s["content"])):
        content_len = len(section["content"].strip())
        gold_len = gold_lengths.get(_normalize_heading_title(section["title"]))
        if content_len < 120 and (gold_len is None or gold_len >= 120):
            add("very short content", section)
    if not citation_count:
        for section in sections:
            if not re.search(r"\[\d+\]|\\cite\{[^}]+\}", section["content"]):
                add("no citations while citation score/count is weak", section)
    for section in sections:
        if gold_titles and _normalize_heading_title(section["title"]) not in gold_titles:
            add("title not present in gold plan", section)
    for section in sections:
        add("first non-empty candidate section fallback", section)

    lines = ["Weak-looking candidate section extracts:"]
    if not selected:
        lines.append("- no non-empty sections available")
    for reason, section in selected:
        lines.append(f"- {section['title']} ({reason}): {_truncate(section['content'], max_worst_section_chars)}")
    return "\n".join(lines)


def _schema_text() -> str:
    return """Return exactly this JSON schema:

{
  "qualitative_assessment": "Brief natural-language judgment. Mention what works, what fails, and the main reason for the verdict.",
  "keep": [
    {
      "point": "A strong point which should be preserved",
      "evidence": "Exact quote or concrete behavior",
      "why_keep": "Why it helps strongly"
    }
  ],
  "problems": [
    {
      "issue": "Concrete problem description",
      "main_impact": "plan | sections content | citations | bibliography | format",
      "priority": "P0 blocker | P1 wrong | P2 minor | P3 polish",
      "impact": "What user/task harm it causes",
      "confidence": "low | medium | high",
      "evidence": "Exact quote, missing requirement, or source mismatch"
    }
  ],
  "uncertainties": [
    {
      "question": "What the judge cannot determine",
      "needed_evidence": "What would resolve it"
    }
  ]
}

The "problems" list must contain no more than 9 items total.
The "keep" list must contain no more than 5 items.
The "uncertainties" list must contain no more than 5 items.
"""


def _instructions_text() -> str:
    return """Evaluate the candidate document.

You are given:
1. DEA scores with interpretation ranges.
2. Other available scores.
3. Gold title, context, plan, and bibliography extracts.
4. Candidate plan, content extracts, and bibliography extracts.
5. Extracts of weak-looking non-empty candidate sections.

Focus on the main causes of weak scores.

For problems:
- Provide at most 3 problems about plan.
- Provide at most 3 problems about sections content.
- Provide at most 3 problems about citations/bibliography.
- Include priority, impact, confidence, and evidence.
- Do not provide recommended fixes.
- Treat "...[truncated]" as an excerpt marker, not as a candidate defect.
- Before calling a section too short, compare it with the matching gold section.
- Prefer recurring mechanism-level patterns with concrete examples over one-off factual corrections.
"""


def _build_judge_prompt(
    *,
    document_content: str,
    solution: dict | None,
    content_type: str,
    dea_scores: dict,
    article_metrics: dict,
    prometheus_scores: dict,
    writehere_scores: dict,
    dea_status: dict | None = None,
    max_prompt_chars: int = 20000,
) -> str:
    instructions = _instructions_text()
    schema = _schema_text()
    fixed_len = len(instructions) + len(schema) + 4
    context_budget = max(0, max_prompt_chars - fixed_len)
    context_parts = [
        _format_score_context(dea_scores, article_metrics, prometheus_scores, writehere_scores, dea_status),
        _extract_gold_context(solution),
        _extract_candidate_context(document_content, content_type),
        _extract_bibliography_context(solution, document_content, content_type),
        f"Citation count from article metrics: {(article_metrics or {}).get('citation_count', 'not available')}",
        _select_weak_section_extracts(document_content, solution, content_type, article_metrics),
    ]
    context = _truncate("\n\n".join(context_parts), context_budget)
    prompt = f"{instructions}\n{context}\n\n{schema}"
    if len(prompt) <= max_prompt_chars:
        return prompt
    overflow = len(prompt) - max_prompt_chars
    context = _truncate(context, max(0, len(context) - overflow))
    return f"{instructions}\n{context}\n\n{schema}"[:max_prompt_chars]


def _extract_json_text(raw: str) -> str:
    text = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    for tag in ("think", "thinking", "analysis", "reasoning"):
        text = re.sub(rf"<{tag}\b[^>]*>.*?</{tag}>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    decoder = json.JSONDecoder()
    fallback = None
    for match in re.finditer(r"\{", text):
        try:
            parsed, end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        candidate = text[match.start(): match.start() + end]
        if REQUIRED_KEYS.issubset(parsed.keys()):
            return candidate
        if fallback is None:
            fallback = candidate

    return fallback or text


def _validate_judge_output(parsed: Any) -> bool:
    if not isinstance(parsed, dict):
        return False
    if not REQUIRED_KEYS.issubset(parsed.keys()):
        return False
    if not isinstance(parsed.get("qualitative_assessment"), str):
        return False
    for key in ("keep", "problems", "uncertainties"):
        if not isinstance(parsed.get(key), list):
            return False
    if len(parsed["keep"]) > 5 or len(parsed["problems"]) > 9 or len(parsed["uncertainties"]) > 5:
        return False

    impact_counts: dict[str, int] = {}

    for keep in parsed.get("keep", []):
        if not isinstance(keep, dict):
            return False
        if not REQUIRED_KEEP_KEYS.issubset(keep.keys()):
            return False

    for problem in parsed.get("problems", []):
        if not isinstance(problem, dict):
            return False
        if not REQUIRED_PROBLEM_KEYS.issubset(problem.keys()):
            return False
        impact = str(problem.get("main_impact", "")).strip().lower()
        impact_counts[impact] = impact_counts.get(impact, 0) + 1
        if impact_counts[impact] > 3:
            return False

    for uncertainty in parsed.get("uncertainties", []):
        if not isinstance(uncertainty, dict):
            return False
        if not REQUIRED_UNCERTAINTY_KEYS.issubset(uncertainty.keys()):
            return False

    return True


def _canonical_main_impact(value: Any) -> str:
    lowered = str(value or "").strip().lower()
    parts = [part.strip() for part in re.split(r"\s*(?:[|/,;]|\band\b)\s*", lowered) if part.strip()]
    for part in parts or [lowered]:
        if part in _MAIN_IMPACT_CANON:
            return _MAIN_IMPACT_CANON[part]
    for needle, canonical in _MAIN_IMPACT_CANON.items():
        if needle in lowered:
            return canonical
    return "format"


def _canonical_priority(value: Any) -> str:
    lowered = str(value or "").strip().lower()
    match = re.search(r"\bp([0-3])\b", lowered)
    if match:
        return _PRIORITY_CANON[f"p{match.group(1)}"]
    return _PRIORITY_CANON.get(lowered, "P2 minor")


def _normalize_problem(problem: dict) -> dict:
    normalized = dict(problem)
    normalized["main_impact"] = _canonical_main_impact(normalized.get("main_impact"))
    normalized["priority"] = _canonical_priority(normalized.get("priority"))
    confidence = str(normalized.get("confidence", "")).strip().lower()
    normalized["confidence"] = confidence if confidence in {"low", "medium", "high"} else "medium"
    normalized["evidence"] = _truncate(normalized.get("evidence", ""), 800)
    return normalized


def _problem_impact_counts_are_valid(problems: list[dict]) -> bool:
    counts: dict[str, int] = {}
    for problem in problems:
        impact = str(problem.get("main_impact", "")).strip().lower()
        counts[impact] = counts.get(impact, 0) + 1
        if counts[impact] > 3:
            return False
    return True


def _parse_judge_response(raw: str) -> dict:
    try:
        parsed = json.loads(_extract_json_text(raw))
    except Exception as exc:
        return _empty_payload("error", error=f"invalid_judge_json: {exc}", raw_response=raw)
    if not _validate_judge_output(parsed):
        return _empty_payload("error", error="invalid_judge_json", raw_response=raw)
    problems = [_normalize_problem(problem) for problem in parsed["problems"]]
    if not _problem_impact_counts_are_valid(problems):
        return _empty_payload("error", error="invalid_judge_json", raw_response=raw)
    return {
        "status": "ok",
        "qualitative_assessment": parsed["qualitative_assessment"],
        "keep": parsed["keep"],
        "problems": problems,
        "uncertainties": parsed["uncertainties"],
        "raw_response": raw,
    }


def _extra_body_from_env(env_var: str = "DEA_JUDGE_EXTRA_BODY_JSON") -> dict[str, Any] | None:
    """Load an optional OpenAI-compatible extra_body payload for DEA judge calls."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{env_var} must contain a JSON object.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{env_var} must contain a JSON object.")
    return payload


def _chat_completion_kwargs(*, prompt: str, system_message: str, model: str | None) -> dict[str, Any]:
    """Build common OpenAI-compatible chat completion kwargs."""
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
    }
    extra_body = _extra_body_from_env()
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


def _call_llm_once(*, prompt: str, system_message: str, model: str | None, client=None, lm=None) -> str:
    if lm is not None:
        raw = lm([{"role": "user", "content": prompt}], temperature=0)
    elif client is not None:
        raw = client.chat.completions.create(
            **_chat_completion_kwargs(prompt=prompt, system_message=system_message, model=model)
        ).choices[0].message.content
    elif model is not None:
        from openai import OpenAI

        client = OpenAI()
        if model.startswith("gpt-5") and hasattr(client, "responses"):
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = getattr(response, "output_text", "")
        else:
            raw = client.chat.completions.create(
                **_chat_completion_kwargs(prompt=prompt, system_message=system_message, model=model)
            ).choices[0].message.content
    else:
        raise ValueError("no judge model/client/lm available")

    if isinstance(raw, str):
        return raw
    if hasattr(raw, "content"):
        return str(raw.content)
    return str(raw)


def run_dea_judge(
    *,
    document_content: str,
    solution: dict | None,
    content_type: str,
    dea_scores: dict,
    article_metrics: dict,
    prometheus_scores: dict,
    writehere_scores: dict,
    dea_status: dict | None = None,
    model: str | None = None,
    client=None,
    lm=None,
    max_prompt_chars: int = 20000,
) -> dict:
    if lm is None and client is None and model is None:
        return _empty_payload("skipped", reason="no judge model/client/lm available")

    prompt = _build_judge_prompt(
        document_content=document_content,
        solution=solution,
        content_type=content_type,
        dea_scores=dea_scores or {},
        article_metrics=article_metrics or {},
        prometheus_scores=prometheus_scores or {},
        writehere_scores=writehere_scores or {},
        dea_status=dea_status,
        max_prompt_chars=max_prompt_chars,
    )
    try:
        raw = _call_llm_once(
            prompt=prompt,
            system_message=SYSTEM_MESSAGE,
            model=model,
            client=client,
            lm=lm,
        )
    except Exception as exc:
        return _empty_payload("error", error=str(exc), raw_response="")
    return _parse_judge_response(raw)
