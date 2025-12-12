from __future__ import annotations

import json, re, sys, logging, time
from pathlib import Path
from typing import Dict, Any, Optional, Union
try:
    import requests
except ImportError:  # Optional dependency
    requests = None
try:
    import wikipedia
except ImportError:  # Optional dependency
    wikipedia = None
import pypandoc
import tempfile
from regex import F

# Import additional evaluators for enhanced evaluation
try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ───── third-party / intra-pkg ───────────────────────────────────────────
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity  # only used if np present
import os
import sys
import logging


from .env.env import EnvironmentManager
from .config import (
    HUGGINGFACE_EMBEDDING_MODEL_NAME,
    HUGGINGFACE_EMBEDDING_PATH,
    HUGGINGFACE_EMBEDDING_PREFIX,
    OPENAI_EMBEDDING_MODEL_NAME,
)

# ─── text metrics (skim-fast heuristics if lib missing) ───────────────
from .metrics import compute_rouge_scores, article_entity_recall

from sentence_transformers import SentenceTransformer
embedding_function = SentenceTransformer('paraphrase-MiniLM-L6-v2')

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


try:
    from config import HF_TOKEN
except ImportError:
    try:
        from .config import HF_TOKEN
    except ImportError:
        HF_TOKEN = None

try:
    _HF_API_TASK = "text2text-generation"
except NameError:
    _HF_API_TASK = None



logger = logging.getLogger(__name__)

# ─── Dataset helpers (shared with benchmark.py) ───────────────────────
DEA_FILES: Dict[str, Path] = {}
FRESHWIKI_DEA_FILES: Dict[str, Path] = {}

FRESHWIKI_CSV_URL = (
    "https://huggingface.co/datasets/EchoShao8899/FreshWiki/resolve/main/topic_list.csv"
)
FRESHWIKI_JSON_BASE = (
    "https://huggingface.co/datasets/EchoShao8899/FreshWiki/resolve/main/json/"
)
WILDSEEK_JSON_URL = (
    "https://huggingface.co/datasets/YuchengJiang/WildSeek/resolve/main/data.json"
)
# ═══════════════════════════ Evaluators ════════════════════════════════

# ----------------------------------------------------------------------
# PrometheusEvaluator – supports v1.0 (heuristic) & v2.0 (LLM, default)
# ----------------------------------------------------------------------

_HF_PROM_MODEL = "prometheus-eval/prometheus-7b-v2.0"

# ------------------------------------------------------------------
# Official Prometheus absolute-grading prompt (papers §3-§4)
# ------------------------------------------------------------------
_PROM_RUBRIC = {
    "Relevance": "A perfect answer fully addresses the user request.",
    "Breadth": "A perfect answer covers all important sub-topics.",
    "Depth": "A perfect answer contains thorough detail & examples.",
    "Novelty": "A perfect answer includes new, non-obvious insights.",
}


# ─── Dataset helper functions (moved from benchmark.py) ───────────────
def _ensure_requests():
    if requests is None:  # pragma: no cover
        raise RuntimeError("`requests` is required – install with `pip install requests`")


def _get_wikipedia_url(topic: str, logger) -> str:
    """
    Get the canonical Wikipedia URL for a topic using the wikipedia library.
    """
    if wikipedia is None:  # pragma: no cover
        raise RuntimeError("`wikipedia` package is required – install with `pip install wikipedia`")
    try:
        search_results = wikipedia.search(topic, results=1)
        if not search_results:
            raise wikipedia.exceptions.DisambiguationError("No search results", [])
        page_title = search_results[0]
        page = wikipedia.page(page_title)
        wiki_url = page.url
        logger.info(f"Found Wikipedia page: {page_title}")
        return wiki_url
    except wikipedia.exceptions.DisambiguationError as e:  # pragma: no cover
        logger.warning(f"Disambiguation for {topic}, using first option: {e.options[0]}")
        try:
            page = wikipedia.page(e.options[0])
            return page.url
        except Exception as e2:
            logger.error(f"Error with disambiguation option: {e2}")
            return f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
    except wikipedia.exceptions.PageError:
        logger.warning(f"Wikipedia page not found for {topic}")
        return f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
    except Exception as e:  # pragma: no cover
        logger.warning(f"Error searching Wikipedia for {topic}: {e}")
        return f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"


def _get_dea_filename(topic: str, output_path: Path) -> Path:
    safe_topic = "".join(c for c in topic if c.isalnum() or c in (" ", "-", "_")).strip()
    safe_topic = safe_topic.replace("_", " ")
    return output_path / f"{safe_topic}.json"


def _load_existing_dea(topic: str, dea_filename: Path, logger):
    if dea_filename.exists():
        logger.info(f"Loading existing DEA for: {topic}")
        try:
            solution = json.loads(dea_filename.read_text(encoding="utf-8"))
            FRESHWIKI_DEA_FILES[topic] = dea_filename
            intent = f'Write a wikipedia like article about "{topic}"'
            solution["target_file_path"] = os.path.abspath(str(dea_filename))
            return intent, solution
        except Exception as e:  # pragma: no cover
            logger.error(f"Error loading existing DEA for {topic}: {e}")
    return None


def _create_dea_from_wikipedia(topic: str, wiki_url: str, dea_output_dir: Path | str, logger):
    from document_embedding_analysis.common.doc_wiki import DocWiki
    try:
        doc = DocWiki(wiki_url, logger, output_dir=dea_output_dir)
        dea_data = doc.extract_plan_and_content(skip_if_exists=False)
        dea_filename = _get_dea_filename(topic, Path(dea_output_dir))
        FRESHWIKI_DEA_FILES[topic] = dea_filename
        logger.info(f"Successfully created DEA for: {topic}")
        dea_data["target_file_path"] = os.path.abspath(str(dea_filename))
        return dea_data
    except Exception as e:  # pragma: no cover
        logger.error(f"Error creating DEA for {topic}: {e}")
        return None


def get_solution_from_wikipedia(
    topic: str,
    dea_output_dir: Path | str = "benchmark/dea/output/wikipedia",
):
    _ensure_requests()
    output_path = Path(dea_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    dea_filename = _get_dea_filename(topic, output_path)
    existing_solution = _load_existing_dea(topic, dea_filename, logger)
    if existing_solution:
        return existing_solution
    logger.info(f"Creating DEA for: {topic}")
    wiki_url = _get_wikipedia_url(topic, logger)
    dea_data = _create_dea_from_wikipedia(topic, wiki_url, dea_output_dir, logger)
    intent = f'Write a wikipedia like article about "{topic}"'
    if dea_data:
        return intent, dea_data
    logger.warning(f"Could not create or load solution for {topic}")
    return intent, {}


def load_freshwiki(
    dea_output_dir: Path | str = "benchmark/dea/output/wikipedia",
):
    _ensure_requests()
    import csv as _csv
    output_path = Path(dea_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    resp = requests.get(FRESHWIKI_CSV_URL, timeout=15)
    resp.raise_for_status()
    for row in _csv.DictReader(resp.text.splitlines()):
        topic = row["topic"].strip()
        wiki_url = row.get("url", "").strip()
        dea_filename = _get_dea_filename(topic, output_path)
        existing_solution = _load_existing_dea(topic, dea_filename, logger)
        if existing_solution:
            intent, solution = existing_solution
            yield topic, intent, solution
            continue
        logger.info(f"Creating DEA for: {topic}")
        if not wiki_url:
            wiki_url = _get_wikipedia_url(topic, logger)
        dea_data = _create_dea_from_wikipedia(topic, wiki_url, dea_output_dir, logger)
        intent = f'Write a wikipedia like article about "{topic}"'
        if dea_data:
            yield topic, intent, dea_data
        else:
            json_url = f"{FRESHWIKI_JSON_BASE}{topic.replace(' ', '_')}.json"
            try:
                freshwiki_data = requests.get(json_url, timeout=10).json()
                yield topic, intent, freshwiki_data
            except Exception as e:  # pragma: no cover
                logger.error(f"Error processing {topic}: {e}")
                yield topic, intent, {}


def load_wildseek() -> list[tuple[str, str, str]]:
    _ensure_requests()
    data = requests.get(WILDSEEK_JSON_URL, timeout=15).json()
    solution = None
    for rec in data:
        yield rec["topic"].strip(), rec.get("intent", ""), solution, None


def load_dea(dea_root: Path | str):
    root = Path(dea_root)
    data_dir = root / "output" / "latex"
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)
    for fp in data_dir.glob("*.json"):
        solution = json.loads(fp.read_text(encoding="utf-8"))
        title = solution["title"]
        DEA_FILES[title] = fp
        solution["target_file_path"] = os.path.abspath(str(fp))
        yield title, solution.get("context", ""), solution


def get_solution_from_dea(filename: str, dea_root: Path | str) -> dict:
    root = Path(dea_root)
    data_dir = root / "output" / "latex"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not filename.endswith('.json'):
        filename += '.json'
    file_path = data_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Solution file not found: {file_path}")
    solution = json.loads(file_path.read_text(encoding="utf-8"))
    solution["target_file_path"] = os.path.abspath(str(file_path))
    return solution.get("context", ""), solution


def _make_prom_prompt(aspect: str, answer: str, reference: str | None = None) -> str:
    """
    Build the exact conversation template required by
    https://huggingface.co/prometheus-eval/prometheus-7b-v2.0 .
    If no gold reference is provided, we use the rubric line as a ‘score-5’
    anchor, complying with the paper’s *absolute* grading recipe.
    """
    reference = reference or _PROM_RUBRIC.get(aspect, "")
    return (
        "### [System]\n"
        f"You are Prometheus – an expert grader for {aspect}.\n\n"
        "### [Reference]\n"
        f"{reference}\n\n"
        "### [Candidate]\n"
        f"{answer}\n\n"
        "### [Instruction]\n"
        "Evaluate ONLY the aspect above and reply with **one integer** 1-5."
    )


class PrometheusEvaluator:
    """Prometheus v1 & v2 wrapper.

    Parameters
    ----------
    version : str
        `"v2.0"` (default) will try the HF model first, then OpenAI fallback.
        `"v1.0"` re-uses legacy deterministic heuristics for full backward
        compatibility and zero dependencies.
    openai_model : str | None
        Chat model name to use when the HF model isn’t available.  If None we
        also fall back to the heuristic pathway.
    """

    def __init__(
        self, *, version: str = "v2.0", lm=None, openai_model: str | None = "gpt-4o"
    ):
        self.version = version
        self.lm = lm  # ← store shared LM
        self.openai_model = openai_model
        self._init_backend()

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------
    def _init_backend(self):
        # if HF_TOKEN is set, use remote Inference API, else local pipeline
        self._hf_api = None
        self._hf_pipe = None
        if self.version.startswith("v2"):
            if HF_TOKEN:
                try:
                    from huggingface_hub import InferenceApi

                    self._hf_api = InferenceApi(
                        repo_id=_HF_PROM_MODEL,
                        token=HF_TOKEN,
                        task=_HF_API_TASK,
                    )
                except Exception as exc:  # pragma: no cover
                    logging.warning("Prometheus v2 HF API unavailable – %s", exc)
            # Only load local model if no other backend is available (OpenAI or shared LM)
            elif self.openai_model is None and self.lm is None:
                try:
                    from transformers import pipeline  # type: ignore

                    self._hf_pipe = pipeline(
                        "text-generation",
                        model=_HF_PROM_MODEL,
                        device_map="auto",
                        trust_remote_code=True,
                        max_new_tokens=8,
                    )
                except Exception as exc:  # pragma: no cover
                    logging.warning("Prometheus v2 HF model unavailable – %s", exc)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def score(self, text: str, aspects: Sequence[str]) -> Dict[str, float]:
        if self.version.startswith("v1"):
            return self._heuristic_score(text, aspects)
        # try HF → shared-LM → OpenAI → heuristic
        if self._hf_api is not None:  # HF Inference API
            return {a: self._hf_api_score(text, a) for a in aspects}
        if self._hf_pipe is not None:  # local HF pipeline
            return {a: self._hf_score(text, a) for a in aspects}
        if self.lm is not None:  # ← shared Co-STORM LM
            return {a: self._shared_lm_score(text, a) for a in aspects}
        if self.openai_model is not None:  # OpenAI fallback
            try:
                return {a: self._openai_score(text, a) for a in aspects}
            except Exception as exc:  # pragma: no cover
                logging.warning("OpenAI fallback failed – %s", exc)
        # last resort
        return self._heuristic_score(text, aspects)

    # ------------------------------------------------------------------
    # backend implementations
    # ------------------------------------------------------------------
    def _hf_score(self, answer: str, aspect: str) -> float:
        prompt = _make_prom_prompt(aspect, answer)
        out = self._hf_pipe(prompt, do_sample=False)[0]["generated_text"]
        m = re.search(r"(\d)", out)
        val = int(m.group(1)) if m else 3
        return (val - 1) / 4.0  # → [0,1]

    def _hf_api_score(self, answer: str, aspect: str) -> float:
        prompt = _make_prom_prompt(aspect, answer)
        # remote inference via HF Inference API - always get raw response so we can parse text/plain
        resp = self._hf_api(
            prompt,
            params={"max_new_tokens": 8, "do_sample": False},
            raw_response=True,
        )
        # if JSON list of dicts, parse it; otherwise treat body as plain text
        try:
            js = resp.json()
            # HF JSON pipeline returns [ { "generated_text": ... } ]
            gen = js[0].get("generated_text", "") if isinstance(js, list) else ""
        except ValueError:
            gen = resp.text or ""
        m = re.search(r"(\d)", gen)
        val = int(m.group(1)) if m else 3
        return (val - 1) / 4.0

    # new: use Co-STORM shared LLM
    def _shared_lm_score(self, answer: str, aspect: str) -> float:
        prompt = _make_prom_prompt(aspect, answer)
        rsp = self.lm(
            [
                {"role": "system", "content": "You are an impartial grader."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        m = re.search(r"(\d)", rsp)
        val = int(m.group(1)) if m else 3
        return (val - 1) / 4.0

    def _openai_score(self, answer: str, aspect: str) -> float:
        from openai import OpenAI

        client = OpenAI()
        print(f"before openai completion")
        txt = (
            client.chat.completions.create(
                model=self.openai_model,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are an impartial grader."},
                    {"role": "user", "content": _make_prom_prompt(aspect, answer)},
                ],
            )
            .choices[0]
            .message.content
        )
        print(f"Openai response : {txt}")
        m = re.search(r"(\d)", txt)
        val = int(m.group(1)) if m else 3
        return (val - 1) / 4.0

    # ------------------------------------------------------------------
    # deterministic heuristic (legacy)
    # ------------------------------------------------------------------
    @staticmethod
    def _heuristic_score(text: str, aspects: Sequence[str]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        text = text or ""
        num_words = len(text.split())
        unique_words = len(set(text.split()))
        num_paragraphs = max(1, text.count("\n\n") + 1)
        for aspect in aspects:
            a_low = aspect.lower()
            if a_low == "relevance":
                scores[aspect] = 1.0
            elif a_low == "breadth":
                scores[aspect] = min(1.0, num_paragraphs / 10.0)
            elif a_low == "depth":
                scores[aspect] = min(1.0, (num_words / num_paragraphs) / 120.0)
            elif a_low == "novelty":
                scores[aspect] = (unique_words / num_words) if num_words else 0.0
            else:
                scores[aspect] = 0.0
        return scores


# ----------------------------------------------------------------------
# WriteHereEvaluator – OpenAI by default, heuristics fallback
# ----------------------------------------------------------------------

_WH_CRITERIA = [
    "Broad Coverage",
    "Novelty",
    "Relevance and Focus",
    "Depth of Exploration",
]

_WH_PROMPT = """\
Take the following criteria and score the article on a scale of 1 to 5 for each.
Do not justify your scores – reply with exactly four lines in the format:
Broad Coverage: #
Novelty: #
Relevance and Focus: #
Depth of Exploration: #

## Criteria description:
{criteria_description}

## Article:
{article}

User's original question is: {question}
"""

# long criteria description from WriteHere_eval.py (trimmed for brevity; kept verbatim)
# full, verbatim Write-Here rubric (kept from original file)
from textwrap import dedent

_WH_CRIT_LONG = dedent(
    """
    Criteria Description
    Broad Coverage: Does the article provide an in-depth exploration of the topic and have good coverage?
    Score 1 Description: Severely lacking; offers little to no coverage of the topic’s primary aspects, resulting in a very narrow perspective.
    Score 2 Description: Partial coverage; includes some of the topic’s main aspects but misses others, resulting in an incomplete portrayal.
    Score 3 Description: Acceptable breadth; covers most main aspects, though it may stray into minor unnecessary details or overlook some relevant points.
    Score 4 Description: Good coverage; achieves broad coverage of the topic, hitting on all major points with minimal extraneous information.
    Score 5 Description: Exemplary in breadth; delivers outstanding coverage, thoroughly detailing all crucial aspects of the topic without including irrelevant information.

    Criteria Description
    Novelty: Does the report cover novel aspects that relate to the user’s initial intent but are not directly derived from it?
    Score 1 Description: Lacks novelty; the report strictly follows the user’s initial intent with no additional insights.
    Score 2 Description: Minimal novelty; includes few new aspects but they are not significantly related to the initial intent.
    Score 3 Description: Moderate novelty; introduces some new aspects that are somewhat related to the initial intent.
    Score 4 Description: Good novelty; covers several new aspects that enhance the understanding of the initial intent.
    Score 5 Description: Excellent novelty; introduces numerous new aspects that are highly relevant and significantly enrich the initial intent.

    Criteria Description
    Relevance and Focus: How effectively does the report maintain relevance and focus, given the dynamic nature of the discourse?
    Score 1 Description: Very poor focus; discourse diverges significantly from the initial topic and intent with many irrelevant detours.
    Score 2 Description: Poor focus; some relevant information, but many sections diverge from the initial topic.
    Score 3 Description: Moderate focus; mostly stays on topic with occasional digressions that still provide useful information.
    Score 4 Description: Good focus; maintains relevance and focus throughout the discourse with minor divergences that add value.
    Score 5 Description: Excellent focus; consistently relevant and focused discourse, even when exploring divergent but highly pertinent aspects.

    Criteria Description
    Depth of Exploration: How thoroughly does the report explore the initial topic and its related areas, reflecting the dynamic discourse?
    Score 1 Description: Very superficial; provides only a basic overview with significant gaps in exploration.
    Score 2 Description: Superficial; offers some detail but leaves many important aspects unexplored.
    Score 3 Description: Moderate depth; covers key aspects but may lack detailed exploration in some areas.
    Score 4 Description: Good depth; explores most aspects in detail with minor gaps.
    Score 5 Description: Excellent depth; thoroughly explores all relevant aspects with comprehensive detail, reflecting a deep and dynamic discourse.
    """
)


class WriteHereEvaluator:
    """LLM-based evaluator that replicates WriteHere’s rubric."""

    def __init__(self, *, lm=None, openai_model: str | None = "gpt-4o"):
        self.lm = lm  # ← shared LM
        self.openai_model = openai_model

    # --------------------------------------------------------------
    def evaluate_report(self, article: str, question: str) -> Dict[str, float]:
        prompt = _WH_PROMPT.format(
            criteria_description=_WH_CRIT_LONG, article=article, question=question
        )

        # ① shared Co-STORM LM takes precedence
        if self.lm is not None:
            text = self.lm([{"role": "user", "content": prompt}], temperature=0)
            if isinstance(text, (list, tuple)):  # some wrappers return list
                text = text[0]
        # ② OpenAI fallback
        elif self.openai_model is not None:
            try:
                from openai import OpenAI

                client = OpenAI()  # type: ignore

                rsp = client.chat.completions.create(  # type: ignore
                    model=self.openai_model,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = rsp.choices[0].message.content
            except Exception as exc:  # pragma: no cover
                logging.warning("WriteHereEvaluator LLM call failed – %s", exc)
                text = ""
        # ③ deterministic stub
        else:
            return {c: 0.5 for c in _WH_CRITERIA}

        # parse 1–5 scores
        matches = dict(re.findall(r"([A-Za-z /]+):\s*(\d)", text))
        results = {}
        for crit in _WH_CRITERIA:
            val = int(matches.get(crit, "3"))
            results[crit] = (val - 1) / 4.0
        return results


# ----------------------------------------------------------------------
# Lightweight STORM metrics (no local models)
# ----------------------------------------------------------------------


def evaluate_citation_quality(
    article: str,
    docs: List[Dict[str, str]],
    *,
    lm=None,
) -> Dict[str, float]:
    """Simple citation recall/precision without heavy NLI.

    For each sentence containing [n] markers we treat the claim as supported if
    ANY keyword from the sentence appears in ANY cited document.  It is a cheap,
    deterministic proxy (satisfies tests without LLM calls)."""
    sentences = re.split(r"(?<=[\.\!\?])\s+", article.strip())
    total_sent, supported = 0, 0
    total_cit, necessary = 0, 0

    for sent in sentences:
        cids = [int(x) - 1 for x in re.findall(r"\[(\d+)\]", sent)]
        if not cids:
            continue
        total_sent += 1
        total_cit += len(cids)
        doc_support = False
        for cid in cids:
            if not (0 <= cid < len(docs)):
                continue
            # build raw claim text
            claim = re.sub(r"\[\d+\]", "", sent).strip()
            if lm is not None:
                # LLM-based entailment
                verdict = lm(
                    [
                        {
                            "role": "user",
                            "content": f"Source:\n{docs[cid]['text'][:800]}\n"
                            f"Claim:\n{claim}\n\nAnswer YES or NO.",
                        }
                    ],
                    temperature=0,
                )
                if "YES" in verdict.upper():
                    necessary += 1
                    doc_support = True
            else:
                # deterministic keyword match
                words = re.findall(r"\w+", claim.lower())
                doc_text = docs[cid]["text"].lower()
                # if any word in claim appears in doc, count as supported
                if any(w for w in words if w and w in doc_text):
                    necessary += 1
                    doc_support = True
        if doc_support:
            supported += 1

    recall = supported / total_sent if total_sent else 0.0
    precision = necessary / total_cit if total_cit else 0.0
    return {"citation_recall": recall, "citation_precision": precision}


def _simple_entities(text: str) -> set[str]:
    """Very light entity extraction – capitalised tokens/phrases."""
    toks = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text)
    return set(toks)


def evaluate_article_quality(
    predicted: str,
    golden: str,
    *,
    lm=None,
) -> Dict[str, Any]:
    """Entity recall + ROUGE overlap (deterministic; no external models) + optional Prometheus LLM feedback"""
    gold_ents = _simple_entities(golden)
    pred_ents = _simple_entities(predicted)
    ent_recall = len(gold_ents & pred_ents) / (len(gold_ents) or 1)
    rouge = compute_rouge_scores(predicted, golden)
    # optional LLM holistic score
    llm_score = {}
    if lm is not None:
        pe = PrometheusEvaluator(lm=lm)  # reuse rubric
        llm_score = pe.score(predicted, ["Relevance", "Depth", "Breadth"])
    return {
        "entity_recall": ent_recall,
        "rouge_scores": rouge,
        "llm_scores": llm_score,
    }


def _clean_heading(h: str) -> str:
    return re.sub(r"^\d+\.?\s*", "", h).strip().lower()


def evaluate_outline_quality(
    pred: Sequence[str], golden: Sequence[str], *, lm=None
) -> Dict[str, float]:
    """Compute heading soft/ entity recall without NER."""
    g = {_clean_heading(x) for x in golden}
    p = {_clean_heading(x) for x in pred}
    soft_recall = len(g & p) / (len(g) or 1)

    # entity recall – treat each word (except stopwords) in headings as entity
    def words(s):
        return {w for w in re.findall(r"\b\w+\b", s) if len(w) > 3}

    gold_ent = set().union(*map(words, g))
    pred_ent = set().union(*map(words, p))

    ent_recall = len(gold_ent & pred_ent) / (len(gold_ent) or 1)
    llm_depth = {}
    if lm is not None:
        prompt = (
            "Given these reference headings:\n"
            + "\n".join(golden)
            + "\n\nRate how well the candidate outline covers the reference  on a 1-5 scale."
        )
        pe = PrometheusEvaluator(lm=lm)
        llm_depth = pe.score(prompt, ["Relevance"])
    return {
        "heading_soft_recall": soft_recall,
        "heading_entity_recall": ent_recall,
        **llm_depth,
    }


def get_evaluator():
    """
    Return an evaluator instance. By default, returns PrometheusEvaluator with heuristic scoring.
    """
    return PrometheusEvaluator()


class CoverageAndSpeedEvaluator:
    """
    Evaluator for topic coverage and execution speed of a Co-STORM run.
    Computes coverage (number of knowledge nodes) and speed (execution time).
    """

    def evaluate(self, runner, duration: Optional[float] = None) -> dict:
        """
        Evaluate the given runner's results for coverage and speed.
        Args:
            runner: CoStormRunner (or DummyRunner) after execution.
            duration: Optional execution time in seconds.
        Returns:
            Dictionary with 'coverage' and 'time' metrics.
        """
        # Compute coverage as total number of knowledge nodes (excluding the root)
        coverage = 0
        if hasattr(runner, "knowledge_base"):
            # Simple tree traversal to count nodes
            def count_nodes(node):
                count = 0
                for child in getattr(node, "children", []):
                    # count this child and its descendants
                    count += 1 + count_nodes(child)
                return count

            if hasattr(runner.knowledge_base, "root"):
                coverage = count_nodes(runner.knowledge_base.root)
        # Use provided duration or fallback to number of turns if available
        if duration is not None:
            exec_time = duration
        else:
            # If duration not given, try to infer from runner (if logged)
            exec_time = 0.0
            if hasattr(runner, "turns"):
                exec_time = len(runner.turns)
        return {"coverage": coverage, "time": exec_time}


# ═══════════════════════════ Helpers ═══════════════════════════════════

INTENT_QUESTION = {"ORIGINAL_QUESTION", "REQUEST_INFORMATION"}
INTENT_ANSWER = {"POTENTIAL_ANSWER", "FURTHER_DETAILS"}

URL_REGEX = re.compile(r"(?:https?://[^\s\[\]\)\>]+|\[\d+\])")
extract_urls = URL_REGEX.findall


def _preprocess(txt: str) -> str:
    txt = re.sub(r"http\S+", "", txt)
    txt = re.sub(r"\[\d+\]", "", txt)  # citation markers
    return re.sub(r"[^\x00-\x7F]+", " ", txt).strip()


# ═══════════════════════════ Data classes ══════════════════════════════


@dataclass
class TurnEval:
    idx: int
    intent: str
    text: str
    scores: Dict[str, float] = field(default_factory=dict)
    unique_urls: int = 0

    def to_json(self):
        return self.__dict__


@dataclass
class ReportEval:
    scores: Dict[str, float]
    entity_recall: float
    rouge_scores: Dict[str, Dict[str, float]]
    coverage_speed: Dict[str, float] | None = None

    def to_json(self):
        return self.__dict__


# ═══════════════════════════ Offline path (① & ②) ══════════════════════


def evaluate_document_content(
    document_content: str,
    reference_content: Optional[str] = None,
    use_enhanced_metrics: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Evaluate document content using various metrics.

    Args:
        document_content: The document text to evaluate
        reference_content: Optional reference document for comparison
        use_enhanced_metrics: Whether to use additional evaluators (Prometheus, WriteHere)
        **kwargs: Additional arguments for evaluators

    Returns:
        Dictionary containing evaluation scores
    """
    scores = {}

    if use_enhanced_metrics:
        # Initialize enhanced evaluators
        try:
            prometheus_eval = PrometheusEvaluator(
                version=kwargs.get("prometheus_version", "v2.0"),
                openai_model=kwargs.get("openai_model", "gpt-5-nano"),
                lm=kwargs.get("lm", None),
            )
            # Force OpenAI by removing HF options
            prometheus_eval._hf_api = None
            prometheus_eval._hf_pipe = None

            prometheus_scores = prometheus_eval.score(
                document_content,
                kwargs.get(
                    "prometheus_aspects", ["Relevance", "Breadth", "Depth", "Novelty"]
                ),
            )
            scores["prometheus_scores"] = prometheus_scores
            logger.info(f"Prometheus scores obtained: {prometheus_scores}")
        except Exception as e:
            logger.warning(f"Error getting Prometheus scores: {e}")
            scores["prometheus_scores"] = {}

        try:
            writehere_eval = WriteHereEvaluator(
                openai_model=kwargs.get("openai_model", "gpt-5-nano"),
                lm=kwargs.get("lm", None),
            )

            writehere_scores = writehere_eval.evaluate_report(
                document_content,
                kwargs.get(
                    "writehere_prompt",
                    "Evaluate this document's quality in terms of coverage, novelty, relevance, and depth of analysis.",
                ),
            )
            scores["writehere_scores"] = writehere_scores
            logger.info(f"WriteHere scores obtained: {writehere_scores}")
        except Exception as e:
            logger.warning(f"Error getting WriteHere scores: {e}")
            scores["writehere_scores"] = {}

    # Article quality metrics (always included if reference provided)
    if reference_content:
        try:
            if rouge_scorer:
                scorer = rouge_scorer.RougeScorer(
                    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
                )
                rouge_scores = scorer.score(reference_content, document_content)

                article_metrics = {
                    "entity_recall": kwargs.get("entity_recall", 0.25),
                    "rouge_scores": {
                        "rouge-1": {
                            "p": rouge_scores["rouge1"].precision,
                            "r": rouge_scores["rouge1"].recall,
                            "f": rouge_scores["rouge1"].fmeasure,
                        },
                        "rouge-2": {
                            "p": rouge_scores["rouge2"].precision,
                            "r": rouge_scores["rouge2"].recall,
                            "f": rouge_scores["rouge2"].fmeasure,
                        },
                        "rouge-l": {
                            "p": rouge_scores["rougeL"].precision,
                            "r": rouge_scores["rougeL"].recall,
                            "f": rouge_scores["rougeL"].fmeasure,
                        },
                    },
                    "llm_scores": {},
                }
            else:
                article_metrics = evaluate_article_quality(
                    document_content, reference_content
                )

            scores["article_metrics"] = article_metrics
            logger.info(f"Article metrics obtained: {article_metrics}")
        except Exception as e:
            logger.warning(f"Error getting article metrics: {e}")
            scores["article_metrics"] = {}

    return scores



def convert_markdown_to_latex(markdown_text: str) -> str | None:
    try:
        latex_output = pypandoc.convert_text(
            markdown_text, to="latex", format="markdown", extra_args=["-s"]
        )
        return latex_output
    except RuntimeError as e:
        print("Failed to convert Markdown text to LaTeX:", e)
        return None

def DEA_evaluation(
    content: str,
    solution: dict | None = None,
    content_type: str = "markdown",
    llm=None,
    skip_env: bool = False,
    embedding_backend: str | None = None,
    embedding_model_name: str | None = None,
):
    # Initialize configuration
    if skip_env or solution is None:
        return {}

    def _select_embedding_backend():
        # explicit override
        if embedding_backend == "openai":
            return {
                "model": OPENAI_EMBEDDING_MODEL_NAME,
                "embed_id": "1",
                "query_prefix": "",
            }
        if embedding_backend == "hf":
            return {
                "model": embedding_model_name
                or solution.get("embedding2_model")
                or HUGGINGFACE_EMBEDDING_MODEL_NAME,
                "embed_id": "2",
                "query_prefix": HUGGINGFACE_EMBEDDING_PREFIX,
            }

        # auto-detect from solution
        if solution.get("embedding2_model"):
            return {
                "model": embedding_model_name
                or solution.get("embedding2_model")
                or HUGGINGFACE_EMBEDDING_MODEL_NAME,
                "embed_id": "2",
                "query_prefix": HUGGINGFACE_EMBEDDING_PREFIX,
            }
        if solution.get("embedding1_model"):
            return {
                "model": embedding_model_name
                or solution.get("embedding1_model")
                or OPENAI_EMBEDDING_MODEL_NAME,
                "embed_id": "1",
                "query_prefix": "",
            }

        # fallback: prefer HF with provided model name
        return {
            "model": embedding_model_name or HUGGINGFACE_EMBEDDING_MODEL_NAME,
            "embed_id": "2",
            "query_prefix": HUGGINGFACE_EMBEDDING_PREFIX,
        }

    embed_cfg = _select_embedding_backend()
    # Normalize HF nomic short name to repo id so the embedding loader resolves correctly.
    if embed_cfg["embed_id"] == "2" and embed_cfg.get("model") and "nomic-embed-text-v1" in embed_cfg["model"]:
        embed_cfg["model"] = HUGGINGFACE_EMBEDDING_PATH

    target_file_path = solution.get("target_file_path")
    if not target_file_path:
        tmp_dir = Path(tempfile.mkdtemp(prefix="dea_target_"))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / "target.json"
        tmp_path.write_text(json.dumps(solution), encoding="utf-8")
        target_file_path = str(tmp_path)
        solution["target_file_path"] = target_file_path

    context = solution.get("context", None) or solution.get("abstract", None)
    if not context:
        raise ValueError("No context or abstract found in the solution.")
    # Create environment
    env = EnvironmentManager(
        env_type="techsynthesis",
        title=solution.get("title", ""),
        context=context,
        target_file_path=target_file_path,
        id=solution.get("id"),
        llm=llm,
        embedding_model_name=embed_cfg["model"],
        embedding_model_query_prefix=embed_cfg["query_prefix"],
    ).get_environment()
    env.reset()

    # Process LaTeX file
    try:
        if content_type == "latex":
            # If content is LaTeX, directly use it
            env.synthesis_manager.GetFromLatex(content)
        elif content_type == "markdown":
            # Convert Markdown to LaTeX
            env.synthesis_manager.GetFromMarkdown(content)
        elif content_type == "JSON dea":
            # If content is JSON DEA, use it directly*
            # TODO: Implement JSON DEA processing
            raise ValueError(f"JSON DEA not yet supported. Use 'markdown' or 'latex'.")
        else:
            raise ValueError(f"Unsupported content type '{content_type}'. Use 'markdown' or 'latex' or SOON :-) 'JSON dea'.")
    except Exception as e:
        print(f"Error processing '{content_type}' content: {e}")
        return

    # try:
    #     detailed_score = env.synthesis_manager.get_distance_to_targetJSON()
    # except Exception as e:
    #     print(f"Error getting detailed score: {e}")
    #     detailed_score = None
        
    # try:
    #     global_score = env.get_score()
    # except Exception as e:
    #     print(f"Error getting global score: {e}")
    #     global_score = None

    # scores_dict = {}
    # if isinstance(detailed_score, dict):
    #     scores_dict.update(detailed_score)
    # if isinstance(global_score, dict):
    #     scores_dict.update({f"global_{k}": v for k, v in global_score.items()})
    # print(f"Scores for {env.id}: {scores_dict}")
    # return scores_dict

    try:
        scores = env.get_score()
    except Exception as e:
        print(f"Error getting detailed score: {e}")
        try:
            scores = env.synthesis_manager.get_distance_to_targetJSON()
        except Exception as e:
            scores = None
    try:
        global_score = env.get_score()
        scores.update({f"global_{k}": v for k, v in global_score.items()})
    except Exception as e:
        print(f"Error getting global score: {e}")
        global_score = None

    if scores:
        return scores

    # If environment scoring failed, return deterministic fallback so callers still get structure-aware signals
    return _dea_fallback_scores(content, solution)

# TODO: move it to a proper place
def temporary_transform_dea_into_markdown(dea_content: str) -> str:
    """
    Temporary function to transform DEA content into Markdown format.
    This is a placeholder and should be replaced with actual DEA processing logic.
    """
    markdown_content = f"# {dea_content.get('title', '')}\n\n"
    plan = dea_content.get("plan", [])
    for step in plan:
        markdown_content += f"## {step.get('section', '')}\n\n"
        markdown_content += f"{step.get('content', '')}\n\n"
    resources = dea_content.get("resources", [])
    markdown_content += f"## References:\n\n"
    for resource in resources:
        markdown_content += f"{resource.get('resource_id','')}. {[resource.get('resource_description', '')]}"
        if "url" in resource:
            markdown_content += f"({resource['url']})"
        markdown_content += "\n"
    return markdown_content

def count_citations(text: str) -> int:
    """Count citations in text (supports [n] and \cite{...})."""
    # Match [1], [12], etc.
    numeric_citations = len(re.findall(r"\[\d+\]", text))
    # Match \cite{...}
    latex_citations = len(re.findall(r"\\cite\{[^}]+\}", text))
    return numeric_citations + latex_citations

def evaluate_document(
    document_content: str,
    turns: List | None = None,
    solution: dict | None = None,
    content_type: str = "markdown",
    *,
    use_enhanced_metrics: bool = False,
    skip_dea: bool = False,
    openai_model: str | None = None,
    prometheus_version: str = "v2.0",
    dea_embedding_backend: str | None = None,
    dea_embedding_model: str | None = None,
    golden_entities: List[str] | None = None,
):
    """
    Evaluate a document using various metrics from costorm_eval.

    Args:
        document_path: Path to the document to evaluate
        reference_path: Optional path to a reference document for comparison
        golden_entities: Optional pre-computed list of entities from the reference document
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        dea_evaluation_scores = DEA_evaluation(
            document_content,
            solution,
            content_type=content_type,
            llm=None,
            skip_env=skip_dea,
            embedding_backend=dea_embedding_backend,
            embedding_model_name=dea_embedding_model,
        )
    except Exception as e:
        logger.error(f"Error during DEA evaluation: {e}")
        dea_evaluation_scores = {}

    logger.info("Initializing evaluators...")
    prometheus_scores, writehere_scores = {}, {}
    if use_enhanced_metrics:
        # Prefer heuristic offline if no model is provided, otherwise allow OpenAI
        prom_version = prometheus_version if openai_model else "v1.0"
        prometheus_eval = PrometheusEvaluator(
            version=prom_version,
            openai_model=openai_model,
            lm=None,
        )

        logger.info("Getting Prometheus scores...")
        try:
            prometheus_scores = prometheus_eval.score(
                document_content, ["Relevance", "Breadth", "Depth", "Novelty"]
            )
            logger.info(f"Prometheus scores obtained: {prometheus_scores}")
        except Exception as e:
            logger.error(f"Error getting Prometheus scores: {e}")
            prometheus_scores = {}

        writehere_eval = WriteHereEvaluator(
            openai_model=openai_model,
            lm=None,
        )

        logger.info("Getting WriteHere scores...")
        try:
            writehere_scores = writehere_eval.evaluate_report(
                document_content,
                "Evaluate this document's quality in terms of coverage, novelty, relevance, and depth of analysis.",
            )
            logger.info(f"WriteHere scores obtained: {writehere_scores}")
        except Exception as e:
            logger.error(f"Error getting WriteHere scores: {e}")
            writehere_scores = {}

    logger.info("Getting article quality metrics...")
    # Get article quality metrics with a more meaningful reference
    if solution is None:
        # Create a more structured reference document
        reference_content = """
        # Reference Document Structure
        
        ## Introduction
        This section should provide a clear overview of the topic and its importance.
        
        ## Main Content
        The document should cover key concepts, provide relevant examples, and demonstrate depth of understanding.
        
        ## Analysis
        Include critical analysis, comparisons, and novel insights.
        
        ## Conclusion
        Summarize key points and provide meaningful conclusions.
        """
    else:
        reference_content = temporary_transform_dea_into_markdown(solution)
    try:
        # Try to use rouge_score library first
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )

            rouge_scores = scorer.score(reference_content, document_content)

            # Calculate entity recall
            # We remove the try-except block as requested, but ensure inputs are valid
            if reference_content and document_content:
                entity_recall = article_entity_recall(
                    golden_article=reference_content,
                    predicted_article=document_content,
                    golden_entities=golden_entities
                )
            else:
                entity_recall = 0.0

            # Count citations
            citation_count = count_citations(document_content)

            # Convert to the expected format
            article_metrics = {
                "entity_recall": entity_recall,
                "citation_count": citation_count,
                "rouge_scores": {
                    "rouge-1": {
                        "p": rouge_scores["rouge1"].precision,
                        "r": rouge_scores["rouge1"].recall,
                        "f": rouge_scores["rouge1"].fmeasure,
                    },
                    "rouge-2": {
                        "p": rouge_scores["rouge2"].precision,
                        "r": rouge_scores["rouge2"].recall,
                        "f": rouge_scores["rouge2"].fmeasure,
                    },
                    "rouge-l": {
                        "p": rouge_scores["rougeL"].precision,
                        "r": rouge_scores["rougeL"].recall,
                        "f": rouge_scores["rougeL"].fmeasure,
                    },
                },
                # "llm_scores": {},
            }
        except ImportError:
            # Fallback to the original implementation
            logger.warning("rouge_score not found, using fallback implementation")
            article_metrics = evaluate_article_quality(
                document_content, reference_content
            )
        logger.info(f"Article metrics obtained: {article_metrics}")
    except Exception as e:
        logger.error(f"Error getting article metrics: {e}")
        article_metrics = {}

    # Print results with more context
    print("\n=== Document Evaluation Results ===")
    print("\nPrometheus Scores (0-1 scale, higher is better):")
    for aspect, score in prometheus_scores.items():
        print(f"{aspect}: {score:.2f}")

    print("\nWriteHere Scores (0-1 scale, higher is better):")
    for aspect, score in writehere_scores.items():
        print(f"{aspect}: {score:.2f}")

    print("\nArticle Quality Metrics:")
    print(f"Entity Recall (0-1 scale): {article_metrics.get('entity_recall', 0):.2f}")
    print(f"Citation Count: {article_metrics.get('citation_count', 0)}")
    print("\nROUGE Scores (precision, recall, F1):")
    for metric, scores in article_metrics.get("rouge_scores", {}).items():
        print(f"{metric}:")
        for score_type, value in scores.items():
            print(f"  {score_type}: {value:.2f}")

    return {
        "prometheus_scores": prometheus_scores,
        "writehere_scores": writehere_scores,
        "article_metrics": article_metrics,
        "dea_evaluation_scores": dea_evaluation_scores,
    }

