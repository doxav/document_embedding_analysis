# Code adapted from Storm/CoStorm, with heavyweight NLP dependencies loaded lazily.
import math
import re
from typing import List, Optional


_tagger = None
_encoder = None


def _simple_entities(text: str) -> list[str]:
    """Lightweight fallback entity extractor.

    Keeps import-time dependencies small. It extracts capitalized tokens/phrases
    and normalizes them to lowercase.
    """
    if not text:
        return []
    entities = re.findall(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b", text)
    return list({e.lower() for e in entities})


def _get_tagger():
    """Load Flair NER only when entity extraction really needs it."""
    global _tagger
    if _tagger is None:
        try:
            from flair.nn import Classifier
        except ImportError as exc:
            raise RuntimeError(
                "`flair` is required for neural NER. Install it only if you need "
                "Flair-based entity recall, or rely on the lightweight fallback."
            ) from exc
        _tagger = Classifier.load("ner")
    return _tagger


def _get_encoder():
    """Load sentence-transformers only when soft heading recall really needs it."""
    global _encoder
    if _encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "`sentence-transformers` is required for embedding-based heading recall. "
                "Install it only if this metric is needed."
            ) from exc
        _encoder = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    return _encoder


def card(l):
    """Soft cardinality.

    Uses sentence-transformers when available. Falls back to exact set cardinality
    when the local embedding backend is unavailable.
    """
    values = list(l)
    if not values:
        return 0.0

    try:
        from sklearn.metrics.pairwise import cosine_similarity

        encoded_l = _get_encoder().encode(values)
        cosine_sim = cosine_similarity(encoded_l)
        soft_count = 1 / cosine_sim.sum(axis=1)
        return soft_count.sum()
    except Exception:
        return float(len(set(values)))


def heading_soft_recall(golden_headings: List[str], predicted_headings: List[str]):
    """
    Given golden headings and predicted headings, compute soft recall.
    Falls back to exact set recall when embedding dependencies are unavailable.
    """
    g = set(golden_headings or [])
    p = set(predicted_headings or [])

    if len(g) == 0:
        return 1.0
    if len(p) == 0:
        return 0.0

    try:
        card_g = card(g)
        card_p = card(p)
        card_intersection = card_g + card_p - card(g.union(p))
        if card_g == 0:
            return 1.0
        return card_intersection / card_g
    except Exception:
        return len(g.intersection(p)) / len(g)


def extract_entities_from_list(l):
    """Extract entities from a list of strings.

    Uses Flair if installed; otherwise uses a deterministic lightweight regex
    fallback. This keeps importing common.metrics cheap.
    """
    texts = [s for s in (l or []) if isinstance(s, str) and s.strip()]
    if not texts:
        return []

    try:
        from flair.data import Sentence

        sentences = [Sentence(s) for s in texts]
        tagger = _get_tagger()
        tagger.predict(sentences, mini_batch_size=32, verbose=True)

        entities = []
        for sent in sentences:
            entities.extend([e.text for e in sent.get_spans("ner")])

        return list({e.lower() for e in entities})
    except Exception:
        entities = []
        for text in texts:
            entities.extend(_simple_entities(text))
        return list(set(entities))


def heading_entity_recall(
    golden_entities: Optional[List[str]] = None,
    predicted_entities: Optional[List[str]] = None,
    golden_headings: Optional[List[str]] = None,
    predicted_headings: Optional[List[str]] = None,
):
    """
    Given golden entities and predicted entities, compute entity recall.
    If explicit entities are not provided, extract them from headings.
    """
    if golden_entities is None:
        assert golden_headings is not None, (
            "golden_headings and golden_entities cannot both be None."
        )
        golden_entities = extract_entities_from_list(golden_headings)

    if predicted_entities is None:
        assert predicted_headings is not None, (
            "predicted_headings and predicted_entities cannot both be None."
        )
        predicted_entities = extract_entities_from_list(predicted_headings)

    g = set(golden_entities)
    p = set(predicted_entities)

    if len(g) == 0:
        return 1
    return len(g.intersection(p)) / len(g)


def article_entity_recall(
    golden_entities: Optional[List[str]] = None,
    predicted_entities: Optional[List[str]] = None,
    golden_article: Optional[str] = None,
    predicted_article: Optional[str] = None,
):
    """
    Given golden entities and predicted entities, compute entity recall.
    If explicit entities are not provided, extract them from article text.
    """
    sentence_splitter = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"

    if golden_entities is None:
        assert golden_article is not None, (
            "golden_article and golden_entities cannot both be None."
        )
        sentences = re.split(sentence_splitter, golden_article)
        golden_entities = extract_entities_from_list(sentences)

    if predicted_entities is None:
        assert predicted_article is not None, (
            "predicted_article and predicted_entities cannot both be None."
        )
        sentences = re.split(sentence_splitter, predicted_article)
        predicted_entities = extract_entities_from_list(sentences)

    g = set(golden_entities)
    p = set(predicted_entities)

    if len(g) == 0:
        return 1
    return len(g.intersection(p)) / len(g)


def get_entities_from_article(article: str) -> List[str]:
    """Extract entities from an article string."""
    sentences = re.split(
        r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s",
        article or "",
    )
    return extract_entities_from_list(sentences)


def _fallback_rouge_scores(golden_answer: str, predicted_answer: str):
    """Small dependency-free ROUGE-like fallback.

    It is not a full ROUGE implementation; it preserves the output shape so
    offline tests and lightweight imports continue to work.
    """
    gold_tokens = re.findall(r"\w+", (golden_answer or "").lower())
    pred_tokens = re.findall(r"\w+", (predicted_answer or "").lower())

    if not gold_tokens and not pred_tokens:
        precision = recall = f1 = 1.0
    elif not gold_tokens or not pred_tokens:
        precision = recall = f1 = 0.0
    else:
        gold_counts = {}
        for tok in gold_tokens:
            gold_counts[tok] = gold_counts.get(tok, 0) + 1

        overlap = 0
        for tok in pred_tokens:
            if gold_counts.get(tok, 0) > 0:
                overlap += 1
                gold_counts[tok] -= 1

        precision = overlap / len(pred_tokens) if pred_tokens else 0.0
        recall = overlap / len(gold_tokens) if gold_tokens else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

    return {
        "ROUGE1_precision": precision,
        "ROUGE1_recall": recall,
        "ROUGE1_f1": f1,
        "ROUGEL_precision": precision,
        "ROUGEL_recall": recall,
        "ROUGEL_f1": f1,
    }


def compute_rouge_scores(golden_answer: str, predicted_answer: str):
    """
    Compute ROUGE score for given output and golden answer.

    Uses rouge_score if installed. Falls back to a lightweight token-overlap
    approximation when the dependency is absent.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return _fallback_rouge_scores(golden_answer, predicted_answer)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(golden_answer or "", predicted_answer or "")

    score_dict = {}
    for metric, metric_score in scores.items():
        score_dict[f"{metric.upper()}_precision"] = metric_score.precision
        score_dict[f"{metric.upper()}_recall"] = metric_score.recall
        score_dict[f"{metric.upper()}_f1"] = metric_score.fmeasure

    return score_dict
