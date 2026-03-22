"""Factual accuracy scoring for extraction benchmark.

Complements the existing embedding-similarity scoring in scoring.py with
metrics that measure whether the correct facts (numbers, entities, topics)
were captured, regardless of wording.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from autoreview.extraction.models import PaperExtraction

logger = structlog.get_logger()

# --- Fact extraction primitives ---

_NUMBER_RE = re.compile(r"\d+\.?\d*")

_PROPER_NOUN_RE = re.compile(r"\b[A-Z][A-Za-z]*(?:[-/][A-Z]?[A-Za-z]*)*\b")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "The",
        "This",
        "These",
        "That",
        "Those",
        "However",
        "Moreover",
        "Furthermore",
        "Additionally",
        "Our",
        "We",
        "They",
        "It",
        "Its",
        "Here",
        "There",
        "When",
        "Where",
        "While",
        "Although",
        "Despite",
        "Between",
        "Among",
        "Within",
        "Without",
        "After",
        "Before",
        "During",
        "Since",
        "Both",
        "Each",
        "Every",
        "Many",
        "Most",
        "Some",
        "Several",
        "Such",
        "Other",
        "More",
        "Less",
        "First",
        "Second",
        "Third",
        "Finally",
        "Overall",
        "Specifically",
        "Notably",
        "Importantly",
        "Interestingly",
        "Table",
        "Figure",
        "Section",
        "Appendix",
        "Results",
        "Methods",
        "Discussion",
        "Introduction",
        "Conclusion",
        "Abstract",
    }
)


def _extract_numbers(text: str | None) -> set[float]:
    """Extract all numeric values from text.

    Returns a set of floats. Percentages are returned as the bare number
    (67.6% -> 67.6). Handles integers and decimals.
    """
    if not text:
        return set()
    nums: set[float] = set()
    for m in _NUMBER_RE.finditer(text):
        try:
            nums.add(float(m.group()))
        except ValueError:
            continue
    return nums


def _extract_facts(text: str) -> set[str]:
    """Extract factual elements: numbers and proper nouns.

    Numbers are stored as their string representation (preserving
    original precision). Proper nouns are filtered against a stopword
    list to remove common English words that happen to be capitalized.
    """
    if not text:
        return set()
    facts: set[str] = set()
    for m in _NUMBER_RE.finditer(text):
        facts.add(m.group())
    for m in _PROPER_NOUN_RE.finditer(text):
        word = m.group()
        if word not in _STOPWORDS and len(word) > 1:
            facts.add(word)
    return facts


def _fact_overlap(gold_facts: set[str], pred_facts: set[str]) -> float:
    """Fraction of gold facts found in predicted facts (recall-oriented)."""
    if not gold_facts:
        return 1.0
    if not pred_facts:
        return 0.0
    matched = sum(1 for gf in gold_facts if gf in pred_facts)
    return matched / len(gold_facts)


# --- Limitation topic taxonomy ---

_LIMITATION_TOPICS: dict[str, list[str]] = {
    "sample_size": ["small sample", "limited sample", "few participants", "n=", "sample size"],
    "generalizability": [
        "not generaliz",
        "limited to",
        "single center",
        "specific to",
        "narrow scope",
    ],
    "language_bias": ["english only", "monolingual", "english-language", "single language"],
    "study_design": ["cross-sectional", "retrospective", "correlational", "no control"],
    "data_quality": [
        "self-report",
        "missing data",
        "incomplete",
        "noisy",
        "annotation quality",
    ],
    "temporal": ["outdated", "fixed date", "point in time", "snapshot", "continual learning"],
    "bias": ["selection bias", "potential bias", "confound", "unbalanced"],
    "reproducibility": ["not open-source", "not reproducible", "proprietary", "not released"],
    "scope": ["narrow scope", "beyond the scope", "did not consider", "excluded", "only consider"],
    "evaluation": ["single rater", "limited evaluation", "no human eval", "automated only"],
    "domain_specificity": ["domain-specific", "not tested on", "single domain"],
    "statistical_power": ["underpowered", "statistical power", "insufficient"],
    "cost_resources": ["expensive", "computational cost", "resource intensive"],
    "safety_ethics": ["safety", "ethical", "bias evaluation", "fairness", "equity"],
    "hallucination": ["hallucination", "factual error", "fabricat"],
    "benchmark_limitations": ["benchmark", "dataset limitation", "test set"],
    "human_subjects": ["irb", "consent", "privacy", "participant"],
    "comparison": ["limited comparison", "no baseline", "few baselines"],
    "future_work": ["future work", "further research", "remains to be"],
    "validity": ["ecological validity", "external validity", "construct validity"],
}


def _detect_limitation_topics(text: str | None) -> set[str]:
    """Detect which limitation topics are mentioned in text.

    Returns a set of topic keys (from _LIMITATION_TOPICS) that are present
    in the text. Matching is case-insensitive substring search.
    """
    if not text:
        return set()
    lower = text.lower()
    found: set[str] = set()
    for topic, keywords in _LIMITATION_TOPICS.items():
        for kw in keywords:
            if kw in lower:
                found.add(topic)
                break
    return found


# --- Per-field factual scoring functions ---


def _factual_key_findings_score(
    pred_claims: list[str],
    gold_claims: list[str],
) -> float:
    """Score key findings by factual element overlap.

    For each gold claim, finds the best-matching predicted claim by
    fact overlap, then averages across all gold claims. Returns 1.0 if
    gold_claims is empty.
    """
    if not gold_claims:
        return 1.0
    if not pred_claims:
        return 0.0

    total = 0.0
    for gold_claim in gold_claims:
        gold_facts = _extract_facts(gold_claim)
        best_overlap = 0.0
        for pred_claim in pred_claims:
            pred_facts = _extract_facts(pred_claim)
            overlap = _fact_overlap(gold_facts, pred_facts)
            if overlap > best_overlap:
                best_overlap = overlap
        total += best_overlap

    return total / len(gold_claims)


def _factual_quant_score(
    pred_quant: str | None,
    gold_quant: str | None,
) -> float:
    """Score quantitative result by numeric value match.

    Computes the fraction of numeric values in gold_quant that also
    appear in pred_quant. Returns 1.0 if gold has no numbers.
    """
    gold_nums = _extract_numbers(gold_quant)
    if not gold_nums:
        return 1.0
    pred_nums = _extract_numbers(pred_quant)
    if not pred_nums:
        return 0.0
    matched = sum(1 for n in gold_nums if n in pred_nums)
    return matched / len(gold_nums)


def _factual_methods_score(pred_text: str | None, gold_text: str | None) -> float:
    """Score methods summary by keyword/entity coverage.

    Extracts factual elements (numbers and proper nouns) from the gold
    methods text, then computes what fraction appear in the predicted
    methods text.
    """
    if not gold_text:
        return 1.0
    gold_facts = _extract_facts(gold_text)
    if not gold_facts:
        return 1.0
    pred_facts = _extract_facts(pred_text or "")
    return _fact_overlap(gold_facts, pred_facts)


def _factual_limitations_score(pred_text: str | None, gold_text: str | None) -> float:
    """Score limitations by limitation topic coverage.

    Detects which limitation topics appear in the gold limitations text,
    then measures what fraction of those topics also appear in the
    predicted limitations text.
    """
    gold_topics = _detect_limitation_topics(gold_text)
    if not gold_topics:
        return 1.0
    pred_topics = _detect_limitation_topics(pred_text)
    if not pred_topics:
        return 0.0
    matched = len(gold_topics & pred_topics)
    return matched / len(gold_topics)


# --- Full per-field extraction scoring ---


def score_extraction_pair_factual(
    predicted: PaperExtraction,
    ground_truth: PaperExtraction,
    similarity_scores: dict[str, float],
) -> dict[str, float]:
    """Compute factual accuracy scores for a predicted vs ground-truth extraction pair.

    Returns a dict of metric names to float scores in [0, 1].

    Factual metrics (computed here):
      - key_findings_factual: factual element overlap across findings
      - quantitative_result_factual: numeric value match for quant results
      - methods_summary_factual: keyword/entity coverage for methods
      - limitations_factual: limitation topic coverage

    Pass-through from similarity_scores (embeddings/string similarity):
      - evidence_strength, study_design, quality_score, sample_size

    For quantitative scoring, each gold finding is matched to the best
    predicted finding by claim fact overlap, then quant results are scored.
    """

    # -- key_findings: average best-match factual overlap --
    pred_claims = [f.claim for f in predicted.key_findings]
    gold_claims = [f.claim for f in ground_truth.key_findings]
    key_findings_factual = _factual_key_findings_score(pred_claims, gold_claims)

    # -- quantitative_result: for each gold finding, find best pred finding
    # by claim overlap, then score quant results --
    quant_scores: list[float] = []
    for gold_finding in ground_truth.key_findings:
        gold_quant = gold_finding.quantitative_result
        # Find best-matching pred finding by claim overlap
        gold_facts = _extract_facts(gold_finding.claim)
        best_pred_finding = None
        best_claim_overlap = -1.0
        for pred_finding in predicted.key_findings:
            pred_facts = _extract_facts(pred_finding.claim)
            overlap = _fact_overlap(gold_facts, pred_facts)
            if overlap > best_claim_overlap:
                best_claim_overlap = overlap
                best_pred_finding = pred_finding
        pred_quant = best_pred_finding.quantitative_result if best_pred_finding else None
        quant_scores.append(_factual_quant_score(pred_quant, gold_quant))

    quantitative_result_factual = sum(quant_scores) / len(quant_scores) if quant_scores else 1.0

    # -- methods_summary --
    methods_summary_factual = _factual_methods_score(
        predicted.methods_summary, ground_truth.methods_summary
    )

    # -- limitations --
    limitations_factual = _factual_limitations_score(
        predicted.limitations, ground_truth.limitations
    )

    scores: dict[str, float] = {
        "key_findings_factual": key_findings_factual,
        "quantitative_result_factual": quantitative_result_factual,
        "methods_summary_factual": methods_summary_factual,
        "limitations_factual": limitations_factual,
    }

    # Pass-through fields from similarity scoring
    for field in ("evidence_strength", "study_design", "quality_score", "sample_size"):
        if field in similarity_scores:
            scores[field] = similarity_scores[field]

    logger.debug(
        "score_extraction_pair_factual",
        paper_id=predicted.paper_id,
        scores=scores,
    )
    return scores
