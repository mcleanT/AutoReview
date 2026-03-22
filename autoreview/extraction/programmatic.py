"""Programmatic (zero-LLM) paper extraction.

Deterministic extraction of structured paper metadata using regex, keyword
rules, and heuristic scoring. Replaces or supplements LLM-based extraction
for speed and reproducibility.
"""

from __future__ import annotations

import math
import re

import structlog

from autoreview.config.models import ExtractionConfig
from autoreview.extraction.extractor import ExtractionFailure
from autoreview.extraction.models import (
    EvidenceStrength,
    Finding,
    PaperExtraction,
    StudyDesign,
)
from autoreview.extraction.truncation import ParsedSection, parse_sections
from autoreview.models.paper import ScreenedPaper

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences using regex heuristics.

    Handles double-newline paragraph breaks and normalises single newlines
    to spaces before splitting on terminal punctuation followed by a capital.
    """
    if not text or not text.strip():
        return []
    # Split on double newlines first (paragraph boundaries)
    paragraphs = re.split(r"\n\n+", text.strip())
    sentences: list[str] = []
    for para in paragraphs:
        # Normalise single newlines to spaces within a paragraph
        para = re.sub(r"\n", " ", para).strip()
        if not para:
            continue
        parts = _SENTENCE_SPLIT_RE.split(para)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


# ---------------------------------------------------------------------------
# Word-overlap similarity
# ---------------------------------------------------------------------------


def word_overlap_similarity(a: str, b: str) -> float:
    """Jaccard similarity on lowercased word sets."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Sentence scoring (for key_findings)
# ---------------------------------------------------------------------------

_POSITION_WEIGHTS: dict[str, float] = {
    "abstract_last": 0.40,
    "abstract_first": 0.30,
    "conclusion": 0.35,
    "results": 0.30,
    "discussion": 0.25,
    "introduction": 0.10,
    "methods": 0.05,
    "other": 0.15,
}

_KEYWORD_PATTERNS: list[str] = [
    "we found",
    "results show",
    "results demonstrate",
    "outperforms",
    "achieves",
    "significant",
    "we propose",
    "we introduce",
    "we demonstrate",
    "we show",
    "our results",
    "our findings",
    "our approach",
    "state-of-the-art",
    "state of the art",
    "novel",
    "superior",
    # Result indicators (Argumentative Zoning literature)
    "we show that",
    "we demonstrate that",
    "we report",
    "our analysis",
    "our experiments",
    "the results suggest",
    # Contribution patterns
    "our main contribution",
    "the key contribution",
    "we make the following contributions",
    # Comparison patterns
    "compared to",
    "outperform",
    "baseline",
    # Specific result patterns (LLM findings tend to cite concrete results)
    "accuracy of",
    "accuracy on",
    "accuracy was",
    "surpasses",
    "exceeds",
    "highest score",
    "best performance",
    "worst performance",
    "lowest score",
    "consistently",
    "notably",
    "reveals that",
    "indicates that",
    "demonstrates that",
    "confirms that",
    "performance gap",
    "performance drops",
    "performance degradation",
    "performance improvement",
    "failure rate",
    "success rate",
    "error rate",
]

# Regex pattern for model name mentions (GPT-4, LLaMA, Claude, etc.)
_MODEL_NAME_RE = re.compile(
    r"\b(?:GPT-?[234o]|GPT-?4o?|Claude|Gemini|LLaMA|Llama|Mistral|Qwen|"
    r"DeepSeek|Falcon|Phi-[0-9]|PaLM|Flan|ChatGPT|Codex|CodeGen|"
    r"BERT|RoBERTa|T5|Vicuna|Alpaca|WizardCoder|StarCoder|CodeLlama|"
    r"GPT-?3\.5)\b",
    re.IGNORECASE,
)

_PERCENTAGE_RE = re.compile(r"\d+\.?\d*\s*%")
_PVALUE_RE = re.compile(r"p\s*[<>=]\s*0?\.\d+", re.IGNORECASE)
_CI_RE = re.compile(r"\b(CI|confidence interval)\b", re.IGNORECASE)
_COMPARISON_RE = re.compile(r"\d+\.?\d*\s*(vs\.?|versus|compared to)\s*\d+\.?\d*", re.IGNORECASE)

_NOVELTY_PATTERNS: list[str] = [
    "novel",
    "first",
    "new approach",
    "for the first time",
]


def _get_title_similarity_weight(sentence: str, title: str) -> float:
    """Compute a title-similarity bonus for a sentence.

    Uses word overlap between the sentence and the paper title.
    Sentences sharing significant words with the title are more likely
    key findings.

    Returns:
        A float in [0.0, 0.15].
    """
    # Filter out very short common words to focus on content words
    stopwords = {
        "a",
        "an",
        "the",
        "of",
        "in",
        "to",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "this",
        "that",
        "it",
        "its",
        "be",
        "has",
        "have",
        "had",
        "not",
        "but",
        "we",
        "our",
        "they",
        "their",
        "he",
        "she",
    }
    title_words = set(title.lower().split()) - stopwords
    sent_words = set(sentence.lower().split()) - stopwords
    if not title_words or not sent_words:
        return 0.0
    overlap = len(title_words & sent_words)
    # Normalize by title word count so shorter titles need fewer matches
    ratio = overlap / len(title_words)
    return min(ratio * 0.20, 0.15)


def score_sentence(
    sentence: str,
    section: str,
    position_in_abstract: float,
    title: str | None = None,
) -> float:
    """Composite score for a sentence's importance as a finding.

    Args:
        sentence: The sentence text.
        section: Normalised section name (e.g. "results", "abstract").
        position_in_abstract: 0.0 (first) to 1.0 (last) position within the
            abstract; ignored for non-abstract sections.
        title: The paper title; used for title-similarity bonus.

    Returns:
        A float score (typically 0.0 – 1.0).
    """
    score = 0.0
    sent_lower = sentence.lower()

    # --- Position weight (0.0 – 0.4) ---
    sec = section.lower()
    if sec == "abstract":
        if position_in_abstract >= 0.8:
            score += _POSITION_WEIGHTS["abstract_last"]
        elif position_in_abstract <= 0.2:
            score += _POSITION_WEIGHTS["abstract_first"]
        else:
            score += 0.20  # mid-abstract
    else:
        for key in ("conclusion", "results", "discussion", "introduction", "methods"):
            if key in sec:
                score += _POSITION_WEIGHTS[key]
                break
        else:
            score += _POSITION_WEIGHTS["other"]

    # --- Keyword weight (0.0 – 0.3, +0.05 per match, capped) ---
    kw_score = 0.0
    for kw in _KEYWORD_PATTERNS:
        if kw in sent_lower:
            kw_score += 0.05
    score += min(kw_score, 0.30)

    # --- Quantitative weight (0.0 – 0.20) ---
    quant = 0.0
    if _PERCENTAGE_RE.search(sentence):
        quant += 0.10
    if _PVALUE_RE.search(sentence):
        quant += 0.10
    if _CI_RE.search(sentence):
        quant += 0.05
    if _COMPARISON_RE.search(sentence):
        quant += 0.05
    score += min(quant, 0.20)

    # --- Novelty weight (0.0 – 0.1) ---
    for pat in _NOVELTY_PATTERNS:
        if pat in sent_lower:
            score += 0.10
            break

    # --- Model name specificity bonus (0.0 – 0.10) ---
    # LLM findings frequently cite specific model names; sentences with
    # model names are more likely to be specific, matchable claims
    if _MODEL_NAME_RE.search(sentence):
        score += 0.10

    # --- Title similarity weight (0.0 – 0.15) ---
    if title:
        score += _get_title_similarity_weight(sentence, title)

    # --- Sentence length normalization ---
    char_len = len(sentence)
    if char_len < 40:
        score *= 0.5
    elif char_len > 400:
        score *= 0.8

    return score


# ---------------------------------------------------------------------------
# Quantitative result extraction
# ---------------------------------------------------------------------------

_RANGE_RE = re.compile(r"(?:between|from)\s+\d+\.?\d*\s+(?:and|to)\s+\d+\.?\d*", re.IGNORECASE)
_RATIO_RE = re.compile(r"\d+\.?\d*\s*/\s*\d+\.?\d*|\d+\.?\d*\s+out\s+of\s+\d+\.?\d*", re.IGNORECASE)
_IMPROVEMENT_RE = re.compile(
    r"\d+\.?\d*\s*%\s*improvement|improved\s+by\s+\d+\.?\d*\s*%", re.IGNORECASE
)

_EMDASH_RANGE_RE = re.compile(r"\d+\.?\d*\s*[–—]\s*\d+\.?\d*", re.IGNORECASE)
_DOLLAR_RE = re.compile(r"\$\d+\.?\d*", re.IGNORECASE)
_TIME_DURATION_RE = re.compile(
    r"\d+\.?\d*\s*(?:hours?|minutes?|seconds?|days?|ms\b)\b", re.IGNORECASE
)
_SCORE_EQUALS_RE = re.compile(
    r"\b(?:score|accuracy|performance|F1|AUC|AUROC|BLEU|ROUGE|precision|recall|PPL|perplexity|MRR|MAP|NDCG)\s*(?:[=:≈]|of)\s*\d+\.?\d*",
    re.IGNORECASE,
)
_FOLD_CHANGE_RE = re.compile(
    r"\d+\.?\d*\s*[-xX×]\s*(?:fold|faster|slower|improvement|speedup)", re.IGNORECASE
)
_DECIMAL_COMPARISON_RE = re.compile(r"\d+\.?\d*\s*(?:vs\.?|versus)\s*\d+\.?\d*", re.IGNORECASE)
_AT_K_RE = re.compile(r"\d+\.?\d*\s*%?\s*(?:at|@)\s*k\s*=\s*\d+", re.IGNORECASE)
_DROPPED_FROM_TO_RE = re.compile(
    r"(?:dropped?|fell?|declin\w+|decreas\w+|increas\w+|improv\w+|rose?)\s+(?:from\s+)?\d+\.?\d*\s*%?\s+to\s+\d+\.?\d*\s*%?",
    re.IGNORECASE,
)
_PERCENTAGE_POINT_RE = re.compile(r"\d+\.?\d*\s*(?:percentage\s+points?|pp\b)", re.IGNORECASE)
_CORRELATION_RE = re.compile(
    r"\b(?:rho|correlation)\s*=\s*[-]?\d+\.?\d*|\br\s*=\s*[-]?0\.\d+", re.IGNORECASE
)

_QUANT_PATTERNS: list[re.Pattern[str]] = [
    _PERCENTAGE_RE,
    _PVALUE_RE,
    _CI_RE,
    re.compile(r"\bN\s*=\s*\d+", re.IGNORECASE),
    re.compile(r"\bmean\s*[±+\-]\s*\d+\.?\d*", re.IGNORECASE),
    re.compile(r"\b(AUC|AUROC|F1|accuracy|precision|recall)\s*[=:]\s*\d+\.?\d*", re.IGNORECASE),
    _RANGE_RE,
    _RATIO_RE,
    _IMPROVEMENT_RE,
    # Additional patterns for ML/benchmark papers
    _EMDASH_RANGE_RE,
    _DOLLAR_RE,
    _TIME_DURATION_RE,
    _SCORE_EQUALS_RE,
    _FOLD_CHANGE_RE,
    _DECIMAL_COMPARISON_RE,
    _AT_K_RE,
    _DROPPED_FROM_TO_RE,
    _PERCENTAGE_POINT_RE,
    _CORRELATION_RE,
]


def _extract_match_with_context(sentence: str, m: re.Match[str], context_words: int = 3) -> str:
    """Extract a regex match with surrounding context words.

    Instead of returning just "67.6%", returns "accuracy of 67.6% on MedQA"
    which improves word-overlap scoring against LLM descriptions.

    Args:
        sentence: The full sentence text.
        m: The regex match object.
        context_words: Number of words to include before and after the match.

    Returns:
        The match text with surrounding context, trimmed to natural boundaries.
    """
    match_text = m.group(0).strip()
    start, end = m.start(), m.end()

    # Get context before match (skip punctuation-only tokens)
    before = sentence[:start]
    before_words = [w for w in before.split() if re.search(r"[a-zA-Z0-9]", w)]
    ctx_before = " ".join(before_words[-context_words:]) if before_words else ""

    # Get context after match (skip punctuation-only tokens)
    after = sentence[end:]
    after_words = [w for w in after.split() if re.search(r"[a-zA-Z0-9]", w)]
    ctx_after = " ".join(after_words[:context_words]) if after_words else ""

    # Build contextualised string
    parts = []
    if ctx_before:
        parts.append(ctx_before)
    parts.append(match_text)
    if ctx_after:
        parts.append(ctx_after)
    return " ".join(parts)


def extract_quantitative_results(sentence: str) -> str | None:
    """Extract quantitative patterns from *sentence* with surrounding context.

    Returns matched patterns (with context words) joined by ``"; "``
    or ``None`` if nothing found. Context words improve word-overlap scoring
    against LLM descriptions that include metric names, model names, etc.
    """
    matches: list[str] = []
    seen_spans: list[tuple[int, int]] = []
    for pat in _QUANT_PATTERNS:
        for m in pat.finditer(sentence):
            # Skip overlapping matches
            span = (m.start(), m.end())
            if any(not (span[1] <= s[0] or span[0] >= s[1]) for s in seen_spans):
                continue
            seen_spans.append(span)
            matches.append(_extract_match_with_context(sentence, m))
    return "; ".join(matches) if matches else None


# ---------------------------------------------------------------------------
# Evidence strength
# ---------------------------------------------------------------------------

# Stronger hedging signals that always indicate uncertainty (not standard academic phrasing)
_STRONG_HEDGING_PATTERNS: list[str] = [
    "preliminary",
    "pilot",
    "small sample",
    "limited sample",
    "further research",
    "warrants further",
    "remains unclear",
    "not yet clear",
    "tentative",
    "speculative",
    "inconclusive",
]

# Claim-specificity markers: sentences with these contain concrete claims
_SPECIFIC_CLAIM_PATTERNS: list[str] = [
    "we found",
    "we show",
    "we demonstrate",
    "results show",
    "results demonstrate",
    "results indicate",
    "our results",
    "our findings",
    "outperforms",
    "outperform",
    "achieves",
    "achieved",
    "state-of-the-art",
    "state of the art",
    "significantly",
    "surpasses",
    "exceeds",
    "best performance",
    "highest accuracy",
    "highest score",
]

# Generic/vague sentence patterns that suggest weak evidence
_GENERIC_CLAIM_PATTERNS: list[str] = [
    "future work",
    "future research",
    "should be explored",
    "could be improved",
    "remains an open",
    "is an important",
    "has attracted",
    "is a promising",
    "has been widely",
    "plays a crucial role",
    "has gained significant attention",
    "is of great importance",
    "has emerged as",
]


def determine_evidence_strength(
    sentence: str,
    has_full_text: bool,
    has_quantitative: bool,
    study_design: StudyDesign | None = None,
    paper_quant_density: float = 0.0,
    section: str | None = None,
) -> EvidenceStrength:
    """Determine the evidence strength for a finding sentence.

    Based on empirical analysis of LLM labelling patterns across 220 papers:
      - No full text -> PRELIMINARY (100% in LLM corpus)
      - Full text + quant -> MODERATE (76%), STRONG only with specific claims (13%)
      - Full text, no quant -> MODERATE (54%), WEAK/PRELIMINARY (40%)

    Target distribution: ~10% strong, ~68% moderate, ~10% weak, ~12% preliminary.

    Additional modifiers:
      - Study design (meta-analyses get small upgrade)
      - Claim specificity (specific result claims can upgrade to STRONG)
      - Generic/hedging language (downgrades one level)
      - Section context (intro/background without quant -> WEAK)
      - Short non-specific sentences without quant -> WEAK

    Args:
        sentence: The finding sentence text.
        has_full_text: Whether full text was available for extraction.
        has_quantitative: Whether this specific finding has quantitative data.
        study_design: Paper-level study design classification.
        paper_quant_density: Fraction of findings with quantitative results (0-1).
        section: Section name where this finding was extracted from.
    """
    sent_lower = sentence.lower()
    sec_lower = (section or "").lower()

    # --- Rule 1: No full text -> always PRELIMINARY ---
    # Empirical: LLM labels 100% of abstract-only findings as preliminary
    if not has_full_text:
        return EvidenceStrength.PRELIMINARY

    # --- From here: has_full_text is True ---

    # --- Rule 2: Base strength from quantitative evidence ---
    # Full text + quant: default MODERATE (76% in LLM corpus)
    # Full text, no quant: default MODERATE (54%), but more likely to be weak
    strength = EvidenceStrength.MODERATE

    # --- Rule 3: Upgrade to STRONG for specific quantitative claims ---
    # Only 13% of full-text+quant findings are STRONG in LLM corpus.
    # Require: quantitative data + specific claim language + sufficient quant
    # density in the paper overall
    has_specific_claim = any(p in sent_lower for p in _SPECIFIC_CLAIM_PATTERNS)

    if has_quantitative and has_specific_claim and paper_quant_density >= 0.3:
        strength = EvidenceStrength.STRONG

    # --- Rule 4: Study design adjustment ---
    high_evidence_designs = {
        StudyDesign.META_ANALYSIS,
        StudyDesign.SYSTEMATIC_REVIEW,
        StudyDesign.RCT,
    }
    if study_design in high_evidence_designs and has_quantitative:
        # Meta-analyses/RCTs with quant data can be STRONG even without
        # specific claim language
        if has_specific_claim or paper_quant_density >= 0.3:
            strength = EvidenceStrength.STRONG

    # --- Rule 5: Downgrade non-quantitative claims from weak sections ---
    # 40% of full-text non-quant findings are WEAK or PRELIMINARY
    if not has_quantitative:
        weak_sections = ("introduction", "background", "related work")
        if any(ws in sec_lower for ws in weak_sections):
            strength = EvidenceStrength.WEAK

    # --- Rule 6: Generic claim check -> WEAK ---
    has_generic_claim = any(p in sent_lower for p in _GENERIC_CLAIM_PATTERNS)
    if has_generic_claim and not has_specific_claim:
        if strength == EvidenceStrength.MODERATE:
            strength = EvidenceStrength.WEAK
        elif strength == EvidenceStrength.STRONG:
            strength = EvidenceStrength.MODERATE

    # --- Rule 7: Hedging downgrade ---
    strong_hedging = any(h in sent_lower for h in _STRONG_HEDGING_PATTERNS)
    mild_hedging = ("may " in sent_lower or "might " in sent_lower) and any(
        uw in sent_lower
        for uw in ("suggest", "indicate", "imply", "possible", "potential", "unclear")
    )

    if strong_hedging or mild_hedging:
        downgrade_map: dict[str, EvidenceStrength] = {
            EvidenceStrength.STRONG: EvidenceStrength.MODERATE,
            EvidenceStrength.MODERATE: EvidenceStrength.WEAK,
            EvidenceStrength.WEAK: EvidenceStrength.PRELIMINARY,
            EvidenceStrength.PRELIMINARY: EvidenceStrength.PRELIMINARY,
        }
        strength = downgrade_map[strength]

    return strength


# ---------------------------------------------------------------------------
# Key findings extraction
# ---------------------------------------------------------------------------


def _is_boilerplate_sentence(sentence: str) -> bool:
    """Detect boilerplate/generic sentences that should be penalised.

    These are sentences that appear in many papers' conclusions or introductions
    and carry little paper-specific information.
    """
    sent_lower = sentence.lower()
    boilerplate_indicators = [
        "in this paper, we",
        "in this work, we",
        "in this study, we",
        "the rest of the paper",
        "the remainder of this paper",
        "this paper is organized",
        "this paper is structured",
        "the paper is organized",
        "the contributions of this paper",
        "we summarize our contributions",
        "we make the following contributions",
        "can be summarized as follows",
        "can be found at",
        "is available at",
        "publicly available at",
        "code is available",
        "data is available",
        "supplementary material",
        "are shown in table",
        "are shown in figure",
        "is shown in table",
        "is shown in figure",
        "as illustrated in figure",
        "as shown in figure",
        "as reported in table",
    ]
    return any(bp in sent_lower for bp in boilerplate_indicators)


def extract_key_findings(
    abstract: str | None,
    full_text: str | None,
    sections: list[ParsedSection],
    paper_id: str,
    title: str | None = None,
    study_design: StudyDesign | None = None,
) -> list[Finding]:
    """Extract and rank key findings from paper text.

    Scores all sentences, deduplicates by word overlap (>70%), and selects
    top N findings (min 3, max 12, calibrated to match LLM output counts
    which average ~8-10 per paper).

    Args:
        abstract: Paper abstract text.
        full_text: Full text of the paper (may be None).
        sections: Parsed sections from full text.
        paper_id: Unique paper identifier.
        title: Paper title for relevance scoring.
        study_design: Paper-level study design for evidence strength.
    """
    has_full_text = bool(full_text)
    scored: list[tuple[float, str, str]] = []  # (score, sentence, section_name)

    # Pre-compute abstract sentences for abstract-as-query matching
    abs_sentences: list[str] = []
    if abstract:
        abs_sentences = split_sentences(abstract)

    # Score abstract sentences
    if abs_sentences:
        n_abs = len(abs_sentences)
        for i, sent in enumerate(abs_sentences):
            pos = i / max(n_abs - 1, 1)
            s = score_sentence(sent, "abstract", pos, title=title)
            scored.append((s, sent, "abstract"))

    # Score section sentences
    for sec in sections:
        sec_sentences = split_sentences(sec.text)
        for sent in sec_sentences:
            s = score_sentence(sent, sec.name, 0.5, title=title)
            # Abstract-as-query matching: boost full-text sentences that
            # echo abstract content (up to 0.15 bonus)
            if abs_sentences:
                max_sim = max(word_overlap_similarity(sent, abs_sent) for abs_sent in abs_sentences)
                s += min(max_sim * 0.20, 0.15)

            # Penalise boilerplate sentences (table/figure references, structure
            # descriptions, availability statements)
            if _is_boilerplate_sentence(sent):
                s *= 0.3

            scored.append((s, sent, sec.name))

    # Sort descending by score
    scored.sort(key=lambda x: x[0], reverse=True)

    # Determine target count: calibrated to match LLM output
    # LLM averages ~10 for full-text papers, ~4-5 for abstract-only,
    # but ranges from 3 to 30. We target 10-14 for full-text to
    # balance precision and recall.
    total_text_len = len(abstract or "") + len(full_text or "")
    if has_full_text:
        # Full-text papers: scale between 8 and 14 based on text length
        target_n = max(8, min(14, 8 + total_text_len // 5000))
    else:
        # Abstract-only: 3-5 findings
        target_n = max(3, min(5, len(abs_sentences)))

    # Deduplicate and select top N (lower threshold catches near-paraphrases)
    findings: list[Finding] = []
    selected_sentences: list[str] = []

    # First pass: collect candidates with quantitative and evidence metadata
    candidates: list[tuple[str, str, str | None, bool]] = []  # (sent, sec, quant, is_specific)
    for _, sent, sec_name in scored:
        if len(candidates) >= target_n * 3:  # collect 3x target to have options
            break
        # Check overlap with already-selected sentences (0.85 threshold)
        is_dup = any(word_overlap_similarity(sent, prev) > 0.85 for prev in selected_sentences)
        if is_dup:
            continue
        quant = extract_quantitative_results(sent)
        has_specific = any(p in sent.lower() for p in _SPECIFIC_CLAIM_PATTERNS)
        candidates.append((sent, sec_name, quant, has_specific))
        selected_sentences.append(sent)

    # Compute paper-level quantitative density for evidence strength
    n_quant_candidates = sum(1 for _, _, q, _ in candidates if q is not None)
    quant_density = n_quant_candidates / max(len(candidates), 1)

    # Second pass: select top N from candidates
    for sent, sec_name, quant, _ in candidates[:target_n]:
        strength = determine_evidence_strength(
            sent,
            has_full_text,
            quant is not None,
            study_design=study_design,
            paper_quant_density=quant_density,
            section=sec_name,
        )

        findings.append(
            Finding(
                claim=sent,
                evidence_strength=strength,
                quantitative_result=quant,
                context=sec_name,
                paper_id=paper_id,
            )
        )

    return findings


# ---------------------------------------------------------------------------
# Section filtering — remove non-content sections before extraction
# ---------------------------------------------------------------------------

# Section names that should be excluded from content extraction.
_NON_CONTENT_SECTION_PATTERNS: list[str] = [
    # Bibliographic sections
    "reference",
    "bibliography",
    "works cited",
    "cited literature",
    # Metadata / navigation
    "acknowledgment",
    "acknowledgement",
    "author contribution",
    "funding",
    "conflict of interest",
    "data availability",
    "code availability",
    "competing interest",
    "declaration",
    "supplementary",
    "appendix",
    # Website / publisher chrome
    "search",
    "browse",
    "refback",
    "journal content",
    "language select",
    "ieee account",
    "purchase detail",
    "profile information",
    "xplore",
    # Copyright / license boilerplate
    "copyright",
    "made available under",
    "creative commons",
    "preprint",
    # Publisher / repo metadata
    "keyword",
    "api integration",
    "published in",
    "country of publisher",
    "read online",
    "hide this message",
    "in all fields",
    # Misc non-content
    "table of contents",
    "contents",
]

# Location / institution signals in section names
_LOCATION_SECTION_SIGNALS: list[str] = [
    "university",
    "institute",
    "department",
    "school of",
    "faculty of",
    "china",
    "japan",
    "usa",
    "germany",
    "france",
    "canada",
    "united kingdom",
    "singapore",
    "hong kong",
    "beijing",
    "london",
    "tokyo",
    "california",
    "ontario",
    "national",
]


# Section names that indicate actual paper content (never filter these)
_KNOWN_CONTENT_SECTION_WORDS: list[str] = [
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "approach",
    "materials",
    "experiment",
    "result",
    "evaluation",
    "discussion",
    "conclusion",
    "limitation",
    "dataset",
    "benchmark",
    "framework",
    "analysis",
    "setup",
    "implementation",
    "model",
    "architecture",
    "training",
    "overview",
    "problem",
    "proposed",
    "system design",
    "future",
    "comparison",
    "ablation",
    "performance",
    "ethical",
    "broader impact",
    "taxonomy",
    "curation",
    "data collection",
    "generation",
]


def _is_junk_section(sec: ParsedSection) -> bool:
    """Detect sections that don't contain paper content.

    Filters out: references, acknowledgments, author affiliations,
    publisher navigation elements, and very short metadata sections.
    Never filters sections whose names match known content section patterns.
    """
    name_lower = sec.name.lower().strip()

    # Never filter known content section names
    if any(word in name_lower for word in _KNOWN_CONTENT_SECTION_WORDS):
        return False

    # Check against known non-content patterns
    if any(pat in name_lower for pat in _NON_CONTENT_SECTION_PATTERNS):
        return True

    # Very short sections (< 100 chars text) with short names are likely
    # author names, affiliations, or other metadata
    text_stripped = sec.text.strip()
    if len(text_stripped) < 100:
        words = sec.name.strip().split()
        # Short name + short text = likely metadata
        if len(words) <= 5:
            # Check if it looks like a person name or affiliation
            if all(w[0].isupper() for w in words if w and w[0].isalpha()):
                return True

    # Sections that are just geographic locations or institutions
    if any(sig in name_lower for sig in _LOCATION_SECTION_SIGNALS):
        return True

    # Section name is a very short generic word
    if name_lower in ("pdf", "search", "browse", "information", "n/a", "detailed"):
        return True

    # Section names that start with "It is" (copyright boilerplate)
    if name_lower.startswith("it is "):
        return True

    # Section names that look like author bylines (contain commas between names)
    if "," in sec.name and len(sec.name.split(",")) >= 2:
        # Check if segments look like names (2-3 words each)
        parts = [p.strip() for p in sec.name.split(",")]
        if all(1 <= len(p.split()) <= 4 for p in parts if p):
            return True

    return False


def filter_content_sections(sections: list[ParsedSection]) -> list[ParsedSection]:
    """Filter sections to only include paper content sections.

    Removes references, metadata, author info, publisher navigation, etc.
    """
    return [sec for sec in sections if not _is_junk_section(sec)]


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------


def _find_section(sections: list[ParsedSection], name_patterns: list[str]) -> ParsedSection | None:
    """Find the first section whose name matches any pattern (case-insensitive substring)."""
    for sec in sections:
        sec_lower = sec.name.lower()
        for pat in name_patterns:
            if pat.lower() in sec_lower:
                return sec
    return None


def _find_section_with_children(
    sections: list[ParsedSection],
    name_patterns: list[str],
    min_text_len: int = 100,
) -> ParsedSection | None:
    """Find a section, aggregating child sub-sections if the parent is near-empty.

    When a parent section heading (e.g. "Materials and methods") has very little
    text because the content is split into sub-sections, this function merges
    consecutive sub-sections that follow it (up to the next top-level heading)
    into a single virtual section.

    Args:
        sections: Parsed sections list.
        name_patterns: Patterns to match section names.
        min_text_len: If the matched section has fewer chars than this,
            try to aggregate children.

    Returns:
        A ParsedSection with aggregated text, or None if not found.
    """
    for idx, sec in enumerate(sections):
        sec_lower = sec.name.lower()
        matched = any(pat.lower() in sec_lower for pat in name_patterns)
        if not matched:
            continue

        # If section text is long enough, return as-is
        if len(sec.text.strip()) >= min_text_len:
            return sec

        # Aggregate child sections: collect sections following this one
        # until we hit another top-level section (common headings that signal
        # a different major section)
        _TOP_LEVEL_NAMES = [
            "introduction",
            "result",
            "discussion",
            "conclusion",
            "acknowledge",
            "reference",
            "appendix",
            "supplement",
            "abstract",
            "related work",
        ]
        child_texts: list[str] = [sec.text.strip()]
        last_child_idx = idx
        for ci in range(idx + 1, len(sections)):
            child = sections[ci]
            child_lower = child.name.lower()
            # Stop if we hit another major section
            if any(top in child_lower for top in _TOP_LEVEL_NAMES):
                break
            # Stop if we hit another section matching the same patterns
            # (avoids merging unrelated methods-like sections)
            if any(pat.lower() in child_lower for pat in name_patterns) and ci > idx + 1:
                break
            child_texts.append(child.text.strip())
            last_child_idx = ci

        merged_text = "\n\n".join(child_texts)
        end_pos = sections[last_child_idx].end if last_child_idx > idx else sec.end
        return ParsedSection(
            name=sec.name,
            start=sec.start,
            end=end_pos,
            text=merged_text,
        )

    return None


def _truncate_at_sentence_boundary(text: str, max_chars: int) -> str:
    """Truncate *text* at the nearest sentence boundary before *max_chars*."""
    if len(text) <= max_chars:
        return text
    # Find last sentence-ending punctuation before max_chars
    truncated = text[:max_chars]
    last_period = max(truncated.rfind(". "), truncated.rfind(".\n"), truncated.rfind("."))
    if last_period > max_chars // 2:
        return text[: last_period + 1]
    return truncated


# ---------------------------------------------------------------------------
# Methods summary
# ---------------------------------------------------------------------------

_METHODS_KEYWORDS: list[str] = [
    "method",
    "approach",
    "technique",
    "algorithm",
    "framework",
    "pipeline",
    "protocol",
    "procedure",
    "dataset",
    "experiment",
    "implementation",
    "model architecture",
    "we used",
    "we employed",
    "we applied",
    "we trained",
]


_METHODS_SCORE_KEYWORDS: list[str] = [
    # Methodology terms
    "method",
    "approach",
    "technique",
    "algorithm",
    "model",
    "architecture",
    "framework",
    "pipeline",
    "protocol",
    "procedure",
    # Data/evaluation terms
    "dataset",
    "benchmark",
    "corpus",
    "training",
    "evaluation",
    "validation",
    "cross-validation",
    "k-fold",
    "test set",
    "train",
    # Specific methodology signals
    "we used",
    "we employed",
    "we applied",
    "we trained",
    "we implemented",
    "we evaluated",
    "we collected",
    "we measured",
    "we analyzed",
    "we propose",
    "we develop",
    "we introduce",
    "we design",
    "was performed",
    "were collected",
    "were used",
    "was used",
    "was trained",
    # Quantitative methodology signals
    "sample size",
    "participants",
    "subjects",
    "patients",
    "learning rate",
    "epochs",
    "batch size",
    "hyperparameter",
    "optimizer",
    "loss function",
    "baseline",
    "metric",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "regression",
    "classification",
]


def _is_non_content_sentence(sentence: str) -> bool:
    """Detect non-content lines: headers, affiliations, ToC entries, metadata.

    These should be filtered out before extractive summarization to avoid
    contaminating methods/limitations summaries with boilerplate.
    """
    stripped = sentence.strip()
    lower = stripped.lower()

    # Very short fragments (headers, labels)
    if len(stripped.split()) < 4:
        return True

    # Table-of-contents patterns: lines with dots, page numbers, section numbers
    if re.search(r"\.{3,}", stripped):
        return True
    # Lines that are just a section number + title (e.g. "3.1 Dataset Construction")
    if re.match(r"^\d+(\.\d+)*\s+[A-Z]", stripped) and len(stripped.split()) <= 6:
        return True
    # Lines with multiple section-number patterns (ToC entries)
    if re.search(r"\d+\.\d+\s+\w+\s+\.\s+\.\s+\.", stripped):
        return True
    # Lines that look like ToC entries with page numbers at end
    if re.search(r"\.\s*\.\s*\d+\s*$", stripped):
        return True
    # Lines starting with appendix/section labels like "G.1", "A.2", "B "
    if re.match(r"^[A-Z]\.\d+\s", stripped) and len(stripped.split()) <= 8:
        return True

    # Affiliation / author metadata patterns
    affiliation_signals = [
        "university",
        "department of",
        "institute",
        "school of",
        "faculty of",
        "@",
        "ontario, canada",
        "germany",
        "japan",
        "usa",
        "china",
        "submitted in partial",
        "bachelor of",
        "master of",
        "doctor of",
    ]
    if any(sig in lower for sig in affiliation_signals) and len(stripped.split()) <= 15:
        return True

    # Copyright / license boilerplate
    copyright_signals = [
        "copyright holder",
        "made available under",
        "creative commons",
        "preprint",
        "arxiv:",
        "doi:",
        "ieee account",
        "purchase details",
        "profile information",
        "xplore",
    ]
    if any(sig in lower for sig in copyright_signals):
        return True

    # NeurIPS / venue checklist boilerplate
    checklist_signals = [
        "question:",
        "answer: [yes]",
        "answer: [no]",
        "answer: [na]",
        "justification:",
        "guidelines:",
        "the answer na means",
    ]
    if any(sig in lower for sig in checklist_signals):
        return True

    # Lines that look like figure/table captions or references only
    if re.match(r"^(figure|fig\.|table)\s+\d+", lower):
        return True

    # Lines that look like bibliographic citations (author et al., year)
    # Multiple "et al" references suggest a citation-heavy or bibliography line
    if lower.count("et al") >= 3:
        return True

    # Lines with excessive special characters (likely formatting artifacts)
    alpha_ratio = sum(1 for c in stripped if c.isalpha()) / max(len(stripped), 1)
    if alpha_ratio < 0.4 and len(stripped) > 10:
        return True

    # Lines that are primarily citation references (e.g. "[1] Author, Title...")
    if re.match(r"^\[\d+\]", stripped):
        return True

    # Lines that look like option/answer choices (benchmark examples, not methodology)
    if re.match(r"^Option [A-Z]:", stripped):
        return True

    return False


def _score_methods_sentence(sentence: str) -> float:
    """Score a sentence for how informative it is about methodology.

    Higher scores indicate more methods-relevant content. Scores based on
    keyword density, specificity signals (numbers, named entities), and
    sentence position heuristics.
    """
    lower = sentence.lower()
    score = 0.0

    # Keyword hits (each unique keyword adds score)
    keyword_hits = sum(1 for kw in _METHODS_SCORE_KEYWORDS if kw in lower)
    score += min(keyword_hits * 0.15, 1.5)  # Cap keyword contribution

    # Bonus for containing numbers (specific quantities, parameters)
    if re.search(r"\d+", sentence):
        score += 0.3

    # Bonus for containing parenthetical details like (n=100) or (p<0.05)
    if re.search(r"\([^)]*\d[^)]*\)", sentence):
        score += 0.2

    # Bonus for containing equals/comparison operators (quantitative details)
    if re.search(r"[=<>≤≥]", sentence):
        score += 0.2

    # Penalty for very short sentences (likely headers or fragments)
    words = sentence.split()
    if len(words) < 5:
        score -= 0.5
    # Bonus for moderate-length sentences (informative but not rambling)
    elif 10 <= len(words) <= 40:
        score += 0.1

    # Penalty for reference-heavy sentences
    if lower.count("et al") >= 2:
        score -= 0.3

    return score


def _extract_methods_from_section(text: str, max_chars: int) -> str:
    """Extract methods summary via hybrid structural + scoring approach.

    Combines structural heuristics (first/last sentences for overview) with
    content scoring to produce a coherent, informative summary. Filters out
    non-content lines (headers, affiliations, ToC entries) before processing.
    """
    stripped = text.strip()

    # Split and filter out non-content sentences
    sentences = split_sentences(stripped)
    sentences = [s for s in sentences if not _is_non_content_sentence(s)]

    if not sentences:
        return _truncate_at_sentence_boundary(stripped, max_chars)

    if len(sentences) <= 3:
        result = " ".join(sentences)
        return _truncate_at_sentence_boundary(result, max_chars)

    if len(" ".join(sentences)) <= max_chars:
        return " ".join(sentences)

    # Hybrid approach: structural anchors + scoring
    # 1. Always include first 2 sentences (overview/introduction of methods)
    # 2. Always include last sentence (summary/conclusion of methods)
    # 3. Fill remaining budget with highest-scoring middle sentences
    structural_indices = set()
    structural_indices.add(0)
    if len(sentences) > 1:
        structural_indices.add(1)
    structural_indices.add(len(sentences) - 1)

    # Score remaining sentences
    middle_scored = [
        (i, sent, _score_methods_sentence(sent))
        for i, sent in enumerate(sentences)
        if i not in structural_indices
    ]
    middle_scored.sort(key=lambda x: x[2], reverse=True)

    # Select top scoring middle sentences to fill budget (up to 7 more)
    remaining_budget = max(0, 10 - len(structural_indices))
    top_middle = middle_scored[:remaining_budget]

    # Combine all selected indices, sort by position for coherence
    all_selected = [(i, sentences[i]) for i in structural_indices]
    all_selected.extend((i, sent) for i, sent, _ in top_middle)
    all_selected.sort(key=lambda x: x[0])

    result = " ".join(s for _, s in all_selected)
    return _truncate_at_sentence_boundary(result, max_chars)


def extract_methods_summary(
    abstract: str | None,
    sections: list[ParsedSection],
    max_chars: int = 800,
) -> str:
    """Extract a methods summary combining abstract context with section detail.

    The LLM ground truth closely echoes abstract methodology content.
    We use abstract-derived method sentences as a foundation, then supplement
    with section-specific detail when a Methods section exists.
    """
    _methods_headings = [
        "method",
        "approach",
        "experimental setup",
        "materials",
        "implementation",
        "experimental design",
        "study design",
        "procedure",
        "protocol",
        "methodology",
        "experimental setting",
        "data collection",
        "evaluation setup",
        "benchmark construction",
        "benchmark design",
        "dataset construction",
        "framework",
        "system design",
        "technical approach",
        "proposed method",
        "our approach",
        "evaluation methodology",
        "research design",
        "data and method",
        "setup",
    ]

    # --- Step 1: Extract abstract method sentences as foundation ---
    abs_method_text = ""
    if abstract:
        abs_sents = split_sentences(abstract)
        abs_sents = [s for s in abs_sents if not _is_non_content_sentence(s)]
        abs_method_sents = [
            s for s in abs_sents if any(kw in s.lower() for kw in _METHODS_KEYWORDS)
        ]
        if not abs_method_sents:
            scored = [(s, _score_methods_sentence(s)) for s in abs_sents]
            scored.sort(key=lambda x: x[1], reverse=True)
            abs_method_sents = [s for s, sc in scored[:3] if sc > 0.0]
        if not abs_method_sents and abs_sents:
            abs_method_sents = abs_sents
        abs_method_text = " ".join(abs_method_sents)

    # --- Step 2: Try to get section-specific methods content ---
    section_method_text = ""

    methods_sec = _find_section_with_children(sections, _methods_headings, min_text_len=100)
    if not methods_sec or len(methods_sec.text.strip()) < 100:
        methods_sec = _find_section(sections, _methods_headings)

    if methods_sec and len(methods_sec.text.strip()) >= 50:
        section_method_text = _extract_methods_from_section(methods_sec.text, max_chars)
    elif sections:
        method_candidates: list[tuple[float, str]] = []
        for sec in sections:
            sec_sents = split_sentences(sec.text)
            for sent in sec_sents:
                if _is_non_content_sentence(sent):
                    continue
                score = _score_methods_sentence(sent)
                if score > 0.3:
                    method_candidates.append((score, sent))
        if len(method_candidates) >= 2:
            method_candidates.sort(key=lambda x: x[0], reverse=True)
            top_sents = [s for _, s in method_candidates[:7]]
            section_method_text = _truncate_at_sentence_boundary(" ".join(top_sents), max_chars)

    # --- Step 3: Combine abstract foundation + section detail ---
    if abs_method_text and section_method_text:
        combined = abs_method_text + " " + section_method_text
        return _truncate_at_sentence_boundary(combined, max_chars)
    if section_method_text:
        return _truncate_at_sentence_boundary(section_method_text, max_chars)
    if abs_method_text:
        return _truncate_at_sentence_boundary(abs_method_text, max_chars)
    if abstract:
        return _truncate_at_sentence_boundary(abstract, max_chars)

    return "Methods not available."


# ---------------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------------

_LIMITATION_KEYWORDS: list[str] = [
    "limitation",
    "shortcoming",
    "weakness",
    "caveat",
    "constraint",
    "drawback",
    "however",
    "although",
    "despite",
    "challenge",
    "future work",
    "further research",
    "not without",
    "small sample",
    "nonetheless",
    "it should be noted",
    "one limitation",
    "a key limitation",
    "threat to validity",
    "threats to validity",
    "not generaliz",
    "may not generaliz",
    "limited to",
    "restricted to",
    "did not consider",
    "does not account",
    "unable to",
    "lack of",
    "potential bias",
    "selection bias",
    "scope of this",
    "beyond the scope",
    "room for improvement",
    "room for enhancement",
    "could be improved",
    "should be interpreted with caution",
    "needs further",
    "requires further",
    "we acknowledge",
    "we recognize",
    "it is important to note",
    "it is worth noting",
    "we did not",
    "we do not",
    "not been validated",
    "not been tested",
]

# Keywords that strongly signal a limitation paragraph when they appear at the start
_LIMITATION_PARA_STARTERS: list[str] = [
    "limitation",
    "however",
    "although",
    "despite",
    "nonetheless",
    "one limitation",
    "a key limitation",
    "a major limitation",
    "our study has",
    "this study has",
    "there are several limitation",
    "our work has",
    "our approach has",
    "this work has",
    "a potential limitation",
    "the main limitation",
    "first, ",
    "second, ",
    "while our",
    "we acknowledge",
    "we recognize",
    "it is important to note",
    "this benchmark",
    "our benchmark",
]

_FALLBACK_LIMITATIONS = "Limitations not explicitly stated in the available text."

# Keywords that signal concrete, specific limitations (vs generic future work)
_CONCRETE_LIMITATION_KEYWORDS: list[str] = [
    "limitation",
    "shortcoming",
    "weakness",
    "drawback",
    "caveat",
    "bias",
    "confound",
    "small sample",
    "limited sample",
    "not generaliz",
    "may not generaliz",
    "cannot generaliz",
    "restricted to",
    "limited to",
    "did not consider",
    "does not account",
    "unable to",
    "lack of",
    "threat to validity",
    "threats to validity",
    "selection bias",
    "potential bias",
    "overfitting",
    "underpowered",
    "missing data",
    "incomplete",
    "not been validated",
    "not been tested",
    "single center",
    "single-center",
    "retrospective",
    "cross-sectional",
    "correlational",
    "self-report",
    "small cohort",
    "narrow scope",
]

# Keywords that signal generic/vague statements (penalize these)
_GENERIC_FUTURE_KEYWORDS: list[str] = [
    "future work should",
    "future research should",
    "further research is needed",
    "it would be interesting",
    "we plan to",
    "in the future",
    "future direction",
    "remains to be",
    "open question",
]


def _score_limitation_sentence(sentence: str) -> float:
    """Score a sentence for how specific and concrete its limitation content is.

    Higher scores indicate concrete, specific limitations. Lower scores for
    generic future-work statements.
    """
    lower = sentence.lower()
    score = 0.0

    # Concrete limitation keyword hits
    concrete_hits = sum(1 for kw in _CONCRETE_LIMITATION_KEYWORDS if kw in lower)
    score += min(concrete_hits * 0.25, 1.5)

    # General limitation keyword hits (weaker signal)
    general_hits = sum(1 for kw in _LIMITATION_KEYWORDS if kw in lower)
    score += min(general_hits * 0.1, 0.5)

    # Penalty for generic future-work statements
    generic_hits = sum(1 for kw in _GENERIC_FUTURE_KEYWORDS if kw in lower)
    score -= generic_hits * 0.3

    # Bonus for specificity signals (numbers, comparisons)
    if re.search(r"\d+", sentence):
        score += 0.2
    if re.search(r"\([^)]*\d[^)]*\)", sentence):
        score += 0.15

    # Penalty for very short or very long sentences
    words = sentence.split()
    if len(words) < 5:
        score -= 0.4
    elif len(words) > 60:
        score -= 0.2

    # Bonus for first-person acknowledgment (strong limitation signal)
    if any(phrase in lower for phrase in ["we acknowledge", "we recognize", "our study"]):
        score += 0.3

    return score


def _select_top_limitation_sentences(
    text: str,
    max_chars: int,
    max_sentences: int = 5,
) -> str:
    """Select the most specific limitation sentences from text.

    Scores each sentence for limitation specificity and selects top 3-5,
    maintaining original order for coherence. Filters out non-content
    and checklist boilerplate sentences.
    """
    sentences = split_sentences(text)
    # Filter out non-content and boilerplate
    sentences = [s for s in sentences if not _is_non_content_sentence(s)]

    if not sentences:
        return _truncate_at_sentence_boundary(text.strip(), max_chars)

    if len(sentences) <= 2:
        result = " ".join(sentences)
        return _truncate_at_sentence_boundary(result, max_chars)

    scored = [(i, sent, _score_limitation_sentence(sent)) for i, sent in enumerate(sentences)]

    # Filter to sentences with positive limitation score
    positive = [s for s in scored if s[2] > 0.0]
    if not positive:
        # Fall back to all sentences if none score positive
        positive = scored

    # Select top N by score, sort by position for coherence
    positive.sort(key=lambda x: x[2], reverse=True)
    top_n = min(max_sentences, len(positive))
    selected = sorted(positive[:top_n], key=lambda x: x[0])

    result = " ".join(s[1] for s in selected)
    return _truncate_at_sentence_boundary(result, max_chars)


# Expanded section headings that may contain limitations
_LIMITATION_SECTION_HEADINGS: list[str] = [
    "limitation",
    "shortcoming",
    "threats to validity",
    "strengths and limitation",
    "discussion and limitation",
    "challenges and limitation",
    "caveats",
    "weaknesses",
]


def _extract_limitations_from_combined_section(
    section_text: str,
    section_name: str,
    max_chars: int,
) -> str | None:
    """Extract limitations from a section that combines conclusions + limitations.

    Handles common patterns like "Conclusion and Limitations", "Conclusions and
    Future Work" where limitations are embedded after the conclusion prose.
    """
    sec_lower = section_name.lower()
    # Only apply special handling for combined sections
    combined_indicators = ["conclusion", "future", "discussion"]
    if not any(ind in sec_lower for ind in combined_indicators):
        return None

    sentences = split_sentences(section_text)
    sentences = [s for s in sentences if not _is_non_content_sentence(s)]

    if not sentences:
        return None

    # Look for limitation content in the section
    lim_candidates: list[tuple[int, str, float]] = []
    for i, sent in enumerate(sentences):
        score = _score_limitation_sentence(sent)
        if score > 0.0:
            lim_candidates.append((i, sent, score))

    if not lim_candidates:
        return None

    # Also include the last few sentences of discussion/conclusion
    # as they often contain embedded limitation acknowledgments
    tail_start = max(0, len(sentences) - 4)
    for i in range(tail_start, len(sentences)):
        sent = sentences[i]
        if not any(i == c[0] for c in lim_candidates):
            score = _score_limitation_sentence(sent)
            # Use a lower threshold for tail sentences since they're
            # positionally likely to contain limitations
            if score > -0.2:
                sent_lower = sent.lower()
                # Check for even weak limitation signals in tail position
                weak_signals = _LIMITATION_KEYWORDS + [
                    "only",
                    "but",
                    "yet",
                    "still",
                    "remain",
                    "not yet",
                    "further",
                    "more research",
                    "more work",
                ]
                if any(kw in sent_lower for kw in weak_signals):
                    lim_candidates.append((i, sent, max(score, 0.05)))

    if not lim_candidates:
        return None

    # Deduplicate and select best
    seen_indices: set[int] = set()
    unique_candidates: list[tuple[int, str, float]] = []
    for idx, sent, score in lim_candidates:
        if idx not in seen_indices:
            seen_indices.add(idx)
            unique_candidates.append((idx, sent, score))

    unique_candidates.sort(key=lambda x: x[2], reverse=True)
    top_n = min(5, len(unique_candidates))
    selected = sorted(unique_candidates[:top_n], key=lambda x: x[0])
    result = " ".join(s for _, s, _ in selected)
    return _truncate_at_sentence_boundary(result, max_chars)


# Scope-limiting patterns for abstract-only papers: the LLM often infers
# limitations from scope-limiting language in the abstract
_SCOPE_LIMITING_PATTERNS: list[str] = [
    "only",
    "limited to",
    "focus on",
    "focused on",
    "restricted to",
    "single",
    "one dataset",
    "one domain",
    "english only",
    "english-only",
    "specific to",
    "preliminary",
    "pilot",
    "small-scale",
    "small scale",
    "case study",
    "proof of concept",
    "proof-of-concept",
    "not consider",
    "do not address",
    "does not address",
    "beyond the scope",
    "outside the scope",
    "we do not",
    "we did not",
    "without consider",
    "excluding",
    "not includ",
]


def extract_limitations(
    abstract: str | None,
    sections: list[ParsedSection],
    max_chars: int = 800,
) -> str:
    """Extract limitations combining section content with abstract context.

    The LLM ground truth for limitations often incorporates abstract context
    (average 0.58 embedding similarity between abstract and LLM limitations).
    We extract section-level limitations and prepend abstract limitation/scope
    context for better semantic coverage.
    """
    # --- Step 1: Extract abstract limitation context ---
    # The LLM often incorporates abstract context in its limitation output.
    # We extract limitation-specific sentences, scope-limiting language, and
    # fall back to abstract summary sentences for general context.
    abs_lim_text = ""
    if abstract:
        abs_sents = split_sentences(abstract)
        abs_sents = [s for s in abs_sents if not _is_non_content_sentence(s)]
        # Explicit limitation keywords in abstract
        abs_lim_sents = [
            s for s in abs_sents if any(kw in s.lower() for kw in _LIMITATION_KEYWORDS)
        ]
        # Scope-limiting language
        if not abs_lim_sents:
            abs_lim_sents = [
                s for s in abs_sents if any(pat in s.lower() for pat in _SCOPE_LIMITING_PATTERNS)
            ]
        # If still nothing, use last 2-3 sentences of abstract as context
        # (abstracts often end with scope/limitation/future-work statements)
        if not abs_lim_sents and len(abs_sents) >= 3:
            abs_lim_sents = abs_sents[-3:]
        elif not abs_lim_sents and abs_sents:
            abs_lim_sents = abs_sents
        abs_lim_text = " ".join(abs_lim_sents)

    # --- Step 2: Try to extract section-level limitation content ---
    section_lim_text = ""

    # Try dedicated limitations section (with child aggregation)
    lim_sec = _find_section_with_children(sections, _LIMITATION_SECTION_HEADINGS, min_text_len=50)
    if lim_sec and len(lim_sec.text.strip()) >= 50:
        section_lim_text = _select_top_limitation_sentences(lim_sec.text.strip(), max_chars)
    else:
        lim_sec = _find_section(sections, _LIMITATION_SECTION_HEADINGS)
        if lim_sec and len(lim_sec.text.strip()) >= 50:
            section_lim_text = _select_top_limitation_sentences(lim_sec.text.strip(), max_chars)

    # Try combined sections (Conclusion and Limitations, etc.)
    if not section_lim_text:
        _combined_headings = [
            "conclusion",
            "concluding",
            "future",
            "ethical",
            "broader impact",
            "societal impact",
            "responsible",
        ]
        for sec in sections:
            sec_lower = sec.name.lower()
            if any(lh in sec_lower for lh in _LIMITATION_SECTION_HEADINGS):
                continue
            if any(ch in sec_lower for ch in _combined_headings):
                result = _extract_limitations_from_combined_section(sec.text, sec.name, max_chars)
                if result and len(result.strip()) >= 30:
                    section_lim_text = result
                    break

    # Discussion/conclusion scanning
    if not section_lim_text:
        for sec_name in ["discussion", "conclusion", "future"]:
            sec = _find_section(sections, [sec_name])
            if not sec:
                continue
            # Paragraphs starting with limitation keywords
            paragraphs = re.split(r"\n\n+", sec.text.strip())
            lim_paras = []
            for para in paragraphs:
                para_stripped = para.strip()
                para_lower = para_stripped.lower()
                if _is_non_content_sentence(para_stripped):
                    continue
                if any(para_lower.startswith(s) for s in _LIMITATION_PARA_STARTERS):
                    lim_paras.append(para_stripped)
            if lim_paras:
                section_lim_text = _select_top_limitation_sentences(" ".join(lim_paras), max_chars)
                break

            # Individual sentences with limitation keywords
            sentences = split_sentences(sec.text)
            sentences = [s for s in sentences if not _is_non_content_sentence(s)]
            lim_sents = [
                s for s in sentences if any(kw in s.lower() for kw in _LIMITATION_KEYWORDS)
            ]
            if lim_sents:
                section_lim_text = _select_top_limitation_sentences(" ".join(lim_sents), max_chars)
                break

            # Tail scanning (last 5 sentences)
            if len(sentences) >= 3:
                tail = sentences[-5:]
                tail_lim = [
                    s
                    for s in tail
                    if any(kw in s.lower() for kw in _LIMITATION_KEYWORDS)
                    or _score_limitation_sentence(s) > 0.2
                ]
                if tail_lim:
                    section_lim_text = _select_top_limitation_sentences(
                        " ".join(tail_lim), max_chars
                    )
                    break

    # All-section strong keyword scan
    if not section_lim_text and sections:
        all_lim_sents: list[str] = []
        strong_kws = [
            "limitation",
            "shortcoming",
            "weakness",
            "drawback",
            "caveat",
            "threat to validity",
            "not generaliz",
            "small sample",
            "potential bias",
            "room for improvement",
            "we acknowledge",
            "should be interpreted with caution",
            "does not account",
            "did not consider",
            "unable to",
            "restricted to",
            "limited to",
            "lack of",
            "we do not",
            "we did not",
            "not been validated",
            "not been tested",
        ]
        for sec in sections:
            sec_sents = split_sentences(sec.text)
            for sent in sec_sents:
                if _is_non_content_sentence(sent):
                    continue
                if any(kw in sent.lower() for kw in strong_kws):
                    all_lim_sents.append(sent)
        if all_lim_sents:
            section_lim_text = _select_top_limitation_sentences(" ".join(all_lim_sents), max_chars)

    # --- Step 3: Combine abstract context + section content ---
    if abs_lim_text and section_lim_text:
        combined = abs_lim_text + " " + section_lim_text
        return _truncate_at_sentence_boundary(combined, max_chars)
    if section_lim_text:
        return _truncate_at_sentence_boundary(section_lim_text, max_chars)
    if abs_lim_text:
        return _truncate_at_sentence_boundary(abs_lim_text, max_chars)

    return _FALLBACK_LIMITATIONS


# ---------------------------------------------------------------------------
# Study design classification (multi-phase)
# ---------------------------------------------------------------------------

# Review/survey signals in the TITLE (strongest indicators)
_TITLE_REVIEW_PATTERNS: list[str] = [
    "a survey",
    "survey of",
    "survey on",
    ": a review",
    "comprehensive review",
    "comprehensive survey",
    "literature review",
    "narrative review",
    "scoping review",
    "overview of",
    "a review of",
    "state of the art",
    "state-of-the-art review",
    "a review:",
    # Common survey title patterns -- must include review/survey/overview word
    "a comprehensive review",
    "a comprehensive survey",
    "a comprehensive overview",
    "a comprehensive analysis",
    "a comprehensive study",
    "a comprehensive evaluation",  # surveys that evaluate a field
    "a comprehensive examination",
]

# Broader review signals in title + abstract
_REVIEW_BODY_PATTERNS: list[str] = [
    "we survey",
    "we review",
    "this survey",
    "this review",
    "in this survey",
    "in this review",
    "our survey",
    "our review",
    "review the literature",
    "review the current",
    "review of the literature",
    "systematically review",
    "systematically survey",
    "survey the landscape",
    "survey the field",
    "survey the current",
    "comprehensive overview",
    "existing literature",
    "body of literature",
    "review recent advances",
    "review recent progress",
    "we present a survey",
    "we present a review",
    "we provide a survey",
    "we provide a review",
    "we conduct a review",
    "we conduct a survey",
    "we offer a comprehensive",
    "panoramic view",
    "provide an overview",
    "literature survey",
    # Technical overview / examination patterns
    "comprehensively examines",
    "comprehensively examine",
    "comprehensive examination",
    "technical overview",
    "this article comprehensively",
    "this paper comprehensively",
    # Systematic study signals that also indicate a review
    "bibliometric",
    "prisma",
    # Additional review signals for papers that discuss/categorize research
    "we categorize existing",
    "we classify existing",
    "we summarize the current",
    "we summarize existing",
    "taxonomy of existing",
]

# Cross-sectional clinical evaluation patterns
_CROSS_SECTIONAL_PATTERNS: list[str] = [
    "cross-sectional",
    "cross sectional",
]

# Patterns that signal a clinical/diagnostic evaluation study
# (LLM evaluated on medical/exam tasks -> cross_sectional)
_CLINICAL_EVAL_TITLE_PATTERNS: list[str] = [
    "performance of chatgpt",
    "performance of gpt",
    # Specific chatgpt/gpt evaluation patterns
    "evaluation of chatgpt",
    "evaluation of gpt",
    # Assessment patterns with specific clinical subjects
    "assessing the accuracy of diagnostic",
    "assessment of chatgpt",
    "assessment of gpt",
    # Clinical-specific patterns (high confidence)
    "diagnostic capabilities of large language",
    "diagnostic capabilities of llm",
    "diagnostic capabilities of chatgpt",
    "diagnostic performance of",
]

# Clinical evaluation body signals (title+abstract)
# These require clinical/medical context to avoid matching general benchmark papers
_CLINICAL_EVAL_BODY_PATTERNS: list[str] = [
    "diagnostic modeling study",
    "diagnostic accuracy of",
    "clinical accuracy of",
    "clinical performance of",
    "evaluated on clinical cases",
    "evaluated on medical cases",
    "tested on clinical cases",
    "tested on clinical data",
    "tested on medical cases",
    "clinical decision-making",
    "clinical decision making",
    # Additional high-confidence clinical patterns
    "medical licensing examination",
    "medical board exam",
]

# Phase 1: exact study design keywords (high confidence, non-computational)
_PHASE1_RULES: list[tuple[list[str], StudyDesign]] = [
    (["meta-analysis", "meta analysis"], StudyDesign.META_ANALYSIS),
    (["systematic review", "systematic literature review"], StudyDesign.SYSTEMATIC_REVIEW),
    (["narrative review", "literature review", "scoping review"], StudyDesign.NARRATIVE_REVIEW),
    (
        [
            "randomized controlled",
            "randomised controlled",
            "rct",
            "randomized trial",
            "randomised trial",
        ],
        StudyDesign.RCT,
    ),
    (["case-control", "case control"], StudyDesign.CASE_CONTROL),
    (_CROSS_SECTIONAL_PATTERNS, StudyDesign.CROSS_SECTIONAL),
    (
        ["cohort study", "cohort analysis", "prospective cohort", "retrospective cohort"],
        StudyDesign.COHORT,
    ),
    (["case series"], StudyDesign.CASE_SERIES),
    (["case report"], StudyDesign.CASE_REPORT),
    (["in vitro", "in-vitro", "cell culture", "cell line"], StudyDesign.IN_VITRO),
]

# Phase 4: computational keywords (only used when paper is NOT a review)
_COMPUTATIONAL_KEYWORDS: list[str] = [
    "computational",
    "simulation",
    "algorithm",
    "deep learning",
    "machine learning",
    "neural network",
    "language model",
    "nlp",
    "natural language processing",
    "benchmark",
    "software",
    "framework",
]

# Phase 4.5: Signals that override computational classification -> OTHER
# These are high-confidence patterns for evaluation/assessment studies that
# use computational tools but are not themselves computational research.
# Only applied when a paper ALSO matches computational keywords.
_EVALUATION_OVERRIDE_PATTERNS: list[str] = [
    "human vs",
    "human versus",
    "comparative effectiveness study",
    "human-calibrated",
    "human calibrated",
    "web-based evaluation tool",
    "web based evaluation tool",
]


def _is_review_paper(title_lower: str, combined_lower: str) -> StudyDesign | None:
    """Detect review/survey papers using title and body signals.

    Returns the appropriate StudyDesign or None if not a review paper.
    Priority: systematic_review > narrative_review (based on signal strength).
    """
    has_title_review_signal = any(pat in title_lower for pat in _TITLE_REVIEW_PATTERNS)
    has_body_review_signal = any(pat in combined_lower for pat in _REVIEW_BODY_PATTERNS)

    # "survey" or "review" in the title is a strong signal
    title_has_survey = "survey" in title_lower
    title_has_review = "review" in title_lower

    if not (
        has_title_review_signal or has_body_review_signal or title_has_survey or title_has_review
    ):
        return None

    # Check for systematic indicators
    systematic_signals = [
        "systematic review",
        "systematic literature review",
        "systematically review",
        "systematically survey",
        "systematic survey",
        "systematic evaluation",
        "bibliometric analysis",
        "bibliometric",
        "prisma",
        "inclusion criteria",
        "exclusion criteria",
        "search strategy",
    ]
    is_systematic = any(sig in combined_lower for sig in systematic_signals)

    if is_systematic:
        return StudyDesign.SYSTEMATIC_REVIEW

    return StudyDesign.NARRATIVE_REVIEW


def _is_clinical_eval(title_lower: str, combined_lower: str) -> bool:
    """Detect clinical/diagnostic evaluation studies of LLMs.

    These are cross-sectional studies where LLMs are evaluated on
    clinical tasks, medical exams, or diagnostic challenges.
    """
    has_title_signal = any(pat in title_lower for pat in _CLINICAL_EVAL_TITLE_PATTERNS)
    has_body_signal = any(pat in combined_lower for pat in _CLINICAL_EVAL_BODY_PATTERNS)
    return has_title_signal or has_body_signal


def classify_study_design(
    title: str,
    abstract: str | None,
    sections: list[ParsedSection],
) -> StudyDesign:
    """Classify study design using multi-phase keyword rules.

    Phase 1: Exact study design keywords (meta-analysis, RCT, etc.)
    Phase 2: Review/survey detection (title + body signals)
    Phase 3: Clinical evaluation detection (cross_sectional)
    Phase 4: Computational keywords (only if not a review)
    Phase 4.5: Override check for evaluation studies -> OTHER
    Phase 5: Fallback to OTHER
    """
    combined = f"{title} {abstract or ''}"
    # Also check methods section
    methods_sec = _find_section(sections, ["method"])
    if methods_sec:
        combined += f" {methods_sec.text}"
    combined_lower = combined.lower()
    title_lower = title.lower()

    # Phase 1: Explicit study design keywords (highest priority)
    for keywords, design in _PHASE1_RULES:
        for kw in keywords:
            if kw in combined_lower:
                return design

    # Phase 2: Review/survey detection
    # This must come BEFORE computational to prevent reviews about
    # "language models", "benchmarks", "frameworks" from being misclassified
    review_design = _is_review_paper(title_lower, combined_lower)
    if review_design is not None:
        return review_design

    # Phase 3: Clinical evaluation studies (cross-sectional)
    if _is_clinical_eval(title_lower, combined_lower):
        return StudyDesign.CROSS_SECTIONAL

    # Phase 4: Computational keywords (only if not a review/survey)
    has_computational = any(kw in combined_lower for kw in _COMPUTATIONAL_KEYWORDS)
    if has_computational:
        # Phase 4.5: Check if this is actually an evaluation study (OTHER)
        # that happens to mention computational terms
        if any(pat in combined_lower for pat in _EVALUATION_OVERRIDE_PATTERNS):
            return StudyDesign.OTHER
        return StudyDesign.COMPUTATIONAL

    return StudyDesign.OTHER


# ---------------------------------------------------------------------------
# Sample size extraction
# ---------------------------------------------------------------------------

# Adjective gap: allows 0-2 intervening words between number and target noun
# e.g. "151 human participants", "20 open-source SLMs", "510 question-answer pairs"
_ADJ_GAP = r"(?:\s+[\w-]+){0,2}\s+"

# High-confidence patterns (traditional study designs)
_SAMPLE_SIZE_HIGH_CONF: list[re.Pattern[str]] = [
    re.compile(r"\bN\s*=\s*(\d[\d,]*)", re.IGNORECASE),
    re.compile(
        r"(\d[\d,]*)" + _ADJ_GAP + r"(?:participants|patients|individuals|respondents)",
        re.IGNORECASE,
    ),
    re.compile(r"sample\s+size\s+of\s+(\d[\d,]*)", re.IGNORECASE),
    # "we collected/gathered/curated X ..." -- signals the paper's own data
    re.compile(
        r"(?:we\s+)?(?:collected|gathered|curated|compiled|assembled)\s+(\d[\d,]*)\s+\w+",
        re.IGNORECASE,
    ),
]

# Medium-confidence patterns (benchmark/ML dataset sizes)
# NOTE: Many patterns use _ADJ_GAP to allow intervening adjectives
# e.g. "510 question-answer pairs", "3,000 daily tasks"
_SAMPLE_SIZE_MED_CONF: list[re.Pattern[str]] = [
    # "consisting/comprising of X [adj] instances/questions/..."
    re.compile(
        r"(?:consisting|comprising|composed)\s+of\s+(\d[\d,]*)"
        + _ADJ_GAP
        + r"(?:instances|questions|items|entries|examples|problems|tasks|articles|projects|pairs|cases|prompts|scenarios|benchmarks|templates|datapoints|data\s*points|test\s*cases|multiple.choice)",
        re.IGNORECASE,
    ),
    # "contains/containing X [adj] instances/..."
    re.compile(
        r"contain(?:s|ing)\s+(\d[\d,]*)"
        + _ADJ_GAP
        + r"(?:instances|questions|items|entries|examples|problems|tasks|articles|projects|pairs|cases|prompts|scenarios|real-world|multiple.choice|datasets?|stories|terms|topics|utterances|judgments?|annotations?|conversations?|instructions?|responses?|encounters?|queries|test\s*pairs?)",
        re.IGNORECASE,
    ),
    # "evaluate/evaluated/evaluating [of] X [adj] benchmarks/datasets" (NOT models/LLMs
    # which produce false positives with small model counts like "12 LLMs")
    re.compile(
        r"(?:evaluat\w+|analyz\w+|assess\w+|investigat\w+|benchmark\w*)\s+(?:of\s+)?(\d[\d,]*)"
        + _ADJ_GAP
        + r"(?:benchmarks?|datasets?|systems?)",
        re.IGNORECASE,
    ),
    # "total of X [adj] items" pattern
    re.compile(
        r"(?:total|totaling)\s+(?:of\s+)?(\d[\d,]*)"
        + _ADJ_GAP
        + r"(?:instances|questions|items|entries|examples|problems|tasks|articles|projects|pairs|cases|prompts|scenarios|papers|studies|terms|multiple.choice|datasets?|stories|topics|utterances|judgments?|annotations?|conversations?|instructions?|responses?|encounters?|queries)",
        re.IGNORECASE,
    ),
    # "X news articles" and similar compound-noun patterns
    re.compile(
        r"(\d[\d,]*)\s+(?:news\s+articles|clinical\s+cases|medical\s+cases|test\s+examples|evaluation\s+instances|benchmark\s+problems|multiple.choice\s+questions|clinical\s+scenarios|examination.style\s+\w+|bug\s+instances|software\s+projects|real.world\s+(?:projects?|utterances?|scenarios?|queries)|solution\s+codes?|test\s+pairs?|query.response\s+pairs?|user\s+queries|interaction\s+records?)",
        re.IGNORECASE,
    ),
    # "dataset of X" -- but only in methods/abstract context
    re.compile(r"(?:dataset|corpus|collection)\s+of\s+(\d[\d,]*)", re.IGNORECASE),
    # "data consists of X [adj] items"
    re.compile(
        r"(?:data|dataset|benchmark)\s+(?:consists?|comprised?)\s+of\s+(\d[\d,]*)",
        re.IGNORECASE,
    ),
    # "X [adj] samples/documents/records/terms/stories/..."
    # Excludes "topics" and "subjects" which are usually taxonomic subdivisions
    re.compile(
        r"(\d[\d,]*)"
        + _ADJ_GAP
        + r"(?:samples|documents|records|images|videos|texts?|sentences|utterances|dialogues?|conversations?|trials?|publications?|terms|stories|queries|instructions?|encounters?|judgments?|annotations?|responses?|transcripts?)",
        re.IGNORECASE,
    ),
    # "includes/including X [adj] questions/tasks"
    re.compile(
        r"(?:includes?|including)\s+(\d[\d,]*)"
        + _ADJ_GAP
        + r"(?:instances|questions|items|examples|problems|tasks|prompts|scenarios|datasets?|test\s*cases|studies|papers?|articles?|bug\s*instances?)",
        re.IGNORECASE,
    ),
    # "across X datasets/tasks/domains" -- exclude subjects/topics/categories
    # to avoid matching subdivision counts like "67 topics" or "13 subjects"
    re.compile(
        r"across\s+(\d[\d,]*)" + _ADJ_GAP + r"(?:datasets?|tasks?|domains?|benchmarks?)",
        re.IGNORECASE,
    ),
    # "benchmark with/of X [adj] ..."
    re.compile(
        r"benchmark\s+(?:with|of|comprises?|comprising)\s+(\d[\d,]*)\s+\w+",
        re.IGNORECASE,
    ),
    # "comprises/encompassing X [adj] tasks/prompts"
    re.compile(
        r"(?:comprises?|encompass\w+)\s+(\d[\d,]*)"
        + _ADJ_GAP
        + r"(?:instances|questions|items|entries|examples|problems|tasks|prompts|scenarios|pairs|cases|datasets?|stories|terms)",
        re.IGNORECASE,
    ),
    # "reviewed/surveyed/examined X [adj] papers/studies"
    re.compile(
        r"(?:review\w*|survey\w*|examin\w*|screen\w*|analyz\w*|retriev\w*)\s+(?:of\s+)?(\d[\d,]*)"
        + _ADJ_GAP
        + r"(?:papers?|studies|articles?|publications?|trials?|primary\s+studies)",
        re.IGNORECASE,
    ),
    # "X-scenario dataset", "X,000-scenario dataset" hyphenated number-noun
    re.compile(
        r"(\d[\d,]*)[.-](?:scenario|question|problem|task|sample|item)\s+(?:dataset|benchmark|corpus)",
        re.IGNORECASE,
    ),
    # "constructed/curated/built X [adj] items" (broader verb set)
    re.compile(
        r"(?:construct\w*|built|creat\w*|develop\w*|prepar\w*|generat\w*|introduc\w*)\s+(?:a\s+)?(?:dataset\s+(?:of|with|comprising|containing)\s+)?(\d[\d,]*)"
        + _ADJ_GAP
        + r"(?:instances|questions|items|entries|examples|problems|tasks|pairs|cases|prompts|scenarios|terms|stories|topics|utterances|judgments?|annotations?|conversations?|instructions?|responses?|encounters?|queries|samples?|test\s*cases)",
        re.IGNORECASE,
    ),
    # "sampled/selected X [adj] items" (require past tense to avoid
    # matching "sample 2,000 risk prompts" where "sample" is a noun/adj)
    re.compile(
        r"(?:sampled|selected|filter\w+|retain\w+)\s+(\d[\d,]*)"
        + _ADJ_GAP
        + r"(?:instances|questions|items|entries|examples|problems|tasks|pairs|cases|prompts|scenarios|terms|stories|utterances|conversations?|instructions?|responses?|encounters?|queries|samples?|papers?|studies|test\s*cases)",
        re.IGNORECASE,
    ),
    # "evaluated on X [adj] questions/items" (passive, data-item targets only)
    re.compile(
        r"evaluated\s+on\s+(\d[\d,]*)"
        + _ADJ_GAP
        + r"(?:instances|questions|items|examples|problems|tasks|prompts|scenarios|pairs|cases)",
        re.IGNORECASE,
    ),
    # "processed/annotated/labeled X questions/items" (past tense verbs)
    re.compile(
        r"(?:process\w*|annotat\w*|label\w*|classif\w*)\s+(\d[\d,]*)"
        + _ADJ_GAP
        + r"(?:instances|questions|items|entries|examples|problems|tasks|pairs|cases|prompts|scenarios|samples?)",
        re.IGNORECASE,
    ),
]

# Low-confidence patterns (broad "X [adj?] LLMs/models" -- only used in abstract)
_SAMPLE_SIZE_LOW_CONF: list[re.Pattern[str]] = [
    re.compile(
        r"(\d[\d,]*)"
        + _ADJ_GAP
        + r"(?:LLMs?|models?|benchmarks?|questions|instances|examples|problems|articles|entries|terms|projects|papers|studies|datasets?|SLMs?)",
        re.IGNORECASE,
    ),
]


def _parse_sample_int(match: re.Match[str], min_val: int = 3) -> int | None:
    """Parse an integer from a regex match group 1, handling commas.

    Args:
        match: Regex match with group(1) containing the number string.
        min_val: Minimum accepted value. Default 3 for high/med confidence;
            use higher thresholds for low-confidence patterns to avoid
            false positives on incidental small counts.
    """
    num_str = match.group(1).replace(",", "")
    try:
        val = int(num_str)
        # Reject implausibly small values (likely not sample sizes)
        if val < min_val:
            return None
        return val
    except ValueError:
        return None


def extract_sample_size(text: str | None, sections: list[ParsedSection]) -> int | None:
    """Extract sample size from methods section, then abstract.

    Uses a tiered pattern approach:
    1. High-confidence patterns (N=X, X participants) in methods, then abstract
    2. Medium-confidence patterns (consisting of X instances) in methods, then abstract
    3. Low-confidence patterns (X models/LLMs) only in abstract

    Full text beyond methods section and abstract is NOT searched to avoid
    picking up incidental numbers from cited studies in survey/review papers.
    """
    # Build search texts: methods section + abstract only (not full text body)
    # This prevents false positives from numbers mentioned in cited studies
    methods_sec = _find_section_with_children(
        sections, ["method", "materials", "experimental setup", "data collection"], min_text_len=50
    )

    # Also look in "evaluation", "dataset", "benchmark" sections which often
    # contain the primary sample size for computational papers
    eval_sec = _find_section(sections, ["evaluation", "dataset", "benchmark", "data"])

    # Look in introduction (often states dataset size for benchmark papers)
    intro_sec = _find_section(sections, ["introduction"])

    # Look in results/experiments (sometimes dataset details are here)
    results_sec = _find_section(sections, ["result", "experiment", "setup", "statistic"])

    # Restrict abstract to first 2000 chars
    abstract_text = text[:2000] if text else ""

    # Build ordered list of texts to search (most reliable first)
    search_texts: list[str] = []
    if methods_sec:
        search_texts.append(methods_sec.text)
    if eval_sec and eval_sec != methods_sec:
        search_texts.append(eval_sec.text)
    if abstract_text:
        search_texts.append(abstract_text)

    # Extended search texts (for tier 1 and 2 only) -- includes intro/results
    extended_texts: list[str] = list(search_texts)
    if intro_sec and intro_sec not in (methods_sec, eval_sec):
        extended_texts.append(intro_sec.text)
    if results_sec and results_sec not in (methods_sec, eval_sec):
        extended_texts.append(results_sec.text)

    # Tier 1: High-confidence patterns (search core + extended sections)
    for search_text in extended_texts:
        for pat in _SAMPLE_SIZE_HIGH_CONF:
            m = pat.search(search_text)
            if m:
                val = _parse_sample_int(m)
                if val is not None:
                    return val

    # Tier 2: Medium-confidence patterns (search core + extended sections)
    for search_text in extended_texts:
        for pat in _SAMPLE_SIZE_MED_CONF:
            m = pat.search(search_text)
            if m:
                val = _parse_sample_int(m)
                if val is not None:
                    return val

    # Tier 3: Low-confidence patterns -- abstract only, higher threshold
    # to avoid false positives on small model counts ("7 LLMs", "12 models")
    if abstract_text:
        for pat in _SAMPLE_SIZE_LOW_CONF:
            m = pat.search(abstract_text)
            if m:
                val = _parse_sample_int(m, min_val=14)
                if val is not None:
                    return val

    return None


# ---------------------------------------------------------------------------
# Quality score
# ---------------------------------------------------------------------------


def compute_quality_score(
    has_full_text: bool,
    full_text_length: int,
    abstract_length: int,
    citation_count: int | None,
    methods_text_length: int,
    results_text_length: int,
    n_quantitative_findings: int,
    n_total_findings: int,
) -> float:
    """Compute a composite quality score in [0, 1].

    Calibrated to match LLM quality assessments. The LLM tends to assign
    scores in the 0.3-0.8 range, with most full-text papers getting 0.5-0.7.

    Weighted components:
        text_completeness (0.25): full-text bonus + length scaling
        structural_depth (0.20): methods + results section richness
        citation_impact (0.20): log-scaled citation count
        quant_density (0.20): fraction of findings with quantitative data
        findings_richness (0.15): number of extractable findings

    After computing the raw composite, applies Bayesian shrinkage toward
    the corpus mean (0.60) to reduce variance on extreme predictions.
    The shrinkage factor was optimized via grid search over the 220-paper
    corpus to minimize mean absolute error vs LLM quality assessments.
    """
    # Text completeness (0-1): full-text papers need ~40K chars for max score
    if has_full_text:
        text_score = min(1.0, full_text_length / 40000)
    else:
        # Abstract-only papers cap at ~0.5
        text_score = min(0.5, abstract_length / 600)

    # Structural depth (0-1): combined methods + results signal
    # Use broad section heading patterns to find methods/results-like content
    meth_sub = min(1.0, methods_text_length / 800)
    res_sub = min(1.0, results_text_length / 1500)
    if meth_sub > 0 and res_sub > 0:
        struct_score = 0.6 * max(meth_sub, res_sub) + 0.4 * min(meth_sub, res_sub)
    elif meth_sub > 0 or res_sub > 0:
        struct_score = max(meth_sub, res_sub) * 0.7
    else:
        # Fallback: use full_text_length as a proxy for structure
        # Many papers have experiments/methods but under non-standard headings
        struct_score = min(0.5, full_text_length / 80000) if has_full_text else 0.0

    # Citation impact (0-1), log-scaled
    cc = citation_count or 0
    cite_score = min(1.0, math.log1p(cc) / math.log1p(200))

    # Quantitative density (0-1)
    quant_score = (n_quantitative_findings / n_total_findings) if n_total_findings > 0 else 0.0

    # Findings richness (0-1): having more findings indicates richer content
    findings_score = min(1.0, n_total_findings / 10)

    raw_composite = (
        0.25 * text_score
        + 0.20 * struct_score
        + 0.20 * cite_score
        + 0.20 * quant_score
        + 0.15 * findings_score
    )

    # Bayesian shrinkage toward corpus mean to reduce variance on extreme
    # predictions. The LLM's quality scores cluster around 0.3-0.8.
    # Shrinkage of 0.35 toward mean 0.62 was optimized via grid search
    # to minimize mean absolute error vs LLM quality assessments.
    corpus_mean = 0.62
    shrinkage = 0.35
    calibrated = raw_composite * (1.0 - shrinkage) + corpus_mean * shrinkage

    return max(0.0, min(1.0, calibrated))


# ---------------------------------------------------------------------------
# ProgrammaticExtractor
# ---------------------------------------------------------------------------


class ProgrammaticExtractor:
    """Deterministic paper extractor using regex and heuristic rules.

    Produces the same :class:`PaperExtraction` schema as the LLM-based
    extractor but without any API calls.
    """

    def __init__(self, config: ExtractionConfig) -> None:
        self.config = config

    def extract(self, sp: ScreenedPaper) -> PaperExtraction:
        """Extract structured data from a single screened paper."""
        paper = sp.paper
        abstract = paper.abstract or ""
        full_text = paper.full_text or ""
        has_full_text = bool(full_text)

        # Parse sections from full text and filter out non-content sections
        # (references, author info, publisher chrome, etc.)
        raw_sections = parse_sections(full_text) if full_text else []
        sections = filter_content_sections(raw_sections)

        # Study design (classified first so it can inform key_findings)
        study_design = classify_study_design(paper.title, abstract or None, sections)

        # Key findings (now receives study_design for evidence strength)
        findings = extract_key_findings(
            abstract,
            full_text or None,
            sections,
            paper.id,
            title=paper.title,
            study_design=study_design,
        )

        # Methods summary
        methods = extract_methods_summary(abstract or None, sections)

        # Limitations
        limitations = extract_limitations(abstract or None, sections)

        # Sample size -- pass abstract (not full combined text) to avoid
        # false positives from incidental numbers in survey paper bodies
        sample_size = extract_sample_size(abstract or None, sections)

        # Quality score -- use broad section heading patterns to find methods/results
        # Many papers use "Experiments", "Evaluation", "Dataset" instead of "Methods"
        _qs_methods_headings = [
            "method",
            "approach",
            "experimental setup",
            "materials",
            "implementation",
            "methodology",
            "experimental setting",
            "experimental design",
            "dataset",
            "benchmark",
            "evaluation setup",
            "our approach",
            "proposed method",
            "framework",
            "system design",
            "technical approach",
            "data collection",
        ]
        _qs_results_headings = [
            "result",
            "experiment",
            "evaluation",
            "findings",
            "analysis",
            "performance",
            "comparison",
            "ablation",
        ]
        methods_sec = _find_section_with_children(sections, _qs_methods_headings, min_text_len=50)
        results_sec = _find_section_with_children(sections, _qs_results_headings, min_text_len=50)
        n_quant = sum(1 for f in findings if f.quantitative_result)

        quality = compute_quality_score(
            has_full_text=has_full_text,
            full_text_length=len(full_text),
            abstract_length=len(abstract),
            citation_count=paper.citation_count,
            methods_text_length=len(methods_sec.text) if methods_sec else 0,
            results_text_length=len(results_sec.text) if results_sec else 0,
            n_quantitative_findings=n_quant,
            n_total_findings=len(findings),
        )

        logger.info(
            "programmatic_extraction.complete",
            paper_id=paper.id,
            title=paper.title[:60],
            findings=len(findings),
            study_design=study_design,
            quality_score=round(quality, 3),
            has_full_text=has_full_text,
        )

        return PaperExtraction(
            paper_id=paper.id,
            key_findings=findings,
            methods_summary=methods,
            limitations=limitations,
            relationships=[],
            methodology_details=None,
            domain_specific_fields={},
            study_design=study_design,
            quality_score=quality,
            sample_size=sample_size,
        )

    def extract_batch(
        self,
        papers: list[ScreenedPaper],
    ) -> tuple[list[PaperExtraction], list[ExtractionFailure]]:
        """Extract from a batch of papers synchronously.

        Returns:
            A 2-tuple of (successful extractions, failures).
        """
        extractions: list[PaperExtraction] = []
        failures: list[ExtractionFailure] = []

        for sp in papers:
            try:
                extraction = self.extract(sp)
                extractions.append(extraction)
            except Exception as exc:
                paper_id = sp.paper.id
                error_msg = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "programmatic_extraction.paper_failed",
                    paper_id=paper_id,
                    error=error_msg,
                )
                failures.append(ExtractionFailure(paper_id=paper_id, error=error_msg))

        logger.info(
            "programmatic_extraction.batch_complete",
            total=len(papers),
            successful=len(extractions),
            failed=len(failures),
        )
        return extractions, failures
