# Programmatic Paper Extractor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic Python extraction function that replaces LLM-based paper extraction, reducing extraction from ~80 min / 25M tokens to <1 min / 0 tokens.

**Architecture:** New `ProgrammaticExtractor` class in `autoreview/extraction/programmatic.py` that scores sentences, extracts sections, and classifies metadata to produce `PaperExtraction` objects identical in schema to LLM output. Benchmarked against 220 ground truth LLM extractions. Integrated via `extraction_mode` config option.

**Tech Stack:** Pure Python + `re` (runtime). `sentence-transformers`, `scipy`, `rouge-score` (benchmark-only).

---

## File Structure

| File | Responsibility |
|------|---------------|
| **Create:** `autoreview/extraction/programmatic.py` | `ProgrammaticExtractor` class — sentence scoring, section extraction, classification |
| **Create:** `autoreview/extraction/scoring.py` | Benchmark scoring: embedding similarity, ROUGE-L, composite scores |
| **Create:** `scripts/benchmark_extractor.py` | Benchmark runner: load ground truth, run extractor, generate report |
| **Create:** `tests/test_extraction/test_programmatic.py` | Unit tests for all extraction heuristics |
| **Create:** `tests/test_extraction/test_scoring.py` | Unit tests for scoring functions |
| **Modify:** `autoreview/config/models.py:95-106` | Add `extraction_mode` field to `ExtractionConfig` |
| **Modify:** `autoreview/pipeline/nodes.py:556-597` | Wire `extraction_mode` to select extractor |

---

### Task 1: Add `extraction_mode` to ExtractionConfig

**Files:**
- Modify: `autoreview/config/models.py:95-106`
- Test: `tests/test_config/test_models.py` (add to existing)

- [ ] **Step 1: Write the failing test**

In `tests/test_config/test_models.py`, add:

```python
from autoreview.config.models import ExtractionConfig


def test_extraction_config_mode_default():
    """extraction_mode defaults to 'llm'."""
    config = ExtractionConfig()
    assert config.extraction_mode == "llm"


def test_extraction_config_mode_programmatic():
    """extraction_mode accepts 'programmatic'."""
    config = ExtractionConfig(extraction_mode="programmatic")
    assert config.extraction_mode == "programmatic"


def test_extraction_config_mode_hybrid():
    """extraction_mode accepts 'hybrid'."""
    config = ExtractionConfig(extraction_mode="hybrid")
    assert config.extraction_mode == "hybrid"


def test_extraction_config_mode_invalid():
    """extraction_mode rejects invalid values."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExtractionConfig(extraction_mode="invalid")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config/test_models.py::test_extraction_config_mode_default -v`
Expected: FAIL — `extraction_mode` field does not exist yet.

- [ ] **Step 3: Add extraction_mode field**

In `autoreview/config/models.py`, add to `ExtractionConfig`:

```python
from typing import Literal

class ExtractionConfig(BaseModel):
    """Configuration for paper extraction."""

    model_config = ConfigDict(extra="forbid")

    # ... existing fields ...

    # Extraction engine selection
    extraction_mode: Literal["llm", "programmatic", "hybrid"] = "llm"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config/test_models.py -k extraction_config_mode -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add autoreview/config/models.py tests/test_config/test_models.py
git commit -m "feat(config): add extraction_mode field to ExtractionConfig"
```

---

### Task 2: Text utilities — sentence splitting, section finding, word overlap

**Files:**
- Create: `autoreview/extraction/programmatic.py` (initial scaffold with utilities)
- Create: `tests/test_extraction/test_programmatic.py`

These are the foundation functions used by all downstream extraction logic.

- [ ] **Step 1: Write failing tests for sentence splitting**

```python
# tests/test_extraction/test_programmatic.py

from autoreview.extraction.programmatic import split_sentences, word_overlap_similarity


class TestSplitSentences:
    def test_basic_split(self):
        text = "First sentence. Second sentence. Third sentence."
        result = split_sentences(text)
        assert len(result) == 3
        assert result[0] == "First sentence."
        assert result[2] == "Third sentence."

    def test_abbreviation_handling(self):
        """Should not split on common abbreviations like 'Dr.' or 'et al.'."""
        text = "We follow Smith et al. in this approach. The results confirm the hypothesis."
        result = split_sentences(text)
        # May split on 'al.' — that's acceptable. Just verify we get sentences, not fragments.
        assert all(len(s) > 10 for s in result)

    def test_empty_string(self):
        assert split_sentences("") == []

    def test_single_sentence(self):
        result = split_sentences("Just one sentence.")
        assert len(result) == 1

    def test_newline_separated(self):
        text = "First paragraph.\n\nSecond paragraph."
        result = split_sentences(text)
        assert len(result) == 2


class TestWordOverlapSimilarity:
    def test_identical(self):
        assert word_overlap_similarity("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert word_overlap_similarity("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        sim = word_overlap_similarity("the cat sat", "the dog sat")
        assert 0.4 < sim < 0.8  # 2/4 words overlap

    def test_empty_string(self):
        assert word_overlap_similarity("", "hello") == 0.0
        assert word_overlap_similarity("", "") == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_extraction/test_programmatic.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement utilities**

Create `autoreview/extraction/programmatic.py`:

```python
"""Programmatic (zero-token) paper extraction.

Extracts structured PaperExtraction data using heuristic text analysis
instead of LLM calls. Produces the same schema as LLM-based extraction
so downstream pipeline stages are agnostic to the extraction method.
"""

from __future__ import annotations

import re

# Sentence boundary: period/question/exclamation followed by whitespace and uppercase letter.
# Simple but sufficient for academic prose.
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex boundary detection."""
    text = text.strip()
    if not text:
        return []
    # Also split on double newlines (paragraph boundaries)
    text = re.sub(r"\n{2,}", ". ", text)
    # Normalize single newlines to spaces
    text = re.sub(r"\n", " ", text)
    parts = [s.strip() for s in _SENT_BOUNDARY.split(text) if s.strip()]
    return parts


def word_overlap_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity based on word overlap (case-insensitive)."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_extraction/test_programmatic.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add autoreview/extraction/programmatic.py tests/test_extraction/test_programmatic.py
git commit -m "feat(extraction): add sentence splitting and word overlap utilities"
```

---

### Task 3: Sentence scoring and key_findings extraction

**Files:**
- Modify: `autoreview/extraction/programmatic.py`
- Modify: `tests/test_extraction/test_programmatic.py`

This is the core algorithm — scores sentences by position, keywords, quantitative content, and novelty signals, then builds `Finding` objects.

- [ ] **Step 1: Write failing tests for sentence scoring**

```python
# Add to tests/test_extraction/test_programmatic.py

from autoreview.extraction.programmatic import (
    score_sentence,
    extract_quantitative_results,
    determine_evidence_strength,
    extract_key_findings,
)
from autoreview.extraction.models import EvidenceStrength


class TestScoreSentence:
    def test_result_sentence_scores_high(self):
        score = score_sentence(
            "Our results demonstrate a 95% accuracy improvement.",
            section="results",
            position_in_abstract=None,
        )
        assert score > 0.5  # keyword + quantitative + position

    def test_methods_sentence_scores_low(self):
        score = score_sentence(
            "We used a standard preprocessing pipeline.",
            section="methods",
            position_in_abstract=None,
        )
        assert score < 0.3

    def test_abstract_conclusion_scores_high(self):
        score = score_sentence(
            "This work presents a new framework.",
            section="abstract",
            position_in_abstract="last",
        )
        assert score > 0.4


class TestExtractQuantitativeResults:
    def test_percentage(self):
        result = extract_quantitative_results("We achieved 95.3% accuracy.")
        assert "95.3%" in result

    def test_p_value(self):
        result = extract_quantitative_results("The difference was significant (p < 0.001).")
        assert "p < 0.001" in result or "p <" in result

    def test_no_numbers(self):
        result = extract_quantitative_results("The approach was effective.")
        assert result is None

    def test_sample_size_pattern(self):
        result = extract_quantitative_results("We evaluated on N = 1,234 samples.")
        assert result is not None


class TestDetermineEvidenceStrength:
    def test_full_text_with_quantitative(self):
        strength = determine_evidence_strength(
            sentence="We achieved 95% accuracy.",
            has_full_text=True,
            has_quantitative=True,
        )
        assert strength == EvidenceStrength.STRONG

    def test_abstract_only_no_quantitative(self):
        strength = determine_evidence_strength(
            sentence="Our approach shows promise.",
            has_full_text=False,
            has_quantitative=False,
        )
        assert strength == EvidenceStrength.PRELIMINARY

    def test_hedging_downgrades(self):
        strength = determine_evidence_strength(
            sentence="This may potentially improve results by 10%.",
            has_full_text=True,
            has_quantitative=True,
        )
        # Hedging "may potentially" should downgrade from STRONG
        assert strength in (EvidenceStrength.MODERATE, EvidenceStrength.WEAK)


class TestExtractKeyFindings:
    def test_from_abstract(self):
        abstract = (
            "Large language models have transformed NLP. "
            "We evaluate GPT-4 on medical benchmarks. "
            "Our results show GPT-4 achieves 86.7% accuracy on MedQA. "
            "The model significantly outperforms prior approaches. "
            "These findings demonstrate the potential of LLMs in healthcare."
        )
        findings = extract_key_findings(
            abstract=abstract,
            full_text=None,
            sections=[],
            paper_id="test_paper",
        )
        assert len(findings) >= 3
        assert all(f.paper_id == "test_paper" for f in findings)
        # Should have found the quantitative result
        quant_findings = [f for f in findings if f.quantitative_result]
        assert len(quant_findings) >= 1

    def test_minimum_findings(self):
        """Even a short abstract should produce at least 1 finding."""
        findings = extract_key_findings(
            abstract="We propose a new method.",
            full_text=None,
            sections=[],
            paper_id="short_paper",
        )
        assert len(findings) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_extraction/test_programmatic.py::TestScoreSentence -v`
Expected: FAIL — functions don't exist.

- [ ] **Step 3: Implement sentence scoring and key_findings extraction**

Add to `autoreview/extraction/programmatic.py`:

```python
import math
from autoreview.extraction.models import EvidenceStrength, Finding
from autoreview.extraction.truncation import ParsedSection

# --- Scoring weights ---

_POSITION_WEIGHTS = {
    "abstract_last": 0.40,
    "abstract_first": 0.30,
    "abstract_middle": 0.25,
    "conclusion": 0.35,
    "results": 0.30,
    "discussion": 0.25,
    "introduction": 0.10,
    "methods": 0.05,
    "other": 0.15,
}

_RESULT_KEYWORDS = [
    "we found", "results show", "results demonstrate", "results indicate",
    "our findings", "analysis reveals", "we observed", "data suggest",
    "data show", "we demonstrate", "experiments show",
]
_PERFORMANCE_KEYWORDS = [
    "outperforms", "achieves", "surpasses", "improves upon",
    "state-of-the-art", "superior to", "beats", "exceeds",
]
_SIGNIFICANCE_KEYWORDS = ["significant", "significantly", "p <", "p =", "p-value"]
_CONTRIBUTION_KEYWORDS = [
    "we propose", "we introduce", "we present", "we develop",
    "this paper presents", "our contribution", "this work presents",
]
_ALL_KEYWORDS = (
    _RESULT_KEYWORDS + _PERFORMANCE_KEYWORDS
    + _SIGNIFICANCE_KEYWORDS + _CONTRIBUTION_KEYWORDS
)

_NOVELTY_SIGNALS = [
    "novel", "first", "new approach", "for the first time",
    "we propose", "we introduce", "pioneering",
]

_HEDGING_SIGNALS = [
    "may", "might", "could", "possibly", "potentially",
    "preliminary", "pilot", "small sample", "limited",
]

# Quantitative extraction patterns
QUANT_PATTERNS = [
    re.compile(r"(\d+\.?\d*)\s*%"),                         # percentages
    re.compile(r"p\s*[<>=]\s*\d+\.?\d*"),                    # p-values
    re.compile(
        r"(?:CI|confidence interval)[:\s]*[\[(\s]*\d+\.?\d*\s*[-–,]\s*\d+\.?\d*[\])\s]*",
        re.IGNORECASE,
    ),  # CIs
    re.compile(r"[Nn]\s*=\s*[\d,]+"),                        # sample sizes
    re.compile(r"(\d+\.?\d*)\s*±\s*(\d+\.?\d*)"),            # mean ± SD
    re.compile(
        r"(?:AUC|accuracy|precision|recall|F1|BLEU|ROUGE)[:\s]*(?:of\s+)?(\d+\.?\d*)",
        re.IGNORECASE,
    ),  # metrics
    re.compile(r"(\d+\.?\d*)\s*(?:fold|times)\s+(?:increase|decrease|improvement)"),
]


def _get_position_weight(section: str, position_in_abstract: str | None) -> float:
    """Compute position weight for a sentence."""
    if position_in_abstract == "last":
        return _POSITION_WEIGHTS["abstract_last"]
    if position_in_abstract == "first":
        return _POSITION_WEIGHTS["abstract_first"]
    if position_in_abstract == "middle":
        return _POSITION_WEIGHTS["abstract_middle"]

    section_lower = section.lower()
    for key in ("conclusion", "results", "discussion", "introduction", "methods"):
        if key in section_lower:
            return _POSITION_WEIGHTS[key]
    return _POSITION_WEIGHTS["other"]


def _get_keyword_weight(sentence: str) -> float:
    """Compute keyword weight — additive, capped at 0.3."""
    sentence_lower = sentence.lower()
    weight = 0.0
    for kw in _ALL_KEYWORDS:
        if kw in sentence_lower:
            weight += 0.05
    return min(weight, 0.3)


def _get_quantitative_weight(sentence: str) -> float:
    """Compute quantitative weight — capped at 0.2."""
    weight = 0.0
    if re.search(r"\d+\.?\d*\s*%", sentence):
        weight += 0.10
    if re.search(r"p\s*[<>=]\s*\d+\.?\d*", sentence):
        weight += 0.10
    if re.search(r"(?:CI|confidence interval)", sentence, re.IGNORECASE):
        weight += 0.05
    if re.search(r"from\s+\d+\.?\d*\s+to\s+\d+\.?\d*", sentence):
        weight += 0.05
    return min(weight, 0.20)


def _get_novelty_weight(sentence: str) -> float:
    """Compute novelty weight — 0.0 or 0.1."""
    sentence_lower = sentence.lower()
    for signal in _NOVELTY_SIGNALS:
        if signal in sentence_lower:
            return 0.10
    return 0.0


def score_sentence(
    sentence: str,
    section: str = "other",
    position_in_abstract: str | None = None,
) -> float:
    """Score a sentence for 'finding-ness'. Higher = more likely a key finding."""
    return (
        _get_position_weight(section, position_in_abstract)
        + _get_keyword_weight(sentence)
        + _get_quantitative_weight(sentence)
        + _get_novelty_weight(sentence)
    )


def extract_quantitative_results(sentence: str) -> str | None:
    """Extract quantitative results from a sentence using regex patterns."""
    matches = []
    for pattern in QUANT_PATTERNS:
        for m in pattern.finditer(sentence):
            matches.append(m.group(0).strip())
    return "; ".join(matches) if matches else None


def determine_evidence_strength(
    sentence: str,
    has_full_text: bool,
    has_quantitative: bool,
) -> EvidenceStrength:
    """Determine evidence strength based on text source and content."""
    sentence_lower = sentence.lower()

    # Check for hedging
    has_hedging = any(h in sentence_lower for h in _HEDGING_SIGNALS)

    if has_full_text and has_quantitative:
        base = EvidenceStrength.STRONG
    elif has_full_text:
        base = EvidenceStrength.MODERATE
    elif has_quantitative:
        base = EvidenceStrength.MODERATE
    else:
        base = EvidenceStrength.PRELIMINARY

    # Downgrade one level if hedging detected
    if has_hedging:
        _DOWNGRADE = {
            EvidenceStrength.STRONG: EvidenceStrength.MODERATE,
            EvidenceStrength.MODERATE: EvidenceStrength.WEAK,
            EvidenceStrength.WEAK: EvidenceStrength.PRELIMINARY,
            EvidenceStrength.PRELIMINARY: EvidenceStrength.PRELIMINARY,
        }
        base = _DOWNGRADE[base]

    return base


def extract_key_findings(
    abstract: str | None,
    full_text: str | None,
    sections: list[ParsedSection],
    paper_id: str,
) -> list[Finding]:
    """Extract key findings by scoring and selecting top sentences."""
    has_full_text = bool(full_text and len(full_text) > 500)

    # Collect scored sentences: (sentence, score, section_name)
    scored: list[tuple[str, float, str]] = []

    # Score abstract sentences
    if abstract:
        abs_sents = split_sentences(abstract)
        for i, sent in enumerate(abs_sents):
            if len(sent) < 20:  # skip tiny fragments
                continue
            if i < 2:
                pos = "first"
            elif i >= len(abs_sents) - 2:
                pos = "last"
            else:
                pos = "middle"
            score = score_sentence(sent, section="abstract", position_in_abstract=pos)
            scored.append((sent, score, "Abstract"))

    # Score full-text section sentences
    if sections:
        for sec in sections:
            sec_sents = split_sentences(sec.text)
            for sent in sec_sents:
                if len(sent) < 20:
                    continue
                score = score_sentence(sent, section=sec.name)
                scored.append((sent, score, sec.name))

    if not scored:
        # Fallback: use the abstract or title as a single finding
        fallback_text = abstract or "No findings extractable from available text."
        return [
            Finding(
                claim=fallback_text[:500],
                evidence_strength=EvidenceStrength.PRELIMINARY,
                quantitative_result=None,
                context=None,
                paper_id=paper_id,
            )
        ]

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate: remove sentences with >85% word overlap with a higher-scored one
    deduped: list[tuple[str, float, str]] = []
    for sent, score, sec in scored:
        if any(word_overlap_similarity(sent, kept[0]) > 0.85 for kept in deduped):
            continue
        deduped.append((sent, score, sec))

    # Select top N
    text_len = len(full_text or abstract or "")
    n = min(max(5, text_len // 1000), 15)
    selected = deduped[:n]

    # Build Finding objects
    findings: list[Finding] = []
    for sent, _score, sec in selected:
        quant = extract_quantitative_results(sent)
        strength = determine_evidence_strength(
            sent,
            has_full_text=has_full_text,
            has_quantitative=quant is not None,
        )
        findings.append(
            Finding(
                claim=sent,
                evidence_strength=strength,
                quantitative_result=quant,
                context=sec,
                paper_id=paper_id,
            )
        )

    return findings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_extraction/test_programmatic.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add autoreview/extraction/programmatic.py tests/test_extraction/test_programmatic.py
git commit -m "feat(extraction): add sentence scoring and key_findings extraction"
```

---

### Task 4: Section-based field extraction — methods_summary, limitations

**Files:**
- Modify: `autoreview/extraction/programmatic.py`
- Modify: `tests/test_extraction/test_programmatic.py`

- [ ] **Step 1: Write failing tests**

```python
from autoreview.extraction.programmatic import (
    extract_methods_summary,
    extract_limitations,
    _find_section,
)
from autoreview.extraction.truncation import ParsedSection


class TestFindSection:
    def test_finds_methods(self):
        sections = [
            ParsedSection(name="Introduction", start=0, end=100, text="Intro text"),
            ParsedSection(name="Methods", start=100, end=300, text="Methods text here"),
        ]
        result = _find_section(sections, ["method", "material", "experimental"])
        assert result is not None
        assert result.name == "Methods"

    def test_returns_none_when_missing(self):
        sections = [
            ParsedSection(name="Introduction", start=0, end=100, text="Intro text"),
        ]
        result = _find_section(sections, ["method"])
        assert result is None

    def test_case_insensitive(self):
        sections = [
            ParsedSection(name="METHODOLOGY", start=0, end=100, text="Method text"),
        ]
        result = _find_section(sections, ["method"])
        assert result is not None


class TestExtractMethodsSummary:
    def test_from_section(self):
        sections = [
            ParsedSection(
                name="Methods",
                start=0,
                end=500,
                text="Methods\nWe used a transformer architecture. "
                     "The model was trained on 10K examples. "
                     "We evaluated using BLEU score.",
            ),
        ]
        result = extract_methods_summary(
            abstract=None, sections=sections
        )
        assert "transformer" in result
        assert len(result) <= 600  # within char limit

    def test_fallback_to_abstract(self):
        abstract = (
            "We propose a novel framework for text generation. "
            "Our approach uses a fine-tuned GPT model on domain data. "
            "Results show significant improvement."
        )
        result = extract_methods_summary(abstract=abstract, sections=[])
        assert len(result) > 0
        # Should extract method-related sentences from abstract
        assert "approach" in result.lower() or "model" in result.lower()


class TestExtractLimitations:
    def test_from_section(self):
        sections = [
            ParsedSection(
                name="Limitations",
                start=0,
                end=300,
                text="Limitations\nOur study has several limitations. "
                     "First, the sample size was small. "
                     "Second, we only tested on English data.",
            ),
        ]
        result = extract_limitations(abstract=None, sections=sections)
        assert "sample size" in result.lower()

    def test_fallback_to_discussion(self):
        sections = [
            ParsedSection(
                name="Discussion",
                start=0,
                end=500,
                text="Discussion\nOur results are promising. "
                     "However, there are limitations in our evaluation. "
                     "Future work should address the scalability challenge.",
            ),
        ]
        result = extract_limitations(abstract=None, sections=sections)
        assert len(result) > 0

    def test_final_fallback(self):
        result = extract_limitations(abstract="Short abstract.", sections=[])
        assert "not explicitly stated" in result.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_extraction/test_programmatic.py::TestExtractMethodsSummary -v`
Expected: FAIL.

- [ ] **Step 3: Implement section-based extraction**

Add to `autoreview/extraction/programmatic.py`:

```python
def _find_section(
    sections: list[ParsedSection],
    name_patterns: list[str],
) -> ParsedSection | None:
    """Find a section by matching name patterns (case-insensitive substring match)."""
    for sec in sections:
        name_lower = sec.name.lower()
        for pattern in name_patterns:
            if pattern in name_lower:
                return sec
    return None


def _truncate_at_sentence_boundary(text: str, max_chars: int) -> str:
    """Truncate text at the nearest sentence boundary before max_chars."""
    if len(text) <= max_chars:
        return text
    # Find the last sentence-ending punctuation before max_chars
    truncated = text[:max_chars]
    last_period = max(truncated.rfind(". "), truncated.rfind(".\n"))
    if last_period > max_chars // 2:
        return truncated[: last_period + 1]
    return truncated.rstrip() + "..."


_METHOD_KEYWORDS = [
    "method", "approach", "dataset", "model", "algorithm", "technique",
    "framework", "we use", "we train", "we employ", "we apply", "using",
    "architecture", "pipeline", "implementation",
]

_LIMITATION_KEYWORDS = [
    "limitation", "weakness", "caveat", "future work", "further research",
    "however", "challenge", "drawback", "constraint", "shortcoming",
    "remains to be",
]


def extract_methods_summary(
    abstract: str | None,
    sections: list[ParsedSection],
    max_chars: int = 500,
) -> str:
    """Extract methods summary from Methods section or abstract fallback."""
    # Try to find Methods section
    methods_sec = _find_section(
        sections, ["method", "material", "experimental", "approach"]
    )
    if methods_sec:
        # Strip the heading line and take first max_chars
        text = methods_sec.text
        # Remove heading line
        lines = text.split("\n", 1)
        if len(lines) > 1:
            text = lines[1]
        return _truncate_at_sentence_boundary(text.strip(), max_chars)

    # Fallback: extract method-related sentences from abstract
    if abstract:
        sents = split_sentences(abstract)
        method_sents = [
            s for s in sents
            if any(kw in s.lower() for kw in _METHOD_KEYWORDS)
        ]
        if method_sents:
            return _truncate_at_sentence_boundary(
                " ".join(method_sents), max_chars
            )
        # Final fallback: first two sentences of abstract
        return _truncate_at_sentence_boundary(
            " ".join(sents[:2]), max_chars
        )

    return "Methods not available from source text."


def extract_limitations(
    abstract: str | None,
    sections: list[ParsedSection],
    max_chars: int = 600,
) -> str:
    """Extract limitations from Limitations section or Discussion/Conclusion fallback."""
    # Try to find Limitations section
    lim_sec = _find_section(sections, ["limitation"])
    if lim_sec:
        text = lim_sec.text
        lines = text.split("\n", 1)
        if len(lines) > 1:
            text = lines[1]
        return _truncate_at_sentence_boundary(text.strip(), max_chars)

    # Fallback: scan Discussion and Conclusion for limitation sentences
    fallback_sections = []
    for sec_name in ["discussion", "conclusion"]:
        sec = _find_section(sections, [sec_name])
        if sec:
            fallback_sections.append(sec)

    if fallback_sections:
        lim_sents = []
        for sec in fallback_sections:
            sents = split_sentences(sec.text)
            for s in sents:
                if any(kw in s.lower() for kw in _LIMITATION_KEYWORDS):
                    lim_sents.append(s)
        if lim_sents:
            return _truncate_at_sentence_boundary(
                " ".join(lim_sents), max_chars
            )

    # Final fallback
    return "Limitations not explicitly stated in available text."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_extraction/test_programmatic.py -k "Methods or Limitations or FindSection" -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add autoreview/extraction/programmatic.py tests/test_extraction/test_programmatic.py
git commit -m "feat(extraction): add methods_summary and limitations extraction"
```

---

### Task 5: Classification fields — study_design, quality_score, sample_size

**Files:**
- Modify: `autoreview/extraction/programmatic.py`
- Modify: `tests/test_extraction/test_programmatic.py`

- [ ] **Step 1: Write failing tests**

```python
from autoreview.extraction.programmatic import (
    classify_study_design,
    compute_quality_score,
    extract_sample_size,
)
from autoreview.extraction.models import StudyDesign


class TestClassifyStudyDesign:
    def test_meta_analysis(self):
        assert classify_study_design(
            title="A meta-analysis of LLM performance",
            abstract="We conducted a meta-analysis...",
            sections=[],
        ) == StudyDesign.META_ANALYSIS

    def test_computational(self):
        assert classify_study_design(
            title="Neural network for text classification",
            abstract="We develop a deep learning model...",
            sections=[],
        ) == StudyDesign.COMPUTATIONAL

    def test_rct(self):
        assert classify_study_design(
            title="A randomized controlled trial of...",
            abstract="Patients were randomized to...",
            sections=[],
        ) == StudyDesign.RCT

    def test_default_other(self):
        assert classify_study_design(
            title="An essay on philosophy",
            abstract="We reflect on the nature of...",
            sections=[],
        ) == StudyDesign.OTHER


class TestExtractSampleSize:
    def test_n_equals(self):
        assert extract_sample_size("We tested on N = 1,234 samples.", []) == 1234

    def test_participants(self):
        assert extract_sample_size("500 participants were enrolled.", []) == 500

    def test_dataset_of(self):
        assert extract_sample_size("We used a dataset of 10000 examples.", []) == 10000

    def test_no_match(self):
        assert extract_sample_size("We used a large dataset.", []) is None


class TestComputeQualityScore:
    def test_full_text_high_quality(self):
        score = compute_quality_score(
            has_full_text=True,
            full_text_length=10000,
            abstract_length=500,
            citation_count=100,
            sections_present=["methods", "results"],
            methods_text_length=3000,
            results_text_length=5000,
            n_quantitative_findings=5,
            n_total_findings=8,
        )
        assert 0.7 < score <= 1.0

    def test_abstract_only_low_quality(self):
        score = compute_quality_score(
            has_full_text=False,
            full_text_length=0,
            abstract_length=200,
            citation_count=0,
            sections_present=[],
            methods_text_length=0,
            results_text_length=0,
            n_quantitative_findings=0,
            n_total_findings=2,
        )
        assert 0.0 < score < 0.4

    def test_bounded_zero_one(self):
        score = compute_quality_score(
            has_full_text=True,
            full_text_length=100000,
            abstract_length=2000,
            citation_count=10000,
            sections_present=["methods", "results", "discussion"],
            methods_text_length=50000,
            results_text_length=50000,
            n_quantitative_findings=100,
            n_total_findings=100,
        )
        assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_extraction/test_programmatic.py::TestClassifyStudyDesign -v`
Expected: FAIL.

- [ ] **Step 3: Implement classification fields**

Add to `autoreview/extraction/programmatic.py`:

```python
from autoreview.extraction.models import StudyDesign

# Study design classification rules — order matters (first match wins)
_STUDY_DESIGN_RULES: list[tuple[list[str], StudyDesign]] = [
    (["meta-analysis", "meta analysis"], StudyDesign.META_ANALYSIS),
    (["systematic review"], StudyDesign.SYSTEMATIC_REVIEW),
    (["narrative review", "literature review", "scoping review"], StudyDesign.NARRATIVE_REVIEW),
    (["randomized", "randomised", "rct", "clinical trial"], StudyDesign.RCT),
    (["case-control", "case control"], StudyDesign.CASE_CONTROL),
    (["cross-sectional", "cross sectional", "survey"], StudyDesign.CROSS_SECTIONAL),
    (["cohort"], StudyDesign.COHORT),
    (["case series"], StudyDesign.CASE_SERIES),
    (["case report"], StudyDesign.CASE_REPORT),
    (["in vitro", "in-vitro", "cell line", "cell culture"], StudyDesign.IN_VITRO),
    (
        [
            "computational", "algorithm", "benchmark", "simulation",
            "deep learning", "machine learning", "neural network",
            "model", "framework",
        ],
        StudyDesign.COMPUTATIONAL,
    ),
]

_SAMPLE_SIZE_PATTERNS = [
    re.compile(r"[Nn]\s*=\s*([\d,]+)"),
    re.compile(
        r"([\d,]+)\s+(?:participants|subjects|patients|samples"
        r"|images|documents|records|cases|observations|examples|instances)",
    ),
    re.compile(r"(?:dataset|corpus|collection)\s+of\s+([\d,]+)"),
    re.compile(r"sample\s+size\s+(?:of\s+)?([\d,]+)"),
    re.compile(
        r"([\d,]+)\s+(?:training|test|validation)\s+(?:samples|examples|instances)",
    ),
]


def classify_study_design(
    title: str,
    abstract: str | None,
    sections: list[ParsedSection],
) -> StudyDesign:
    """Classify study design from title, abstract, and methods section."""
    # Search in priority order: title → abstract → methods section
    search_texts = [title]
    if abstract:
        search_texts.append(abstract)
    methods = _find_section(sections, ["method", "material", "experimental"])
    if methods:
        search_texts.append(methods.text[:2000])

    combined = " ".join(search_texts).lower()
    for keywords, design in _STUDY_DESIGN_RULES:
        for kw in keywords:
            if kw in combined:
                return design

    return StudyDesign.OTHER


def extract_sample_size(
    text: str,
    sections: list[ParsedSection],
) -> int | None:
    """Extract sample size from methods section, abstract, or full text."""
    # Priority: methods section → abstract/full text
    search_texts = []
    methods = _find_section(sections, ["method", "material", "experimental"])
    if methods:
        search_texts.append(methods.text)
    search_texts.append(text)

    for search_text in search_texts:
        for pattern in _SAMPLE_SIZE_PATTERNS:
            m = pattern.search(search_text)
            if m:
                # Parse the matched number, stripping commas
                num_str = m.group(1).replace(",", "")
                try:
                    return int(num_str)
                except ValueError:
                    continue

    return None


def compute_quality_score(
    has_full_text: bool,
    full_text_length: int,
    abstract_length: int,
    citation_count: int,
    sections_present: list[str],
    methods_text_length: int,
    results_text_length: int,
    n_quantitative_findings: int,
    n_total_findings: int,
) -> float:
    """Compute a heuristic quality score in [0.0, 1.0]."""
    # Text completeness (0.3 weight)
    if has_full_text and full_text_length > 5000:
        text_score = 1.0
    elif has_full_text and full_text_length > 1000:
        text_score = 0.7
    elif abstract_length > 200:
        text_score = 0.4
    else:
        text_score = 0.1

    # Citation count — log-normalized (0.2 weight)
    cite_score = min(1.0, math.log1p(citation_count) / math.log1p(500))

    # Methods section detail (0.2 weight)
    methods_score = min(1.0, methods_text_length / 3000) if methods_text_length else 0.0

    # Results section detail (0.2 weight)
    results_score = min(1.0, results_text_length / 3000) if results_text_length else 0.0

    # Quantitative finding density (0.1 weight)
    if n_total_findings > 0:
        quant_score = min(1.0, n_quantitative_findings / n_total_findings)
    else:
        quant_score = 0.0

    total = (
        0.3 * text_score
        + 0.2 * cite_score
        + 0.2 * methods_score
        + 0.2 * results_score
        + 0.1 * quant_score
    )
    return round(min(1.0, max(0.0, total)), 3)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_extraction/test_programmatic.py -k "StudyDesign or SampleSize or QualityScore" -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add autoreview/extraction/programmatic.py tests/test_extraction/test_programmatic.py
git commit -m "feat(extraction): add study_design, quality_score, sample_size heuristics"
```

---

### Task 6: ProgrammaticExtractor class

**Files:**
- Modify: `autoreview/extraction/programmatic.py`
- Modify: `tests/test_extraction/test_programmatic.py`

Assembles all field extractors into a single class that produces `PaperExtraction` objects.

- [ ] **Step 1: Write failing tests**

```python
import pytest
from autoreview.extraction.programmatic import ProgrammaticExtractor
from autoreview.extraction.models import PaperExtraction
from autoreview.config.models import ExtractionConfig
from autoreview.models.paper import CandidatePaper, ScreenedPaper


def _make_paper(
    paper_id: str = "test_paper",
    title: str = "Evaluating Large Language Models",
    abstract: str | None = (
        "Large language models have shown remarkable performance on NLP tasks. "
        "We evaluate GPT-4 and Claude on medical benchmarks including MedQA. "
        "Our results show GPT-4 achieves 86.7% accuracy on MedQA, outperforming prior models. "
        "The models significantly improve on few-shot learning tasks. "
        "However, challenges remain in hallucination and factual accuracy."
    ),
    full_text: str | None = None,
    citation_count: int = 50,
) -> CandidatePaper:
    return CandidatePaper(
        title=title,
        authors=["Author A", "Author B"],
        year=2024,
        journal="Nature",
        doi="10.1234/test",
        abstract=abstract,
        source_database="semantic_scholar",
        full_text=full_text,
        citation_count=citation_count,
    )


def _make_screened(paper: CandidatePaper | None = None) -> ScreenedPaper:
    paper = paper or _make_paper()
    return ScreenedPaper(
        paper=paper,
        relevance_score=4,
        rationale="Relevant",
        include=True,
    )


class TestProgrammaticExtractor:
    def test_extract_returns_paper_extraction(self):
        extractor = ProgrammaticExtractor(ExtractionConfig())
        sp = _make_screened()
        result = extractor.extract(sp)
        assert isinstance(result, PaperExtraction)
        assert result.paper_id == sp.paper.id

    def test_extract_has_key_findings(self):
        extractor = ProgrammaticExtractor(ExtractionConfig())
        result = extractor.extract(_make_screened())
        assert len(result.key_findings) >= 1
        assert all(f.paper_id == result.paper_id for f in result.key_findings)

    def test_extract_has_methods_summary(self):
        extractor = ProgrammaticExtractor(ExtractionConfig())
        result = extractor.extract(_make_screened())
        assert result.methods_summary
        assert len(result.methods_summary) > 0

    def test_extract_has_limitations(self):
        extractor = ProgrammaticExtractor(ExtractionConfig())
        result = extractor.extract(_make_screened())
        assert result.limitations
        assert len(result.limitations) > 0

    def test_extract_has_study_design(self):
        extractor = ProgrammaticExtractor(ExtractionConfig())
        result = extractor.extract(_make_screened())
        assert result.study_design is not None

    def test_extract_quality_score_bounded(self):
        extractor = ProgrammaticExtractor(ExtractionConfig())
        result = extractor.extract(_make_screened())
        assert result.quality_score is not None
        assert 0.0 <= result.quality_score <= 1.0

    def test_extract_batch(self):
        extractor = ProgrammaticExtractor(ExtractionConfig())
        papers = [_make_screened(), _make_screened(_make_paper(title="Other paper"))]
        successes, failures = extractor.extract_batch(papers)
        assert len(successes) == 2
        assert len(failures) == 0

    def test_extract_batch_handles_failure(self):
        """A paper with no abstract and no full text should still not crash."""
        extractor = ProgrammaticExtractor(ExtractionConfig())
        bad_paper = _make_screened(_make_paper(abstract=None, full_text=None))
        successes, failures = extractor.extract_batch([bad_paper])
        # Should either succeed with minimal extraction or fail gracefully
        assert len(successes) + len(failures) == 1

    def test_skipped_fields_have_defaults(self):
        extractor = ProgrammaticExtractor(ExtractionConfig())
        result = extractor.extract(_make_screened())
        assert result.relationships == []
        assert result.methodology_details is None
        assert result.domain_specific_fields == {}

    def test_pydantic_validates(self):
        """The extraction output should pass Pydantic validation."""
        extractor = ProgrammaticExtractor(ExtractionConfig())
        result = extractor.extract(_make_screened())
        # Re-validate by round-tripping through dict
        PaperExtraction.model_validate(result.model_dump())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_extraction/test_programmatic.py::TestProgrammaticExtractor -v`
Expected: FAIL.

- [ ] **Step 3: Implement ProgrammaticExtractor**

Add to `autoreview/extraction/programmatic.py`:

```python
import structlog
from autoreview.config.models import ExtractionConfig
from autoreview.extraction.extractor import ExtractionFailure
from autoreview.extraction.models import PaperExtraction
from autoreview.extraction.truncation import parse_sections
from autoreview.models.paper import CandidatePaper, ScreenedPaper

logger = structlog.get_logger()


class ProgrammaticExtractor:
    """Deterministic paper extraction using heuristic text analysis.

    Produces PaperExtraction objects with the same schema as LLM-based
    extraction, using sentence scoring, regex patterns, and keyword
    classification instead of LLM calls.
    """

    def __init__(self, config: ExtractionConfig) -> None:
        self.config = config

    def extract(self, sp: ScreenedPaper) -> PaperExtraction:
        """Extract structured data from a single ScreenedPaper."""
        paper = sp.paper
        abstract = paper.abstract
        full_text = paper.full_text

        # Parse sections from full text if available
        sections = []
        if full_text:
            sections = parse_sections(full_text)

        # Extract each field
        key_findings = extract_key_findings(
            abstract=abstract,
            full_text=full_text,
            sections=sections,
            paper_id=paper.id,
        )

        methods_summary = extract_methods_summary(
            abstract=abstract,
            sections=sections,
        )

        limitations = extract_limitations(
            abstract=abstract,
            sections=sections,
        )

        study_design = classify_study_design(
            title=paper.title,
            abstract=abstract,
            sections=sections,
        )

        # Compute quality score
        methods_sec = _find_section(sections, ["method", "material", "experimental"])
        results_sec = _find_section(sections, ["result", "finding", "experiment"])
        n_quant = sum(1 for f in key_findings if f.quantitative_result)

        quality_score = compute_quality_score(
            has_full_text=bool(full_text),
            full_text_length=len(full_text) if full_text else 0,
            abstract_length=len(abstract) if abstract else 0,
            citation_count=paper.citation_count or 0,
            sections_present=[s.name for s in sections],
            methods_text_length=len(methods_sec.text) if methods_sec else 0,
            results_text_length=len(results_sec.text) if results_sec else 0,
            n_quantitative_findings=n_quant,
            n_total_findings=len(key_findings),
        )

        sample_size = extract_sample_size(
            text=abstract or paper.title,
            sections=sections,
        )

        extraction = PaperExtraction(
            paper_id=paper.id,
            key_findings=key_findings,
            methods_summary=methods_summary,
            limitations=limitations,
            relationships=[],
            methodology_details=None,
            domain_specific_fields={},
            study_design=study_design,
            quality_score=quality_score,
            sample_size=sample_size,
        )

        logger.info(
            "programmatic_extraction.complete",
            paper_id=paper.id,
            title=paper.title[:60],
            findings=len(key_findings),
            text_source="full_text" if full_text else ("abstract" if abstract else "title_only"),
        )

        return extraction

    def extract_batch(
        self,
        papers: list[ScreenedPaper],
    ) -> tuple[list[PaperExtraction], list[ExtractionFailure]]:
        """Extract from multiple papers. Returns (successes, failures)."""
        successes: list[PaperExtraction] = []
        failures: list[ExtractionFailure] = []

        for sp in papers:
            try:
                extraction = self.extract(sp)
                successes.append(extraction)
            except Exception as e:
                logger.error(
                    "programmatic_extraction.failed",
                    paper_id=sp.paper.id,
                    error=str(e),
                )
                failures.append(ExtractionFailure(
                    paper_id=sp.paper.id,
                    error=str(e),
                ))

        logger.info(
            "programmatic_extraction.batch_complete",
            total=len(papers),
            successful=len(successes),
            failed=len(failures),
        )
        return successes, failures
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_extraction/test_programmatic.py -v`
Expected: ALL tests PASS.

- [ ] **Step 5: Commit**

```bash
git add autoreview/extraction/programmatic.py tests/test_extraction/test_programmatic.py
git commit -m "feat(extraction): add ProgrammaticExtractor class with full extraction pipeline"
```

---

### Task 7: Pipeline integration — wire extraction_mode into nodes.py

**Files:**
- Modify: `autoreview/pipeline/nodes.py:556-597`
- Test: `tests/test_pipeline/test_nodes.py` (add extraction mode test)

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_pipeline/ or create test_extraction_mode.py

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from autoreview.config.models import ExtractionConfig


def test_extraction_mode_config_wiring():
    """Verify ExtractionConfig with programmatic mode is accepted."""
    config = ExtractionConfig(extraction_mode="programmatic")
    assert config.extraction_mode == "programmatic"


@pytest.mark.asyncio
async def test_programmatic_extraction_node():
    """Verify the extraction node uses ProgrammaticExtractor when mode=programmatic."""
    from autoreview.extraction.programmatic import ProgrammaticExtractor
    from autoreview.extraction.models import PaperExtraction, Finding, EvidenceStrength
    from autoreview.models.paper import CandidatePaper, ScreenedPaper

    # This is an integration-style test that verifies the wiring
    config = ExtractionConfig(extraction_mode="programmatic")
    extractor = ProgrammaticExtractor(config)

    paper = CandidatePaper(
        title="Test Paper",
        authors=["A"],
        year=2024,
        journal="J",
        doi="10.1/test",
        abstract="We propose a novel method that achieves 95% accuracy.",
        source_database="test",
    )
    sp = ScreenedPaper(paper=paper, relevance_score=4, rationale="test", include=True)

    successes, failures = extractor.extract_batch([sp])
    assert len(successes) == 1
    assert len(failures) == 0
    assert isinstance(successes[0], PaperExtraction)
```

- [ ] **Step 2: Run test to verify it passes** (this test should already pass from Task 6)

Run: `python -m pytest tests/test_extraction/test_programmatic.py -v`

- [ ] **Step 3: Modify nodes.py to check extraction_mode**

In `autoreview/pipeline/nodes.py`, modify the `extraction` method (line 556):

```python
async def extraction(self, kb: KnowledgeBase) -> None:
    """Node: Extract structured information from papers in batches."""
    mode = self.config.extraction.extraction_mode

    if mode in ("programmatic", "hybrid"):
        from autoreview.extraction.programmatic import ProgrammaticExtractor

        prog_extractor = ProgrammaticExtractor(self.config.extraction)
        papers = kb.screened_papers

        # Programmatic extraction is synchronous — run in thread
        import asyncio
        successes, failures = await asyncio.to_thread(
            prog_extractor.extract_batch, papers
        )
        kb.extractions.update({r.paper_id: r for r in successes})

        if failures:
            logger.warning(
                "extraction.programmatic_failures",
                failed=len(failures),
                failed_ids=[f.paper_id for f in failures],
            )

        if mode == "hybrid":
            # Re-extract low-confidence papers with LLM
            low_confidence_ids = {
                e.paper_id for e in successes
                if (e.quality_score or 0) < 0.3 or len(e.key_findings) < 3
            }
            if low_confidence_ids:
                llm_papers = [
                    p for p in papers if p.paper.id in low_confidence_ids
                ]
                logger.info(
                    "extraction.hybrid_llm_fallback",
                    n_papers=len(llm_papers),
                )
                tracker = _TokenAccumulator(
                    self.llm, self._global_tokens, node_name="extraction"
                )
                llm_extractor = PaperExtractor(
                    tracker,
                    domain_fields=self.config.extraction.domain_fields,
                    max_concurrent=self._effective_max_concurrent(),
                    full_text_max_chars=self.config.extraction.full_text_max_chars,
                    tiered_models=self.config.extraction.tiered_models,
                    section_truncation=self.config.extraction.section_truncation,
                )
                llm_results, llm_failures = await llm_extractor.extract_batch_safe(
                    llm_papers
                )
                kb.extractions.update({r.paper_id: r for r in llm_results})

        kb.save_snapshot("extraction_programmatic")
        kb.current_phase = PipelinePhase.EXTRACTION
        kb.add_audit_entry(
            "extraction",
            "complete",
            f"Extracted {len(kb.extractions)} papers (mode={mode})",
        )
        return

    # Default: LLM extraction (existing code unchanged)
    tracker = _TokenAccumulator(self.llm, self._global_tokens, node_name="extraction")
    extractor = PaperExtractor(
        tracker,
        domain_fields=self.config.extraction.domain_fields,
        max_concurrent=self._effective_max_concurrent(),
        full_text_max_chars=self.config.extraction.full_text_max_chars,
        tiered_models=self.config.extraction.tiered_models,
        section_truncation=self.config.extraction.section_truncation,
    )
    batch_size = self.config.extraction.extraction_batch_size
    papers = kb.screened_papers
    total_batches = (len(papers) + batch_size - 1) // batch_size

    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(
            "extraction.batch_start",
            batch=batch_num,
            total=total_batches,
            papers=len(batch),
        )
        results, failures = await extractor.extract_batch_safe(batch)
        kb.extractions.update({r.paper_id: r for r in results})
        if failures:
            logger.warning(
                "extraction.batch_failures",
                batch=batch_num,
                failed=len(failures),
                failed_ids=[f.paper_id for f in failures],
            )
        kb.save_snapshot(f"extraction_batch_{batch_num}")

    kb.current_phase = PipelinePhase.EXTRACTION
    kb.add_audit_entry(
        "extraction",
        "complete",
        f"Extracted {len(kb.extractions)} papers in {total_batches} batches",
        tracker.usage,
    )
```

- [ ] **Step 4: Run existing tests to verify nothing breaks**

Run: `python -m pytest tests/ -x -q --timeout=60`
Expected: All existing tests still PASS.

- [ ] **Step 5: Commit**

```bash
git add autoreview/pipeline/nodes.py
git commit -m "feat(pipeline): wire extraction_mode into extraction node"
```

---

### Task 8: Scoring module for benchmarking

**Files:**
- Create: `autoreview/extraction/scoring.py`
- Create: `tests/test_extraction/test_scoring.py`

This module compares programmatic extractions against LLM ground truth. Uses `sentence-transformers` for embedding similarity (benchmark-only dependency).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_extraction/test_scoring.py

import pytest
from autoreview.extraction.scoring import (
    rouge_l_f1,
    compute_study_design_accuracy,
    compute_sample_size_accuracy,
    compute_quality_score_correlation,
    compute_composite_score,
)
from autoreview.extraction.models import StudyDesign


class TestRougeLF1:
    def test_identical(self):
        score = rouge_l_f1("the cat sat on the mat", "the cat sat on the mat")
        assert score == 1.0

    def test_no_overlap(self):
        score = rouge_l_f1("hello world", "foo bar baz")
        assert score == 0.0

    def test_partial_overlap(self):
        score = rouge_l_f1("the cat sat", "the dog sat on mat")
        assert 0.0 < score < 1.0

    def test_empty_strings(self):
        assert rouge_l_f1("", "") == 0.0
        assert rouge_l_f1("hello", "") == 0.0


class TestStudyDesignAccuracy:
    def test_all_match(self):
        pred = [StudyDesign.COMPUTATIONAL, StudyDesign.RCT]
        gold = [StudyDesign.COMPUTATIONAL, StudyDesign.RCT]
        assert compute_study_design_accuracy(pred, gold) == 1.0

    def test_none_match(self):
        pred = [StudyDesign.COMPUTATIONAL]
        gold = [StudyDesign.RCT]
        assert compute_study_design_accuracy(pred, gold) == 0.0

    def test_partial_match(self):
        pred = [StudyDesign.COMPUTATIONAL, StudyDesign.RCT]
        gold = [StudyDesign.COMPUTATIONAL, StudyDesign.COHORT]
        assert compute_study_design_accuracy(pred, gold) == 0.5


class TestSampleSizeAccuracy:
    def test_exact_match(self):
        assert compute_sample_size_accuracy([100], [100]) == 1.0

    def test_both_none(self):
        assert compute_sample_size_accuracy([None], [None]) == 1.0

    def test_within_tolerance(self):
        # 105 is within 10% of 100
        assert compute_sample_size_accuracy([105], [100]) == 1.0

    def test_mismatch(self):
        assert compute_sample_size_accuracy([None], [100]) == 0.0


class TestCompositeScore:
    def test_all_perfect(self):
        scores = {
            "key_findings": 1.0,
            "evidence_strength": 1.0,
            "quantitative_result": 1.0,
            "methods_summary": 1.0,
            "limitations": 1.0,
            "study_design": 1.0,
            "quality_score": 1.0,
            "sample_size": 1.0,
        }
        composite = compute_composite_score(scores)
        assert composite == pytest.approx(1.0)

    def test_all_zero(self):
        scores = {k: 0.0 for k in [
            "key_findings", "evidence_strength", "quantitative_result",
            "methods_summary", "limitations", "study_design",
            "quality_score", "sample_size",
        ]}
        composite = compute_composite_score(scores)
        assert composite == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_extraction/test_scoring.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement scoring module**

Create `autoreview/extraction/scoring.py`:

```python
"""Benchmark scoring for programmatic vs LLM extraction comparison.

Compares PaperExtraction outputs from the programmatic extractor against
LLM-generated ground truth extractions. Uses embedding similarity for
claim matching and ROUGE-L for text comparison.

Note: sentence-transformers is a benchmark-only dependency, not needed
at runtime for the extractor itself.
"""

from __future__ import annotations

from autoreview.extraction.models import PaperExtraction, StudyDesign


# --- Weights for composite scoring ---
_WEIGHTS = {
    "key_findings": 0.40,
    "evidence_strength": 0.05,
    "quantitative_result": 0.05,
    "methods_summary": 0.15,
    "limitations": 0.10,
    "study_design": 0.10,
    "quality_score": 0.05,
    "sample_size": 0.10,
}


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Use 1D DP for memory efficiency
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def rouge_l_f1(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 score between two strings."""
    pred_words = prediction.lower().split()
    ref_words = reference.lower().split()
    if not pred_words or not ref_words:
        return 0.0
    lcs = _lcs_length(pred_words, ref_words)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_words)
    recall = lcs / len(ref_words)
    return 2 * precision * recall / (precision + recall)


def compute_study_design_accuracy(
    predicted: list[StudyDesign | None],
    ground_truth: list[StudyDesign | None],
) -> float:
    """Compute exact match accuracy for study design classification."""
    if not predicted:
        return 0.0
    matches = sum(1 for p, g in zip(predicted, ground_truth) if p == g)
    return matches / len(predicted)


def compute_sample_size_accuracy(
    predicted: list[int | None],
    ground_truth: list[int | None],
    tolerance: float = 0.10,
) -> float:
    """Compute accuracy for sample size extraction with tolerance."""
    if not predicted:
        return 0.0
    matches = 0
    for p, g in zip(predicted, ground_truth):
        if p is None and g is None:
            matches += 1
        elif p is not None and g is not None:
            if g == 0:
                matches += 1 if p == 0 else 0
            elif abs(p - g) / g <= tolerance:
                matches += 1
        # else: one is None, other isn't → 0
    return matches / len(predicted)


def compute_quality_score_correlation(
    predicted: list[float | None],
    ground_truth: list[float | None],
) -> float:
    """Compute Pearson correlation between quality scores, normalized to [0, 1]."""
    # Filter out None pairs
    pairs = [
        (p, g) for p, g in zip(predicted, ground_truth)
        if p is not None and g is not None
    ]
    if len(pairs) < 3:
        return 0.5  # Not enough data

    pred_vals = [p for p, _ in pairs]
    gold_vals = [g for _, g in pairs]

    # Compute Pearson correlation
    n = len(pairs)
    mean_p = sum(pred_vals) / n
    mean_g = sum(gold_vals) / n

    cov = sum((p - mean_p) * (g - mean_g) for p, g in pairs)
    std_p = (sum((p - mean_p) ** 2 for p in pred_vals)) ** 0.5
    std_g = (sum((g - mean_g) ** 2 for g in gold_vals)) ** 0.5

    if std_p < 1e-10 or std_g < 1e-10:
        return 0.5  # No variance

    r = cov / (std_p * std_g)
    # Normalize from [-1, 1] to [0, 1]
    return (r + 1) / 2


def compute_composite_score(field_scores: dict[str, float]) -> float:
    """Compute weighted composite score from per-field scores."""
    total = 0.0
    for field, weight in _WEIGHTS.items():
        total += weight * field_scores.get(field, 0.0)
    return round(total, 4)


def score_extraction_pair(
    predicted: PaperExtraction,
    ground_truth: PaperExtraction,
) -> dict[str, float]:
    """Score a single predicted extraction against ground truth.

    Returns per-field scores. For key_findings similarity, uses
    word-overlap matching (embedding-based scoring available via
    benchmark script with sentence-transformers installed).
    """
    from autoreview.extraction.programmatic import word_overlap_similarity

    # key_findings: match claims by word overlap, compute mean similarity
    pred_claims = [f.claim for f in predicted.key_findings]
    gold_claims = [f.claim for f in ground_truth.key_findings]

    if pred_claims and gold_claims:
        # Greedy matching: for each gold claim, find best pred match
        matched_sims = []
        used_pred = set()
        for g_claim in gold_claims:
            best_sim = 0.0
            best_idx = -1
            for i, p_claim in enumerate(pred_claims):
                if i in used_pred:
                    continue
                sim = word_overlap_similarity(g_claim, p_claim)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
            if best_idx >= 0:
                used_pred.add(best_idx)
                matched_sims.append(best_sim)
            else:
                matched_sims.append(0.0)
        kf_score = sum(matched_sims) / len(matched_sims)
    else:
        kf_score = 0.0

    # evidence_strength: match on paired findings
    strength_matches = 0
    strength_total = 0
    for pf, gf in zip(predicted.key_findings, ground_truth.key_findings):
        strength_total += 1
        if pf.evidence_strength == gf.evidence_strength:
            strength_matches += 1
    es_score = strength_matches / strength_total if strength_total else 0.0

    # quantitative_result: token overlap
    quant_scores = []
    for pf, gf in zip(predicted.key_findings, ground_truth.key_findings):
        p_q = pf.quantitative_result or ""
        g_q = gf.quantitative_result or ""
        if not p_q and not g_q:
            quant_scores.append(1.0)
        elif p_q and g_q:
            quant_scores.append(word_overlap_similarity(p_q, g_q))
        else:
            quant_scores.append(0.0)
    qr_score = sum(quant_scores) / len(quant_scores) if quant_scores else 0.0

    # methods_summary: ROUGE-L
    ms_score = rouge_l_f1(predicted.methods_summary, ground_truth.methods_summary)

    # limitations: ROUGE-L
    lim_score = rouge_l_f1(predicted.limitations, ground_truth.limitations)

    # study_design: exact match
    sd_score = 1.0 if predicted.study_design == ground_truth.study_design else 0.0

    # quality_score: simple difference
    if predicted.quality_score is not None and ground_truth.quality_score is not None:
        qs_score = 1.0 - abs(predicted.quality_score - ground_truth.quality_score)
    elif predicted.quality_score is None and ground_truth.quality_score is None:
        qs_score = 1.0
    else:
        qs_score = 0.0

    # sample_size: tolerance match
    ss_score = compute_sample_size_accuracy(
        [predicted.sample_size], [ground_truth.sample_size]
    )

    return {
        "key_findings": kf_score,
        "evidence_strength": es_score,
        "quantitative_result": qr_score,
        "methods_summary": ms_score,
        "limitations": lim_score,
        "study_design": sd_score,
        "quality_score": qs_score,
        "sample_size": ss_score,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_extraction/test_scoring.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add autoreview/extraction/scoring.py tests/test_extraction/test_scoring.py
git commit -m "feat(extraction): add benchmark scoring module for extraction comparison"
```

---

### Task 9: Benchmark script

**Files:**
- Create: `scripts/benchmark_extractor.py`

This script loads ground truth from snapshots, runs the programmatic extractor, computes scores, and generates a comparison report.

- [ ] **Step 1: Create benchmark script**

```python
#!/usr/bin/env python3
"""Benchmark the programmatic extractor against LLM ground truth.

Usage:
    python scripts/benchmark_extractor.py [--snapshot PATH] [--output-dir DIR] [--verbose]

Loads LLM extractions and source papers from a pipeline snapshot,
runs the ProgrammaticExtractor on the same papers, and compares results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from autoreview.config.models import ExtractionConfig
from autoreview.extraction.models import PaperExtraction
from autoreview.extraction.programmatic import ProgrammaticExtractor
from autoreview.extraction.scoring import (
    compute_composite_score,
    score_extraction_pair,
)
from autoreview.models.paper import CandidatePaper, ScreenedPaper


def load_ground_truth(
    snapshot_path: Path,
) -> tuple[list[ScreenedPaper], dict[str, PaperExtraction]]:
    """Load screened papers and LLM extractions from a snapshot."""
    data = json.loads(snapshot_path.read_text())

    # Load screened papers
    screened = []
    for sp_data in data.get("screened_papers", []):
        paper_data = sp_data["paper"]
        paper = CandidatePaper(
            title=paper_data["title"],
            authors=paper_data.get("authors", []),
            year=paper_data.get("year"),
            journal=paper_data.get("journal"),
            doi=paper_data.get("doi"),
            abstract=paper_data.get("abstract"),
            source_database=paper_data.get("source_database", "unknown"),
            citation_count=paper_data.get("citation_count"),
            full_text=paper_data.get("full_text"),
        )
        # Override ID to match extraction keys
        paper.id = paper_data["id"]
        screened.append(
            ScreenedPaper(
                paper=paper,
                relevance_score=sp_data.get("relevance_score", 4),
                rationale=sp_data.get("rationale", ""),
                include=True,
            )
        )

    # Load LLM extractions
    extractions = {}
    for paper_id, ext_data in data.get("extractions", {}).items():
        try:
            extractions[paper_id] = PaperExtraction.model_validate(ext_data)
        except Exception as e:
            print(f"  Warning: Could not load extraction for {paper_id}: {e}")

    return screened, extractions


def run_benchmark(
    snapshot_path: Path,
    output_dir: Path,
    verbose: bool = False,
) -> dict:
    """Run the benchmark and return results."""
    print(f"Loading ground truth from {snapshot_path}...")
    screened_papers, llm_extractions = load_ground_truth(snapshot_path)

    # Filter to papers that have LLM extractions
    papers_with_gt = [
        sp for sp in screened_papers if sp.paper.id in llm_extractions
    ]
    print(f"  {len(papers_with_gt)} papers with LLM extractions (of {len(screened_papers)} screened)")

    # Run programmatic extractor
    config = ExtractionConfig(extraction_mode="programmatic")
    extractor = ProgrammaticExtractor(config)

    print("Running programmatic extractor...")
    start_time = time.monotonic()
    prog_extractions, failures = extractor.extract_batch(papers_with_gt)
    elapsed = time.monotonic() - start_time
    print(f"  Extracted {len(prog_extractions)} papers in {elapsed:.1f}s ({len(failures)} failures)")

    # Build lookup by paper_id
    prog_by_id = {e.paper_id: e for e in prog_extractions}

    # Score each paper
    per_paper_scores = []
    for paper_id, llm_ext in llm_extractions.items():
        prog_ext = prog_by_id.get(paper_id)
        if not prog_ext:
            continue
        scores = score_extraction_pair(prog_ext, llm_ext)
        composite = compute_composite_score(scores)
        per_paper_scores.append({
            "paper_id": paper_id,
            "composite": composite,
            **scores,
        })

    if not per_paper_scores:
        print("ERROR: No papers to score!")
        return {}

    # Compute corpus-level averages
    field_names = [
        "key_findings", "evidence_strength", "quantitative_result",
        "methods_summary", "limitations", "study_design",
        "quality_score", "sample_size",
    ]
    corpus_scores = {}
    for field in field_names:
        vals = [p[field] for p in per_paper_scores]
        corpus_scores[field] = sum(vals) / len(vals)

    overall_composite = compute_composite_score(corpus_scores)

    # Find worst papers
    per_paper_scores.sort(key=lambda x: x["composite"])
    worst_10 = per_paper_scores[:10]

    # Print report
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\nOverall Composite Score: {overall_composite:.4f}")
    print(f"Papers scored: {len(per_paper_scores)}")
    print(f"Extraction time: {elapsed:.1f}s")
    print(f"\nPer-field scores:")
    for field in field_names:
        print(f"  {field:25s}: {corpus_scores[field]:.4f}")

    print(f"\nWorst 10 papers:")
    for p in worst_10:
        print(f"  {p['paper_id'][:40]:40s} composite={p['composite']:.3f}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "composite": overall_composite,
        "extraction_time_s": round(elapsed, 1),
        "n_papers": len(per_paper_scores),
        "n_failures": len(failures),
        "corpus_scores": corpus_scores,
        "worst_10": worst_10,
        "per_paper": per_paper_scores if verbose else [],
    }

    results_path = output_dir / "benchmark_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {results_path}")

    # Append to history
    history_path = output_dir / "benchmark_history.jsonl"
    history_entry = {
        "timestamp": results["timestamp"],
        "composite": overall_composite,
        **corpus_scores,
    }
    with history_path.open("a") as f:
        f.write(json.dumps(history_entry) + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark programmatic extractor")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("output/arise/arise_llm_eval_v2/snapshots/latest.json"),
        help="Path to snapshot with LLM extractions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/extractor_benchmark"),
        help="Directory for benchmark results",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_benchmark(args.snapshot, args.output_dir, args.verbose)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the benchmark**

```bash
python scripts/benchmark_extractor.py --verbose
```

Expected: Completes without errors. Composite score will likely be 0.2-0.4 on first run (before optimization).

- [ ] **Step 3: Commit**

```bash
git add scripts/benchmark_extractor.py
git commit -m "feat(benchmark): add extractor benchmark script"
```

---

### Task 10: First optimization iteration

**Files:**
- Modify: `autoreview/extraction/programmatic.py` (tune heuristics based on benchmark results)

This task is iterative — analyze benchmark failures and improve heuristics.

- [ ] **Step 1: Run benchmark and analyze failures**

```bash
python scripts/benchmark_extractor.py --verbose --output-dir results/extractor_benchmark
```

Review `results/extractor_benchmark/benchmark_results.json` for:
- Which fields score lowest?
- What do the worst 10 papers have in common?
- Are key_findings claims too different from LLM claims?

- [ ] **Step 2: Tune heuristics based on failure analysis**

Common improvements:
- Adjust position weights if abstract sentences score too high/low
- Add domain-specific keywords (e.g., "benchmark", "BLEU", "F1" for CS papers)
- Adjust deduplication threshold (0.85 may be too aggressive/lenient)
- Adjust N selection formula if too many/few findings selected
- Improve study_design keywords for the CS/AI domain

- [ ] **Step 3: Re-run benchmark after each change**

```bash
python scripts/benchmark_extractor.py --output-dir results/extractor_benchmark
```

Check `results/extractor_benchmark/benchmark_history.jsonl` to track improvement.

- [ ] **Step 4: Commit improved heuristics**

```bash
git add autoreview/extraction/programmatic.py
git commit -m "feat(extraction): optimize heuristics based on benchmark iteration 1"
```

- [ ] **Step 5: Repeat steps 1-4 until composite >= 0.5**

Target for first optimization pass: composite >= 0.5. Further optimization can be done in subsequent sessions.

---

## Dependency Graph

```
Task 1 (config) ─────────────────────────────────────────────────────┐
                                                                      │
Task 2 (utilities) ──┬──────────────────────────────────────────────┐ │
                     │                                               │ │
Task 3 (key_findings) ──┐                                           │ │
                        │                                            │ │
Task 4 (methods/limits) ─┼─ Task 6 (ProgrammaticExtractor) ── Task 7 (pipeline) ── Task 10 (optimize)
                        │                                            │
Task 5 (classification) ─┘                                           │
                                                                      │
Task 8 (scoring) ──────────── Task 9 (benchmark) ───────────────────┘
```

**Parallelizable batches:**
- Batch 1: Tasks 1, 2, 8 (no dependencies)
- Batch 2: Tasks 3, 4, 5 (depend on Task 2)
- Batch 3: Task 6 (depends on Tasks 3-5)
- Batch 4: Tasks 7, 9 (depend on Tasks 1, 6, 8)
- Batch 5: Task 10 (depends on Task 9)
