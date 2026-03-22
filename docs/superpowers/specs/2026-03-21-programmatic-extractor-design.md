# Programmatic Paper Extractor — Design Spec

> **For agentic workers:** This spec defines a deterministic Python extraction function that replaces LLM-based paper extraction. Read the full spec before implementing. Key files: `autoreview/extraction/programmatic.py` (new), `autoreview/extraction/scoring.py` (new), `autoreview/extraction/models.py` (existing — `PaperExtraction`, `Finding`, `StudyDesign`, `EvidenceStrength`), `autoreview/extraction/truncation.py` (existing — `parse_sections`, `ParsedSection`), `autoreview/config/models.py` (modify `ExtractionConfig`).

**Date:** 2026-03-21
**Status:** Draft
**Author:** Claude (brainstormed with user)

**Goal:** Reduce paper extraction from ~80 minutes and ~25M tokens (LLM-based) to <1 minute and 0 tokens (deterministic Python), while maintaining sufficient quality for downstream pipeline stages.

**Architecture:** New `ProgrammaticExtractor` class producing identical `PaperExtraction` objects, selected via `extraction_mode` config option, benchmarked against 220 LLM extractions as ground truth.

**Tech Stack:** Pure Python + regex for the extractor itself (no new runtime dependencies). `sentence-transformers` and `scipy` for benchmark scoring only.

---

## Problem Statement

The AutoReview pipeline extracts structured data from scientific papers using an LLM (Claude) at ~50K tokens per paper. For a 500-paper corpus, this costs ~25M tokens and ~80 minutes of wall-clock time. Extraction is the single most expensive pipeline stage.

Most extracted fields (methods summary, limitations, study design, sample size) follow predictable patterns that can be captured with heuristics. Even the hardest field — `key_findings` — can be approximated by sentence-level scoring, since the LLM is fundamentally selecting and reformulating sentences from the source text.

We have 220 LLM extractions with source papers available as ground truth for benchmarking.

## Design Principles

1. **Same interface, different engine.** `ProgrammaticExtractor` returns `PaperExtraction` objects identical to what `PaperExtractor` produces. Downstream stages are completely agnostic.
2. **Good enough beats perfect.** Heuristic extraction will differ in style from LLM output. The target is functional equivalence (downstream review quality), not textual similarity.
3. **Zero runtime dependencies.** The extractor uses only Python stdlib + regex. Embedding-based scoring is benchmark-only, not required at runtime.
4. **Leverage existing infrastructure.** Reuse `truncation.py` section detection (`parse_sections`, `ParsedSection`) rather than reimplementing text parsing.

## Downstream Consumer Analysis

Understanding which fields matter and to whom drives prioritization:

| Field | Consumers | Priority |
|-------|-----------|----------|
| `key_findings` | Clustering, gap search, outline, narrative, citation selection, section writing | **Critical** (6/7 downstream stages) |
| `methods_summary` | Outline, narrative, citation selection, section writing | **High** (4 consumers) |
| `limitations` | Narrative, section writing | **Medium** (2 consumers) |
| `study_design` | Contradiction resolver only | **Low** (1 consumer) |
| `quality_score` | Contradiction resolver only | **Low** (1 consumer) |
| `sample_size` | Contradiction resolver only | **Low** (1 consumer) |
| `relationships` | Evidence chain builder | **Skip** (safe to return `[]`) |
| `methodology_details` | None (confirmed unused) | **Skip** (safe to return `None`) |
| `domain_specific_fields` | None (confirmed unused) | **Skip** (safe to return `{}`) |

---

## Architecture

### New Files

| File | Purpose |
|------|---------|
| `autoreview/extraction/programmatic.py` | `ProgrammaticExtractor` class — deterministic extraction |
| `autoreview/extraction/scoring.py` | Benchmark scoring functions (embedding similarity, ROUGE-L, exact match) |
| `scripts/benchmark_extractor.py` | Benchmark runner — loads ground truth, runs extractor, generates comparison report |

### Modified Files

| File | Change |
|------|--------|
| `autoreview/config/models.py` | Add `extraction_mode: Literal["llm", "programmatic", "hybrid"]` to `ExtractionConfig` |
| `autoreview/pipeline/nodes.py` | Check `extraction_mode` in extraction node, instantiate appropriate extractor |

### Class Design

```python
class ProgrammaticExtractor:
    """Deterministic paper extraction using heuristic text analysis.

    Produces PaperExtraction objects with the same schema as LLM-based
    extraction, using sentence scoring, regex patterns, and keyword
    classification instead of LLM calls.
    """

    def __init__(self, config: ExtractionConfig) -> None:
        self.config = config

    def extract(self, paper: ScreenedPaper) -> PaperExtraction:
        """Extract structured data from a single paper.

        Uses paper.abstract, paper.full_text (if available), and paper metadata
        to produce a PaperExtraction without any LLM calls.
        """
        ...

    def extract_batch(
        self, papers: list[ScreenedPaper]
    ) -> tuple[list[PaperExtraction], list[ExtractionFailure]]:
        """Extract from multiple papers. Returns (successes, failures)."""
        ...
```

The class is synchronous — no LLM calls means no async I/O needed. The pipeline node wraps it with `asyncio.to_thread()` if called from an async context.

---

## Field Extraction Strategies

### `key_findings` — Sentence Scoring Algorithm

This is the hardest and most important field. The strategy is to score every sentence in the paper and select the top N as findings.

**Input text priority:**
1. Full text (if available) — parse into sections via `truncation.parse_sections()`
2. Abstract only — all sentences are candidates

**Sentence scoring function:**

Each sentence receives a composite score from four signal categories:

```
score = position_weight + keyword_weight + quantitative_weight + novelty_weight
```

#### Position Weight (0.0 — 0.4)

| Location | Weight |
|----------|--------|
| Abstract — last 2 sentences | 0.40 |
| Abstract — first 2 sentences | 0.30 |
| Abstract — middle sentences | 0.25 |
| Conclusion section | 0.35 |
| Results section | 0.30 |
| Discussion section | 0.25 |
| Introduction section | 0.10 |
| Methods section | 0.05 |
| Other / unknown section | 0.15 |

#### Keyword Weight (0.0 — 0.3)

Additive, capped at 0.3. Each matching keyword/phrase contributes +0.05:

- **Result indicators:** "we found", "results show", "results demonstrate", "results indicate", "our findings", "analysis reveals", "we observed", "data suggest"
- **Performance claims:** "outperforms", "achieves", "surpasses", "improves upon", "state-of-the-art", "superior to"
- **Significance markers:** "significant", "significantly", "p <", "p =", "p-value"
- **Contribution claims:** "we propose", "we introduce", "we present", "we develop", "this paper presents", "our contribution"

#### Quantitative Weight (0.0 — 0.2)

- Sentence contains a percentage (e.g., "67.6%"): +0.10
- Sentence contains a p-value (e.g., "p < 0.001"): +0.10
- Sentence contains a confidence interval: +0.05
- Sentence contains comparison numbers (e.g., "from X to Y"): +0.05
- Capped at 0.20

#### Novelty Weight (0.0 — 0.1)

- Sentence contains novelty signals: "novel", "first", "new approach", "for the first time", "we propose", "we introduce": +0.10

**Selection:**
- Score all sentences
- Deduplicate: if two sentences have cosine similarity > 0.85 (using simple word overlap, not embeddings), keep the higher-scored one
- Select top N sentences where N = `min(max(5, paper_length // 1000), 15)`
  - Short papers (< 5K chars): 5 findings
  - Medium papers (5K-15K chars): 5-15 findings (1 per 1000 chars)
  - Long papers (> 15K chars): 15 findings (cap)

**Finding construction from selected sentences:**

```python
Finding(
    claim=sentence_text,           # The selected sentence as-is
    evidence_strength=...,         # See evidence strength heuristic below
    quantitative_result=...,       # Regex-extracted numbers (see below)
    context=section_header,        # Section name from parse_sections()
    paper_id=paper.paper_id,
)
```

#### Evidence Strength Heuristic

Determined per-finding based on text source and content:

| Condition | Strength |
|-----------|----------|
| Full text available AND sentence contains quantitative data (numbers, p-values, CIs) | `STRONG` |
| Full text available AND sentence is from Results/Discussion section | `MODERATE` |
| Full text available AND sentence is from Abstract only | `MODERATE` |
| Abstract only AND sentence contains quantitative data | `MODERATE` |
| Abstract only AND no quantitative data | `PRELIMINARY` |
| Sentence contains hedging language ("may", "might", "preliminary", "pilot", "small sample") | Downgrade one level |

#### Quantitative Result Extraction

Regex patterns applied to each selected sentence:

```python
QUANT_PATTERNS = [
    r"(\d+\.?\d*)\s*%",                          # percentages
    r"p\s*[<>=]\s*\d+\.?\d*",                     # p-values
    r"(?:CI|confidence interval)[:\s]*\[?\(?\d+\.?\d*\s*[-–,]\s*\d+\.?\d*\]?\)?",  # CIs
    r"[Nn]\s*=\s*[\d,]+",                         # sample sizes
    r"(\d+\.?\d*)\s*±\s*(\d+\.?\d*)",             # mean ± SD
    r"(?:AUC|accuracy|precision|recall|F1)[:\s]*(?:of\s+)?(\d+\.?\d*)",  # metrics
    r"(\d+\.?\d*)\s*(?:fold|times)\s+(?:increase|decrease|improvement)",  # fold changes
]
```

Matched patterns are concatenated with "; " separator into `quantitative_result`. If no patterns match, field is `None`.

---

### `methods_summary`

**Strategy:** Extract and truncate the Methods section text.

1. Parse sections via `truncation.parse_sections()`
2. Find section matching Methods/Methodology/Materials/Experimental (case-insensitive)
3. Take the first 500 characters, truncated at sentence boundary
4. **Fallback** (no Methods section found or abstract-only): Extract sentences from abstract containing method keywords: "method", "approach", "dataset", "model", "algorithm", "technique", "framework", "we use", "we train", "we employ", "we apply", "using"
5. Concatenate matched abstract sentences, truncate to 500 chars at sentence boundary

---

### `limitations`

**Strategy:** Extract Limitations section or infer from Discussion/Conclusion.

1. Find section matching "Limitation" (case-insensitive, partial match)
2. If found: take first 600 characters, truncated at sentence boundary
3. **Fallback:** Scan Discussion and Conclusion sections for sentences containing: "limitation", "weakness", "caveat", "future work", "further research", "however", "challenge", "drawback", "constraint", "shortcoming", "remains to be"
4. Concatenate matched sentences, truncate to 600 chars at sentence boundary
5. **Final fallback** (abstract only): Return "Limitations not explicitly stated in available text."

---

### `study_design`

**Strategy:** Keyword classification against the `StudyDesign` enum.

```python
STUDY_DESIGN_RULES: list[tuple[list[str], StudyDesign]] = [
    # Order matters — first match wins. More specific patterns first.
    (["meta-analysis", "meta analysis"], StudyDesign.META_ANALYSIS),
    (["systematic review"], StudyDesign.SYSTEMATIC_REVIEW),
    (["narrative review", "literature review", "scoping review"], StudyDesign.NARRATIVE_REVIEW),
    (["randomized", "randomised", "RCT", "clinical trial"], StudyDesign.RCT),
    (["case-control", "case control"], StudyDesign.CASE_CONTROL),
    (["cross-sectional", "cross sectional", "survey"], StudyDesign.CROSS_SECTIONAL),
    (["cohort"], StudyDesign.COHORT),
    (["case series"], StudyDesign.CASE_SERIES),
    (["case report"], StudyDesign.CASE_REPORT),
    (["in vitro", "in-vitro", "cell line", "cell culture"], StudyDesign.IN_VITRO),
    (["computational", "algorithm", "benchmark", "simulation",
      "deep learning", "machine learning", "neural network",
      "model", "framework"], StudyDesign.COMPUTATIONAL),
]
```

**Search order:** Title → Abstract → Methods section (first match wins). If no match: `StudyDesign.OTHER`.

**Note:** The `COMPUTATIONAL` bucket is intentionally broad as a catch-all for non-clinical papers. The dual-keyword requirement ("computational" alone matches, "model" alone matches) reflects the dominance of computational papers in many AutoReview corpora.

---

### `quality_score`

**Strategy:** Composite heuristic score in [0.0, 1.0].

```python
def compute_quality_score(
    paper: ScreenedPaper,
    sections: list[ParsedSection],
    findings: list[Finding],
) -> float:
    scores = []

    # Text completeness (0.3 weight)
    if paper.full_text and len(paper.full_text) > 5000:
        text_score = 1.0
    elif paper.full_text and len(paper.full_text) > 1000:
        text_score = 0.7
    elif paper.abstract and len(paper.abstract) > 200:
        text_score = 0.4
    else:
        text_score = 0.1
    scores.append(("text_completeness", 0.3, text_score))

    # Citation count — log-normalized (0.2 weight)
    cite_count = paper.citation_count or 0
    cite_score = min(1.0, math.log1p(cite_count) / math.log1p(500))
    scores.append(("citation_count", 0.2, cite_score))

    # Methods section detail (0.2 weight)
    methods_section = _find_section(sections, ["method", "material", "experimental"])
    methods_score = min(1.0, len(methods_section.text) / 3000) if methods_section else 0.0
    scores.append(("methods_detail", 0.2, methods_score))

    # Results section detail (0.2 weight)
    results_section = _find_section(sections, ["result", "finding", "experiment"])
    results_score = min(1.0, len(results_section.text) / 3000) if results_section else 0.0
    scores.append(("results_detail", 0.2, results_score))

    # Quantitative finding density (0.1 weight)
    quant_findings = sum(1 for f in findings if f.quantitative_result)
    quant_score = min(1.0, quant_findings / max(len(findings), 1))
    scores.append(("quant_density", 0.1, quant_score))

    return sum(weight * score for _, weight, score in scores)
```

---

### `sample_size`

**Strategy:** Regex extraction from Methods section, then Abstract.

```python
SAMPLE_SIZE_PATTERNS = [
    r"[Nn]\s*=\s*([\d,]+)",                                    # N = 123, n=456
    r"(\d[\d,]*)\s+(?:participants|subjects|patients|samples"
    r"|images|documents|records|cases|observations)",           # 123 participants
    r"(?:dataset|corpus|collection)\s+of\s+([\d,]+)",           # dataset of 123
    r"sample\s+size\s+(?:of\s+)?([\d,]+)",                     # sample size of 123
    r"(\d[\d,]*)\s+(?:training|test|validation)\s+(?:samples|examples|instances)",  # ML datasets
]
```

**Search order:** Methods section → Abstract → Full text (first match wins). Parse matched string as integer (strip commas). If no match: `None`.

---

### Skipped Fields

These fields are set to safe defaults without extraction logic:

```python
relationships = []              # Evidence chains still build without them
methodology_details = None      # Confirmed unused downstream
domain_specific_fields = {}     # Confirmed unused downstream
```

---

## Configuration

### ExtractionConfig Extension

```python
class ExtractionConfig(BaseModel):
    """Configuration for paper extraction."""

    model_config = ConfigDict(extra="forbid")

    # Existing fields (unchanged)
    domain_fields: dict[str, bool] = {}
    max_concurrent: int = 10
    ollama_max_concurrent: int = 2
    full_text_max_chars: int = 80_000
    extraction_batch_size: int = 20
    tiered_models: TieredModelConfig = Field(default_factory=TieredModelConfig)
    section_truncation: SectionTruncationConfig = Field(default_factory=SectionTruncationConfig)

    # NEW — extraction mode selection
    extraction_mode: Literal["llm", "programmatic", "hybrid"] = "llm"
```

**Mode behavior:**

| Mode | Behavior |
|------|----------|
| `llm` | Current behavior — LLM extraction for all papers (default) |
| `programmatic` | Deterministic extraction for all papers — zero LLM tokens |
| `hybrid` | Programmatic first; LLM fallback for papers with low confidence (quality_score < 0.3 or < 3 findings extracted) |

---

## Benchmark Scoring System

### Module: `autoreview/extraction/scoring.py`

Compares programmatic extractions against LLM ground truth extractions to measure quality.

### Field-Level Scoring

#### key_findings similarity (weight: 0.40)

The most complex scoring function, using embedding-based semantic matching:

1. Embed all claims from both extraction sets using `sentence-transformers` (`all-MiniLM-L6-v2`)
2. Compute pairwise cosine similarity matrix between programmatic and LLM claims
3. Apply Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) for optimal 1:1 matching
4. Score = mean cosine similarity of matched pairs

Additional diagnostics:
- **Precision:** Fraction of programmatic claims that have a matched LLM claim (similarity > 0.5)
- **Recall:** Fraction of LLM claims that are covered by a programmatic claim (similarity > 0.5)

#### evidence_strength accuracy (weight: 0.05)

Exact match percentage across matched finding pairs from the key_findings matching above. Only computed for pairs with cosine similarity > 0.5.

#### quantitative_result overlap (weight: 0.05)

For matched finding pairs: compute token-level F1 overlap between `quantitative_result` strings. Handle `None` vs `None` as a match, `None` vs non-`None` as 0.0.

#### methods_summary similarity (weight: 0.15)

Dual scoring:
- ROUGE-L F1 score (lexical overlap)
- Embedding cosine similarity (semantic overlap)
- Final = average of both

#### limitations similarity (weight: 0.10)

Same dual scoring as `methods_summary`.

#### study_design accuracy (weight: 0.10)

Exact match: 1.0 if both extractions produce the same `StudyDesign` enum value, 0.0 otherwise.

#### quality_score correlation (weight: 0.05)

Pearson correlation coefficient across all papers in the benchmark set. Computed at the corpus level (not per-paper), then normalized to [0, 1] via `(r + 1) / 2`.

#### sample_size accuracy (weight: 0.10)

Per-paper: 1.0 if exact match or within 10% tolerance. 0.0 if one is `None` and the other is not. `None`-`None` pairs count as 1.0.

### Composite Score

```python
composite = (
    0.40 * key_findings_similarity
    + 0.05 * evidence_strength_accuracy
    + 0.05 * quantitative_result_overlap
    + 0.15 * methods_summary_similarity
    + 0.10 * limitations_similarity
    + 0.10 * study_design_accuracy
    + 0.05 * quality_score_correlation
    + 0.10 * sample_size_accuracy
)
```

Weights sum to 1.0 and reflect downstream consumer importance.

---

## Benchmark Dataset

### Ground Truth Location

```
output/arise/arise_llm_eval_v2/snapshots/
├── extraction_batch_1/
│   └── *.json          # PaperExtraction JSON files
├── extraction_batch_2/
│   └── ...
├── ...
└── extraction_batch_11/
```

- 220 LLM extractions total across 11 batches
- Source papers: metadata (including abstracts) in the screened papers list; full texts retrievable via `full_text_source` field
- Each JSON file contains a serialized `PaperExtraction`

### Benchmark Script: `scripts/benchmark_extractor.py`

```
Usage: python scripts/benchmark_extractor.py [--output-dir results/] [--verbose]

Steps:
1. Load all 220 LLM extractions from snapshot batches
2. Load corresponding source papers (abstract + full text where available)
3. Run ProgrammaticExtractor on each source paper
4. Compute per-paper and corpus-level scores using scoring.py
5. Generate comparison report:
   - Overall composite score
   - Per-field score breakdown
   - Worst 10 papers (lowest composite) with failure analysis
   - Per-field histograms (score distributions)
   - Common error patterns
```

---

## Optimization Loop

Following a Karpathy-style auto-research iteration pattern:

```
┌─────────────────────────────────────────────────┐
│  1. Run programmatic extractor on 220 papers    │
│  2. Score against LLM ground truth              │
│  3. Generate failure analysis report            │
│  4. Identify worst fields and common patterns   │
│  5. Implement targeted heuristic improvements   │
│  6. Re-run and re-score                         │
│  7. Repeat until composite ≥ 0.7                │
└─────────────────────────────────────────────────┘
```

**Iteration tracking:** Each run writes scores to `results/benchmark_history.jsonl` for tracking improvement over iterations. Fields:

```json
{
    "timestamp": "2026-03-21T14:00:00Z",
    "iteration": 3,
    "composite": 0.65,
    "key_findings": 0.58,
    "methods_summary": 0.72,
    "limitations": 0.69,
    "study_design": 0.85,
    "quality_score": 0.61,
    "sample_size": 0.78,
    "notes": "Added position weighting for abstract conclusion sentences"
}
```

---

## Integration Plan

### Pipeline Integration

The extraction pipeline node checks `ExtractionConfig.extraction_mode`:

```python
# In autoreview/pipeline/nodes.py — extraction node

async def extraction(self, ...) -> ...:
    mode = self.config.extraction.extraction_mode

    if mode == "programmatic":
        extractor = ProgrammaticExtractor(self.config.extraction)
        extractions, failures = extractor.extract_batch(papers)
    elif mode == "hybrid":
        extractor = ProgrammaticExtractor(self.config.extraction)
        extractions, failures = extractor.extract_batch(papers)
        # Re-extract low-confidence papers with LLM
        low_confidence = [
            e for e in extractions
            if (e.quality_score or 0) < 0.3 or len(e.key_findings) < 3
        ]
        if low_confidence:
            paper_ids = {e.paper_id for e in low_confidence}
            llm_papers = [p for p in papers if p.paper_id in paper_ids]
            llm_extractions, llm_failures = await self._llm_extract(llm_papers)
            # Replace programmatic with LLM for re-extracted papers
            extractions = [
                e for e in extractions if e.paper_id not in paper_ids
            ] + llm_extractions
            failures = [
                f for f in failures if f.paper_id not in paper_ids
            ] + llm_failures
    else:  # "llm" — current behavior
        extractions, failures = await self._llm_extract(papers)
```

### Domain YAML Configuration

Users can select extraction mode in their domain YAML:

```yaml
extraction:
  extraction_mode: programmatic  # or "llm" or "hybrid"
```

No other config changes needed — the programmatic extractor uses the same `ExtractionConfig` for `full_text_max_chars`, `section_truncation`, etc.

---

## Dependencies

### Runtime (extractor itself)

None. Pure Python + `re` module. The extractor uses:
- `re` for regex patterns
- `math` for log normalization
- `autoreview.extraction.truncation.parse_sections` for section detection (already in the codebase)

### Benchmark only

| Package | Purpose | Already in environment? |
|---------|---------|------------------------|
| `sentence-transformers` | Embedding-based claim similarity | No — install for benchmarking |
| `scipy` | Hungarian algorithm for optimal matching, Pearson correlation | Yes |
| `rouge-score` | ROUGE-L computation | No — install for benchmarking |

### Sentence Tokenization

Use a simple regex-based sentence splitter rather than adding `nltk` or `spacy` as dependencies:

```python
SENT_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

def split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex boundary detection."""
    return [s.strip() for s in SENT_BOUNDARY.split(text) if s.strip()]
```

This handles the common case (English academic prose with standard punctuation) and avoids heavy NLP library dependencies.

---

## Implementation Order

| Step | Task | Dependencies | Estimated Effort |
|------|------|-------------|-----------------|
| 1 | Add `extraction_mode` to `ExtractionConfig` | None | Small |
| 2 | Implement sentence splitter + scoring functions in `programmatic.py` | Step 1 | Medium |
| 3 | Implement `key_findings` extraction with all scoring signals | Step 2 | Large |
| 4 | Implement `methods_summary`, `limitations`, `study_design`, `quality_score`, `sample_size` | Step 2 | Medium |
| 5 | Assemble `ProgrammaticExtractor` class with `extract()` and `extract_batch()` | Steps 3-4 | Small |
| 6 | Implement `scoring.py` with all field-level scoring functions | None (parallel with 2-5) | Medium |
| 7 | Write `scripts/benchmark_extractor.py` | Steps 5-6 | Medium |
| 8 | Run initial benchmark, analyze failures | Step 7 | Small |
| 9 | Iterate on heuristics (optimization loop) | Step 8 | Large (iterative) |
| 10 | Wire into pipeline node (`hybrid` and `programmatic` modes) | Step 5 | Small |
| 11 | Integration tests | Step 10 | Medium |

Steps 2-5 and 6 can proceed in parallel.

---

## Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Composite benchmark score | >= 0.7 | Weighted average across all fields |
| Key findings recall | >= 0.6 | Fraction of LLM claims covered (similarity > 0.5) |
| Methods/limitations similarity | >= 0.7 | ROUGE-L F1 score |
| Study design exact match | >= 0.8 | Exact enum match percentage |
| Extraction speed | < 60 seconds for 500 papers | Wall-clock time |
| End-to-end review quality | ARISE score within 10 points of LLM extraction | Full pipeline run comparison |
| Zero new runtime dependencies | Confirmed | No new packages in `pyproject.toml` runtime deps |

---

## Non-Goals

- **Perfect replication of LLM output.** Heuristic extraction will produce different phrasing and may select different sentences. The goal is functional equivalence for downstream stages, not textual identity.
- **Relationship extraction.** The `relationships` field requires cross-paper reasoning that is inherently non-deterministic. Returning `[]` is safe — evidence chains still build from other signals.
- **Domain-specific field extraction.** Confirmed unused by any downstream consumer. Returns `{}`.
- **Methodology details extraction.** Confirmed unused downstream. Returns `None`.
- **Replacing scoring with a simpler metric.** Embedding-based scoring is essential for measuring claim similarity. Lexical overlap (BLEU/ROUGE alone) would penalize valid paraphrases.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Key findings scoring misses domain-specific claims | Low recall on specialized topics | Optimization loop with per-domain failure analysis; hybrid mode as safety net |
| Sentence splitter fails on non-standard formatting | Malformed findings | Graceful fallback: if < 3 sentences detected, use abstract sentences directly |
| Section detection fails (no headings found) | Missing methods/limitations | Already handled: `truncation.parse_sections()` returns `[]`, triggering abstract-based fallback for all fields |
| Quality score heuristic diverges from LLM scores | Contradiction resolver affected | Low impact — contradiction resolver uses quality_score as a tiebreaker, not a gate. Pearson correlation target (0.5+) ensures directional agreement |
| Hybrid mode negates speed gains | Partial token savings only | Set confidence thresholds conservatively; monitor hybrid fallback rate |

---

## Testing Strategy

### Unit Tests

- Sentence scoring: verify position, keyword, quantitative, and novelty weights with known inputs
- Each field extractor: test with sample paper text (abstract-only and full-text variants)
- Study design classifier: test each keyword rule fires correctly
- Sample size regex: test all pattern variants with edge cases (commas, ranges)
- Quality score: verify weights sum to 1.0, boundary conditions

### Integration Tests

- Run `ProgrammaticExtractor.extract()` on 5 fixture papers, verify `PaperExtraction` schema compliance
- Run `extract_batch()` on fixture papers, verify failures are captured (not raised)
- Pipeline integration: run extraction node with `extraction_mode: "programmatic"`, verify downstream stages accept output
- Hybrid mode: verify LLM fallback triggers for low-confidence papers

### Benchmark Tests

- Scoring functions: verify with synthetic ground truth pairs (known similarity scores)
- Hungarian matching: verify optimal assignment with small known matrices
- Full benchmark: run on 220 papers, verify composite score tracking and report generation
