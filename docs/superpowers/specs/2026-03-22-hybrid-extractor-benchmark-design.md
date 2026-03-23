# Hybrid Extractor + Dual-Layer Benchmark — Design Spec

> **For agentic workers:** This spec defines (1) a factual accuracy scoring layer added to the existing benchmark, and (2) a hybrid extraction strategy that uses the programmatic extractor as a pre-filter for a cheap LLM refinement pass. Key files: `autoreview/extraction/scoring.py` (modify), `autoreview/extraction/hybrid.py` (new), `scripts/benchmark_extractor.py` (modify), `autoreview/llm/prompts/hybrid_extraction.py` (new).

**Date:** 2026-03-22
**Status:** Draft
**Author:** Claude (brainstormed with user)

**Goal:** Close the quality gap between the programmatic extractor (composite 0.70) and full-LLM extraction (composite 1.0 by definition) at minimal cost, while building a benchmark that properly measures factual accuracy — not just textual similarity.

**Context:** The programmatic extractor reached a ceiling at composite ~0.70 after 92 optimization experiments. The gap is structural: embedding cosine similarity between raw sentences and LLM paraphrases caps at ~0.80, and the extractor cannot parse tables or synthesize across sentences. A hybrid approach sends condensed programmatic output to a cheap LLM for refinement.

---

## Problem Statement

The current benchmark measures how similar our text is to LLM-generated text (embedding cosine similarity). This penalizes correct extractions that use different wording and rewards wrong extractions that happen to sound like the LLM. It also creates a ceiling for any non-LLM extractor.

Meanwhile, the programmatic extractor already identifies the right sentences and sections — it just can't synthesize them into clean claims. A cheap LLM refinement pass could bridge that gap at a fraction of the cost of full LLM extraction.

## Two Workstreams

### Workstream 1: Dual-Layer Benchmark Scoring
Add a factual accuracy scoring layer alongside the existing embedding similarity layer.

### Workstream 2: Hybrid Extractor
Use programmatic extraction as a pre-filter, then refine with a cheap LLM.

### Evaluation Matrix
Benchmark 5 strategies on the same 220-paper corpus using both scoring layers:

| ID | Strategy | Input to LLM | LLM Model | Est. Cost/200 papers |
|----|----------|-------------|-----------|---------------------|
| A | Programmatic only | — | — | $0.00 |
| B | Hybrid → Haiku | ~2K tokens | Haiku | ~$0.30 |
| C | Hybrid → Sonnet | ~2K tokens | Sonnet | ~$3.60 |
| D | Direct → Haiku | ~18K tokens | Haiku | ~$1.12 |
| E | Direct → Sonnet (gold) | ~18K tokens | Sonnet | ~$13.42 (already done) |

Strategy E is the existing gold corpus — no new runs needed.

---

## Workstream 1: Dual-Layer Benchmark Scoring

### Current Scoring (Similarity Layer)
Retained as-is. Per-field scores using embedding cosine similarity, ROUGE-L, word overlap, exact match, and tolerance matching. Composite weighted sum.

### New Scoring (Factual Layer)

Each field gets a factual accuracy score measuring whether the extraction captured the correct **information**, independent of wording.

#### key_findings — Entity/Number Recall

Extract factual elements from each gold claim:
- **Named entities**: model names, dataset names, system names (regex + spaCy NER if available)
- **Numbers**: all numeric values with optional units (67.6%, 540B, 17.3%)
- **Relationships**: directional comparisons (X outperforms Y, X increases from A to B)

Score: For each gold claim, find the predicted claim with highest factual overlap (Jaccard on extracted elements). Average across gold claims. This rewards finding the right facts regardless of phrasing.

**v1 scope**: Numbers + capitalized proper nouns only (no dependency on spaCy). Relationship extraction deferred to v2.

```python
_NUMBER_RE = re.compile(r"\d+\.?\d*\s*%?")
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][A-Za-z]*(?:[-/][A-Z][A-Za-z]*)*\b")
# Filter out common English words that happen to be capitalized
_STOPWORDS = {"The", "This", "These", "That", "However", "Our", "We", ...}


def _extract_facts(text: str) -> set[str]:
    """Extract factual elements: numbers and proper nouns (no spaCy needed)."""
    facts = set()
    for m in _NUMBER_RE.finditer(text):
        facts.add(m.group().strip())
    for m in _PROPER_NOUN_RE.finditer(text):
        word = m.group()
        if word not in _STOPWORDS and len(word) > 1:
            facts.add(word)
    return facts


def _fact_overlap(gold_facts: set[str], pred_facts: set[str]) -> float:
    """Jaccard-like overlap: fraction of gold facts found in predicted."""
    if not gold_facts:
        return 1.0
    matched = sum(1 for gf in gold_facts if gf in pred_facts)
    return matched / len(gold_facts)


def _factual_key_findings_score(pred_claims: list[str], gold_claims: list[str]) -> float:
    """Score key findings by factual element overlap, not textual similarity."""
    if not gold_claims:
        return 1.0 if not pred_claims else 0.0

    gold_facts = [_extract_facts(claim) for claim in gold_claims]
    pred_facts = [_extract_facts(claim) for claim in pred_claims]

    scores = []
    for gf in gold_facts:
        best = max(
            (_fact_overlap(gf, pf) for pf in pred_facts),
            default=0.0,
        )
        scores.append(best)
    return sum(scores) / len(scores)
```

#### quantitative_result — Numeric Extraction Accuracy

Parse all numbers from both predicted and gold quantitative_result strings. Score based on whether the same numeric values appear (within 1% tolerance), ignoring surrounding text.

```python
def _factual_quant_score(pred_quant: str, gold_quant: str) -> float:
    """Score quantitative result by numeric value match, not text overlap."""
    pred_nums = _extract_numbers(pred_quant)  # {67.6, 17.3, 50.3}
    gold_nums = _extract_numbers(gold_quant)  # {67.6, 17.3, 50.3}
    if not gold_nums and not pred_nums:
        return 1.0
    if not gold_nums or not pred_nums:
        return 0.0
    # Fraction of gold numbers found in predicted (within 1% tolerance)
    matched = sum(1 for g in gold_nums if any(abs(g - p) / max(abs(g), 1e-9) < 0.01 for p in pred_nums))
    return matched / len(gold_nums)
```

#### methods_summary — Keyword Checklist

Extract key nouns from gold methods text: dataset names, model names, technique names, metric names. Score by coverage: what fraction of gold keywords appear in our predicted text?

Uses a lightweight noun-phrase extractor (regex-based, no dependencies):
- Capitalized multi-word phrases (proper nouns): "MedQA", "Flan-PaLM", "HealthSearchQA"
- Technical terms: "chain-of-thought", "self-consistency", "instruction tuning"
- Numbers with units: "540B parameters", "3,173 questions", "9 clinicians"

#### limitations — Topic Coverage

Extract limitation topics from gold text using keyword clustering:
- Group limitation sentences by topic (sample size, generalizability, language bias, etc.)
- Score by topic recall: did our extraction mention the same limitation topics?

Predefined topic taxonomy (matched by keyword presence):

1. sample_size ("small sample", "limited sample", "N=", "few participants")
2. generalizability ("not generaliz", "limited to", "single center", "specific to")
3. language_bias ("english only", "monolingual", "english-language")
4. study_design ("cross-sectional", "retrospective", "correlational", "no control")
5. data_quality ("self-report", "missing data", "incomplete", "noisy")
6. temporal ("outdated", "fixed date", "point in time", "snapshot")
7. bias ("selection bias", "potential bias", "confound", "unbalanced")
8. reproducibility ("not open-source", "not reproducible", "proprietary")
9. scope ("narrow scope", "beyond the scope", "did not consider", "excluded")
10. evaluation ("single rater", "limited evaluation", "no human eval", "automated only")
11. domain_specificity ("domain-specific", "not tested on", "single domain")
12. statistical_power ("underpowered", "statistical power", "insufficient")
13. cost_resources ("expensive", "computational cost", "resource intensive")
14. safety_ethics ("safety", "ethical", "bias evaluation", "fairness")
15. hallucination ("hallucination", "factual errors", "fabricat")
16. benchmark_limitations ("benchmark", "dataset limitations", "test set")
17. human_subjects ("IRB", "consent", "privacy", "participant")
18. comparison ("limited comparison", "no baseline", "few baselines")
19. future_work ("future work", "further research", "remains to be")
20. validity ("ecological validity", "external validity", "construct validity")

#### Other Fields — Pass-Through

`evidence_strength`, `study_design`, `quality_score`, `sample_size` keep their existing scoring. For the factual composite, these fields' similarity scores are **passed through** so the factual composite uses all 8 fields (not just the 4 with new factual metrics).

### Composite Formula

```python
def score_extraction_pair_factual(
    predicted: PaperExtraction,
    ground_truth: PaperExtraction,
    similarity_scores: dict[str, float],
) -> dict[str, float]:
    """Factual accuracy scores. Pass-through fields use similarity scores."""
    scores = {}
    # New factual metrics for 4 fields
    scores["key_findings"] = _factual_key_findings_score(pred_claims, gold_claims)
    scores["quantitative_result"] = _factual_quant_score(...)
    scores["methods_summary"] = _factual_methods_score(...)
    scores["limitations"] = _factual_limitations_score(...)
    # Pass-through: reuse similarity scores for unchanged fields
    for field in ("evidence_strength", "study_design", "quality_score", "sample_size"):
        scores[field] = similarity_scores[field]
    return scores


def compute_dual_composite(
    similarity_scores: dict[str, float],
    factual_scores: dict[str, float],
    alpha: float = 0.5,
) -> dict[str, float]:
    """Compute dual-layer composite.

    Returns dict with 'similarity', 'factual', and 'combined' composites.
    Both score dicts must contain all 8 fields (factual passes through
    unchanged fields from similarity).
    """
    sim_composite = compute_composite_score(similarity_scores)
    fact_composite = compute_composite_score(factual_scores)
    combined = alpha * sim_composite + (1 - alpha) * fact_composite
    return {
        "similarity": sim_composite,
        "factual": fact_composite,
        "combined": combined,
    }
```

The `alpha` parameter is configurable (default 0.5, CLI flag `--alpha`). The benchmark reports all three composites so we can see how strategies perform on each layer.

### New Files

| File | Purpose |
|------|---------|
| `autoreview/extraction/scoring_factual.py` | Factual accuracy scoring functions |

### Modified Files

| File | Change |
|------|--------|
| `autoreview/extraction/scoring.py` | Add `score_extraction_pair_factual()`, `compute_dual_composite()` |
| `scripts/benchmark_extractor.py` | Report both scoring layers, add `--scoring-mode` flag |

---

## Workstream 2: Hybrid Extractor

### Architecture

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────┐
│ ProgrammaticExtractor│────▶│  HybridExtractor     │────▶│ PaperExtraction
│ .extract(paper)      │     │  .refine(draft, paper)│     │ (refined)    │
│                      │     │                       │     │              │
│ Returns:             │     │ Sends to LLM:         │     │ LLM improves:│
│ - 1000 raw findings  │     │ - Draft extraction    │     │ - 10 claims  │
│ - Abstract as methods│     │ - Abstract            │     │ - Methods    │
│ - Raw limitations    │     │ - Top 10 sentences    │     │ - Limitations│
│ - study_design ✓     │     │ - Section excerpts    │     │ - Quant      │
│ - quality_score ✓    │     │                       │     │ - Evidence   │
│ - sample_size        │     │ Keeps programmatic:   │     │              │
│                      │     │ - study_design        │     │              │
│                      │     │ - quality_score       │     │              │
└─────────────────────┘     └──────────────────────┘     └─────────────┘
```

### HybridExtractor Class

```python
class HybridExtractor:
    """Refines programmatic extraction output using a cheap LLM pass."""

    def __init__(
        self,
        programmatic: ProgrammaticExtractor,
        llm: Any,  # LLMClient or claude_code provider
        refine_model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self.programmatic = programmatic
        self.llm = llm
        self.refine_model = refine_model

    async def extract(self, sp: ScreenedPaper) -> PaperExtraction:
        """Full hybrid extraction: programmatic → LLM refinement."""
        # Step 1: Programmatic extraction (free, instant, synchronous)
        draft = self.programmatic.extract(sp)

        # Step 1.5: Short-circuit for abstract-only papers
        # The LLM adds little value when all it has is the abstract
        if not sp.paper.full_text:
            return draft

        # Step 2: Build condensed context for LLM
        context = self._build_refinement_context(draft, sp)

        # Step 3: LLM refinement (cheap, fast)
        refined = await self._refine_with_llm(draft, context, sp.paper.id)

        # Step 4: Keep programmatic fields that don't need LLM
        refined.study_design = draft.study_design
        refined.quality_score = draft.quality_score
        # sample_size: use LLM's if it found one, otherwise keep programmatic
        if refined.sample_size is None:
            refined.sample_size = draft.sample_size

        return refined

    async def _refine_with_llm(
        self,
        draft: PaperExtraction,
        context: str,
        paper_id: str,
    ) -> PaperExtraction:
        """Send condensed context to LLM and parse structured output.

        Supports two backends:
        - API mode: uses llm.generate_structured(response_model=PaperExtraction)
        - claude -p mode: parses raw JSON response via PaperExtraction.model_validate_json()
        """
        prompt = _build_refinement_prompt(draft, context)

        if hasattr(self.llm, "generate_structured"):
            # API mode: structured output with Pydantic validation
            response = await self.llm.generate_structured(
                prompt=prompt,
                response_model=PaperExtraction,
                system=HYBRID_SYSTEM_PROMPT,
                model_override=self.refine_model,
            )
            extraction = response.parsed
        else:
            # claude -p mode: raw text → JSON parse
            raw = await self.llm.generate(
                prompt=prompt,
                system=HYBRID_SYSTEM_PROMPT,
                model_override=self.refine_model,
            )
            # Extract JSON from response (may be wrapped in markdown code fence)
            json_str = _extract_json_block(raw)
            try:
                extraction = PaperExtraction.model_validate_json(json_str)
            except Exception:
                logger.warning("hybrid_refinement.parse_failed", paper_id=paper_id)
                return draft  # Fallback to programmatic draft on parse failure

        extraction.paper_id = paper_id
        return extraction
```

**Sync/async bridging**: `ProgrammaticExtractor.extract()` is synchronous and called directly in step 1. The `async` boundary is only at `_refine_with_llm`. The benchmark runner must become async when running hybrid/direct strategies — `run_benchmark` wraps in `asyncio.run()` at the `main()` level, with the inner loop using `await` only for LLM-based strategies.

### Refinement Context Builder

Builds a ~2K token context from the programmatic output and source paper:

```python
def _build_refinement_context(
    self,
    draft: PaperExtraction,
    sp: ScreenedPaper,
) -> str:
    """Build condensed context for LLM refinement (~2K tokens)."""
    paper = sp.paper
    parts = []

    # 1. Paper title
    parts.append(f"Title: {paper.title}")

    # 2. Abstract (full — typically 200-300 words)
    if paper.abstract:
        parts.append(f"Abstract: {paper.abstract}")

    # 3. Top 10 key findings from programmatic extraction (best sentences)
    parts.append("Top extracted sentences:")
    for i, f in enumerate(draft.key_findings[:10]):
        quant_str = f" [{f.quantitative_result}]" if f.quantitative_result else ""
        parts.append(f"  {i+1}. {f.claim[:200]}{quant_str}")

    # 4. Methods section excerpt (first 500 chars)
    sections = parse_sections(paper.full_text) if paper.full_text else []
    methods_sec = _find_section_with_children(sections, ["method", "approach", ...])
    if methods_sec:
        parts.append(f"Methods section excerpt: {methods_sec.text[:500]}")

    # 5. Limitations section excerpt (first 500 chars)
    lim_sec = _find_section_with_children(sections, ["limitation", ...])
    if lim_sec:
        parts.append(f"Limitations section excerpt: {lim_sec.text[:500]}")

    return "\n\n".join(parts)
```

### LLM Refinement Prompt

```python
HYBRID_REFINEMENT_PROMPT = """You are refining a draft paper extraction. Below is a programmatic extraction
and source excerpts from the paper. Produce a refined extraction with:

1. key_findings: 8-12 synthesized claims (combine related sentences, add specific numbers)
2. methods_summary: 3-5 sentence structured summary of methodology
3. limitations: Numbered list of study-specific limitations
4. For each finding: evidence_strength (strong/moderate/weak/preliminary) and quantitative_result

Source material:
{context}

Draft extraction (top 10 findings):
{draft_findings}

Draft methods: {draft_methods}
Draft limitations: {draft_limitations}

Output as JSON matching the PaperExtraction schema."""
```

### LLM Backend Configuration

```python
# Dev mode: uses claude -p (claude_code provider)
hybrid = HybridExtractor(
    programmatic=ProgrammaticExtractor(config),
    llm=claude_code_provider,
    refine_model="sonnet",  # claude -p model selection
)

# Production mode: uses Anthropic API
hybrid = HybridExtractor(
    programmatic=ProgrammaticExtractor(config),
    llm=api_client,
    refine_model="claude-haiku-4-5-20251001",
)
```

### New Files

| File | Purpose |
|------|---------|
| `autoreview/extraction/hybrid.py` | `HybridExtractor` class |
| `autoreview/llm/prompts/hybrid_extraction.py` | Refinement prompt template |

### Modified Files

| File | Change |
|------|--------|
| `scripts/benchmark_extractor.py` | Add `--strategy` flag (A/B/C/D), `--alpha` flag, async support, token tracking, run comparison matrix |
| `autoreview/config/models.py` | Add `hybrid` to extraction mode options |

**Note**: The `--scoring-mode` and `--strategy` flags are orthogonal — any strategy can be scored with any scoring mode (similarity, factual, or both).

---

## Benchmark Runner Changes

```bash
# Run individual strategy
python scripts/benchmark_extractor.py --strategy programmatic     # Strategy A
python scripts/benchmark_extractor.py --strategy hybrid-haiku     # Strategy B
python scripts/benchmark_extractor.py --strategy hybrid-sonnet    # Strategy C
python scripts/benchmark_extractor.py --strategy direct-haiku     # Strategy D

# Run full comparison matrix
python scripts/benchmark_extractor.py --compare-all

# Output: results/extractor_benchmark/comparison_matrix.json
```

Output format:
```json
{
  "strategies": {
    "A_programmatic": {
      "similarity_composite": 0.6955,
      "factual_composite": 0.???,
      "combined_composite": 0.???,
      "per_field_similarity": { ... },
      "per_field_factual": { ... },
      "cost_per_paper": 0.0,
      "elapsed_seconds": 178
    },
    "B_hybrid_haiku": { ... },
    ...
  }
}
```

---

## Implementation Order

1. **Factual scoring layer** (`scoring_factual.py`) — independent of hybrid extractor, can benchmark programmatic immediately
2. **Benchmark runner updates** — dual-layer reporting, strategy flags
3. **Hybrid extractor** (`hybrid.py`) — the LLM refinement path
4. **Refinement prompt** — engineer and iterate on the prompt
5. **Run comparison matrix** — benchmark all 5 strategies
6. **Analysis** — decide which strategy to use in production

Steps 1-2 unblock immediate re-evaluation of the programmatic extractor with the new factual scoring. Steps 3-4 can proceed in parallel. Step 5 requires all previous steps.

---

## Testing

### Unit Tests (`tests/test_scoring_factual.py`)
- `_extract_facts`: known inputs → expected entity/number sets
- `_factual_key_findings_score`: synthetic pred/gold claims with known overlap
- `_factual_quant_score`: matching/non-matching number sets
- `_factual_methods_score`: keyword coverage on known texts
- `_factual_limitations_score`: topic detection on known limitation texts
- `compute_dual_composite`: verify alpha blending and field pass-through

### Unit Tests (`tests/test_hybrid_extractor.py`)
- `HybridExtractor.extract`: mock LLM returning valid JSON → verify refined output
- `HybridExtractor.extract`: mock LLM returning garbage → verify fallback to draft
- `HybridExtractor.extract`: abstract-only paper → verify short-circuit (no LLM call)
- `_build_refinement_context`: verify token budget stays under 2.5K tokens
- `_refine_with_llm`: test both API and claude-p parsing paths

### Integration Tests (`tests/test_hybrid_integration.py`)
- Run hybrid extractor on 3 corpus papers with mocked LLM
- Verify output passes `PaperExtraction` validation
- Verify study_design/quality_score preserved from programmatic draft

## Token Usage Tracking

The benchmark runner tracks actual token counts for LLM-based strategies:

```python
# In comparison_matrix.json output:
"B_hybrid_haiku": {
    ...,
    "total_input_tokens": 440000,
    "total_output_tokens": 176000,
    "avg_input_tokens_per_paper": 2000,
    "avg_output_tokens_per_paper": 800,
    "actual_cost_usd": 0.33,
}
```

---

## Success Criteria

- Factual scoring layer gives meaningfully different rankings than similarity-only (validates the two-layer approach)
- Hybrid → Haiku (B) scores within 10% of Direct → Sonnet (E) on combined composite
- Hybrid → Haiku (B) costs <5% of Direct → Sonnet (E)
- If B achieves >0.85 combined composite, adopt as default extraction strategy

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Factual scoring is noisy (regex-based entity extraction misses entities) | Start with numbers + capitalized phrases (high precision). Iterate. |
| Hybrid LLM doesn't follow the refinement prompt format | Use structured output (Pydantic model) with the API; for claude -p, validate JSON output |
| 2K token context is insufficient for the LLM to produce good output | Monitor per-paper quality; increase context budget if needed (still much cheaper than 18K) |
| Haiku quality too low for key_findings synthesis | Strategy C (Sonnet) is the fallback; cost is still 4x lower than full Sonnet |
