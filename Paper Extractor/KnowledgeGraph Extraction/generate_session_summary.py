"""Generate session summary PDF with embedded figures."""

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import markdown
from weasyprint import HTML

OUT_DIR = Path(__file__).parent
FIG1 = OUT_DIR / "KnowledgeGraph Extraction" / "extraction_comparison.png"
FIG2 = OUT_DIR / "KnowledgeGraph Extraction" / "extraction_comparison_v2.png"


def img_to_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def build_markdown() -> str:
    fig1_uri = img_to_data_uri(FIG1)
    fig2_uri = img_to_data_uri(FIG2)

    return f"""
# Mycelium Extraction Framework: Development & Testing Report

**Author:** McLean Taggart
**Date:** March 24, 2026
**Status:** Framework validated, ready for pilot extraction

---

## 1. Executive Summary

This session formalized the Mycelium claim extraction architecture, built a strict Pydantic schema and extraction prompt, and tested extraction on a real protein-interaction paper using three Claude models (Opus, Sonnet, Haiku). Key outcomes:

- **Schema created**: 880-line Pydantic schema defining ExtractionResult with AssertionDraft, EvidenceUnit, and CitationContext models, 12 StrEnum types
- **Architectural decision**: Background claims replaced with CitationContext signals — edges emerge from cross-paper semantic comparison and explicit citation signals, not from duplicating prior claims as assertions
- **Model recommendation**: Sonnet as primary extraction model (100% scope completeness, rich citation contexts, best cost-quality balance)
- **Cost estimate**: ~$10K (Haiku Batch) to ~$19K (two-tier) for full bioRxiv corpus (~320K papers)

---

## 2. Architecture: Three-Layer Pipeline

The extraction framework sits in Layer 1 of a three-layer pipeline:

- **Layer 1 — Extraction** (this session): LLM processes a single paper → outputs ExtractionResult
- **Layer 2 — Projection** (future): Resolves claim identity across papers, detects edges, produces canonical Assertion nodes
- **Layer 3 — Graph** (future): PostgreSQL (metadata, embeddings) + Neo4j (graph traversal, confidence propagation)

### What the LLM Extracts Per Paper

| Component | Purpose | Count (typical) |
|-----------|---------|-----------------|
| **AssertionDraft** | Novel scientific claims at the mechanism level | 11–22 per paper |
| **EvidenceUnit** | Distinct experimental results with full statistical detail | 16–23 per paper |
| **CitationContext** | How this paper references specific prior findings (edge signals) | 4–12 per paper |
| **PaperProvenance** | Paper-level metadata for source independence computation | 1 per paper |

### The CitationContext Decision

In v1, background claims were extracted as full assertion drafts (is_primary=false). This inflated the graph with second-hand interpretations of claims that already exist as primary findings from their source papers.

**v2 approach**: Background claims are replaced with lightweight CitationContext signals that capture the exact citing sentence, what the author says the cited paper found, the relationship type (supports, contradicts, extends, qualifies, contextualizes, replicates, mentions), and links to this paper's novel assertions. This gives Layer 2 high-confidence edge candidates without node pollution.

Edges emerge two ways:

1. **Explicit**: Citation context signals (high precision, author-stated)
2. **Implicit**: Semantic comparison across papers' novel assertions (high recall, computationally discovered)

---

## 3. Schema Design

The Pydantic schema (`src/mycelium/extraction_schema.py`) defines 12 enumerations and 16 models.

### Enumerations

| Enum | Values |
|------|--------|
| AssertionTypeEnum | mechanistic_causal, correlational, comparative, existence, absence, conditional, methodological |
| CausalTypeEnum | necessary, sufficient, necessary_and_sufficient, contributory, modulatory |
| HedgeLevelEnum | high, medium, low |
| SectionEnum | abstract, introduction, methods, results, discussion, conclusion |
| FunctionEnum | novel_finding, background (QA only), replication, hypothesis, interpretation, limitation, methodological_note |
| EntityTypeEnum | gene, protein, chemical, disease, pathway, cell_type, organism, method, other |
| EvidenceStrengthEnum | systematic_review_meta_analysis → expert_opinion (8 levels) |
| ResultDirectionEnum | positive, negative, null_powered, null_underpowered, not_reported |
| CitationRelationshipEnum | supports, contradicts, extends, qualifies, contextualizes, replicates, mentions |

### Key Design Decisions

1. **Scope is first-class** — species, tissue, cell_type, disease, condition, developmental_stage, in_vitro are mandatory fields
2. **Hedging preserved verbatim** — the author's exact language is captured alongside structured classification
3. **Section context determines epistemic function** — same sentence means different things in Introduction vs Results
4. **Lazy entity normalization** — surface forms at extraction time, ontology IDs populated during enrichment
5. **strict=False** — Pydantic v2 strict mode breaks JSON validation for StrEnum; LLM output produces strings, not enum instances

---

## 4. Test Paper

**Title:** Rai14 is a novel interactor of Invariant chain that regulates macropinocytosis
**Authors:** Lobos Patorniti, Zulkefli, McAdam, Vargas, Bakke, Progida
**Journal:** Frontiers in Immunology (2023)
**DOI:** 10.3389/fimmu.2023.1182180

Selected because it is a primary research paper with clear protein-protein interaction claims (Rai14–Ii, Rai14–myosin II), multiple experimental methods (Y2H, co-IP, live imaging, knockdown, migration assays), quantitative results, and is open access.

---

## 5. Extraction Results: v1 (Opus vs Sonnet)

The first test compared Opus and Sonnet on the v1 schema, which included background assertions as full assertion drafts.

<img src="{fig1_uri}" style="width:100%"/>

*Figure 1: v1 Opus vs Sonnet comparison. Panel C (Scope Completeness) shows the most dramatic difference — Sonnet fills scope fields at ~100% vs Opus at ~60%.*

### Key Findings (v1)

| Metric | Opus | Sonnet |
|--------|------|--------|
| Assertion drafts | 37 | 29 |
| Evidence units | 23 | 21 |
| Novel findings | 22 | 22 |
| Background claims | 15 | 7 |
| Scope completeness (species) | 59% | 100% |
| Hedging captured | 70% | 100% |
| Entity surface forms | 42 (verbose) | 15 (clean) |
| Cross-reference integrity | Perfect | Perfect |

**Opus** extracted more assertions overall, but the advantage was entirely in background claims. Both models found the same 22 novel findings. Opus's entity extraction produced verbose compound phrases that would complicate entity resolution.

**Sonnet** had significantly better field discipline — 100% scope fill, 100% hedging capture. Every assertion knew what species, cell type, and experimental context it applied to.

---

## 6. Extraction Results: v2 (Three-Model Comparison)

The v2 schema replaced background assertions with CitationContext. We tested Sonnet and Haiku on the updated framework.

<img src="{fig2_uri}" style="width:100%"/>

*Figure 2: v2 three-model comparison. Panel C shows both Sonnet and Haiku achieve ~100% scope completeness. Panel D shows Sonnet extracts 3x more citation contexts with richer relationship diversity.*

### Three-Model Comparison

| Metric | Opus (v1) | Sonnet (v2) | Haiku (v2) |
|--------|-----------|-------------|------------|
| Assertion drafts | 37 | 17 | 11 |
| Evidence units | 23 | 19 | 16 |
| Citation contexts | 0 (old schema) | 12 | 4 |
| Novel findings | 22 | 17 | 11 |
| Background assertions | 15 | 0 | 0 |
| Scope completeness | 59% | 100% | 100% |
| Avg entity name length | 15.1 chars | 14.4 chars | 11.5 chars |

### Citation Context Quality

Sonnet extracted 3x more citation contexts than Haiku (12 vs 4), with richer relationship diversity:

- **Sonnet**: contextualizes (9), extends (2), supports (1)
- **Haiku**: contextualizes (4) only

Haiku captures basic contextual references but misses when a prior finding is being extended or supported.

---

## 7. Cost Analysis

### Per-Paper Extraction Cost

| Paper Length | Sonnet | Sonnet Batch | Haiku | Haiku Batch |
|-------------|--------|-------------|-------|-------------|
| Short (5 pg) | $0.22 | $0.11 | $0.04 | $0.02 |
| Medium (10 pg) | $0.33 | $0.16 | $0.06 | $0.03 |
| Typical (15 pg) | $0.37 | $0.18 | $0.07 | $0.03 |
| Long (25 pg) | $0.56 | $0.28 | $0.10 | $0.05 |

### Full bioRxiv Corpus (~320K papers)

| Strategy | Cost | Notes |
|----------|------|-------|
| Sonnet Batch | $53K–$60K | Best quality |
| Haiku Batch | $10K–$11K | Cheapest viable |
| Two-tier (Haiku all + Sonnet top 20%) | ~$19K | Recommended |
| Fine-tuned 8B (future) | $2K–$5K inference | Requires $20K+ training investment first |

### Fine-Tuning vs API

At bioRxiv scale (320K papers), the API is cheaper ($10–19K) than fine-tuning ($21–44K total). Fine-tuning only pays off at PubMed scale (36M papers), where API costs reach $1M+ but a fine-tuned model runs for ~$3K.

### Input Format

bioRxiv provides full text through its API in structured format — no PDF parsing needed for the target corpus. The extraction results from our test (which used parsed text) are representative.

---

## 8. Framework Improvements Identified

1. **Add `biological_process` to EntityTypeEnum** — both models mapped macropinocytosis/cell migration to `pathway` as a workaround
2. **Clarify assertion_type for binding evidence** — co-IP/pulldown = existence of interaction; knockdown + phenotype = mechanistic_causal
3. **Enforce single-entity rule** — subject_entity and object_entity must be individual biological entities
4. **Standardize figure references** — enforce consistent format (e.g., "Figure 1A")
5. **Add prompt caching** — system prompt is ~7,650 tokens, identical across papers; caching saves ~90% on input cost

---

## 9. Recommendations

### For the Pilot (Next Steps)

1. Apply the 5 framework improvements to the schema and prompt
2. Run Sonnet v2 extraction on 50–100 papers from the PPI pilot domain to validate at scale
3. Begin gold standard annotation on the first 10 papers (McLean + 2 domain experts)
4. Build Layer 2 projection pipeline (claim identity resolution, edge detection)

### For Production Extraction

- **Primary model**: Sonnet (best balance of coverage, scope discipline, citation context quality)
- **Corpus strategy**: Haiku Batch on full bioRxiv ($10K), Sonnet re-extract on top 50K papers ($9K)
- **Input format**: bioRxiv API provides structured text — no PDF parsing needed

### For the Paper

- The extraction comparison (v1 vs v2, three models) is publishable as a methods evaluation
- The CitationContext decision is a novel contribution — most knowledge graphs extract all claims indiscriminately
- Calibration data: extraction quality by paper section, claim type, and method type

---

## 10. Files Produced

| File | Description |
|------|-------------|
| `src/mycelium/extraction_schema.py` | Pydantic schema (880 lines, 12 enums, 16 models) |
| `prompts/mycelium_extraction_prompt.md` | Three-stage extraction prompt |
| `outputs/extraction_test_opus.json` | Opus v1 extraction (37 assertions, 23 EU) |
| `outputs/extraction_test_sonnet_v2.json` | Sonnet v2 extraction (17 assertions, 19 EU, 12 citations) |
| `outputs/extraction_test_haiku.json` | Haiku v2 extraction (11 assertions, 16 EU, 4 citations) |
| `outputs/extraction_comparison.png` | v1 comparison figure (6 panels) |
| `outputs/extraction_comparison_v2.png` | v2 three-model comparison figure (6 panels) |
| `outputs/extraction_report_opus_v1.pdf` | Full Opus extraction report |
| `outputs/extraction_report_sonnet_v2.pdf` | Full Sonnet extraction report |
| `outputs/extraction_report_haiku_v2.pdf` | Full Haiku extraction report |
"""


CSS = """
body { font-family: Arial, Helvetica, sans-serif; font-size: 11pt; line-height: 1.6; margin: 0; padding: 0; color: #1a1a1a; }
h1 { color: #0072B2; border-bottom: 3px solid #0072B2; padding-bottom: 8px; font-size: 22pt; margin-top: 0; }
h2 { color: #333; border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-top: 30px; font-size: 16pt; page-break-before: auto; }
h3 { color: #D55E00; margin-top: 20px; font-size: 13pt; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 10pt; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
th { background: #f5f5f5; font-weight: bold; }
tr:nth-child(even) { background: #fafafa; }
img { max-width: 100%; height: auto; margin: 15px 0; }
em { color: #555; font-size: 10pt; }
strong { color: #222; }
code { background: #f4f4f4; padding: 2px 4px; border-radius: 3px; font-size: 10pt; }
hr { border: none; border-top: 1px solid #ddd; margin: 25px 0; }
ol, ul { margin: 8px 0; padding-left: 24px; }
li { margin: 4px 0; }
@page { margin: 0.9in; size: letter; @bottom-center { content: counter(page); font-size: 9pt; color: #999; } }
"""


def main():
    md_text = build_markdown()

    md_path = OUT_DIR / "session_summary_extraction_framework.md"
    md_path.write_text(md_text)
    print(f"Markdown written: {md_path}")

    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc"])
    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{html_body}</body></html>"""

    pdf_path = OUT_DIR / "session_summary_extraction_framework.pdf"
    HTML(string=full_html).write_pdf(str(pdf_path))
    print(f"PDF written: {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
