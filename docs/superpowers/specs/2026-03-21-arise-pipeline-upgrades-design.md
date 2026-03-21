# ARISE Pipeline Quality Upgrades — Design Spec

**Date:** 2026-03-21
**Goal:** Upgrade the AutoReview 17-node pipeline to 20 nodes, automatically producing ARISE 90+ scoring review papers by generating figures, tables, and quality-enhancing prose sections.
**Motivation:** Manual post-hoc improvements to the v3 RAG review raised its ARISE score from 84 to 91-93/100. This spec integrates those improvements into the pipeline so every future run benefits automatically.

---

## 1. Scope

### In Scope
- 3 new DAG nodes: `figure_generation`, `table_generation`, `visual_audit`
- Modifications to 5 existing nodes: `final_outline`, `narrative_planning`, `section_writing` (via outline/directives only), `assembly`, `final_polish`
- New KB fields for figure and table metadata
- Deterministic figure generation using `SchematicEngine` + matplotlib

### Out of Scope
- PDF generation (handled by existing output formatter)
- Changes to search, screening, extraction, or clustering stages
- ARISE self-evaluation (could be a future node but not in this spec)
- LaTeX/docx output format changes (markdown figures use standard `![](path)` syntax)

---

## 2. Updated DAG Architecture (20 nodes)

```
query_expansion → search → screening → full_text_retrieval → extraction
→ clustering → figure_generation* → gap_search → draft_outline
→ contextual_enrichment → corpus_expansion → final_outline
→ table_generation* → narrative_planning → citation_selection
→ section_writing → passage_search → assembly → visual_audit*
→ final_polish
```

Nodes marked `*` are new.

### Dependency Graph for New Nodes

| Node | Dependencies | Why | Timeout |
|------|-------------|-----|---------|
| `figure_generation` | `clustering` | Needs evidence_map for Fig 4, screened_papers for Fig 2 | 60s |
| `table_generation` | `final_outline` | Needs outline sections for Table 4 (takeaways), theme data for Tables 1-3 | 120s |
| `visual_audit` | `assembly` | Needs full draft to verify figure/table references and accuracy | 300s |

**Dependency note:** `narrative_planning` must depend on BOTH `table_generation` AND `figure_generation` (not just `final_outline` as in the current pipeline) so that visual-reference directives can read `kb.figures` and `kb.tables`. Registration in `build_pipeline()`:
```python
dag.add_node("narrative_planning", nodes.narrative_planning,
             dependencies=["table_generation"])
# figure_generation → gap_search → ... → table_generation already chains it
```

### Node Summary Strings

**Node summary strings** for `_node_summary()` in runner.py:
- `figure_generation`: `f"{len(kb.figures)} figures generated"`
- `table_generation`: `f"{len(kb.tables)} tables generated"`
- `visual_audit`: `f"audit: {len(kb.visual_audit_report.get('issues', [])) if kb.visual_audit_report else 0} issues"`

---

## 3. New Node: `figure_generation`

### Purpose
Generate 4 publication-quality PNG figures and store metadata in the KnowledgeBase so downstream nodes (section_writing, assembly) can reference them.

### Figures Produced

| Key | Tool | Data Source | Content |
|-----|------|-------------|---------|
| `fig1_rag_pipeline` | SchematicEngine | Static template | RAG pipeline architecture: Query → Retrieval → Reranking → Context Integration → LLM → Output. Three tier brackets: Naive, Advanced, Agentic. |
| `fig2_temporal` | matplotlib | `kb.screened_papers` year counts | Bar chart of papers by publication year. Annotate ChatGPT-era inflection (2023). |
| `fig3_taxonomy` | SchematicEngine | Static + clustering themes | Hierarchical tree: RAG Architectures → Naive/Advanced/Modular/Agentic → representative systems |
| `fig4_evidence` | matplotlib | `kb.evidence_map.themes[].evidence_strength_distribution` | Stacked horizontal bars: strong/moderate/weak/preliminary per theme |

### Implementation Details
- **No LLM calls needed** — all figures are deterministic from KB data + static templates
- **SchematicEngine** imported at runtime from `~/.claude/skills/programmatic-schematics/schematic_engine.py`
- **Colorblind-safe palette**: WONG dict from engine for schematics, `["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]` for matplotlib charts
- **Output**: PNGs saved to `{kb.output_dir}/figures/`, 300 DPI, Arial font
- **Fallback**: If SchematicEngine import fails, fall back to basic matplotlib rectangles+arrows for fig1/fig3 with a warning log

**Import strategy:** Use `importlib.util.spec_from_file_location` to attempt loading from `~/.claude/skills/programmatic-schematics/schematic_engine.py`. Catch `ImportError`, `OSError`, and `FileNotFoundError`. On failure, set a module-level `HAS_SCHEMATIC_ENGINE = False` flag and use matplotlib `FancyBboxPatch` + `FancyArrowPatch` for fig1/fig3 instead. Log `structlog.warning('schematic_engine.unavailable', fallback='matplotlib')`.

```python
try:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        'schematic_engine',
        Path.home() / '.claude/skills/programmatic-schematics/schematic_engine.py'
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    SchematicEngine = _mod.SchematicEngine
    WONG = _mod.WONG
    TINTS = _mod.TINTS
    HAS_SCHEMATIC_ENGINE = True
except (ImportError, OSError, FileNotFoundError, AttributeError):
    HAS_SCHEMATIC_ENGINE = False
```

### KB Updates
```python
kb.figures["fig1_rag_pipeline"] = FigureMetadata(
    key="fig1_rag_pipeline",
    path="figures/fig1_rag_pipeline.png",
    caption="Figure 1. Canonical RAG pipeline architecture showing the progression from naive single-retrieval to agentic multi-tool systems.",
    anchor=VisualInsertionAnchor(section_id="sec_1", position="after"),  # After Introduction
)
# ... similar for fig2-fig4
```

### Snapshot
Saves as `{output_dir}/snapshots/{timestamp}_figure_generation.json` containing figure metadata (not the PNG binary).

---

## 4. New Node: `table_generation`

### Purpose
Generate 4 markdown tables and store them in the KnowledgeBase for insertion during assembly.

### Tables Produced

| Key | Data Source | Content | LLM Required |
|-----|-------------|---------|-------------|
| `table1_retrieval` | Static + extraction study designs | Retrieval paradigm comparison: Dense, Sparse, Hybrid, Late-interaction, Learned-sparse. Columns: Method, Mechanism, Key Systems, Strengths, Limitations | No |
| `table2_domains` | Clustering theme paper counts | Domain application summary: Biomedical, Legal, Financial, Education, Enterprise, Multimodal. Columns: Domain, Papers, Key Challenges, Representative Systems | No |
| `table3_evaluation` | Static + extraction data | Evaluation framework comparison: RAGAS, ARES, RAGBench, ASTRID, Human eval. Columns: Framework, Metrics, Scope, Automation, Key Limitation | No |
| `table4_takeaways` | Final outline section descriptions | Key takeaway per body section (10 rows). Columns: Section, Key Takeaway | Yes (lightweight, ~200 tokens per section) |

### Implementation Details
- Tables 1-3 are **template-based with data fill** — static structure, dynamic paper counts and system names from KB
- Table 4 uses a **single LLM call** to distill each outline section description into a one-sentence takeaway
- Tables stored as raw markdown strings in `kb.tables`

### KB Updates
```python
kb.tables["table1_retrieval"] = TableMetadata(
    key="table1_retrieval",
    markdown="| Method | Mechanism | ... |",
    caption="Table 1. Comparison of retrieval paradigms for RAG systems.",
    anchor=VisualInsertionAnchor(section_id="sec_3_1", position="before"),  # Before first subsection of Retrieval
)
```

---

## 5. New Node: `visual_audit`

### Purpose
Post-assembly quality check verifying that all figures and tables are properly referenced in the draft text and that data-driven visuals still accurately reflect the assembled content.

### Checks Performed
1. **Reference completeness**: Every figure/table in `kb.figures`/`kb.tables` has at least one prose reference in `kb.full_draft`
2. **Caption consistency**: Figure/table captions match the surrounding discussion text
3. **Data accuracy**: For data-driven figures (fig2, fig4), verify the numbers mentioned in prose match the actual KB data
4. **Orphan detection**: No `![Figure` or `**Table` references in the draft that don't correspond to a KB entry

### Implementation
- Single lightweight LLM call scanning the draft against figure/table metadata
- Produces a report stored in `kb.visual_audit_report`
- If critical issues found (orphaned references, wrong numbers), flags them for `final_polish` to fix
- **Does not modify the draft** — audit only

**Integration with `final_polish`:** If `kb.visual_audit_report` contains entries with `severity='critical'` (e.g., orphaned figure reference, incorrect data claim), `final_polish` appends these as extra instructions to its LLM polish prompt: 'Additionally, fix the following visual reference issues: {issues}'. Non-critical issues (caption style, minor wording) are logged but not acted on.

---

## 6. Existing Node Modifications

### 6.1 `final_outline` — Add Quality Sections to Outline Template

**What changes:** When generating the outline, the prompt must include these additional sections:

1. **"Prior Survey Comparison"** — a paragraph in the Introduction comparing to prior RAG surveys (Gao et al. 2024, Zhao et al. 2024, Fan et al. 2024) and stating this review's differentiators (scale, temporal coverage, analytical framework)
2. **"Ethical and Societal Considerations"** — a Discussion subsection covering: retrieval corpus bias, intellectual property, privacy-preserving RAG, environmental cost, epistemic authority shifts
3. **"Limitations of This Review"** — a Discussion subsection covering: abstract-only papers percentage, language/venue bias, preprint dominance, no formal meta-analysis, search cutoff

**How:** Modify the outline generation prompt in `autoreview/llm/prompts/outline.py` to include these sections as required elements in the outline template. The outline critique loop will validate their presence. Add ethics, limitations, and prior-survey-comparison to `OutlineConfig.required_sections` defaults in `autoreview/config/models.py` rather than hardcoding them in the prompt string.

**Section type classification:** "ethics" and "limitations" are type `discussion`; "prior survey comparison" is type `introduction`.

### 6.2 `narrative_planning` — Add Quality Directives

**What changes:** The narrative plan must include directives for:

1. **Productive Tensions Framework**: Name the contradiction-resolution approach explicitly. Directive: "Frame the four contested contradictions as the Productive Tensions Framework — a named analytical contribution of this review."
2. **Methodology expansion**: Directive for the Methods section writer: "Describe the evidence weighting methodology: study design classification, quality scoring (0-1), sample size weighting. Explain how contradictions were identified through structured resolution."
3. **Limitations directive**: Key points to cover (abstract-only %, language bias, preprint dominance, no meta-analysis, search cutoff)
4. **Ethics directive**: Key points to cover (corpus bias, IP, privacy, environmental cost, epistemic authority)
5. **Figure/table references**: Directives instructing section writers to reference specific figures/tables where relevant (e.g., "Reference Figure 1 when describing the pipeline architecture", "Reference Table 1 when comparing retrieval methods")

**How:** Modify the narrative planning prompt in `autoreview/llm/prompts/narrative.py` to include these directives. The directive generation reads `kb.figures` and `kb.tables` metadata to know which visuals are available.

### 6.3 `section_writing` — No Code Changes

The new sections (ethics, limitations, prior survey comparison) will be written naturally because they appear in the outline and have narrative directives. Writers will reference figures/tables because the narrative directives instruct them to. No changes to the section writing code itself.

### 6.4 `assembly` — Insert Visuals and Polish Abstract

**What changes (3 additions):**

1. **Visual insertion:** `DraftAssembler.assemble()` already produces markdown with `# N. Section Title` headings. The `_insert_visuals(draft, figures, tables)` helper locates insertion points by regex-matching section headings (e.g., `^# 3\.` for `insert_before_section='sec_3'`, or matching the heading *after* `sec_1` for `insert_after_section='sec_1'`). This uses the existing heading structure — no new markers needed. The helper:
   1. Builds a position map: `{section_id: char_offset}` by scanning for `# N.` headings
   2. For each figure, inserts the `![caption](path)` markdown + caption prose AFTER the matched section's last paragraph (before the next `#` heading)
   3. For each table, inserts the table markdown BEFORE the matched section heading
   4. Inserts in reverse position order to preserve character offsets

2. **Abstract objectives**: After assembly, scan the abstract for the review's purpose statement and make objectives explicit with a four-part enumeration: "(1) mapping the technical landscape...; (2) assessing domain-specific adaptation...; (3) resolving contested contradictions...; (4) identifying critical gaps..."

3. **Navigation guidance**: Append a "How to read this review" sentence to the end of the Introduction section.

**How:** Add a `_insert_visuals(draft, figures, tables)` helper function. Add `_polish_abstract(draft)` and `_add_navigation(draft)` helpers. Call all three after the existing assembly logic but before the holistic critique loop (so the critique evaluates the complete document including visuals).

**Variable scope:** These three helpers operate on the local `full_draft` string variable inside the `assembly()` method, BEFORE it enters the holistic critique loop. They do NOT read or write `kb.full_draft`, which is only set at the end of assembly from the post-critique `final_draft`. The implementation pattern:
```python
full_draft = assembler.assemble(outline, section_drafts)
full_draft = _insert_visuals(full_draft, kb.figures, kb.tables)
full_draft = _polish_abstract(full_draft)
full_draft = _add_navigation(full_draft)
# ... then holistic critique loop operates on this full_draft ...
kb.full_draft = final_draft  # only set at the end
```

### 6.5 `final_polish` — Supplementary Corpus Note Only

**What changes (1 addition):**

The existing `OutputFormatter._resolve_citations()` already converts `[@paper_id]` markers to sequential `[N]` numbers in first-appearance order. Citation renumbering is therefore **not needed** in `final_polish`.

The only addition is the supplementary corpus note. However, since `final_polish` operates on `kb.full_draft` which still uses `[@paper_id]` markers, and the note references numbered citations, the corpus note should be added in `OutputFormatter.format_markdown()` instead — after `_resolve_citations()` resolves the markers. Add `_add_corpus_note(text, corpus_size, cited_count)` as a formatter helper that inserts at the top of the References section:

*'This review synthesises {corpus_size} papers; the references below list the {cited_count} works directly cited in the text. The full corpus is available in the supplementary data.'*

Where `corpus_size = len(kb.screened_papers)` and `cited_count` is the length of `cited_ids` returned by `_resolve_citations()`.

---

## 7. Data Model Changes

### New Models (in `autoreview/models/`)

```python
class VisualInsertionAnchor(AutoReviewModel):
    section_id: str               # e.g., 'sec_3' or 'sec_3_1'
    position: Literal['before', 'after'] = 'after'

class FigureMetadata(AutoReviewModel):
    key: str
    path: str
    caption: str
    anchor: VisualInsertionAnchor
    data_driven: bool = False

class TableMetadata(AutoReviewModel):
    key: str
    markdown: str
    caption: str
    anchor: VisualInsertionAnchor
```

### KnowledgeBase Additions

```python
class KnowledgeBase(TimestampedModel):
    # ... existing fields ...
    figures: dict[str, FigureMetadata] = Field(default_factory=dict)
    tables: dict[str, TableMetadata] = Field(default_factory=dict)
    visual_audit_report: dict[str, Any] | None = None
```

### PipelinePhase Additions

```python
class PipelinePhase(StrEnum):
    # ... existing phases ...
    FIGURE_GENERATION = "figure_generation"
    TABLE_GENERATION = "table_generation"
    VISUAL_AUDIT = "visual_audit"
```

---

## 8. File Changes Summary

| File | Change Type | What Changes |
|------|------------|-------------|
| `autoreview/pipeline/nodes.py` | Add methods | `figure_generation()`, `table_generation()`, `visual_audit()` |
| `autoreview/pipeline/runner.py` | Modify | Register 3 new nodes in `build_pipeline()`, add to `_node_summary()` |
| `autoreview/models/knowledge_base.py` | Add fields | `figures`, `tables`, `visual_audit_report` + `FigureMetadata`, `TableMetadata` models + 3 new `PipelinePhase` values |
| `autoreview/llm/prompts/outline.py` | Modify | Add ethics, limitations, prior-survey sections to outline template |
| `autoreview/llm/prompts/narrative.py` | Modify | Add Productive Tensions, methodology, ethics, limitations, visual-reference directives |
| `autoreview/pipeline/nodes.py` (assembly) | Modify | Add `_insert_visuals()`, `_polish_abstract()`, `_add_navigation()` helpers |
| `autoreview/pipeline/nodes.py` (final_polish) | Modify (minor) | No citation renumbering; corpus note moved to formatter |
| `autoreview/figures/generators.py` | Create | Figure generation functions using SchematicEngine + matplotlib |
| `autoreview/figures/__init__.py` | Create | Package init |
| `autoreview/tables/generators.py` | Create | Table generation functions |
| `autoreview/tables/__init__.py` | Create | Package init |
| `autoreview/output/formatter.py` | Modify — add `_add_corpus_note()` helper after `_resolve_citations()` |
| `tests/test_pipeline/test_figure_generation.py` | Create | Unit tests for figure generation |
| `tests/test_pipeline/test_table_generation.py` | Create | Unit tests for table generation |
| `tests/test_pipeline/test_visual_audit.py` | Create | Unit tests for visual audit |
| `tests/test_pipeline/test_citation_renumber.py` | Create | Unit tests for citation renumbering |

---

## 9. Testing Strategy

### Unit Tests
- **Figure generation**: Mock KB data, verify PNGs are created at expected paths, verify metadata is correct
- **Table generation**: Mock KB data, verify markdown tables have correct row/column counts, verify captions
- **Visual audit**: Mock draft + metadata, verify orphan detection and reference completeness
- **Citation renumbering**: Test with known input/output pairs, including grouped citations `[3, 7]`, edge cases (single citation, no citations)
- **Outline modifications**: Verify ethics/limitations sections appear in generated outlines
- **Assembly visual insertion**: Verify figures/tables appear at correct locations in assembled draft

### Integration Test
- Run full pipeline on a small fixture corpus (10-20 papers) and verify:
  - 4 PNG files exist in `output/figures/`
  - 4 tables appear in `review.md`
  - Ethics and limitations sections present
  - Citations numbered sequentially
  - Supplementary corpus note present

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| SchematicEngine not available on target machine | Fallback to basic matplotlib rectangles+arrows; log warning |
| Figure generation slows pipeline | Figures are deterministic (no LLM), should complete in <5 seconds |
| Table 4 LLM call fails | Fall back to using outline section descriptions directly (no distillation) |
| Visual insertion breaks section numbering | Insert visuals BETWEEN sections (after/before markers), not inside sections |
| Citation renumbering breaks grouped citations | Regex handles `[N, M, ...]` patterns; unit tested |
| Outline prompt too long with new sections | Ethics/limitations are 2-3 sentences each in the prompt; minimal token increase |
