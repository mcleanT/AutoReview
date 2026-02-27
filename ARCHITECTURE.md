# AutoReview Architecture

AutoReview is a fully autonomous, domain-agnostic pipeline that generates publication-ready scientific review papers. Given a topic or research question, it searches the literature, extracts structured evidence, synthesizes findings, writes a complete review, and self-critiques iteratively until quality thresholds are met — with no human intervention required.

**Core philosophy**: The quality of generated text is the #1 priority. Everything else — formatting, speed, cost — is secondary. A good review paper synthesizes rather than summarizes, traces contradictions, identifies gaps, and tells a coherent narrative.

**Architecture**: DAG-based pipeline with multi-stage critique. Designed to evolve toward a multi-agent system when the DAG's quality ceiling is reached.

---

## Pipeline DAG

```
[Query Expansion] → [Multi-Source Search] → [Screen & Deduplicate]
                                                     ↓
                          [Parallel Extraction (per paper)]
                                                     ↓
                    [Thematic Clustering + Contradiction Detection]
                                                     ↓
                              [Outline Generation] → [Outline Critique] ←→ [Revise Outline]
                                                     ↓
                              [Gap-Aware Supplementary Search] (conditional)
                                                     ↓
                         [Section Writing (with cross-section context)]
                                                     ↓
                              [Per-Section Critique] ←→ [Section Revision]
                                                     ↓
                    [Assemble Draft] → [Holistic Critique] ←→ [Cross-Section Revision]
                                                     ↓
                              [Final Polish] → [Format Output]
```

Each DAG node is an async Python callable with typed Pydantic inputs/outputs. The pipeline state is serialized to JSON after every node for crash recovery and debugging.

---

## Core Data Models

All inter-stage communication uses Pydantic v2 models. These are the foundation of the system.

### Paper Models (`models/paper.py`)

- **`CandidatePaper`** — Raw search result: title, authors, year, journal, DOI, abstract, source database
- **`ScreenedPaper`** — CandidatePaper + relevance score (1-5) + screening rationale

### Extraction Models (`extraction/models.py`)

- **`Finding`** — A single claim extracted from a paper: `claim`, `evidence_strength` (strong/moderate/weak/preliminary), `supports_or_contradicts` (relationship to other findings), `quantitative_result`
- **`RelationshipClaim`** — How one paper relates to another: `target_paper_id`, `relationship_type` (supports/contradicts/extends/replicates), `description`
- **`MethodologyRecord`** — Approach, datasets used, metrics, reproducibility indicators
- **`PaperExtraction`** — Full structured extraction: paper metadata, `key_findings: list[Finding]`, `methods_summary`, `limitations`, `relationships: list[RelationshipClaim]`, `methodology_details`, `domain_specific_fields: dict[str, Any]`

### Evidence Models (`analysis/evidence_map.py`)

- **`Theme`** — Named cluster of related findings: `name`, `description`, `paper_ids`, `sub_themes`
- **`ConsensusClaim`** — A claim supported by multiple papers
- **`Contradiction`** — Opposing claims with supporting papers and a hypothesis about why they differ
- **`IdentifiedGap`** — Expected sub-topic with insufficient evidence
- **`EvidenceMap`** — Full evidence landscape: `themes`, `consensus_claims`, `contradictions`, `gaps`, `evidence_chains`

### Critique Models (`critique/models.py`)

- **`CritiqueIssue`** — `severity` (critical/major/minor), `location` (section reference), `description`, `suggested_fix`
- **`CritiqueReport`** — `target` (outline/section/full_draft), `passed: bool`, `overall_score: float`, `dimension_scores: dict[str, float]`, `issues: list[CritiqueIssue]`, `identified_gaps`

### Central State (`models/knowledge_base.py`)

- **`KnowledgeBase`** — Holds everything: search queries, candidate papers, screened papers, extractions, evidence map, outline, section drafts, full draft, critique history, revision history, current phase, iteration counts, and a full agent/action audit log. Serialized to JSON after every DAG node.

---

## Pipeline Stages

### 1. Query Expansion
**Input**: Topic string (e.g., "the role of gut microbiome in neurodegenerative diseases")
**Output**: Structured search queries + scope document
**Logic**: LLM generates Boolean queries for PubMed, semantic queries for Semantic Scholar, and natural language questions for Perplexity. Also generates a "scope document" defining what the review should cover, its boundaries, and expected sub-topics.

### 2. Multi-Source Search
**Input**: Search queries
**Output**: Unified list of CandidatePapers
**Logic**: Concurrent API calls to PubMed, Semantic Scholar, OpenAlex, and Perplexity Sonar. Results are unified into the CandidatePaper schema and deduplicated by DOI. Target: 200-500 raw candidates.

### 3. Screening
**Input**: CandidatePapers
**Output**: ScreenedPapers (filtered, scored)
**Logic**: LLM scores each abstract for relevance (1-5) in batches. Configurable threshold (default: 3). Target working corpus: 50-200 papers.

### 4. Parallel Extraction
**Input**: ScreenedPapers (full text where available, otherwise abstract)
**Output**: PaperExtraction per paper
**Logic**: Each paper extracted independently and concurrently. The LLM populates the PaperExtraction Pydantic model using structured output. Domain-specific fields are configured per domain YAML.

### 5. Thematic Clustering + Contradiction Detection
**Input**: All PaperExtractions
**Output**: EvidenceMap
**Logic**: Clusters findings into themes/sub-themes. Identifies consensus claims, contradictions, and gaps (where the scope document expected coverage but none was found). This is the analytical core of the pipeline.

### 6. Outline Generation + Critique
**Input**: EvidenceMap + scope document
**Output**: Validated hierarchical outline
**Logic**: LLM generates a section outline from the evidence map. A separate critique step evaluates completeness, logical ordering, and granularity. Up to 2 revision cycles.

### 7. Gap-Aware Supplementary Search (Conditional)
**Input**: IdentifiedGaps from evidence map
**Output**: Additional PaperExtractions
**Logic**: If gaps were found, targeted search queries fill specific holes. New papers go through extraction and are added to the evidence map. Runs at most once.

### 8. Section Writing
**Input**: Outline + relevant extractions per section + full outline context
**Output**: Draft sections
**Logic**: Each section written with: its assigned findings, the full outline (for cross-section awareness), and explicit synthesis directives. The writer is instructed to synthesize, not summarize — to trace patterns, weigh contradictions, and build narrative.

### 9. Per-Section Critique + Revision
**Input**: Each draft section
**Output**: Revised sections
**Logic**: Each section critiqued for: citation accuracy, synthesis quality (not paper-by-paper summary), connection to adjacent sections, internal coherence. Failing sections are revised. Up to 2 cycles per section.

### 10. Holistic Critique + Cross-Section Revision
**Input**: Assembled full draft
**Output**: Revised full draft
**Logic**: Evaluates: narrative arc, redundancy across sections, transition quality, introduction/conclusion alignment, overall balance, fair treatment of conflicting findings. Up to 3 revision cycles or until score converges.

### 11. Final Polish + Format
**Input**: Final draft
**Output**: Formatted document (Markdown, LaTeX, or DOCX)
**Logic**: Language polishing, terminology consistency, bibliography assembly from structured citation data, template application via Jinja2 + Pandoc.

---

## Literature Search

### Data Sources
| Source | Package/API | Purpose |
|---|---|---|
| PubMed | `biopython` (`Bio.Entrez`) | Biomedical literature, structured Boolean queries |
| Semantic Scholar | `semanticscholar` | Semantic/embedding-based search, citation graphs |
| OpenAlex | `pyalex` | Broad academic coverage across all domains |
| Perplexity Sonar | `httpx` (REST API) | AI-powered discovery of recent or under-indexed work |

### Search Strategy
- LLM generates domain-appropriate queries for each source
- All sources queried concurrently via `asyncio`
- Results unified into `CandidatePaper` schema, deduplicated by DOI
- Gap-aware second pass after extraction/clustering fills identified holes
- Each `SearchSource` implements a protocol: `async def search(queries: list[str]) -> list[CandidatePaper]`

---

## LLM Integration

### Provider Abstraction (`llm/provider.py`)

```python
class LLMProvider(Protocol):
    async def generate(self, prompt: str, system: str = "", max_tokens: int = 4096) -> str: ...
    async def generate_structured(self, prompt: str, response_model: type[T], system: str = "") -> T: ...
```

### Implementation Notes
- Primary: Claude via `anthropic` async SDK
- Use Claude's structured output / tool-use capabilities for all extraction and critique stages — return Pydantic models directly
- The `LLMProvider` protocol enables swapping to other providers without changing pipeline logic
- All prompts are constructed programmatically in `llm/prompts/` modules, not as raw string templates

---

## Self-Critique System

### Three Critique Levels

| Level | Target | Evaluates | When |
|---|---|---|---|
| Outline critique | Generated outline | Completeness vs scope, logical ordering, granularity | After outline generation |
| Per-section critique | Each section individually | Citation accuracy, synthesis quality, coherence, connection to neighbors | After section writing |
| Holistic critique | Full assembled draft | Narrative arc, redundancy, balance, transitions, intro/conclusion alignment | After draft assembly |

### Critique Dimensions (configurable per domain)
- **Coverage**: Are important sub-topics addressed?
- **Synthesis**: Does it synthesize across papers or summarize one-by-one?
- **Accuracy**: Are claims properly attributed to sources?
- **Balance**: Are conflicting findings fairly presented?
- **Narrative**: Does the text flow logically with clear transitions?
- **Gaps**: Are research gaps and future directions identified?

### Termination Criteria
1. Overall score exceeds threshold (default: 0.8)
2. Maximum revision count reached (default: 3)
3. Score stops improving between iterations (convergence detection)

---

## Domain Configuration

Domains are configured via YAML files in `config/defaults/`. No code changes needed to add a new domain.

```yaml
domain: biomedical
databases:
  primary: [pubmed, semantic_scholar]
  secondary: [openalex]
  discovery: [perplexity]
search:
  date_range: "2015-2025"
  max_results_per_source: 500
  relevance_threshold: 3
extraction:
  domain_fields:
    sample_size: true
    study_design: true
    p_values: true
critique:
  rubric_weights:
    coverage: 0.25
    synthesis: 0.30
    accuracy: 0.20
    balance: 0.15
    narrative: 0.10
writing:
  style: academic_biomedical
  citation_format: vancouver
outline:
  required_sections:
    - Introduction
    - Methods of Review
    - Results
    - Discussion
    - Future Directions
```

Default domain configs ship for: `biomedical`, `cs_ai`, `chemistry`. Additional domains are added by creating a new YAML file.

---

## Technology Stack

| Component | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Ecosystem, async support, scientific tooling |
| Async | `asyncio` | Concurrent search and extraction |
| LLM SDK | `anthropic` (async) | Claude-first, wrapped in abstract protocol |
| Data models | Pydantic v2 | Type safety, serialization, validation |
| HTTP client | `httpx` | Async, modern, used for Perplexity + fallback APIs |
| CLI | `typer` | Clean CLI with minimal boilerplate |
| Metadata storage | SQLite | No server needed, sufficient for single-user research tool |
| Pipeline state | JSON files | Snapshots after every DAG node for recovery/debugging |
| Output templating | `jinja2` | Flexible template rendering |
| Format conversion | `pypandoc` | Markdown → LaTeX / DOCX |
| Testing | `pytest` + `pytest-asyncio` | Standard, async-aware |
| Logging | `structlog` | Structured logging with token usage tracking |
