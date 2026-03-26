# Knowledge Graph Prototype Design

**Date:** 2026-03-25
**Status:** Approved
**Author:** McLean Taggart + Claude

## Summary

Build a three-tier knowledge graph from the gastruloid corpus extractions (303 papers, ~2,900 assertions, ~3,100 evidence units, ~3,400 unique entities). **Tier 1** (assertions): merged, deduplicated claims at the level of biological mechanism — the unit of scientific discourse. **Tier 2** (evidence): individual experimental demonstrations grounding each assertion — the unit of evidential weight. **Tier 3** (provenance): paper-level metadata for independence and credibility assessment. Supports exploration/discovery, contradiction detection, and gap analysis. Uses NetworkX in-memory with Pydantic data contracts, designed for graduation to SQLite/Neo4j.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Use case | All: exploration, contradictions, gaps | Maximize value from existing extractions |
| Code location | `autoreview/knowledge_graph/` | Tight integration with extraction schema, models, search |
| Architecture | B+C hybrid: layered pipeline + Pydantic models | Clean separation, testable modules, graduation path baked in |
| Persistence | NetworkX in-memory (Tier 1) | Fast iteration for prototype; graduation path documented |
| Entity dedup | Hybrid: ontology ID + fuzzy string matching | Balances precision vs. recall without API cost |
| Assertion dedup | Predicate normalization + (subject, predicate, object) merging | Collapses cross-paper claims about the same mechanism into single edges |
| Confidence model | Beta-Binomial with evidence independence weighting | Simple, interpretable, incrementally updatable; discounts correlated sources |

## Module Structure

```
autoreview/knowledge_graph/
├── __init__.py          # Public API: build_graph(), load_graph()
├── models.py            # Pydantic: KGEntity, KGEdge, KGEvidenceLink, KGCitation, BetaPosterior
├── ingest.py            # Parse extraction JSONs → flat Pydantic records
├── dedup.py             # Entity resolution (ontology ID + fuzzy matching)
├── graph.py             # NetworkX MultiDiGraph construction + serialization
├── confidence.py        # Beta-Binomial edge scoring
├── analysis.py          # Community detection, contradiction finder, gap analysis
└── viz.py               # Graph visualization helpers (matplotlib + GraphML export)
```

## Data Flow

```
extraction JSONs → ingest.py → raw KGEntity/KGEdge/KGEvidence records
    → dedup.py → deduplicated entity registry (canonical ID → KGEntity)
               → predicate normalization (synonym families → canonical predicates)
               → assertion merging (same subject+predicate+object → single edge, evidence accumulates)
    → graph.py → NetworkX MultiDiGraph (entities=nodes, merged assertions=edges, evidence=edge attrs)
    → confidence.py → Beta(α,β) posteriors on each edge (with evidence independence weighting)
    → analysis.py → queries, metrics, subgraph extraction
    → viz.py → figures, GraphML for Gephi/Cytoscape
```

**Expected edge reduction:** ~2,900 raw assertions → estimated ~1,500–2,000 merged edges after entity dedup + predicate normalization + assertion merging. Each edge backed by 1–15+ evidence units from multiple papers.

## Pydantic Models (`models.py`)

### EntityType Enum

```python
class EntityType(str, Enum):
    gene = "gene"
    protein = "protein"
    pathway = "pathway"
    biological_process = "biological_process"
    cell_type = "cell_type"
    chemical = "chemical"
    organism = "organism"
    disease = "disease"
    method = "method"
    other = "other"
```

### KGEntity

All KG models extend `AutoReviewModel` (from `autoreview/models/base.py`) for consistency with project conventions (`extra="ignore"`, `use_enum_values=True`, `validate_default=True`).

```python
class KGEntity(AutoReviewModel):
    """A deduplicated node in the knowledge graph."""
    entity_id: str                    # deterministic hash of canonical_name + entity_type
    canonical_name: str               # "anterior-posterior axis symmetry breaking"
    entity_type: EntityType           # gene, protein, pathway, biological_process, cell_type, ...
    ontology_id: str | None           # "GO:0009855" (normalized: uppercase source prefix)
    ontology_source: str | None       # normalized to uppercase: "GO", "CL", "CHEBI", etc.
    aliases: list[str]                # accumulated from all papers (list for JSON serialization)
    paper_count: int                  # how many papers reference this entity
    source_paper_ids: list[str]       # paper hashes for provenance (list for JSON serialization)
```

### AssertionType Enum

```python
class AssertionType(str, Enum):
    mechanistic_causal = "mechanistic_causal"
    existence = "existence"
    comparative = "comparative"
    methodological = "methodological"
    correlational = "correlational"
    absence = "absence"
    conditional = "conditional"
```

### KGEdge

```python
class KGEdge(AutoReviewModel):
    """A typed assertion between two entities, grounded in evidence."""
    edge_id: str
    subject_id: str                   # → KGEntity.entity_id
    object_id: str                    # → KGEntity.entity_id
    predicate: str                    # "is_required_for", "induces", "inhibits", ...
    direction: str | None             # "positive", "negative", or None (null → unknown)
    assertion_type: AssertionType     # typed enum for filtering
    confidence: BetaPosterior         # α, β, mean, 95% CI
    evidence_links: list[KGEvidenceLink]
    source_assertions: list[str]      # draft_ids for traceability
    publication_date: str | None      # earliest paper date for temporal analysis
```

**Edge construction rules:**
- Assertions with empty-string predicates are assigned `"related_to"` as a default and flagged in the ingest log
- Assertions with `None` direction are stored as-is (direction=None means "unknown"). String `"null"` values are coerced to `None` during ingest.
- Self-loops (subject_id == object_id) are allowed — biologically valid (e.g., autoregulation)

### BetaPosterior

```python
class BetaPosterior(AutoReviewModel):
    """Beta-Binomial confidence score for an edge."""
    alpha: float = 1.0               # prior + supporting evidence count
    beta_param: float = 1.0          # prior + contradicting evidence count (beta_param to avoid shadowing)

    @computed_field
    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta_param)

    @computed_field
    @property
    def ci_95(self) -> tuple[float, float]:
        from scipy.stats import beta as beta_dist
        return beta_dist.interval(0.95, self.alpha, self.beta_param)
```

Uses `@computed_field` (Pydantic v2) so `mean` and `ci_95` appear in `.model_dump()` and JSON serialization.

### KGEvidenceLink

```python
class KGEvidenceLink(AutoReviewModel):
    """Links an edge to its experimental grounding."""
    evidence_id: str
    paper_id: str
    evidence_strength: EvidenceStrength  # typed enum matching actual data
    evidence_direction: str           # "supports", "contradicts"
    experiment_summary: str
    model_system: str | None
    sample_size: str | None
    key_figure: str | None
    publication_date: str | None      # from paper_provenance, for temporal gap analysis
```

### KGCitation

```python
class KGCitation(AutoReviewModel):
    """A citation context linking papers through claim references."""
    citation_id: str
    citing_paper_id: str              # paper hash of the citing paper
    cited_source_doi: str | None       # matches extraction field name
    cited_source_pmid: str | None      # matches extraction field name
    citing_sentence: str
    cited_claim_paraphrase: str | None
    relationship: str                 # "supports", "contradicts", "extends", etc.
    linked_assertion_ids: list[str]   # assertion draft_ids this citation relates to
    section: str | None               # where in the paper this citation appears
```

## Entity Deduplication (`dedup.py`)

Three-pass strategy:

### Pass 1: Exact Ontology Match
Normalize ontology sources first (split on `;` and `,`, strip whitespace, deduplicate, case-fold to uppercase — handles `"GO; UniProt"`, `"GO, UBERON"`, `"CL; CL"`). Then entities sharing the same `ontology_id` merge immediately. Cheapest, highest confidence.

### Pass 2: Canonical Name Normalization
Lowercase, strip whitespace, expand common abbreviations (e.g., "A-P" to "anterior-posterior"). Exact match after normalization merges.

### Pass 3: Fuzzy Matching Within Entity Type
Only compare entities of the same `entity_type` (genes vs genes, not genes vs pathways). Two metrics:
- Token overlap ratio (Jaccard on word tokens) with threshold >= 0.75
- Levenshtein ratio >= 0.85 on canonical names

Merge candidates in the ambiguous zone (0.75-0.90) are flagged for manual review.

**Handling the `other` bucket:** The `other` entity type contains ~1,400 entries (25% of all entities), making brute-force pairwise comparison expensive. Strategy: sub-cluster `other` entities by first significant word token before fuzzy matching, capping comparison to entities sharing at least one word token. This reduces the effective comparison set from O(n^2) to manageable blocked comparisons.

### Merge Behavior
- Entity with an ontology ID wins as canonical
- Aliases accumulate from all merged entities
- Paper counts sum
- A `merge_log` records every merge decision for auditability

### Output
`EntityRegistry`: dict mapping `entity_id -> KGEntity`, plus a reverse index `surface_form -> entity_id` for fast lookup during edge construction.

## Predicate Normalization (`dedup.py`)

Before assertion merging, predicates are normalized to canonical forms using a synonym table. This collapses surface-level variation in how papers describe the same relationship.

### Predicate Synonym Families

| Canonical Predicate | Synonyms |
|---|---|
| `is_required_for` | `is_necessary_for`, `is_essential_for`, `is_critical_for`, `is_needed_for` |
| `induces` | `activates`, `triggers`, `initiates`, `promotes`, `stimulates`, `upregulates` |
| `inhibits` | `suppresses`, `blocks`, `represses`, `downregulates`, `prevents`, `attenuates` |
| `expresses` | `produces`, `encodes`, `transcribes` |
| `regulates` | `modulates`, `controls`, `mediates` |
| `differentiates_into` | `gives_rise_to`, `develops_into`, `matures_into`, `becomes` |
| `interacts_with` | `binds_to`, `associates_with`, `complexes_with` |
| `is_marker_for` | `marks`, `identifies`, `labels`, `characterizes` |
| `contains` | `comprises`, `includes`, `consists_of` |
| `is_located_in` | `localizes_to`, `is_expressed_in`, `is_found_in` |

**Strategy:** Two-pass normalization:
1. **Exact synonym lookup**: O(1) dictionary lookup against the table above
2. **Fuzzy predicate matching**: For predicates not in the synonym table, compute Levenshtein ratio against all canonical predicates. Match at >= 0.85, flag at 0.70–0.85 for manual review, leave as-is below 0.70.

Predicates that don't match any family are kept as-is — the synonym table is intentionally conservative. A `predicate_normalization_log` records all mappings for auditability.

**Bootstrap approach:** The synonym table above is seeded from the top predicates in the corpus. After the first full graph build, review the predicate frequency distribution and expand the table as needed.

## Assertion Merging (`dedup.py`)

After entity dedup and predicate normalization, assertions that describe the same biological relationship across papers are merged into a single `KGEdge` with accumulated evidence.

### Merge Key

Two assertions merge when they share the same `(subject_id, canonical_predicate, object_id)` triple after entity dedup and predicate normalization. This is the **Tier 1 assertion identity** — the unit of scientific discourse that papers can independently provide evidence for or against.

### Merge Behavior

- `edge_id`: deterministic hash of `(subject_id, canonical_predicate, object_id)`
- `evidence_links`: accumulate all evidence units from all merged assertions
- `source_assertions`: accumulate all draft_ids for traceability
- `assertion_type`: majority vote across merged assertions (ties broken by earliest publication)
- `direction`: if all merged assertions agree, use that direction; if they disagree, set to `None` and flag as a potential contradiction (this feeds contradiction detection in `analysis.py`)
- `publication_date`: earliest date across all merged assertions
- An `assertion_merge_log` records every merge decision, including the source papers and original predicates

### Expected Impact

With ~2,900 raw assertions, entity dedup (~3,400 → ~2,000 entities) and predicate normalization will cause many assertions about the same entities with synonymous predicates to collapse. Conservative estimate: **~1,500–2,000 unique edges**, each backed by 1–15+ evidence units. This is where the graph becomes useful — you discover that "Wnt signaling induces mesoderm" has been demonstrated in 12 different model systems across 8 labs, even though no single paper frames it that way.

### Direction Conflict as Signal

When assertions merge but disagree on `direction` (e.g., one says "positive", another says "negative"), this is valuable signal, not noise. The merged edge stores `direction=None` and the conflicting directions are preserved in the individual evidence links. This naturally feeds into contradiction detection without requiring a separate mechanism.

## Confidence Scoring (`confidence.py`)

### EvidenceStrength Enum and Weight Table

Evidence strength categories match the actual extraction data values:

```python
class EvidenceStrength(str, Enum):
    direct_experimental = "direct_experimental"
    observational_controlled = "observational_controlled"
    observational_uncontrolled = "observational_uncontrolled"
    computational_prediction = "computational_prediction"
    expert_opinion = "expert_opinion"
```

| Evidence Strength | Support Weight (+alpha) | Contradict Weight (+beta) | Count in Corpus |
|---|---|---|---|
| `direct_experimental` | 1.0 | 1.0 | ~2,479 |
| `observational_controlled` | 0.7 | 0.7 | ~64 |
| `observational_uncontrolled` | 0.4 | 0.4 | ~156 |
| `computational_prediction` | 0.3 | 0.3 | ~142 |
| `expert_opinion` | 0.2 | 0.2 | ~236 |

Each evidence unit for an edge updates the Beta posterior based on its `evidence_direction` (supports -> alpha, contradicts -> beta) scaled by its `evidence_strength` weight. Prior is Beta(1,1) (uniform/uninformative). Unknown evidence strength values are mapped to `expert_opinion` (lowest weight) with a warning log.

### Evidence Independence Weighting

Evidence from the same research group should count less than independent replications. Two papers from the same lab using the same cell line are not independent evidence — they share systematic biases.

**Source grouping:** Evidence units are grouped by `paper_id`. Papers sharing the same first/last author combination are flagged as potentially non-independent (using author lists from `paper_provenance`).

**Independence discount:** When multiple evidence units for the same edge come from the same author group:
- First evidence unit from a group: full weight (per evidence strength table)
- Subsequent units from the same group: weight multiplied by `0.5` (diminishing returns)

This means 3 evidence units from 3 independent labs contribute `3.0` to alpha, while 3 units from the same lab contribute `1.0 + 0.5 + 0.25 = 1.75`. The discount is applied during Beta posterior updates in `confidence.py`, not during ingest.

**Provenance fields used:** `paper_provenance.authors` (first + last author), `paper_provenance.journal`, `paper_provenance.funding`. For the prototype, only author overlap is used; institutional/funding correlation is deferred to v2.

### Derived Metrics Per Edge
- `confidence_mean`: alpha / (alpha + beta)
- `confidence_ci_95`: 95% credible interval from Beta distribution
- `evidence_count`: total evidence units
- `independent_source_count`: number of distinct author groups contributing evidence
- `controversy_score`: min(alpha, beta) / max(alpha, beta) -- high when evidence is split (safe with Beta(1,1) prior; alpha and beta are always >= 1.0)
- `evidence_diversity`: number of distinct papers contributing evidence

## Analysis Layer (`analysis.py`)

### Exploration/Discovery
- Community detection (Louvain algorithm) to find claim clusters
- Hub entities: highest degree centrality (most-connected concepts)
- Subgraph extraction by entity type, pathway, or keyword

### Contradiction Detection
- Edges with `controversy_score > 0.5` (substantial disagreement)
- Edges where different papers have opposing `evidence_direction`
- Contradiction subgraph: extract the neighborhood around contested claims

### Gap Analysis
- Low-evidence entities: high degree but low total evidence count (well-connected but poorly grounded)
- Missing edges: entity pairs that co-occur in papers but have no assertion connecting them
- Temporal gaps: claims supported only by old evidence (no recent replication). Uses `publication_date` propagated from `paper_provenance` through `KGEvidenceLink.publication_date`

## Visualization (`viz.py`)

- `export_graphml()`: full graph with all attributes for Gephi/Cytoscape
- `plot_subgraph()`: matplotlib network plot, nodes colored by entity type, edges colored by confidence, edge width by evidence count. Colorblind-safe palette per project conventions.
- `plot_confidence_distribution()`: histogram of Beta posteriors across all edges
- `plot_controversy_map()`: highlight high-controversy edges in the network

All figures follow project standards: colorblind-safe palettes, 300 DPI, Arial/Helvetica typography, `constrained_layout=True`.

## Public API (`__init__.py`)

```python
def build_graph(extraction_dir: Path) -> nx.MultiDiGraph:
    """Full pipeline: ingest → dedup → graph → confidence. Returns annotated graph."""

def load_graph(path: Path) -> nx.MultiDiGraph:
    """Load a previously built graph from pickle format."""

def save_graph(graph: nx.MultiDiGraph, path: Path) -> None:
    """Serialize graph to pickle (fast reload) and GraphML (interop)."""
```

Serialization uses pickle for fast Python-native reload. GraphML export is available via `viz.export_graphml()` for external tools (Gephi/Cytoscape). Both formats are written on `save_graph()`.

## Graduation Path

Only Tier 1 is implemented in this prototype. Tiers 2 and 3 are documented for future reference.

| Tier | Storage | Trigger | What Changes |
|---|---|---|---|
| **1 (now)** | NetworkX + GraphML/pickle | Prototype | Everything |
| **2** | SQLite (entities/edges/evidence tables) | >50K edges or need concurrent queries | Swap `graph.py` to `storage_sqlite.py`; Pydantic models serialize via ORM |
| **3** | Neo4j | Need Cypher queries, web UI, or multi-user access | Swap storage layer; add Cypher query interface in `analysis.py` |

The Pydantic models in `models.py` are the graduation contract. Upstream modules (ingest, dedup) and downstream modules (confidence, analysis, viz) remain unchanged when swapping the storage layer.

## Input Data Summary

Source: `Paper Extractor/KnowledgeGraph Extraction/gastruloid_run/extractions/`

| Metric | Count |
|--------|-------|
| Valid extraction JSONs | 303 |
| Total assertions | ~2,900 |
| Total evidence units | ~3,100 |
| Total citation contexts | ~650 |
| Unique entities (canonical name) | ~3,400 |
| Top entity types | biological_process (~2,200), cell_type (~840), protein (~600), gene (~120), pathway (~90) |
| Top predicates | is_required_for (137), induces (94), expresses (66), contains (64), generates (57) |
| Assertions with empty predicate | ~54 (assigned `"related_to"` default) |
| Assertions with null direction | ~37 (stored as None) |
| Evidence strength distribution | direct_experimental (~2,479), expert_opinion (~236), observational_uncontrolled (~156), computational_prediction (~142), observational_controlled (~64) |

### Input JSON Schema

Each extraction JSON has the following top-level structure (from mycelium's `ExtractionResult`):
- `paper_provenance`: DOI, PMID, title, authors, journal, publication_date, peer_reviewed, funding
- `evidence_units[]`: experimental evidence with evidence_id, assertion_draft_ids, evidence_direction, evidence_strength, experiment details, results, methodology tags
- `assertion_drafts[]`: claims with draft_id, natural_language/canonical_form, subject_entity, object_entity, predicate, direction, assertion_type, scope, hedging, evidence_unit_ids
- `citation_contexts[]`: cross-paper references with citation_id, citing_sentence, cited_source_doi, relationship, linked_assertion_draft_ids

## Dependencies

- `networkx` (graph construction and algorithms)
- `scipy` (Beta distribution for confidence intervals)
- `python-Levenshtein` or `rapidfuzz` (fuzzy string matching)
- `networkx.algorithms.community` (Louvain built-in since NetworkX 2.7, no external dep needed)
- `matplotlib` (visualization)
- `pydantic` (data models -- already in project)
- `structlog` (logging -- already in project)
