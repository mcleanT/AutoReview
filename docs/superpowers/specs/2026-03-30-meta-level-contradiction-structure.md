# Meta-Level Contradiction Structure — Design Spec

**Date**: 2026-03-30
**Status**: Approved
**Builds on**: claims-graph-v2-condition-aware-merging, post-extraction-normalization-layer, bayesian-inference-upgrade

## Problem

The current KG pipeline detects contradictions at the individual claim edge level — pairwise between `KGEdge` objects sharing entities. Field-level scientific contradictions operate at a higher level: two papers may have dozens of individually-consistent granular claims but reach opposite conclusions about a biological question. The system has no way to represent or detect these meta-level contradictions.

## Decision

Graph-time hierarchical clustering (no extraction schema changes). The prompt already extracts interpretive claims from Discussion sections. The finding layer is constructed algorithmically from the graph structure.

**Literature basis**: SemMedDB groups predications by shared triples for contradiction detection at PubMed scale (Kilicoglu et al. 2019). PSL/HL-MRF supports cluster-level truth variables as a natural extension. GraphRAG/LeanRAG use community detection → cluster summarization → inter-cluster analysis. The hierarchical approach is the strongest fit for contradiction resolution downstream.

## Architecture

### Graph Hierarchy

```
Community (Louvain — already implemented)
  └── TopicCluster (subject_id, predicate_class, object_id)
        └── Finding (topic_cluster + direction + condition_group)
              └── KGEdge (individual claim)
```

- **TopicCluster** represents a biological question ("What is the relationship between BMP4 and mesoderm specification?")
- **Finding** represents a coherent directional assertion within a topic cluster under specific experimental conditions
- **KGEdge** remains the atomic unit of evidence (unchanged)

### Predicate Class Collapse Table

Canonical predicates collapse into predicate classes so that related assertions group together:

| Predicate Class | Member Predicates |
|---|---|
| `activating` | `induces`, `is_sufficient_for`, `phosphorylates`, `stabilizes` |
| `inhibiting` | `inhibits`, `degrades` |
| `regulatory` | `regulates`, `is_required_for`, `modifies`, `maintains` |
| `associative` | `correlates_with`, `interacts_with`, `colocalizes_with` |
| `structural` | `is_component_of`, `is_located_in`, `is_marker_of` |
| `transformative` | `differentiates_into`, `converts`, `transports` |

This means "BMP4 induces mesoderm" and "BMP4 is_sufficient_for mesoderm" land in the same topic cluster — correct, since they're assertions about the same activating relationship.

## Data Model

### TopicCluster

```python
@dataclass
class TopicCluster:
    cluster_id: str            # SHA1(subject_id|predicate_class|object_id)[:16]
    subject_id: str
    object_id: str
    predicate_class: str       # From collapse table
    member_predicates: set[str] # Original predicates present
    edge_ids: list[str]
    finding_ids: list[str]     # Populated after finding formation
```

### Finding

```python
@dataclass
class Finding:
    finding_id: str            # SHA1(cluster_id|direction|condition_group_key)[:16]
    topic_cluster_id: str
    direction: str             # "positive" or "negative"
    condition_group: str       # Merged condition group key
    member_edge_ids: list[str]
    anchor_edge_id: str        # Best interpretive claim, or best empirical
    anchor_text: str           # natural_language of anchor edge
    confidence: BetaPosterior  # Aggregated from member edges
    paper_ids: set[str]
    evidence_count: int
    organism_class: str        # Species-level (e.g., "human", "mouse")
    in_vitro: bool
```

### FindingContradiction

```python
@dataclass
class FindingContradiction:
    finding_a_id: str
    finding_b_id: str
    topic_cluster_id: str
    contradiction_type: str    # "directional" | "boundary" | "interpretive"
    severity: float            # 0.0-1.0
    condition_coupling: float  # Reuse existing condition_coupling logic
    anchor_text_a: str
    anchor_text_b: str
    paper_ids_a: set[str]
    paper_ids_b: set[str]
```

## Finding Formation Algorithm

### Step 1 — Build topic clusters

Iterate all edges in the graph. For each edge, compute `predicate_class` from the collapse table. Group by `(subject_id, predicate_class, object_id)`. Discard singleton clusters (1 edge — no contradiction possible).

### Step 2 — Partition into findings

Within each topic cluster, group edges by `(direction, condition_group)`. Condition groups are formed by clustering `condition_signature` values: signatures sharing `organism_class` and `in_vitro` status merge into one condition group. `organism_class` is derived from the edge's `organism` field by mapping to species-level labels (e.g., "Mus musculus" → "mouse", "Homo sapiens" → "human") via a small lookup table with a fallback to lowercase genus. This means mouse-in-vivo and human-in-vitro remain separate findings even about the same triple.

### Step 3 — Assign anchors

For each finding, select the anchor claim:
1. If any member edge has `section_source == "interpretive"`, pick the one with highest `confidence_mean`
2. Otherwise, pick the empirical edge with highest `confidence_mean`
3. The anchor's `natural_language` becomes the finding's human-readable statement

### Step 4 — Compute finding confidence

Aggregate member edges' Beta posteriors by summing `(alpha - 1, beta - 1)` contributions (conjugate prior update), producing a single `BetaPosterior` for the finding.

## Meta-Level Contradiction Detection

Contradictions are detected between findings within the same topic cluster:

### Type 1 — Directional Contradiction

Two findings in the same topic cluster with opposite `direction` and overlapping condition groups. Classic case: "Paper A says X activates Y, Paper B says X does not activate Y" under similar experimental conditions.

Severity = `condition_coupling` score (reuse existing `condition_coupling()` logic from `structural_contradictions.py`).

### Type 2 — Cross-Condition Boundary

Two findings with opposite directions but non-overlapping condition groups. Flagged as `BOUNDARY_CONDITION` — a scope-dependent difference, not a true contradiction. Lower severity.

### Type 3 — Interpretive Conflict

Two findings where both anchors are interpretive claims (`section_source == "interpretive"`) from different papers, regardless of direction agreement at the empirical level. Captures the case where empirical claims are individually consistent but Discussion sections reach different conclusions.

Detected by: `anchor_a.section_source == "interpretive"` AND `anchor_b.section_source == "interpretive"` AND `paper_ids_a ∩ paper_ids_b == ∅`.

## HL-MRF Integration

Findings become first-class variables in the MRF alongside edges.

### New variable type

Each finding gets a truth variable `f_i ∈ [0,1]`, initialized to its aggregated `confidence_mean`.

### New rules

1. **Upward aggregation** (edge → finding):
   `weight × (f_i - mean(member_edges))²`
   Anchors finding truth to its constituent evidence.
   Weight = `evidence_weight` (10.0).

2. **Finding contradiction** (finding ↔ finding):
   `weight × max(0, f_a + f_b - 1)²`
   Same hinge-loss form as edge contradictions, but between findings.
   Weight = `finding_contradiction_weight` (default 12.0 — higher than edge-level 8.0 because finding-level contradictions are more meaningful).

3. **Downward propagation** (finding → edge):
   `weight × max(0, f_i - e_j)²` for each member edge `e_j`
   If a finding's truth value drops due to contradiction, its member edges should also decrease.
   Weight = `propagation_weight` (default 3.0 — lighter touch, evidence anchors dominate).

Edge-level contradiction rules remain for within-paper conflicts. Finding-level rules handle cross-paper conflicts. The MRF optimizes both simultaneously.

### MRF Config additions

```python
# New fields in MRFConfig
finding_contradiction_weight: float = 12.0
propagation_weight: float = 3.0
enable_finding_layer: bool = True  # Toggle for A/B comparison
```

## File Structure

### New file: `autoreview/knowledge_graph/cluster.py`

- `PREDICATE_CLASS_TABLE` — the collapse mapping dict
- `TopicCluster`, `Finding`, `FindingContradiction` dataclasses
- `build_topic_clusters(graph) → list[TopicCluster]`
- `form_findings(clusters, graph) → list[Finding]`
- `detect_finding_contradictions(findings, graph) → list[FindingContradiction]`
- `get_predicate_class(predicate: str) → str`

### Modified files

- **`hlmrf.py`** — Add finding variables to the optimization. Extend `ground_rules()` with upward aggregation, finding contradiction, and downward propagation rules. Extend `solve()` to include finding variables in L-BFGS-B. Add finding posteriors to `MRFResult`.
- **`mrf_scoring.py`** — Add `MRFConfig` fields: `finding_contradiction_weight`, `propagation_weight`, `enable_finding_layer`.
- **`analysis.py`** — Add `score_finding_contradiction_centrality()` to compute per-entity centrality from finding-level contradictions. Add `summarize_topic_clusters()` for reporting.

### Integration point

`build_topic_clusters()` + `form_findings()` runs after `build_nx_graph()` and before MRF solving. Findings are a parallel data structure, not graph nodes. The MRF solver receives both the graph and the findings list.

## Testing Strategy

### Unit tests (`tests/test_knowledge_graph/test_cluster.py`)

- Predicate class collapse: verify all canonical predicates map correctly
- Topic cluster formation: 3 edges with same subject/object but different activating predicates → 1 cluster
- Finding partitioning: edges with opposite directions → 2 findings; edges with same direction but different organisms → 2 findings
- Anchor selection: interpretive claim preferred over empirical; highest confidence wins within same section_source
- Condition grouping: same organism_class + in_vitro → same group; different → separate groups
- Singleton cluster filtering: clusters with 1 edge are excluded

### Unit tests for HL-MRF changes (`tests/test_knowledge_graph/test_hlmrf.py`)

- Finding variable grounding: verify finding variables appear in optimization
- Upward aggregation: finding truth tracks mean of member edges
- Downward propagation: low finding truth pulls member edges down
- Finding contradiction: two opposing findings can't both be high
- Toggle: `enable_finding_layer=False` produces identical results to current behavior

### Integration test

Small fixture graph: 3 papers, ~15 edges forming 2 topic clusters with 1 directional contradiction and 1 interpretive conflict. Verify:
1. Clusters and findings form correctly
2. Contradictions detected at finding level
3. MRF resolves: contradicted finding's truth value decreases
4. Downward propagation reduces member edge posteriors
5. Non-contradicted findings/edges unaffected

### Regression

All existing edge-level MRF tests pass unchanged when `enable_finding_layer=False`.
