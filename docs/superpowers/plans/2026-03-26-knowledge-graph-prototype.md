# Knowledge Graph Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a three-tier knowledge graph from 303 gastruloid corpus extractions, with entity/assertion deduplication, Beta-Binomial confidence scoring, and analysis capabilities (exploration, contradiction detection, gap analysis).

**Architecture:** Layered pipeline (ingest → dedup → graph → confidence → analysis → viz) with Pydantic data contracts serving as the graduation interface for future storage backends. NetworkX MultiDiGraph for in-memory graph representation. All modules live under `autoreview/knowledge_graph/`.

**Tech Stack:** NetworkX (graph), rapidfuzz (fuzzy matching), scipy (Beta distribution), matplotlib (viz), Pydantic v2 (models), structlog (logging)

**Spec:** `docs/superpowers/specs/2026-03-25-knowledge-graph-prototype-design.md`

---

## File Structure

```
autoreview/knowledge_graph/
├── __init__.py          # Public API: build_graph(), load_graph(), save_graph()
├── models.py            # KGEntity, KGEdge, KGEvidenceLink, KGCitation, BetaPosterior, enums
├── ingest.py            # Parse extraction JSONs → flat Pydantic records
├── dedup.py             # Entity resolution + predicate normalization + assertion merging
├── graph.py             # NetworkX MultiDiGraph construction + serialization
├── confidence.py        # Beta-Binomial edge scoring with independence weighting
├── analysis.py          # Community detection, contradiction finder, gap analysis
└── viz.py               # Graph visualization helpers (matplotlib + GraphML export)

tests/test_knowledge_graph/
├── conftest.py          # Shared fixtures: sample entities, edges, evidence, extraction JSONs
├── test_models.py       # Model creation, validation, serialization
├── test_ingest.py       # JSON parsing, edge cases (empty predicates, null direction)
├── test_dedup.py        # Entity dedup (3 passes), predicate normalization, assertion merging
├── test_graph.py        # Graph construction, serialization round-trip
├── test_confidence.py   # Beta posterior updates, independence weighting, derived metrics
├── test_analysis.py     # Community detection, contradictions, gaps
└── test_viz.py          # GraphML export, figure generation
```

**Dependencies to add to `pyproject.toml`:**
- `"networkx>=3.0"` (graph construction and algorithms)

**Already in `pyproject.toml`:**
- `rapidfuzz>=3.0`, `scipy>=1.17.1`, `pydantic>=2.10`, `structlog>=24.0`, `matplotlib`

---

## Input Data Format

The KG ingests extraction JSONs from `Paper Extractor/KnowledgeGraph Extraction/gastruloid_run/extractions/`. Each JSON has this structure (NOT the AutoReview `PaperExtraction` model — these use the mycelium `ExtractionResult` schema):

```
Top-level keys: paper_provenance, evidence_units, assertion_drafts, citation_contexts, extraction_metadata

assertion_drafts[]: draft_id, natural_language, canonical_form, subject_entity, object_entity,
    predicate, direction, assertion_type, scope, hedging, evidence_unit_ids, ...
    subject_entity: surface_form, canonical_name, ontology_id, ontology_source, entity_type, aliases
    object_entity: (same structure)

evidence_units[]: evidence_id, assertion_draft_ids, evidence_direction, evidence_strength,
    experiment{description, model_system, organism, ...}, results{effect_size, sample_size, key_figure, ...},
    methodological_tags{...}, source_section, source_text_span

paper_provenance: doi, pmid, title, authors[{name, orcid, affiliations, role}],
    journal, publication_date, peer_reviewed, funding_sources, conflicts_of_interest

citation_contexts[]: citation_id, citing_sentence, cited_source_doi, cited_source_pmid,
    cited_claim_paraphrase, relationship, linked_assertion_draft_ids, section
```

---

## Task 1: Add networkx dependency + scaffold package

**Files:**
- Modify: `pyproject.toml` (add networkx)
- Create: `autoreview/knowledge_graph/__init__.py`
- Create: `tests/test_knowledge_graph/__init__.py`
- Create: `tests/test_knowledge_graph/conftest.py`

- [ ] **Step 1: Add networkx to pyproject.toml**

In `pyproject.toml` dependencies list, add:
```
"networkx>=3.0",
```

- [ ] **Step 2: Install dependencies**

Run: `conda run -n autoreview pip install networkx rapidfuzz`

- [ ] **Step 3: Create package scaffold**

Create `autoreview/knowledge_graph/__init__.py`:
```python
"""Knowledge graph construction and analysis from extraction data."""

from __future__ import annotations
```

Create `tests/test_knowledge_graph/__init__.py` (empty).

- [ ] **Step 4: Create test conftest with shared fixtures**

Create `tests/test_knowledge_graph/conftest.py` with fixtures for sample entities, edges, evidence, and a minimal extraction JSON dict that matches the real schema. These fixtures will be used across all test files.

The conftest must include:
- `sample_paper_provenance()` — dict matching provenance schema with 2 authors
- `sample_assertion_draft()` — dict with subject_entity, object_entity, predicate, direction, evidence_unit_ids
- `sample_evidence_unit()` — dict with evidence_id, evidence_direction, evidence_strength, experiment, results
- `sample_citation_context()` — dict matching citation schema
- `sample_extraction_json()` — complete extraction dict combining the above
- `sample_extraction_dir(tmp_path)` — writes 3 extraction JSONs to a temp dir, returns path
- `sample_kg_entity()` — returns a `KGEntity` instance
- `sample_kg_edge()` — returns a `KGEdge` instance
- `sample_entity_registry()` — returns a dict of entity_id → KGEntity with 5 entities

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def sample_paper_provenance() -> dict:
    return {
        "doi": "10.1038/s41586-020-0001",
        "pmid": "12345678",
        "title": "Wnt signaling in gastruloid development",
        "authors": [
            {"name": "Smith, Alice", "orcid": None, "affiliations": ["Lab A, MIT"], "role": "first_author"},
            {"name": "Jones, Bob", "orcid": None, "affiliations": ["Lab A, MIT"], "role": "last_author"},
        ],
        "journal": "Nature",
        "publication_date": "2023-01-15",
        "peer_reviewed": True,
        "preprint_doi": None,
        "funding_sources": ["NIH R01"],
        "conflicts_of_interest": [],
        "data_availability": "GEO: GSE12345",
    }


@pytest.fixture
def sample_assertion_draft() -> dict:
    return {
        "draft_id": "a_001",
        "natural_language": "Wnt signaling is required for mesoderm formation in gastruloids",
        "canonical_form": "Wnt signaling -> is_required_for -> mesoderm formation",
        "negatable_form": "Wnt signaling is not required for mesoderm formation",
        "subject_entity": {
            "surface_form": "Wnt signaling",
            "canonical_name": "Wnt signaling pathway",
            "ontology_id": "GO:0016055",
            "ontology_source": "GO",
            "entity_type": "pathway",
            "aliases": ["Wnt pathway", "canonical Wnt"],
        },
        "object_entity": {
            "surface_form": "mesoderm formation",
            "canonical_name": "mesoderm formation",
            "ontology_id": "GO:0001707",
            "ontology_source": "GO",
            "entity_type": "biological_process",
            "aliases": ["mesoderm development"],
        },
        "predicate": "is_required_for",
        "direction": "positive",
        "assertion_type": "mechanistic_causal",
        "causal_type": "necessary",
        "scope": "gastruloid_specific",
        "conditions": [],
        "hedging": None,
        "epistemic_status": "established",
        "evidence_unit_ids": ["e_001"],
        "parent_assertion_ids": [],
        "provenance": {"extraction_method": "llm", "confidence": 0.9},
    }


@pytest.fixture
def sample_evidence_unit() -> dict:
    return {
        "evidence_id": "e_001",
        "assertion_draft_ids": ["a_001"],
        "evidence_direction": "supports",
        "evidence_strength": "direct_experimental",
        "experiment": {
            "description": "CHIR99021 (Wnt agonist) treatment of gastruloids",
            "model_system": "Mouse ESC-derived gastruloids",
            "organism": "Mus musculus",
            "perturbation_type": "chemical",
            "readout": "T/Brachyury immunofluorescence",
            "control_description": "Untreated gastruloids",
        },
        "results": {
            "result_direction": "positive",
            "effect_description": "CHIR treatment induced robust mesoderm marker expression",
            "effect_size": "3.5-fold increase",
            "sample_size": "n=50 gastruloids per condition",
            "key_figure": "Fig. 2A",
        },
        "methodological_tags": {
            "approach_category": "in_vitro_model",
            "assay_types": ["immunofluorescence"],
            "blinding_reported": None,
            "randomization_reported": None,
        },
        "limitations_stated_by_authors": [],
        "source_section": "results",
        "source_text_span": "Treatment with CHIR99021...",
    }


@pytest.fixture
def sample_citation_context() -> dict:
    return {
        "citation_id": "c_001",
        "citing_sentence": "Previous work demonstrated that Wnt signaling is essential for mesoderm specification (Smith et al., 2020).",
        "cited_source_doi": "10.1016/j.cell.2020.0001",
        "cited_source_pmid": "87654321",
        "cited_claim_paraphrase": "Wnt is essential for mesoderm",
        "relationship": "supports",
        "linked_assertion_draft_ids": ["a_001"],
        "section": "introduction",
    }


@pytest.fixture
def sample_extraction_json(
    sample_paper_provenance: dict,
    sample_assertion_draft: dict,
    sample_evidence_unit: dict,
    sample_citation_context: dict,
) -> dict:
    return {
        "paper_provenance": sample_paper_provenance,
        "assertion_drafts": [sample_assertion_draft],
        "evidence_units": [sample_evidence_unit],
        "citation_contexts": [sample_citation_context],
        "extraction_metadata": {"model": "haiku", "timestamp": "2025-01-01T00:00:00Z"},
    }


@pytest.fixture
def sample_extraction_dir(
    tmp_path: Path,
    sample_extraction_json: dict,
) -> Path:
    """Write 3 extraction JSONs to a temp dir with slight variations."""
    for i, paper_hash in enumerate(["aaa111", "bbb222", "ccc333"]):
        data = json.loads(json.dumps(sample_extraction_json))  # deep copy
        data["paper_provenance"]["doi"] = f"10.1038/paper-{i}"
        data["paper_provenance"]["title"] = f"Paper {i} on gastruloids"
        data["assertion_drafts"][0]["draft_id"] = f"a_{i:03d}"
        data["evidence_units"][0]["evidence_id"] = f"e_{i:03d}"
        data["evidence_units"][0]["assertion_draft_ids"] = [f"a_{i:03d}"]
        data["assertion_drafts"][0]["evidence_unit_ids"] = [f"e_{i:03d}"]

        # Paper 2: same entities, different predicate synonym -> should merge after normalization
        if i == 1:
            data["assertion_drafts"][0]["predicate"] = "is_necessary_for"

        # Paper 3: different entities entirely
        if i == 2:
            data["assertion_drafts"][0]["subject_entity"]["canonical_name"] = "BMP signaling pathway"
            data["assertion_drafts"][0]["subject_entity"]["ontology_id"] = "GO:0030509"
            data["assertion_drafts"][0]["object_entity"]["canonical_name"] = "dorsal-ventral axis specification"
            data["assertion_drafts"][0]["object_entity"]["ontology_id"] = "GO:0009953"
            data["assertion_drafts"][0]["predicate"] = "induces"

        (tmp_path / f"{paper_hash}.json").write_text(json.dumps(data, indent=2))

    # Also write a non-JSON file that should be skipped
    (tmp_path / "debug_raw.txt").write_text("raw extraction text")

    return tmp_path
```

- [ ] **Step 5: Verify scaffold**

Run: `conda run -n autoreview python -c "import autoreview.knowledge_graph"`
Expected: no error

- [ ] **Step 6: Commit**

```bash
git add autoreview/knowledge_graph/__init__.py tests/test_knowledge_graph/ pyproject.toml
git commit -m "feat(kg): scaffold knowledge_graph package with test fixtures"
```

---

## Task 2: Pydantic models (`models.py`)

**Files:**
- Create: `autoreview/knowledge_graph/models.py`
- Create: `tests/test_knowledge_graph/test_models.py`

**Reference:** Spec §"Pydantic Models" — `EntityType`, `AssertionType`, `EvidenceStrength`, `KGEntity`, `KGEdge`, `BetaPosterior`, `KGEvidenceLink`, `KGCitation`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_knowledge_graph/test_models.py`:
```python
"""Tests for knowledge graph Pydantic models."""

from __future__ import annotations

import pytest


class TestEntityType:
    def test_all_expected_values(self):
        from autoreview.knowledge_graph.models import EntityType

        expected = {"gene", "protein", "pathway", "biological_process", "cell_type",
                    "chemical", "organism", "disease", "method", "other"}
        assert set(EntityType) == expected


class TestEvidenceStrength:
    def test_all_expected_values(self):
        from autoreview.knowledge_graph.models import EvidenceStrength

        expected = {"direct_experimental", "observational_controlled",
                    "observational_uncontrolled", "computational_prediction", "expert_opinion"}
        assert set(EvidenceStrength) == expected


class TestBetaPosterior:
    def test_default_uniform_prior(self):
        from autoreview.knowledge_graph.models import BetaPosterior

        bp = BetaPosterior()
        assert bp.alpha == 1.0
        assert bp.beta_param == 1.0
        assert bp.mean == pytest.approx(0.5)

    def test_mean_computation(self):
        from autoreview.knowledge_graph.models import BetaPosterior

        bp = BetaPosterior(alpha=3.0, beta_param=1.0)
        assert bp.mean == pytest.approx(0.75)

    def test_ci_95_in_serialization(self):
        from autoreview.knowledge_graph.models import BetaPosterior

        bp = BetaPosterior(alpha=10.0, beta_param=2.0)
        data = bp.model_dump()
        assert "mean" in data
        assert "ci_95" in data
        assert len(data["ci_95"]) == 2
        assert data["ci_95"][0] < data["ci_95"][1]

    def test_json_round_trip(self):
        from autoreview.knowledge_graph.models import BetaPosterior

        bp = BetaPosterior(alpha=5.0, beta_param=3.0)
        json_str = bp.model_dump_json()
        bp2 = BetaPosterior.model_validate_json(json_str)
        assert bp2.alpha == bp.alpha
        assert bp2.beta_param == bp.beta_param


class TestKGEntity:
    def test_creation(self):
        from autoreview.knowledge_graph.models import KGEntity

        entity = KGEntity(
            entity_id="abc123",
            canonical_name="Wnt signaling pathway",
            entity_type="pathway",
            ontology_id="GO:0016055",
            ontology_source="GO",
            aliases=["Wnt pathway", "canonical Wnt"],
            paper_count=5,
            source_paper_ids=["p1", "p2"],
        )
        assert entity.entity_id == "abc123"
        assert entity.entity_type == "pathway"

    def test_extra_fields_ignored(self):
        from autoreview.knowledge_graph.models import KGEntity

        entity = KGEntity(
            entity_id="abc123",
            canonical_name="test",
            entity_type="gene",
            ontology_id=None,
            ontology_source=None,
            aliases=[],
            paper_count=1,
            source_paper_ids=["p1"],
            unknown_field="should be ignored",
        )
        assert not hasattr(entity, "unknown_field")


class TestKGEdge:
    def test_creation_with_evidence(self):
        from autoreview.knowledge_graph.models import BetaPosterior, KGEdge, KGEvidenceLink

        evidence = KGEvidenceLink(
            evidence_id="e1",
            paper_id="p1",
            evidence_strength="direct_experimental",
            evidence_direction="supports",
            experiment_summary="CHIR treatment of gastruloids",
            model_system="mouse ESC gastruloids",
            sample_size="n=50",
            key_figure="Fig. 2A",
            publication_date="2023-01-15",
        )
        edge = KGEdge(
            edge_id="edge1",
            subject_id="ent1",
            object_id="ent2",
            predicate="is_required_for",
            direction="positive",
            assertion_type="mechanistic_causal",
            confidence=BetaPosterior(alpha=2.0, beta_param=1.0),
            evidence_links=[evidence],
            source_assertions=["a_001"],
            publication_date="2023-01-15",
        )
        assert edge.predicate == "is_required_for"
        assert len(edge.evidence_links) == 1
        assert edge.confidence.mean == pytest.approx(2 / 3)

    def test_null_direction_allowed(self):
        from autoreview.knowledge_graph.models import BetaPosterior, KGEdge

        edge = KGEdge(
            edge_id="edge1",
            subject_id="ent1",
            object_id="ent2",
            predicate="related_to",
            direction=None,
            assertion_type="existence",
            confidence=BetaPosterior(),
            evidence_links=[],
            source_assertions=[],
            publication_date=None,
        )
        assert edge.direction is None


class TestKGCitation:
    def test_creation(self):
        from autoreview.knowledge_graph.models import KGCitation

        cit = KGCitation(
            citation_id="c1",
            citing_paper_id="p1",
            cited_source_doi="10.1016/j.cell.2020.0001",
            cited_source_pmid=None,
            citing_sentence="Previous work showed...",
            cited_claim_paraphrase="Wnt is essential",
            relationship="supports",
            linked_assertion_ids=["a_001"],
            section="introduction",
        )
        assert cit.relationship == "supports"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n autoreview python -m pytest tests/test_knowledge_graph/test_models.py -v`
Expected: ImportError — `autoreview.knowledge_graph.models` does not exist

- [ ] **Step 3: Implement models.py**

Create `autoreview/knowledge_graph/models.py`:
```python
"""Pydantic models for the knowledge graph.

Three-tier architecture:
- Tier 1 (KGEdge): Assertions — mechanism-level claims, the unit of scientific discourse
- Tier 2 (KGEvidenceLink): Evidence — experimental demonstrations grounding assertions
- Tier 3 (paper_provenance via paper_id): Provenance — source credibility and independence
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import computed_field

from autoreview.models.base import AutoReviewModel


class EntityType(StrEnum):
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


class AssertionType(StrEnum):
    mechanistic_causal = "mechanistic_causal"
    existence = "existence"
    comparative = "comparative"
    methodological = "methodological"
    correlational = "correlational"
    absence = "absence"
    conditional = "conditional"


class EvidenceStrength(StrEnum):
    direct_experimental = "direct_experimental"
    observational_controlled = "observational_controlled"
    observational_uncontrolled = "observational_uncontrolled"
    computational_prediction = "computational_prediction"
    expert_opinion = "expert_opinion"


class BetaPosterior(AutoReviewModel):
    """Beta-Binomial confidence score for an edge."""

    alpha: float = 1.0
    beta_param: float = 1.0

    @computed_field
    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta_param)

    @computed_field
    @property
    def ci_95(self) -> tuple[float, float]:
        from scipy.stats import beta as beta_dist

        return beta_dist.interval(0.95, self.alpha, self.beta_param)


class KGEntity(AutoReviewModel):
    """A deduplicated node in the knowledge graph."""

    entity_id: str
    canonical_name: str
    entity_type: EntityType
    ontology_id: str | None
    ontology_source: str | None
    aliases: list[str]
    paper_count: int
    source_paper_ids: list[str]


class KGEvidenceLink(AutoReviewModel):
    """Links an edge to its experimental grounding (Tier 2)."""

    evidence_id: str
    paper_id: str
    evidence_strength: EvidenceStrength
    evidence_direction: str
    experiment_summary: str
    model_system: str | None
    sample_size: str | None
    key_figure: str | None
    publication_date: str | None


class KGEdge(AutoReviewModel):
    """A typed assertion between two entities, grounded in evidence (Tier 1)."""

    edge_id: str
    subject_id: str
    object_id: str
    predicate: str
    direction: str | None
    assertion_type: AssertionType
    confidence: BetaPosterior
    evidence_links: list[KGEvidenceLink]
    source_assertions: list[str]
    publication_date: str | None


class KGCitation(AutoReviewModel):
    """A citation context linking papers through claim references."""

    citation_id: str
    citing_paper_id: str
    cited_source_doi: str | None
    cited_source_pmid: str | None
    citing_sentence: str
    cited_claim_paraphrase: str | None
    relationship: str
    linked_assertion_ids: list[str]
    section: str | None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n autoreview python -m pytest tests/test_knowledge_graph/test_models.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/knowledge_graph/models.py tests/test_knowledge_graph/test_models.py
git commit -m "feat(kg): add Pydantic models for KG entities, edges, evidence, citations"
```

---

## Task 3: Ingestion (`ingest.py`)

**Files:**
- Create: `autoreview/knowledge_graph/ingest.py`
- Create: `tests/test_knowledge_graph/test_ingest.py`

**Reference:** Spec §"Data Flow" — parse extraction JSONs into flat Pydantic records

The ingest module reads extraction JSON files and produces raw (pre-dedup) entity dicts, assertion dicts, evidence dicts, and citation records. It does NOT do dedup or graph construction — just parsing and normalization of the raw extraction format into KG-ready records.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_knowledge_graph/test_ingest.py`:
```python
"""Tests for extraction JSON ingestion."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestIngestSingleExtraction:
    def test_parses_entities(self, sample_extraction_json: dict):
        from autoreview.knowledge_graph.ingest import ingest_extraction

        result = ingest_extraction(sample_extraction_json, paper_hash="aaa111")
        # Should extract subject + object = 2 entities
        assert len(result.entities) == 2
        assert result.entities[0]["canonical_name"] == "Wnt signaling pathway"
        assert result.entities[0]["entity_type"] == "pathway"
        assert result.entities[0]["ontology_id"] == "GO:0016055"

    def test_parses_assertions(self, sample_extraction_json: dict):
        from autoreview.knowledge_graph.ingest import ingest_extraction

        result = ingest_extraction(sample_extraction_json, paper_hash="aaa111")
        assert len(result.assertions) == 1
        assert result.assertions[0]["predicate"] == "is_required_for"
        assert result.assertions[0]["direction"] == "positive"

    def test_parses_evidence(self, sample_extraction_json: dict):
        from autoreview.knowledge_graph.ingest import ingest_extraction

        result = ingest_extraction(sample_extraction_json, paper_hash="aaa111")
        assert len(result.evidence_units) == 1
        assert result.evidence_units[0]["evidence_strength"] == "direct_experimental"
        assert result.evidence_units[0]["evidence_direction"] == "supports"

    def test_parses_citations(self, sample_extraction_json: dict):
        from autoreview.knowledge_graph.ingest import ingest_extraction

        result = ingest_extraction(sample_extraction_json, paper_hash="aaa111")
        assert len(result.citations) == 1

    def test_coerces_null_string_direction(self, sample_extraction_json: dict):
        from autoreview.knowledge_graph.ingest import ingest_extraction

        sample_extraction_json["assertion_drafts"][0]["direction"] = "null"
        result = ingest_extraction(sample_extraction_json, paper_hash="aaa111")
        assert result.assertions[0]["direction"] is None

    def test_assigns_related_to_for_empty_predicate(self, sample_extraction_json: dict):
        from autoreview.knowledge_graph.ingest import ingest_extraction

        sample_extraction_json["assertion_drafts"][0]["predicate"] = ""
        result = ingest_extraction(sample_extraction_json, paper_hash="aaa111")
        assert result.assertions[0]["predicate"] == "related_to"

    def test_normalizes_ontology_source(self, sample_extraction_json: dict):
        from autoreview.knowledge_graph.ingest import ingest_extraction

        sample_extraction_json["assertion_drafts"][0]["subject_entity"]["ontology_source"] = "go; UniProt"
        result = ingest_extraction(sample_extraction_json, paper_hash="aaa111")
        # Should normalize to uppercase, split on separators
        ent = result.entities[0]
        assert ent["ontology_source"] == "GO"  # First source after normalization

    def test_unknown_evidence_strength_maps_to_expert_opinion(self, sample_extraction_json: dict):
        from autoreview.knowledge_graph.ingest import ingest_extraction

        sample_extraction_json["evidence_units"][0]["evidence_strength"] = "anecdotal"
        result = ingest_extraction(sample_extraction_json, paper_hash="aaa111")
        assert result.evidence_units[0]["evidence_strength"] == "expert_opinion"


class TestIngestDirectory:
    def test_ingests_all_jsons(self, sample_extraction_dir: Path):
        from autoreview.knowledge_graph.ingest import ingest_directory

        result = ingest_directory(sample_extraction_dir)
        assert result.paper_count == 3
        assert len(result.all_entities) >= 4  # 2 per paper, some shared
        assert len(result.all_assertions) == 3
        assert len(result.all_evidence_units) == 3

    def test_skips_non_json_files(self, sample_extraction_dir: Path):
        from autoreview.knowledge_graph.ingest import ingest_directory

        result = ingest_directory(sample_extraction_dir)
        # debug_raw.txt should be skipped
        assert result.paper_count == 3

    def test_handles_malformed_json(self, sample_extraction_dir: Path):
        from autoreview.knowledge_graph.ingest import ingest_directory

        # Write a malformed JSON
        (sample_extraction_dir / "bad.json").write_text("{invalid json")
        result = ingest_directory(sample_extraction_dir)
        assert result.paper_count == 3  # Still processes the valid ones
        assert len(result.parse_errors) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n autoreview python -m pytest tests/test_knowledge_graph/test_ingest.py -v`
Expected: ImportError

- [ ] **Step 3: Implement ingest.py**

Create `autoreview/knowledge_graph/ingest.py`. Key responsibilities:
- `ingest_extraction(data: dict, paper_hash: str) -> ExtractionRecord` — parse one JSON
- `ingest_directory(extraction_dir: Path) -> CorpusIngestion` — parse all JSONs in a dir
- Normalize: coerce `"null"` direction → `None`, empty predicate → `"related_to"`, unknown evidence_strength → `"expert_opinion"`, ontology source uppercase/split
- Return simple dataclass/NamedTuple results (not Pydantic KG models yet — those are created during dedup/graph)

The `ExtractionRecord` contains flat lists of dicts for entities, assertions, evidence, citations, plus paper provenance. The `CorpusIngestion` accumulates all records across papers.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n autoreview python -m pytest tests/test_knowledge_graph/test_ingest.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/knowledge_graph/ingest.py tests/test_knowledge_graph/test_ingest.py
git commit -m "feat(kg): add extraction JSON ingestion with normalization"
```

---

## Task 4: Entity deduplication (`dedup.py` — entity resolution)

**Files:**
- Create: `autoreview/knowledge_graph/dedup.py`
- Create: `tests/test_knowledge_graph/test_dedup.py`

**Reference:** Spec §"Entity Deduplication" — 3-pass strategy (ontology match, canonical name normalization, fuzzy matching)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_knowledge_graph/test_dedup.py`:
```python
"""Tests for entity deduplication, predicate normalization, and assertion merging."""

from __future__ import annotations

import pytest


class TestEntityDedup:
    """Test the three-pass entity resolution strategy."""

    def test_pass1_ontology_id_merge(self):
        """Entities sharing ontology_id merge regardless of name."""
        from autoreview.knowledge_graph.dedup import deduplicate_entities

        entities = [
            {"canonical_name": "Wnt signaling pathway", "entity_type": "pathway",
             "ontology_id": "GO:0016055", "ontology_source": "GO",
             "aliases": ["Wnt pathway"], "paper_ids": ["p1"]},
            {"canonical_name": "Wnt signalling", "entity_type": "pathway",
             "ontology_id": "GO:0016055", "ontology_source": "GO",
             "aliases": ["canonical Wnt"], "paper_ids": ["p2"]},
        ]
        registry = deduplicate_entities(entities)
        assert len(registry.entities) == 1
        ent = list(registry.entities.values())[0]
        assert ent.paper_count == 2
        assert "Wnt pathway" in ent.aliases
        assert "canonical Wnt" in ent.aliases

    def test_pass2_canonical_name_normalization(self):
        """Entities with same normalized name merge."""
        from autoreview.knowledge_graph.dedup import deduplicate_entities

        entities = [
            {"canonical_name": "Anterior-Posterior Axis", "entity_type": "biological_process",
             "ontology_id": None, "ontology_source": None,
             "aliases": [], "paper_ids": ["p1"]},
            {"canonical_name": "anterior-posterior axis", "entity_type": "biological_process",
             "ontology_id": None, "ontology_source": None,
             "aliases": [], "paper_ids": ["p2"]},
        ]
        registry = deduplicate_entities(entities)
        assert len(registry.entities) == 1

    def test_pass3_fuzzy_match_within_type(self):
        """Similar names within same entity type merge above threshold."""
        from autoreview.knowledge_graph.dedup import deduplicate_entities

        entities = [
            {"canonical_name": "mesoderm specification", "entity_type": "biological_process",
             "ontology_id": None, "ontology_source": None,
             "aliases": [], "paper_ids": ["p1"]},
            {"canonical_name": "mesoderm specifications", "entity_type": "biological_process",
             "ontology_id": None, "ontology_source": None,
             "aliases": [], "paper_ids": ["p2"]},
        ]
        registry = deduplicate_entities(entities)
        assert len(registry.entities) == 1

    def test_no_cross_type_fuzzy_merge(self):
        """Similar names across different entity types do NOT merge."""
        from autoreview.knowledge_graph.dedup import deduplicate_entities

        entities = [
            {"canonical_name": "BMP4", "entity_type": "gene",
             "ontology_id": None, "ontology_source": None,
             "aliases": [], "paper_ids": ["p1"]},
            {"canonical_name": "BMP4", "entity_type": "protein",
             "ontology_id": None, "ontology_source": None,
             "aliases": [], "paper_ids": ["p2"]},
        ]
        registry = deduplicate_entities(entities)
        assert len(registry.entities) == 2

    def test_ontology_id_entity_wins_canonical(self):
        """Entity with ontology ID becomes canonical in merge."""
        from autoreview.knowledge_graph.dedup import deduplicate_entities

        entities = [
            {"canonical_name": "wnt signaling", "entity_type": "pathway",
             "ontology_id": None, "ontology_source": None,
             "aliases": [], "paper_ids": ["p1"]},
            {"canonical_name": "Wnt signaling pathway", "entity_type": "pathway",
             "ontology_id": "GO:0016055", "ontology_source": "GO",
             "aliases": [], "paper_ids": ["p2"]},
        ]
        registry = deduplicate_entities(entities)
        ent = list(registry.entities.values())[0]
        assert ent.ontology_id == "GO:0016055"
        assert ent.canonical_name == "Wnt signaling pathway"

    def test_surface_form_reverse_index(self):
        """Registry provides surface_form → entity_id lookup."""
        from autoreview.knowledge_graph.dedup import deduplicate_entities

        entities = [
            {"canonical_name": "Wnt signaling pathway", "entity_type": "pathway",
             "ontology_id": "GO:0016055", "ontology_source": "GO",
             "aliases": ["Wnt pathway"], "paper_ids": ["p1"]},
        ]
        registry = deduplicate_entities(entities)
        assert "wnt signaling pathway" in registry.surface_to_id
        assert "wnt pathway" in registry.surface_to_id

    def test_merge_log_records_decisions(self):
        """All merge decisions are logged for auditability."""
        from autoreview.knowledge_graph.dedup import deduplicate_entities

        entities = [
            {"canonical_name": "Wnt signaling pathway", "entity_type": "pathway",
             "ontology_id": "GO:0016055", "ontology_source": "GO",
             "aliases": [], "paper_ids": ["p1"]},
            {"canonical_name": "Wnt signalling", "entity_type": "pathway",
             "ontology_id": "GO:0016055", "ontology_source": "GO",
             "aliases": [], "paper_ids": ["p2"]},
        ]
        registry = deduplicate_entities(entities)
        assert len(registry.merge_log) >= 1
        assert registry.merge_log[0]["pass"] == "ontology_id"

    def test_other_type_uses_token_blocking(self):
        """'other' entity type uses token-based blocking to avoid O(n^2) comparisons."""
        from autoreview.knowledge_graph.dedup import deduplicate_entities

        entities = [
            {"canonical_name": "gastruloid elongation morphogenesis", "entity_type": "other",
             "ontology_id": None, "ontology_source": None,
             "aliases": [], "paper_ids": ["p1"]},
            {"canonical_name": "gastruloid elongation process", "entity_type": "other",
             "ontology_id": None, "ontology_source": None,
             "aliases": [], "paper_ids": ["p2"]},
            {"canonical_name": "culture medium composition", "entity_type": "other",
             "ontology_id": None, "ontology_source": None,
             "aliases": [], "paper_ids": ["p3"]},
        ]
        registry = deduplicate_entities(entities)
        # First two share "gastruloid" + "elongation" tokens and are similar → merge
        # Third shares no tokens with the others → stays separate
        assert len(registry.entities) == 2

    def test_ambiguous_zone_flagged_for_review(self):
        """Entities in the 0.75-0.90 ambiguous zone are flagged, not auto-merged."""
        from autoreview.knowledge_graph.dedup import deduplicate_entities

        entities = [
            {"canonical_name": "anterior patterning signal", "entity_type": "biological_process",
             "ontology_id": None, "ontology_source": None,
             "aliases": [], "paper_ids": ["p1"]},
            {"canonical_name": "anterior patterning process", "entity_type": "biological_process",
             "ontology_id": None, "ontology_source": None,
             "aliases": [], "paper_ids": ["p2"]},
        ]
        registry = deduplicate_entities(entities)
        # These have high Jaccard (2/4 shared tokens = 0.5 < 0.75) so should NOT merge,
        # but if Levenshtein is in ambiguous zone, should be flagged
        assert any(entry.get("flagged_ambiguous") for entry in registry.merge_log) or len(registry.entities) == 2


class TestPredicateNormalization:
    """Test predicate synonym resolution."""

    def test_exact_synonym_match(self):
        from autoreview.knowledge_graph.dedup import normalize_predicate

        assert normalize_predicate("is_necessary_for") == "is_required_for"
        assert normalize_predicate("activates") == "induces"
        assert normalize_predicate("suppresses") == "inhibits"
        assert normalize_predicate("binds_to") == "interacts_with"

    def test_canonical_predicate_unchanged(self):
        from autoreview.knowledge_graph.dedup import normalize_predicate

        assert normalize_predicate("is_required_for") == "is_required_for"
        assert normalize_predicate("induces") == "induces"

    def test_unknown_predicate_kept_as_is(self):
        from autoreview.knowledge_graph.dedup import normalize_predicate

        assert normalize_predicate("totally_novel_predicate") == "totally_novel_predicate"

    def test_related_to_default_unchanged(self):
        from autoreview.knowledge_graph.dedup import normalize_predicate

        assert normalize_predicate("related_to") == "related_to"

    def test_normalization_log_records_mappings(self):
        """All predicate normalizations are logged for auditability."""
        from autoreview.knowledge_graph.dedup import PredicateNormalizer

        normalizer = PredicateNormalizer()
        normalizer.normalize("is_necessary_for")
        normalizer.normalize("activates")
        normalizer.normalize("totally_novel_predicate")
        assert len(normalizer.log) == 3
        assert normalizer.log[0]["original"] == "is_necessary_for"
        assert normalizer.log[0]["canonical"] == "is_required_for"
        assert normalizer.log[2]["original"] == "totally_novel_predicate"
        assert normalizer.log[2]["canonical"] == "totally_novel_predicate"  # unchanged


class TestAssertionMerging:
    """Test merging of assertions with same (subject, predicate, object) after dedup."""

    def test_same_triple_merges(self):
        from autoreview.knowledge_graph.dedup import merge_assertions

        assertions = [
            {"subject_id": "ent1", "object_id": "ent2", "predicate": "is_required_for",
             "direction": "positive", "assertion_type": "mechanistic_causal",
             "draft_id": "a_001", "evidence_unit_ids": ["e_001"],
             "paper_id": "p1", "publication_date": "2023-01-15"},
            {"subject_id": "ent1", "object_id": "ent2", "predicate": "is_required_for",
             "direction": "positive", "assertion_type": "mechanistic_causal",
             "draft_id": "a_002", "evidence_unit_ids": ["e_002"],
             "paper_id": "p2", "publication_date": "2022-06-01"},
        ]
        merged = merge_assertions(assertions)
        assert len(merged) == 1
        assert len(merged[0]["source_assertions"]) == 2
        assert merged[0]["publication_date"] == "2022-06-01"  # earliest

    def test_different_triples_stay_separate(self):
        from autoreview.knowledge_graph.dedup import merge_assertions

        assertions = [
            {"subject_id": "ent1", "object_id": "ent2", "predicate": "is_required_for",
             "direction": "positive", "assertion_type": "mechanistic_causal",
             "draft_id": "a_001", "evidence_unit_ids": ["e_001"],
             "paper_id": "p1", "publication_date": "2023-01-15"},
            {"subject_id": "ent3", "object_id": "ent4", "predicate": "induces",
             "direction": "positive", "assertion_type": "mechanistic_causal",
             "draft_id": "a_002", "evidence_unit_ids": ["e_002"],
             "paper_id": "p2", "publication_date": "2022-06-01"},
        ]
        merged = merge_assertions(assertions)
        assert len(merged) == 2

    def test_direction_conflict_sets_none(self):
        from autoreview.knowledge_graph.dedup import merge_assertions

        assertions = [
            {"subject_id": "ent1", "object_id": "ent2", "predicate": "induces",
             "direction": "positive", "assertion_type": "mechanistic_causal",
             "draft_id": "a_001", "evidence_unit_ids": ["e_001"],
             "paper_id": "p1", "publication_date": "2023-01-15"},
            {"subject_id": "ent1", "object_id": "ent2", "predicate": "induces",
             "direction": "negative", "assertion_type": "mechanistic_causal",
             "draft_id": "a_002", "evidence_unit_ids": ["e_002"],
             "paper_id": "p2", "publication_date": "2022-06-01"},
        ]
        merged = merge_assertions(assertions)
        assert len(merged) == 1
        assert merged[0]["direction"] is None  # conflict → None
        assert merged[0]["direction_conflict"] is True

    def test_merge_log_records_decisions(self):
        """All assertion merge decisions are logged for auditability."""
        from autoreview.knowledge_graph.dedup import merge_assertions

        assertions = [
            {"subject_id": "ent1", "object_id": "ent2", "predicate": "is_required_for",
             "direction": "positive", "assertion_type": "mechanistic_causal",
             "draft_id": "a_001", "evidence_unit_ids": ["e_001"],
             "paper_id": "p1", "publication_date": "2023-01-15"},
            {"subject_id": "ent1", "object_id": "ent2", "predicate": "is_required_for",
             "direction": "positive", "assertion_type": "mechanistic_causal",
             "draft_id": "a_002", "evidence_unit_ids": ["e_002"],
             "paper_id": "p2", "publication_date": "2022-06-01"},
        ]
        merged = merge_assertions(assertions)
        # merge_assertions returns a MergeResult with merged list + log
        assert len(merged.merge_log) >= 1
        assert merged.merge_log[0]["merged_draft_ids"] == ["a_001", "a_002"]
        assert merged.merge_log[0]["papers"] == ["p1", "p2"]

    def test_self_loop_allowed(self):
        """Assertions where subject == object are valid (e.g., autoregulation)."""
        from autoreview.knowledge_graph.dedup import merge_assertions

        assertions = [
            {"subject_id": "ent1", "object_id": "ent1", "predicate": "regulates",
             "direction": "positive", "assertion_type": "mechanistic_causal",
             "draft_id": "a_001", "evidence_unit_ids": ["e_001"],
             "paper_id": "p1", "publication_date": "2023-01-15"},
        ]
        merged = merge_assertions(assertions)
        assert len(merged.assertions) == 1
        assert merged.assertions[0]["subject_id"] == merged.assertions[0]["object_id"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n autoreview python -m pytest tests/test_knowledge_graph/test_dedup.py -v`
Expected: ImportError

- [ ] **Step 3: Implement dedup.py**

Create `autoreview/knowledge_graph/dedup.py`. Key classes and functions:

**`EntityRegistry`** (dataclass):
- `entities: dict[str, KGEntity]` — entity_id → KGEntity
- `surface_to_id: dict[str, str]` — lowercased surface form → entity_id
- `merge_log: list[dict]` — audit trail with `pass`, `merged_entities`, `reason` fields

**`PredicateNormalizer`** (class):
- `SYNONYM_TABLE: dict[str, str]` — maps synonyms to canonical predicates (from spec table)
- `normalize(predicate: str) -> str` — exact lookup, then fuzzy fallback (Levenshtein >= 0.85)
- `log: list[dict]` — records every normalization: `{"original", "canonical", "method", "score"}`

**`deduplicate_entities(entities: list[dict]) -> EntityRegistry`**:
- Pass 1: Group by `ontology_id`, merge groups
- Pass 2: Group by `normalize_name(canonical_name)`, merge groups
- Pass 3: Within each `entity_type`, compute pairwise fuzzy similarity. For `other` type: **block by shared word tokens first** — only compare entities sharing at least one significant word token (>3 chars). This avoids O(n^2) on the ~1,400 `other` entities.
- Merge at Levenshtein >= 0.85 AND Jaccard >= 0.75. Flag 0.70-0.85 zone as ambiguous in merge_log.
- Uses `rapidfuzz.fuzz.ratio` for Levenshtein, set intersection for Jaccard token overlap.

**`merge_assertions(assertions: list[dict]) -> MergeResult`**:
- `MergeResult` (dataclass): `assertions: list[dict]`, `merge_log: list[dict]`
- Group by `(subject_id, predicate, object_id)` tuple
- For each group: accumulate `source_assertions`, `evidence_unit_ids`; resolve `direction` (unanimous → keep, conflict → None + flag); `assertion_type` by majority vote; `publication_date` = earliest
- Self-loops (subject_id == object_id) are allowed
- `merge_log` records: `{"merged_draft_ids", "papers", "original_predicates", "direction_conflict"}`

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n autoreview python -m pytest tests/test_knowledge_graph/test_dedup.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/knowledge_graph/dedup.py tests/test_knowledge_graph/test_dedup.py
git commit -m "feat(kg): add entity dedup, predicate normalization, assertion merging"
```

---

## Task 5: Graph construction (`graph.py`)

**Files:**
- Create: `autoreview/knowledge_graph/graph.py`
- Create: `tests/test_knowledge_graph/test_graph.py`

**Reference:** Spec §"Data Flow" — build NetworkX MultiDiGraph from deduplicated entities and merged assertions

- [ ] **Step 1: Write the failing tests**

Create `tests/test_knowledge_graph/test_graph.py`:
```python
"""Tests for NetworkX graph construction and serialization."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestBuildGraph:
    def test_entities_become_nodes(self):
        from autoreview.knowledge_graph.graph import build_nx_graph
        from autoreview.knowledge_graph.models import BetaPosterior, KGEdge, KGEntity

        entities = {
            "ent1": KGEntity(entity_id="ent1", canonical_name="Wnt", entity_type="pathway",
                             ontology_id="GO:0016055", ontology_source="GO",
                             aliases=[], paper_count=1, source_paper_ids=["p1"]),
            "ent2": KGEntity(entity_id="ent2", canonical_name="mesoderm", entity_type="biological_process",
                             ontology_id=None, ontology_source=None,
                             aliases=[], paper_count=1, source_paper_ids=["p1"]),
        }
        edges = [
            KGEdge(edge_id="e1", subject_id="ent1", object_id="ent2",
                   predicate="is_required_for", direction="positive",
                   assertion_type="mechanistic_causal", confidence=BetaPosterior(),
                   evidence_links=[], source_assertions=["a1"], publication_date=None),
        ]
        G = build_nx_graph(entities, edges)
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 1
        assert G.nodes["ent1"]["canonical_name"] == "Wnt"
        assert G.nodes["ent1"]["entity_type"] == "pathway"

    def test_edge_attributes_stored(self):
        from autoreview.knowledge_graph.graph import build_nx_graph
        from autoreview.knowledge_graph.models import BetaPosterior, KGEdge, KGEntity

        entities = {
            "ent1": KGEntity(entity_id="ent1", canonical_name="A", entity_type="gene",
                             ontology_id=None, ontology_source=None,
                             aliases=[], paper_count=1, source_paper_ids=["p1"]),
            "ent2": KGEntity(entity_id="ent2", canonical_name="B", entity_type="gene",
                             ontology_id=None, ontology_source=None,
                             aliases=[], paper_count=1, source_paper_ids=["p1"]),
        }
        edges = [
            KGEdge(edge_id="e1", subject_id="ent1", object_id="ent2",
                   predicate="induces", direction="positive",
                   assertion_type="mechanistic_causal",
                   confidence=BetaPosterior(alpha=3.0, beta_param=1.0),
                   evidence_links=[], source_assertions=["a1"], publication_date="2023-01-01"),
        ]
        G = build_nx_graph(entities, edges)
        edge_data = G.edges["ent1", "ent2", "e1"]
        assert edge_data["predicate"] == "induces"
        assert edge_data["confidence_mean"] == pytest.approx(0.75)

    def test_self_loop_allowed(self):
        """Self-loops (autoregulation) are biologically valid and must be accepted."""
        from autoreview.knowledge_graph.graph import build_nx_graph
        from autoreview.knowledge_graph.models import BetaPosterior, KGEdge, KGEntity

        entities = {
            "ent1": KGEntity(entity_id="ent1", canonical_name="SIRT1", entity_type="gene",
                             ontology_id=None, ontology_source=None,
                             aliases=[], paper_count=1, source_paper_ids=["p1"]),
        }
        edges = [
            KGEdge(edge_id="e1", subject_id="ent1", object_id="ent1",
                   predicate="regulates", direction="positive",
                   assertion_type="mechanistic_causal", confidence=BetaPosterior(),
                   evidence_links=[], source_assertions=["a1"], publication_date=None),
        ]
        G = build_nx_graph(entities, edges)
        assert G.number_of_nodes() == 1
        assert G.number_of_edges() == 1
        assert G.has_edge("ent1", "ent1")


class TestSerializationRoundTrip:
    def test_pickle_round_trip(self, tmp_path: Path):
        from autoreview.knowledge_graph.graph import build_nx_graph, load_graph, save_graph
        from autoreview.knowledge_graph.models import BetaPosterior, KGEdge, KGEntity

        entities = {
            "ent1": KGEntity(entity_id="ent1", canonical_name="A", entity_type="gene",
                             ontology_id=None, ontology_source=None,
                             aliases=[], paper_count=1, source_paper_ids=["p1"]),
        }
        G = build_nx_graph(entities, [])
        save_graph(G, tmp_path / "test_graph")
        G2 = load_graph(tmp_path / "test_graph.pkl")
        assert G2.number_of_nodes() == 1
        assert G2.nodes["ent1"]["canonical_name"] == "A"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n autoreview python -m pytest tests/test_knowledge_graph/test_graph.py -v`
Expected: ImportError

- [ ] **Step 3: Implement graph.py**

Create `autoreview/knowledge_graph/graph.py`. Key functions:
- `build_nx_graph(entities: dict[str, KGEntity], edges: list[KGEdge]) -> nx.MultiDiGraph`
- `save_graph(G: nx.MultiDiGraph, path: Path) -> None` — writes `.pkl` (pickle) + `.graphml`
- `load_graph(path: Path) -> nx.MultiDiGraph` — loads from pickle

Node attributes: all `KGEntity` fields stored as node attrs.
Edge attributes: `predicate`, `direction`, `assertion_type`, `confidence_mean`, `evidence_count`, `source_assertions`, `publication_date` stored as edge attrs. The full `KGEdge` model is stored under `_kg_edge` attr for rich access.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n autoreview python -m pytest tests/test_knowledge_graph/test_graph.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/knowledge_graph/graph.py tests/test_knowledge_graph/test_graph.py
git commit -m "feat(kg): add NetworkX graph construction and pickle/GraphML serialization"
```

---

## Task 6: Confidence scoring (`confidence.py`)

**Files:**
- Create: `autoreview/knowledge_graph/confidence.py`
- Create: `tests/test_knowledge_graph/test_confidence.py`

**Reference:** Spec §"Confidence Scoring" — Beta-Binomial with evidence strength weights and independence discounting

- [ ] **Step 1: Write the failing tests**

Create `tests/test_knowledge_graph/test_confidence.py`:
```python
"""Tests for Beta-Binomial confidence scoring with independence weighting."""

from __future__ import annotations

import pytest


class TestEvidenceWeights:
    def test_weight_table_values(self):
        from autoreview.knowledge_graph.confidence import EVIDENCE_WEIGHTS

        assert EVIDENCE_WEIGHTS["direct_experimental"] == 1.0
        assert EVIDENCE_WEIGHTS["observational_controlled"] == 0.7
        assert EVIDENCE_WEIGHTS["observational_uncontrolled"] == 0.4
        assert EVIDENCE_WEIGHTS["computational_prediction"] == 0.3
        assert EVIDENCE_WEIGHTS["expert_opinion"] == 0.2


class TestScoreEdge:
    def test_single_supporting_evidence(self):
        from autoreview.knowledge_graph.confidence import score_edge

        evidence = [
            {"evidence_direction": "supports", "evidence_strength": "direct_experimental",
             "paper_id": "p1", "first_author": "Smith", "last_author": "Jones"},
        ]
        posterior = score_edge(evidence)
        assert posterior.alpha == pytest.approx(2.0)  # 1.0 prior + 1.0 weight
        assert posterior.beta_param == pytest.approx(1.0)  # unchanged

    def test_contradicting_evidence(self):
        from autoreview.knowledge_graph.confidence import score_edge

        evidence = [
            {"evidence_direction": "contradicts", "evidence_strength": "direct_experimental",
             "paper_id": "p1", "first_author": "Smith", "last_author": "Jones"},
        ]
        posterior = score_edge(evidence)
        assert posterior.alpha == pytest.approx(1.0)  # unchanged
        assert posterior.beta_param == pytest.approx(2.0)  # 1.0 prior + 1.0 weight

    def test_mixed_evidence(self):
        from autoreview.knowledge_graph.confidence import score_edge

        evidence = [
            {"evidence_direction": "supports", "evidence_strength": "direct_experimental",
             "paper_id": "p1", "first_author": "Smith", "last_author": "Jones"},
            {"evidence_direction": "contradicts", "evidence_strength": "observational_controlled",
             "paper_id": "p2", "first_author": "Lee", "last_author": "Park"},
        ]
        posterior = score_edge(evidence)
        assert posterior.alpha == pytest.approx(2.0)  # 1.0 + 1.0
        assert posterior.beta_param == pytest.approx(1.7)  # 1.0 + 0.7


class TestIndependenceWeighting:
    def test_same_author_group_discounted(self):
        from autoreview.knowledge_graph.confidence import score_edge

        evidence = [
            {"evidence_direction": "supports", "evidence_strength": "direct_experimental",
             "paper_id": "p1", "first_author": "Smith", "last_author": "Jones"},
            {"evidence_direction": "supports", "evidence_strength": "direct_experimental",
             "paper_id": "p2", "first_author": "Smith", "last_author": "Jones"},
            {"evidence_direction": "supports", "evidence_strength": "direct_experimental",
             "paper_id": "p3", "first_author": "Smith", "last_author": "Jones"},
        ]
        posterior = score_edge(evidence)
        # First: 1.0, second: 0.5, third: 0.25 → total alpha = 1.0 + 1.75
        assert posterior.alpha == pytest.approx(2.75)

    def test_independent_labs_full_weight(self):
        from autoreview.knowledge_graph.confidence import score_edge

        evidence = [
            {"evidence_direction": "supports", "evidence_strength": "direct_experimental",
             "paper_id": "p1", "first_author": "Smith", "last_author": "Jones"},
            {"evidence_direction": "supports", "evidence_strength": "direct_experimental",
             "paper_id": "p2", "first_author": "Lee", "last_author": "Park"},
            {"evidence_direction": "supports", "evidence_strength": "direct_experimental",
             "paper_id": "p3", "first_author": "Chen", "last_author": "Wang"},
        ]
        posterior = score_edge(evidence)
        # 3 independent groups: 1.0 + 1.0 + 1.0 → alpha = 1.0 + 3.0
        assert posterior.alpha == pytest.approx(4.0)


class TestDerivedMetrics:
    def test_controversy_score(self):
        from autoreview.knowledge_graph.confidence import compute_derived_metrics
        from autoreview.knowledge_graph.models import BetaPosterior

        bp = BetaPosterior(alpha=3.0, beta_param=2.5)
        metrics = compute_derived_metrics(bp, evidence_count=5, paper_ids=["p1", "p2"],
                                          author_groups=2)
        assert metrics["controversy_score"] == pytest.approx(2.5 / 3.0)
        assert metrics["evidence_diversity"] == 2
        assert metrics["independent_source_count"] == 2

    def test_controversy_zero_for_unanimous(self):
        from autoreview.knowledge_graph.confidence import compute_derived_metrics
        from autoreview.knowledge_graph.models import BetaPosterior

        bp = BetaPosterior(alpha=5.0, beta_param=1.0)  # only prior beta
        metrics = compute_derived_metrics(bp, evidence_count=4, paper_ids=["p1", "p2"],
                                          author_groups=2)
        assert metrics["controversy_score"] == pytest.approx(1.0 / 5.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n autoreview python -m pytest tests/test_knowledge_graph/test_confidence.py -v`
Expected: ImportError

- [ ] **Step 3: Implement confidence.py**

Create `autoreview/knowledge_graph/confidence.py`. Key functions:
- `EVIDENCE_WEIGHTS: dict[str, float]` — maps EvidenceStrength values to numeric weights
- `score_edge(evidence: list[dict]) -> BetaPosterior` — compute Beta posterior with independence weighting
- `compute_derived_metrics(posterior, evidence_count, paper_ids, author_groups) -> dict` — controversy_score, evidence_diversity, independent_source_count
- `score_all_edges(edges: list[KGEdge], provenance: dict) -> list[KGEdge]` — batch scoring

Independence weighting: group evidence by `(first_author, last_author)`, apply 0.5x decay per additional unit from same group.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n autoreview python -m pytest tests/test_knowledge_graph/test_confidence.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/knowledge_graph/confidence.py tests/test_knowledge_graph/test_confidence.py
git commit -m "feat(kg): add Beta-Binomial confidence scoring with independence weighting"
```

---

## Task 7: Analysis layer (`analysis.py`)

**Files:**
- Create: `autoreview/knowledge_graph/analysis.py`
- Create: `tests/test_knowledge_graph/test_analysis.py`

**Reference:** Spec §"Analysis Layer" — community detection, contradiction finder, gap analysis

- [ ] **Step 1: Write the failing tests**

Create `tests/test_knowledge_graph/test_analysis.py`:
```python
"""Tests for graph analysis: communities, contradictions, gaps."""

from __future__ import annotations

import networkx as nx
import pytest


def _make_test_graph() -> nx.MultiDiGraph:
    """Build a small test graph with known properties."""
    G = nx.MultiDiGraph()
    # Cluster 1: Wnt-mesoderm
    G.add_node("wnt", canonical_name="Wnt signaling", entity_type="pathway", paper_count=5)
    G.add_node("meso", canonical_name="mesoderm formation", entity_type="biological_process", paper_count=4)
    G.add_node("bra", canonical_name="Brachyury", entity_type="gene", paper_count=3)
    G.add_edge("wnt", "meso", key="e1", predicate="is_required_for",
               confidence_mean=0.8, evidence_count=5, controversy_score=0.2,
               evidence_diversity=3, independent_source_count=3)
    G.add_edge("wnt", "bra", key="e2", predicate="induces",
               confidence_mean=0.7, evidence_count=3, controversy_score=0.3,
               evidence_diversity=2, independent_source_count=2)
    # Cluster 2: BMP-dorsal
    G.add_node("bmp", canonical_name="BMP signaling", entity_type="pathway", paper_count=3)
    G.add_node("dorsal", canonical_name="dorsal-ventral axis", entity_type="biological_process", paper_count=2)
    G.add_edge("bmp", "dorsal", key="e3", predicate="induces",
               confidence_mean=0.6, evidence_count=2, controversy_score=0.7,
               evidence_diversity=2, independent_source_count=2)
    # Cross-cluster link
    G.add_edge("wnt", "bmp", key="e4", predicate="inhibits",
               confidence_mean=0.5, evidence_count=1, controversy_score=0.9,
               evidence_diversity=1, independent_source_count=1)
    return G


class TestCommunityDetection:
    def test_finds_communities(self):
        from autoreview.knowledge_graph.analysis import detect_communities

        G = _make_test_graph()
        communities = detect_communities(G)
        assert len(communities) >= 1
        # All nodes should be assigned to a community
        all_nodes = set()
        for comm in communities:
            all_nodes.update(comm)
        assert all_nodes == set(G.nodes)


class TestHubEntities:
    def test_hub_ranking(self):
        from autoreview.knowledge_graph.analysis import find_hub_entities

        G = _make_test_graph()
        hubs = find_hub_entities(G, top_n=3)
        # Wnt has the most edges (3 outgoing)
        assert hubs[0][0] == "wnt"


class TestContradictionDetection:
    def test_finds_high_controversy_edges(self):
        from autoreview.knowledge_graph.analysis import find_contradictions

        G = _make_test_graph()
        contradictions = find_contradictions(G, threshold=0.5)
        # Edges e3 (0.7) and e4 (0.9) are above threshold
        assert len(contradictions) >= 2
        edge_ids = [c["edge_key"] for c in contradictions]
        assert "e4" in edge_ids


class TestGapAnalysis:
    def test_low_evidence_entities(self):
        from autoreview.knowledge_graph.analysis import find_low_evidence_entities

        G = _make_test_graph()
        gaps = find_low_evidence_entities(G, min_degree=2, max_evidence=3)
        # Entities with high degree but low total evidence
        # This depends on the specific graph, so just check structure
        assert isinstance(gaps, list)

    def test_temporal_gaps(self):
        from autoreview.knowledge_graph.analysis import find_temporal_gaps

        G = _make_test_graph()
        # Add publication dates to edges
        for u, v, k in G.edges(keys=True):
            G.edges[u, v, k]["publication_date"] = "2018-01-01"
        gaps = find_temporal_gaps(G, cutoff_year=2020)
        assert len(gaps) >= 1  # All edges are before 2020
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n autoreview python -m pytest tests/test_knowledge_graph/test_analysis.py -v`
Expected: ImportError

- [ ] **Step 3: Implement analysis.py**

Create `autoreview/knowledge_graph/analysis.py`. Key functions:
- `detect_communities(G) -> list[set[str]]` — Louvain community detection on undirected projection
- `find_hub_entities(G, top_n=20) -> list[tuple[str, float]]` — degree centrality ranking
- `find_contradictions(G, threshold=0.5) -> list[dict]` — edges with `controversy_score > threshold`
- `find_low_evidence_entities(G, min_degree, max_evidence) -> list[dict]` — high connectivity, low grounding
- `find_temporal_gaps(G, cutoff_year) -> list[dict]` — edges with no evidence newer than cutoff
- `extract_subgraph(G, node_ids) -> nx.MultiDiGraph` — extract neighborhood subgraph

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n autoreview python -m pytest tests/test_knowledge_graph/test_analysis.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/knowledge_graph/analysis.py tests/test_knowledge_graph/test_analysis.py
git commit -m "feat(kg): add analysis layer — communities, contradictions, gap detection"
```

---

## Task 8: Visualization (`viz.py`)

**Files:**
- Create: `autoreview/knowledge_graph/viz.py`
- Create: `tests/test_knowledge_graph/test_viz.py`

**Reference:** Spec §"Visualization" — GraphML export, matplotlib network plots, confidence distribution

- [ ] **Step 1: Write the failing tests**

Create `tests/test_knowledge_graph/test_viz.py`:
```python
"""Tests for graph visualization and export."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest


def _make_viz_graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    G.add_node("a", canonical_name="Gene A", entity_type="gene", paper_count=3)
    G.add_node("b", canonical_name="Process B", entity_type="biological_process", paper_count=2)
    G.add_edge("a", "b", key="e1", predicate="induces",
               confidence_mean=0.8, evidence_count=3)
    return G


class TestGraphMLExport:
    def test_export_creates_file(self, tmp_path: Path):
        from autoreview.knowledge_graph.viz import export_graphml

        G = _make_viz_graph()
        out = tmp_path / "test.graphml"
        export_graphml(G, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_exported_graphml_readable(self, tmp_path: Path):
        from autoreview.knowledge_graph.viz import export_graphml

        G = _make_viz_graph()
        out = tmp_path / "test.graphml"
        export_graphml(G, out)
        G2 = nx.read_graphml(out)
        assert G2.number_of_nodes() == 2


class TestPlotSubgraph:
    def test_generates_figure(self, tmp_path: Path):
        from autoreview.knowledge_graph.viz import plot_subgraph

        G = _make_viz_graph()
        out = tmp_path / "subgraph.png"
        plot_subgraph(G, output_path=out)
        assert out.exists()


class TestConfidenceDistribution:
    def test_generates_figure(self, tmp_path: Path):
        from autoreview.knowledge_graph.viz import plot_confidence_distribution

        G = _make_viz_graph()
        out = tmp_path / "confidence.png"
        plot_confidence_distribution(G, output_path=out)
        assert out.exists()


class TestControversyMap:
    def test_generates_figure(self, tmp_path: Path):
        from autoreview.knowledge_graph.viz import plot_controversy_map

        G = _make_viz_graph()
        # Add controversy_score to the edge
        G.edges["a", "b", "e1"]["controversy_score"] = 0.8
        out = tmp_path / "controversy.png"
        plot_controversy_map(G, output_path=out, threshold=0.5)
        assert out.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n autoreview python -m pytest tests/test_knowledge_graph/test_viz.py -v`
Expected: ImportError

- [ ] **Step 3: Implement viz.py**

Create `autoreview/knowledge_graph/viz.py`. Key functions:
- `export_graphml(G, path)` — write GraphML for Gephi/Cytoscape (strips non-serializable attrs)
- `plot_subgraph(G, output_path, node_ids=None)` — matplotlib network plot, nodes colored by entity_type, edges by confidence, width by evidence_count. Colorblind-safe palette.
- `plot_confidence_distribution(G, output_path)` — histogram of confidence_mean across all edges
- `plot_controversy_map(G, output_path, threshold=0.5)` — highlight high-controversy edges

All figures: 300 DPI, Arial/Helvetica, `constrained_layout=True`, colorblind-safe palette `["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n autoreview python -m pytest tests/test_knowledge_graph/test_viz.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/knowledge_graph/viz.py tests/test_knowledge_graph/test_viz.py
git commit -m "feat(kg): add visualization — GraphML export, network plots, confidence histograms"
```

---

## Task 9: Public API and full pipeline (`__init__.py`)

**Files:**
- Modify: `autoreview/knowledge_graph/__init__.py`
- Create: `tests/test_knowledge_graph/test_pipeline.py`

**Reference:** Spec §"Public API" — `build_graph()`, `load_graph()`, `save_graph()`

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_knowledge_graph/test_pipeline.py`:
```python
"""Integration tests for the full KG pipeline: ingest → dedup → graph → confidence."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestBuildGraph:
    def test_full_pipeline(self, sample_extraction_dir: Path):
        from autoreview.knowledge_graph import build_graph

        G = build_graph(sample_extraction_dir)

        # Should have nodes (entities) and edges (merged assertions)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

        # Papers 0 and 1 have same entities + synonymous predicates → should merge into 1 edge
        # Paper 2 has different entities → separate edge
        # So expect: ~4 unique entities, ~2 unique edges
        assert G.number_of_nodes() >= 3
        assert G.number_of_edges() >= 2

        # Edges should have confidence scores
        for u, v, k, data in G.edges(keys=True, data=True):
            assert "confidence_mean" in data
            assert 0.0 <= data["confidence_mean"] <= 1.0

    def test_save_and_load_round_trip(self, sample_extraction_dir: Path, tmp_path: Path):
        from autoreview.knowledge_graph import build_graph, load_graph, save_graph

        G = build_graph(sample_extraction_dir)
        save_graph(G, tmp_path / "test_kg")

        assert (tmp_path / "test_kg.pkl").exists()
        assert (tmp_path / "test_kg.graphml").exists()

        G2 = load_graph(tmp_path / "test_kg.pkl")
        assert G2.number_of_nodes() == G.number_of_nodes()
        assert G2.number_of_edges() == G.number_of_edges()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n autoreview python -m pytest tests/test_knowledge_graph/test_pipeline.py -v`
Expected: ImportError (build_graph not defined)

- [ ] **Step 3: Implement public API in __init__.py**

Update `autoreview/knowledge_graph/__init__.py`:
```python
"""Knowledge graph construction and analysis from extraction data.

Public API:
    build_graph(extraction_dir) -> nx.MultiDiGraph
    load_graph(path) -> nx.MultiDiGraph
    save_graph(graph, path) -> None
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx


def build_graph(extraction_dir: Path) -> nx.MultiDiGraph:
    """Full pipeline: ingest → dedup → graph → confidence.

    Args:
        extraction_dir: Path to directory containing extraction JSON files.

    Returns:
        Annotated NetworkX MultiDiGraph with confidence-scored edges.
    """
    from autoreview.knowledge_graph.confidence import score_all_edges
    from autoreview.knowledge_graph.dedup import (
        deduplicate_entities,
        merge_assertions,
        normalize_predicate,
    )
    from autoreview.knowledge_graph.graph import build_nx_graph
    from autoreview.knowledge_graph.ingest import ingest_directory

    # 1. Ingest
    corpus = ingest_directory(extraction_dir)

    # 2. Entity dedup
    registry = deduplicate_entities(corpus.all_entities)

    # 3. Predicate normalization + remap entity IDs in assertions
    normalized_assertions = []
    for assertion in corpus.all_assertions:
        subj_surface = assertion["subject_canonical_name"].lower()
        obj_surface = assertion["object_canonical_name"].lower()
        subj_id = registry.surface_to_id.get(subj_surface)
        obj_id = registry.surface_to_id.get(obj_surface)
        if subj_id and obj_id:
            assertion["subject_id"] = subj_id
            assertion["object_id"] = obj_id
            assertion["predicate"] = normalize_predicate(assertion["predicate"])
            normalized_assertions.append(assertion)

    # 4. Assertion merging
    merged = merge_assertions(normalized_assertions)

    # 5. Build graph
    edges = _merged_to_kg_edges(merged, corpus)
    G = build_nx_graph(registry.entities, edges)

    # 6. Confidence scoring
    G = score_all_edges(G, corpus.provenance_by_paper)

    return G


def load_graph(path: Path) -> nx.MultiDiGraph:
    """Load a previously built graph from pickle format."""
    from autoreview.knowledge_graph.graph import load_graph as _load

    return _load(path)


def save_graph(graph: nx.MultiDiGraph, path: Path) -> None:
    """Serialize graph to pickle (fast reload) and GraphML (interop)."""
    from autoreview.knowledge_graph.graph import save_graph as _save

    _save(graph, path)
```

The `_merged_to_kg_edges` helper converts merged assertion dicts to `KGEdge` model instances with evidence links attached.

- [ ] **Step 4: Run integration test to verify it passes**

Run: `conda run -n autoreview python -m pytest tests/test_knowledge_graph/test_pipeline.py -v`
Expected: all PASS

- [ ] **Step 5: Run full test suite**

Run: `conda run -n autoreview python -m pytest tests/test_knowledge_graph/ -v`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add autoreview/knowledge_graph/__init__.py tests/test_knowledge_graph/test_pipeline.py
git commit -m "feat(kg): add public API and full pipeline integration — build_graph(), save/load"
```

---

## Task 10: Run against real corpus

**Files:**
- No new files — this validates the implementation against the actual 303-paper corpus

- [ ] **Step 1: Run build_graph on the real extraction directory**

```bash
conda run -n autoreview python -c "
from pathlib import Path
from autoreview.knowledge_graph import build_graph, save_graph

extraction_dir = Path('Paper Extractor/KnowledgeGraph Extraction/gastruloid_run/extractions')
G = build_graph(extraction_dir)
print(f'Nodes: {G.number_of_nodes()}')
print(f'Edges: {G.number_of_edges()}')

# Save for reuse
save_graph(G, Path('output/knowledge_graph/gastruloid_kg'))
print('Saved to output/knowledge_graph/')
"
```

Expected output (approximate):
- Nodes: ~2,000–2,500 (after entity dedup from ~3,400)
- Edges: ~1,500–2,000 (after assertion merging from ~2,900)

- [ ] **Step 2: Verify graph statistics**

```bash
conda run -n autoreview python -c "
from autoreview.knowledge_graph import load_graph
from autoreview.knowledge_graph.analysis import detect_communities, find_contradictions, find_hub_entities

G = load_graph('output/knowledge_graph/gastruloid_kg.pkl')
print(f'Communities: {len(detect_communities(G))}')
print(f'Contradictions (>0.5): {len(find_contradictions(G, threshold=0.5))}')
hubs = find_hub_entities(G, top_n=10)
print('Top 10 hub entities:')
for name, score in hubs:
    print(f'  {G.nodes[name][\"canonical_name\"]}: {score:.3f}')
"
```

- [ ] **Step 3: Generate visualizations**

```bash
conda run -n autoreview python -c "
from pathlib import Path
from autoreview.knowledge_graph import load_graph
from autoreview.knowledge_graph.viz import (
    export_graphml, plot_confidence_distribution, plot_controversy_map,
)

G = load_graph('output/knowledge_graph/gastruloid_kg.pkl')
out = Path('output/knowledge_graph/figures')
out.mkdir(parents=True, exist_ok=True)

export_graphml(G, out / 'gastruloid_kg.graphml')
plot_confidence_distribution(G, out / 'confidence_distribution.png')
plot_controversy_map(G, out / 'controversy_map.png')
print('Visualizations saved')
"
```

- [ ] **Step 4: Commit results**

```bash
git add output/knowledge_graph/
git commit -m "feat(kg): initial gastruloid corpus graph — nodes, edges, figures"
```

---

## Dependency Graph

```
Task 1 (scaffold)
  ↓
Task 2 (models)
  ↓
Task 3 (ingest) ──→ Task 4 (dedup) ──→ Task 5 (graph) ──→ Task 6 (confidence)
                                                               ↓
                                                          Task 7 (analysis) ──→ Task 8 (viz)
                                                               ↓
                                                          Task 9 (public API + integration)
                                                               ↓
                                                          Task 10 (real corpus run)
```

**Parallelizable batches:**
- Batch 1: Task 1 (scaffold)
- Batch 2: Task 2 (models)
- Batch 3: Task 3 (ingest)
- Batch 4: Task 4 (dedup — depends on ingest output format defined by Task 3's entity dict schema)
- Batch 5: Task 5 (graph — needs dedup output)
- Batch 6: Task 6 (confidence — needs graph)
- Batch 7: Tasks 7 + 8 in parallel (both read from graph, don't depend on each other)
- Batch 8: Task 9 (wires everything together)
- Batch 9: Task 10 (real corpus validation)
