"""Pydantic models for the knowledge graph.

Three-tier architecture:
- Tier 1 (KGEdge): Assertions — mechanism-level claims, the unit of scientific discourse
- Tier 2 (KGEvidenceLink): Evidence — experimental demonstrations grounding assertions
- Tier 3 (paper_provenance via paper_id): Provenance — source credibility and independence
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field, computed_field

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
    rna = "rna"
    small_molecule = "small_molecule"
    phenotype = "phenotype"
    cellular_compartment = "cellular_compartment"
    tissue = "tissue"
    other = "other"


class AssertionType(StrEnum):
    mechanistic_causal = "mechanistic_causal"
    existence = "existence"
    comparative = "comparative"
    methodological = "methodological"
    correlational = "correlational"
    absence = "absence"
    conditional = "conditional"


class SectionSource(StrEnum):
    """Where the claim originates — determines epistemic weight."""

    primary_empirical = "primary_empirical"
    interpretive = "interpretive"
    attributed_prior = "attributed_prior"
    methodological = "methodological"


class Certainty(StrEnum):
    """Linguistic certainty level — maps to Beta prior width."""

    high = "high"
    medium = "medium"
    low = "low"


class EvidenceStrength(StrEnum):
    direct_experimental = "direct_experimental"
    indirect_experimental = "indirect_experimental"
    observational = "observational"
    computational = "computational"
    review_citation = "review_citation"


class BetaPosterior(AutoReviewModel):
    """Beta-Binomial confidence score for an edge."""

    alpha: float = 1.0
    beta_param: float = 1.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta_param)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ci_95(self) -> tuple[float, float]:
        from scipy.stats import beta as beta_dist

        return tuple(beta_dist.interval(0.95, self.alpha, self.beta_param))


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
    citing_sentence: str | None = None
    source_doi: str | None = None
    section: str | None = None


class QuantitativeContext(AutoReviewModel):
    """Structured dose/time/concentration context for a claim."""

    concentration: str | None = None
    timepoint: str | None = None
    dose: str | None = None


class ConditionContext(AutoReviewModel):
    """Structured summary of the experimental context for a condition-partitioned edge."""

    organism: str | None = None
    model_system_class: str | None = None
    in_vitro: bool | None = None
    cell_types: list[str] = Field(default_factory=list)
    treatments: list[str] = Field(default_factory=list)
    stages: list[str] = Field(default_factory=list)


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
    # v5 context fields
    certainty: Certainty | None = None
    section_source: SectionSource | None = None
    model_system: str | None = None
    organism: str | None = None
    in_vitro: bool | None = None
    conditions: dict[str, Any] | None = None
    hedging: str | None = None
    negatable_form: str | None = None
    natural_language: str | None = None
    quantitative_context: QuantitativeContext | None = None
    # v2 condition-aware merging fields
    condition_signature: str | None = None
    condition_context: ConditionContext | None = None


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
