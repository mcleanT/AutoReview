"""
KG-specific extraction schema for knowledge graph construction.

This schema is SEPARATE from the AutoReview pipeline ExtractionResult schema.
It targets ~100 tokens per claim (vs ~300 in the full schema) for high-density
extraction optimized for graph edge comparison and contradiction detection.

Design priorities:
- Claim density: more claims per token budget
- Every field enables graph edges or contradiction detection
- Flat condition strings instead of nested ontology objects
- Literal types for enums (no StrEnum classes)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class KGEntity(BaseModel):
    """Minimal entity for graph nodes."""

    name: str = Field(..., description="Canonical name (e.g., 'BMP4', 'mesoderm differentiation')")
    type: Literal[
        "protein",
        "gene",
        "rna",
        "small_molecule",
        "pathway",
        "biological_process",
        "phenotype",
        "cellular_compartment",
        "organism",
        "cell_type",
        "disease",
        "tissue",
        "method",
        "other",
    ] = Field(..., description="Entity type for graph node classification")
    ontology_id: str | None = Field(
        None, description="Best-effort ontology ID (UniProt, GO, CL, UBERON, etc.)"
    )


class KGConditions(BaseModel):
    """Experimental context that scopes a claim. Essential for determining whether two claims are truly comparable."""

    species: list[str] = Field(default_factory=list, description="e.g., ['Mus musculus']")
    cell_type: list[str] = Field(default_factory=list, description="e.g., ['mESC', 'E14Tg2a']")
    tissue: list[str] = Field(default_factory=list, description="e.g., ['neural tube']")
    treatment: list[str] = Field(
        default_factory=list, description="e.g., ['10ng/mL BMP4', '48h culture']"
    )
    developmental_stage: str | None = Field(None, description="e.g., 'E8.5', 'day 5', 'adult'")
    in_vitro: bool | None = Field(None, description="True for cell culture, False for in vivo")


class KGEvidenceLink(BaseModel):
    """Link from a claim to an evidence unit with directional qualifier."""

    evidence_id: str = Field(..., description="Reference to evidence unit ID")
    direction: Literal["supports", "refutes", "mixed", "not_applicable"] = Field(
        ...,
        description="Whether this evidence supports, refutes, or has mixed bearing on the claim",
    )


class KGClaim(BaseModel):
    """A single atomic scientific claim — the core unit of the knowledge graph."""

    claim_id: str = Field(..., description="Sequential: c_001, c_002, ...")
    natural_language: str = Field(..., description="One-sentence claim in natural language")
    subject: KGEntity
    predicate: str = Field(
        ...,
        description="Controlled vocabulary predicate (e.g., 'induces', 'inhibits', 'is_required_for')",
    )
    object: KGEntity
    direction: Literal["positive", "negative"] = Field(
        ...,
        description="Whether the predicate holds (positive) or does not hold (negative)",
    )
    claim_type: Literal[
        "mechanistic_causal",
        "correlational",
        "comparative",
        "existence",
        "absence",
        "conditional",
        "methodological",
    ] = Field(..., description="Type of assertion")
    causal_type: (
        Literal[
            "necessary",
            "sufficient",
            "necessary_and_sufficient",
            "contributory",
            "modulatory",
        ]
        | None
    ) = Field(None, description="Only for mechanistic_causal claims")
    conditions: KGConditions = Field(
        default_factory=KGConditions,
        description="Experimental context scoping this claim",
    )
    evidence_strength: Literal[
        "direct_experimental",
        "indirect_experimental",
        "observational",
        "computational",
        "review_citation",
    ] = Field(..., description="Strength of evidence supporting this claim")
    certainty: Literal["high", "medium", "low"] = Field(
        ...,
        description="Author's certainty: high=demonstrates/shows, medium=suggests/indicates, low=may/might/could",
    )
    section_source: Literal[
        "primary_empirical",
        "interpretive",
        "attributed_prior",
        "methodological",
    ] = Field(
        ...,
        description="Epistemic status based on which section the claim comes from",
    )
    source_doi: str | None = Field(
        None, description="For attributed_prior claims: DOI of the cited work"
    )
    model_system: str | None = Field(
        None, description="e.g., 'mouse ESC gastruloids' — required by prompt for all claims"
    )
    organism: str | None = Field(None, description="Latin binomial, e.g., 'Mus musculus'")
    quantitative_context: dict[str, Any] | None = Field(
        None,
        description="Dose/time context: {concentration, timepoint, dose} — required for mechanistic/comparative claims",
    )
    evidence_links: list[KGEvidenceLink] = Field(
        default_factory=list,
        description="Links to evidence units with directional qualifiers (supports/refutes/mixed)",
    )


class KGEvidence(BaseModel):
    """A distinct experiment or analysis — one per figure panel or table."""

    evidence_id: str = Field(..., description="Sequential: e_001, e_002, ...")
    description: str = Field(..., description="Brief description of the experiment")
    result_summary: str = Field(
        ...,
        description="One-sentence conclusion: what the experiment PROVED or DISPROVED (not what was done)",
    )
    model_system: str = Field(..., description="e.g., 'mouse ESC gastruloids'")
    organism: str = Field(..., description="Latin binomial, e.g., 'Mus musculus'")
    perturbation: str | None = Field(
        None,
        description="e.g., 'CRISPR KO of Brachyury', 'siRNA knockdown of BMP4'",
    )
    readout: str = Field(..., description="What was measured")
    result_direction: Literal["positive", "negative", "null", "not_reported"] = Field(
        ..., description="Outcome direction"
    )
    effect_size: str | None = Field(None, description="Verbatim from paper")
    p_value: str | None = Field(None, description="Verbatim from paper")
    sample_size: str | None = Field(None, description="Verbatim from paper")
    key_figure: str | None = Field(None, description="Figure/table reference, e.g., 'Figure 2A'")
    approach: Literal[
        "biochemical_assay",
        "cell_biology",
        "genetics",
        "omics",
        "imaging",
        "computational",
        "clinical",
        "animal_model",
        "in_vitro_model",
        "structural_biology",
        "pharmacology",
        "citation_reference",
    ] = Field(..., description="Methodological category")
    assay_types: list[str] = Field(
        default_factory=list,
        description="Specific assays, e.g., ['qRT-PCR', 'Western_blot']",
    )
    evidence_strength: (
        Literal[
            "direct_experimental",
            "indirect_experimental",
            "observational",
            "computational",
            "review_citation",
        ]
        | None
    ) = Field(None, description="Strength tier for this evidence unit (used by citation stubs)")
    citing_sentence: str | None = Field(
        None, description="Verbatim sentence from paper citing this evidence"
    )
    source_doi: str | None = Field(
        None, description="DOI of the cited work (for attributed_prior evidence)"
    )


class KGExtraction(BaseModel):
    """Top-level extraction result for one paper."""

    doi: str = Field(..., description="Paper DOI")
    title: str = Field(..., description="Full paper title")
    journal: str = Field(..., description="Journal name")
    publication_date: str = Field(..., description="YYYY-MM-DD format")
    claims: list[KGClaim] = Field(default_factory=list, description="All extracted claims")
    evidence: list[KGEvidence] = Field(default_factory=list, description="All evidence units")
    citation_contexts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Cross-paper citation relationships extracted from the paper",
    )
    extraction_model: str = Field(default="", description="Model ID used for extraction")
    extraction_timestamp: str = Field(default="", description="ISO 8601 UTC timestamp")
