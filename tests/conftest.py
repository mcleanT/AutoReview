from __future__ import annotations

import pytest

from autoreview.models.paper import CandidatePaper, ScreenedPaper
from autoreview.extraction.models import (
    EvidenceStrength,
    Finding,
    PaperExtraction,
    MethodologyRecord,
)
from autoreview.models.knowledge_base import KnowledgeBase


@pytest.fixture
def sample_paper() -> CandidatePaper:
    return CandidatePaper(
        title="Gut Microbiome and Neurodegeneration: A Systematic Review",
        authors=["Smith J", "Doe A", "Zhang L"],
        year=2023,
        journal="Nature Reviews Neuroscience",
        doi="10.1038/s41583-023-0001",
        abstract="This systematic review examines the role of gut microbiota in neurodegenerative diseases...",
        source_database="pubmed",
        external_ids={"pmid": "12345678"},
        citation_count=42,
    )


@pytest.fixture
def sample_screened_paper(sample_paper: CandidatePaper) -> ScreenedPaper:
    return ScreenedPaper(
        paper=sample_paper,
        relevance_score=4,
        rationale="Directly addresses gut-brain axis in neurodegeneration with strong methodology",
        include=True,
    )


@pytest.fixture
def sample_finding() -> Finding:
    return Finding(
        claim="Gut dysbiosis precedes motor symptom onset in Parkinson's disease",
        evidence_strength=EvidenceStrength.MODERATE,
        quantitative_result="OR=2.3, 95% CI [1.4-3.8], p<0.001",
        context="Longitudinal cohort study, n=1200",
        paper_id="abc123",
    )


@pytest.fixture
def sample_extraction(sample_finding: Finding) -> PaperExtraction:
    return PaperExtraction(
        paper_id="abc123",
        key_findings=[sample_finding],
        methods_summary="Longitudinal cohort study with 16S rRNA sequencing",
        limitations="Single-center study, limited ethnic diversity",
        methodology_details=MethodologyRecord(
            approach="16S rRNA sequencing + clinical assessment",
            datasets=["UK Biobank subset"],
            metrics=["Shannon diversity index", "Bray-Curtis dissimilarity"],
        ),
        domain_specific_fields={"sample_size": 1200, "study_design": "longitudinal cohort"},
    )


@pytest.fixture
def sample_kb() -> KnowledgeBase:
    return KnowledgeBase(
        topic="gut microbiome in neurodegenerative diseases",
        domain="biomedical",
        output_dir="/tmp/autoreview_test",
    )
