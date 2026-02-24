from __future__ import annotations

import json
import os
import tempfile

import pytest

from autoreview.models.base import AutoReviewModel, TimestampedModel
from autoreview.models.paper import CandidatePaper, ScreenedPaper
from autoreview.extraction.models import (
    EvidenceStrength,
    Finding,
    RelationshipClaim,
    RelationshipType,
    PaperExtraction,
)
from autoreview.analysis.evidence_map import (
    Theme,
    EvidenceMap,
    Contradiction,
    IdentifiedGap,
    GapSeverity,
    ConsensusClaim,
)
from autoreview.critique.models import (
    CritiqueReport,
    CritiqueIssue,
    CritiqueSeverity,
    CritiqueTarget,
)
from autoreview.models.knowledge_base import KnowledgeBase, PipelinePhase


class TestBaseModels:
    def test_timestamped_model_has_id_and_created_at(self):
        model = TimestampedModel()
        assert model.id is not None
        assert len(model.id) == 32  # uuid4 hex
        assert model.created_at is not None

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            TimestampedModel(unknown_field="bad")


class TestPaperModels:
    def test_doi_normalization_strips_prefix(self):
        paper = CandidatePaper(
            title="Test",
            authors=["A"],
            doi="https://doi.org/10.1234/TEST",
            source_database="pubmed",
        )
        assert paper.doi == "10.1234/test"

    def test_doi_normalization_dx_prefix(self):
        paper = CandidatePaper(
            title="Test",
            authors=["A"],
            doi="http://dx.doi.org/10.5678/Something",
            source_database="s2",
        )
        assert paper.doi == "10.5678/something"

    def test_doi_none_allowed(self):
        paper = CandidatePaper(
            title="Test Preprint",
            authors=["A"],
            doi=None,
            source_database="s2",
        )
        assert paper.doi is None

    def test_full_text_excluded_from_serialization(self):
        paper = CandidatePaper(
            title="Test",
            authors=["A"],
            source_database="pubmed",
            full_text="Very long full text content...",
        )
        dumped = paper.model_dump()
        assert "full_text" not in dumped

    def test_screened_paper_score_range(self):
        paper = CandidatePaper(title="T", authors=["A"], source_database="s2")
        with pytest.raises(Exception):
            ScreenedPaper(paper=paper, relevance_score=6, rationale="test", include=True)
        with pytest.raises(Exception):
            ScreenedPaper(paper=paper, relevance_score=0, rationale="test", include=True)


class TestExtractionModels:
    def test_finding_creation(self, sample_finding):
        assert sample_finding.evidence_strength == EvidenceStrength.MODERATE
        assert "dysbiosis" in sample_finding.claim

    def test_relationship_claim(self):
        rel = RelationshipClaim(
            source_paper_id="abc",
            target_paper_id="def",
            relationship_type=RelationshipType.CONTRADICTS,
            description="Different conclusions on microbiome diversity",
        )
        assert rel.relationship_type == RelationshipType.CONTRADICTS

    def test_paper_extraction_serialization(self, sample_extraction):
        data = sample_extraction.model_dump()
        restored = PaperExtraction.model_validate(data)
        assert len(restored.key_findings) == 1
        assert restored.domain_specific_fields["sample_size"] == 1200


class TestEvidenceMap:
    def test_empty_evidence_map(self):
        em = EvidenceMap()
        assert em.coverage_score == 0.0
        assert em.themes == []

    def test_contradiction(self):
        c = Contradiction(
            claim_a="Microbiome diversity decreases in PD",
            claim_b="Microbiome diversity unchanged in PD",
            paper_ids_a=["p1", "p2"],
            paper_ids_b=["p3"],
            possible_explanation="Different disease stages sampled",
        )
        assert len(c.paper_ids_a) == 2

    def test_gap_severity(self):
        gap = IdentifiedGap(
            expected_topic="Viral microbiome role",
            current_coverage="No papers found",
            severity=GapSeverity.MAJOR,
            suggested_queries=["viral metagenomics neurodegeneration"],
        )
        assert gap.severity == GapSeverity.MAJOR


class TestCritiqueModels:
    def test_critique_report_score_range(self):
        with pytest.raises(Exception):
            CritiqueReport(
                target=CritiqueTarget.OUTLINE,
                passed=False,
                overall_score=1.5,  # Out of range
            )

    def test_critique_with_issues(self):
        report = CritiqueReport(
            target=CritiqueTarget.SECTION,
            target_id="3.2",
            passed=False,
            overall_score=0.65,
            dimension_scores={"synthesis": 0.4, "accuracy": 0.9},
            issues=[
                CritiqueIssue(
                    severity=CritiqueSeverity.CRITICAL,
                    location="Section 3.2",
                    description="Paper-by-paper summary instead of synthesis",
                    suggested_fix="Group findings by theme, compare across papers",
                ),
            ],
        )
        assert len(report.issues) == 1
        assert report.issues[0].severity == CritiqueSeverity.CRITICAL


class TestKnowledgeBase:
    def test_initial_state(self, sample_kb):
        assert sample_kb.current_phase == PipelinePhase.INITIALIZED
        assert sample_kb.candidate_papers == []
        assert sample_kb.audit_log == []

    def test_add_audit_entry(self, sample_kb):
        sample_kb.add_audit_entry(
            "search", "query_expansion", "Generated 5 queries",
            token_usage={"input_tokens": 500, "output_tokens": 200},
        )
        assert len(sample_kb.audit_log) == 1
        assert sample_kb.audit_log[0].node_name == "search"

    def test_total_tokens(self, sample_kb):
        sample_kb.add_audit_entry("n1", "a1", token_usage={"input_tokens": 100, "output_tokens": 50})
        sample_kb.add_audit_entry("n2", "a2", token_usage={"input_tokens": 200, "output_tokens": 75})
        totals = sample_kb.total_tokens()
        assert totals["input_tokens"] == 300
        assert totals["output_tokens"] == 125

    def test_snapshot_round_trip(self, sample_kb, sample_paper):
        sample_kb.candidate_papers.append(sample_paper)
        sample_kb.current_phase = PipelinePhase.SEARCH

        with tempfile.TemporaryDirectory() as tmpdir:
            sample_kb.output_dir = tmpdir
            sample_kb.save_snapshot("search")

            # Find the snapshot file
            snap_dir = os.path.join(tmpdir, "snapshots")
            files = os.listdir(snap_dir)
            json_files = [f for f in files if f.endswith("_search.json")]
            assert len(json_files) == 1

            # Load from latest.json
            latest_path = os.path.join(snap_dir, "latest.json")
            loaded = KnowledgeBase.load_snapshot(latest_path)
            assert loaded.topic == "gut microbiome in neurodegenerative diseases"
            assert len(loaded.candidate_papers) == 1
            assert loaded.current_phase == PipelinePhase.SEARCH
