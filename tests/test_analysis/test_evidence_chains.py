"""Tests for evidence chain building and synthesis directives."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from autoreview.analysis.evidence_chains import (
    EnrichedContradiction,
    EvidenceChain,
    EvidenceChainBuilder,
    EvidenceStrengthDistribution,
    SynthesisDirectiveBundle,
    TemporalProgression,
)
from autoreview.analysis.evidence_map import (
    ConsensusClaim,
    Contradiction,
    EvidenceMap,
    Theme,
)
from autoreview.extraction.models import (
    EvidenceStrength,
    Finding,
    PaperExtraction,
    RelationshipClaim,
    RelationshipType,
)


def _make_extraction(
    paper_id: str,
    claim: str = "Test claim",
    strength: str = "moderate",
    methods: str = "Standard methods",
) -> PaperExtraction:
    return PaperExtraction(
        paper_id=paper_id,
        key_findings=[
            Finding(
                claim=claim,
                evidence_strength=strength,
                paper_id=paper_id,
            )
        ],
        methods_summary=methods,
        limitations="None noted",
    )


class TestComputeStrengthDistribution:
    def test_basic_distribution(self):
        extractions = {
            "p1": _make_extraction("p1", strength="strong"),
            "p2": _make_extraction("p2", strength="moderate"),
            "p3": _make_extraction("p3", strength="moderate"),
            "p4": _make_extraction("p4", strength="weak"),
        }
        builder = EvidenceChainBuilder(llm=None)
        dist = builder.compute_strength_distribution(
            ["p1", "p2", "p3", "p4"], extractions,
        )

        assert dist.strong == 1
        assert dist.moderate == 2
        assert dist.weak == 1
        assert dist.preliminary == 0
        assert dist.dominant_type == "moderate"
        assert "moderate" in dist.writer_guidance.lower()

    def test_empty_papers(self):
        builder = EvidenceChainBuilder(llm=None)
        dist = builder.compute_strength_distribution([], {})
        assert dist.strong == 0
        assert dist.moderate == 0
        assert dist.dominant_type == ""
        assert dist.writer_guidance == ""

    def test_all_strong(self):
        extractions = {
            "p1": _make_extraction("p1", strength="strong"),
            "p2": _make_extraction("p2", strength="strong"),
        }
        builder = EvidenceChainBuilder(llm=None)
        dist = builder.compute_strength_distribution(["p1", "p2"], extractions)
        assert dist.dominant_type == "strong"
        assert dist.strong == 2

    def test_missing_papers_skipped(self):
        extractions = {"p1": _make_extraction("p1", strength="strong")}
        builder = EvidenceChainBuilder(llm=None)
        dist = builder.compute_strength_distribution(
            ["p1", "nonexistent"], extractions,
        )
        assert dist.strong == 1


class TestDetectTemporalProgressions:
    def test_detect_progression_across_years(self):
        extractions = {
            f"p{i}": _make_extraction(f"p{i}", claim=f"Claim {i}")
            for i in range(1, 9)
        }
        paper_years = {
            "p1": 2015, "p2": 2016, "p3": 2017, "p4": 2018,
            "p5": 2020, "p6": 2021, "p7": 2022, "p8": 2023,
        }
        builder = EvidenceChainBuilder(llm=None)
        progs = builder.detect_temporal_progressions(
            list(extractions.keys()), extractions, paper_years,
        )

        assert len(progs) == 1
        prog = progs[0]
        assert "2015" in prog.early_period
        assert "2023" in prog.late_period
        assert len(prog.paper_ids) == 8

    def test_too_few_papers_returns_empty(self):
        extractions = {
            "p1": _make_extraction("p1"),
            "p2": _make_extraction("p2"),
        }
        paper_years = {"p1": 2020, "p2": 2021}
        builder = EvidenceChainBuilder(llm=None)
        progs = builder.detect_temporal_progressions(
            ["p1", "p2"], extractions, paper_years,
        )
        assert progs == []

    def test_narrow_year_range_returns_empty(self):
        extractions = {f"p{i}": _make_extraction(f"p{i}") for i in range(1, 6)}
        paper_years = {f"p{i}": 2023 for i in range(1, 6)}
        builder = EvidenceChainBuilder(llm=None)
        progs = builder.detect_temporal_progressions(
            list(extractions.keys()), extractions, paper_years,
        )
        assert progs == []


class TestBuildSectionDirectives:
    @pytest.mark.asyncio
    async def test_formats_correctly(self):
        extractions = {
            "p1": _make_extraction("p1", strength="strong"),
            "p2": _make_extraction("p2", strength="moderate"),
        }
        evidence_map = EvidenceMap(
            themes=[],
            consensus_claims=[
                ConsensusClaim(
                    claim="X is true",
                    supporting_paper_ids=["p1", "p2"],
                    strength="strong",
                    evidence_count=2,
                )
            ],
            contradictions=[],
        )

        builder = EvidenceChainBuilder(llm=None)
        bundle = await builder.build_section_directives(
            section_id="s1",
            section_paper_ids=["p1", "p2"],
            evidence_map=evidence_map,
            extractions=extractions,
            paper_years={"p1": 2020, "p2": 2021},
        )

        assert bundle.section_id == "s1"
        assert bundle.strength_distribution is not None
        assert bundle.strength_distribution.strong == 1
        assert "X is true" in bundle.consensus_claims_text
        assert "Evidence Profile" in bundle.directive_text

    @pytest.mark.asyncio
    async def test_with_evidence_chains(self):
        extractions = {
            "p1": _make_extraction("p1"),
            "p2": _make_extraction("p2"),
        }
        evidence_map = EvidenceMap(
            themes=[],
            consensus_claims=[],
            contradictions=[],
            evidence_chains=[
                EvidenceChain(
                    chain_id="c1",
                    theme="theme1",
                    description="Progressive understanding",
                    chain_type="progressive",
                    paper_ids=["p1", "p2"],
                ).model_dump()
            ],
        )

        builder = EvidenceChainBuilder(llm=None)
        bundle = await builder.build_section_directives(
            section_id="s1",
            section_paper_ids=["p1", "p2"],
            evidence_map=evidence_map,
            extractions=extractions,
            paper_years={},
        )

        assert len(bundle.evidence_chains) == 1
        assert "Evidence Chains" in bundle.directive_text


class TestEnrichedContradiction:
    @pytest.mark.asyncio
    async def test_has_framing_strategy(self):
        """Test that enriched contradictions include framing strategy via mock LLM."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.parsed = MagicMock()
        mock_response.parsed.methodological_differences = "Different sample sizes"
        mock_response.parsed.framing_strategy = "The discrepancy reflects population differences"
        mock_llm.generate_structured = AsyncMock(return_value=mock_response)

        builder = EvidenceChainBuilder(mock_llm)
        contradictions = [
            Contradiction(
                claim_a="X causes Y",
                claim_b="X does not cause Y",
                paper_ids_a=["p1"],
                paper_ids_b=["p2"],
            )
        ]
        extractions = {
            "p1": _make_extraction("p1", methods="RCT, n=100"),
            "p2": _make_extraction("p2", methods="Observational, n=50"),
        }

        enriched = await builder.enrich_contradictions(contradictions, extractions)
        assert len(enriched) == 1
        assert enriched[0].framing_strategy == "The discrepancy reflects population differences"
        assert enriched[0].methodological_differences == "Different sample sizes"
