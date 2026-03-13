from __future__ import annotations

from autoreview.analysis.clustering import ThematicClusterer, _format_findings_for_clustering
from autoreview.analysis.evidence_map import (
    EvidenceMap,
    Theme,
)
from autoreview.analysis.gap_detector import GapDetector
from autoreview.extraction.models import EvidenceStrength, Finding, PaperExtraction
from autoreview.llm.prompts.clustering import (
    ClusteringResult,
    ConsensusClaimResult,
    ContradictionItem,
    ContradictionResult,
    GapAnalysisResult,
    GapItem,
    SubThemeCluster,
    ThemeCluster,
)
from autoreview.llm.provider import LLMStructuredResponse


class MockClusteringLLM:
    """Mock LLM for clustering tests."""

    async def generate_structured(
        self,
        prompt,
        response_model,
        system="",
        max_tokens=4096,
        temperature=0.0,
        model_override=None,
    ):
        if response_model == ClusteringResult:
            return LLMStructuredResponse(
                parsed=ClusteringResult(
                    themes=[
                        ThemeCluster(
                            name="Gut-Brain Axis",
                            description="Mechanisms linking gut to brain",
                            paper_ids=["p1", "p2"],
                            sub_themes=[
                                SubThemeCluster(
                                    name="Vagal Pathways",
                                    description="Via vagus nerve",
                                    paper_ids=["p1"],
                                ),
                            ],
                        ),
                        ThemeCluster(
                            name="Microbiome Composition",
                            description="Changes in microbial composition",
                            paper_ids=["p2", "p3"],
                        ),
                    ]
                ),
                input_tokens=800,
                output_tokens=400,
            )
        elif response_model == ContradictionResult:
            return LLMStructuredResponse(
                parsed=ContradictionResult(
                    consensus_claims=[
                        ConsensusClaimResult(
                            claim="Gut dysbiosis is common in PD",
                            supporting_paper_ids=["p1", "p2"],
                            strength="moderate",
                            evidence_count=2,
                        ),
                    ],
                    contradictions=[
                        ContradictionItem(
                            claim_a="Diversity decreases",
                            claim_b="Diversity unchanged",
                            paper_ids_a=["p1"],
                            paper_ids_b=["p2"],
                            possible_explanation="Different disease stages",
                        ),
                    ],
                ),
                input_tokens=600,
                output_tokens=300,
            )
        elif response_model == GapAnalysisResult:
            return LLMStructuredResponse(
                parsed=GapAnalysisResult(
                    gaps=[
                        GapItem(
                            expected_topic="Viral microbiome",
                            current_coverage="No papers",
                            severity="major",
                            suggested_queries=["viral metagenomics neurodegeneration"],
                        ),
                    ],
                    coverage_score=0.75,
                ),
                input_tokens=400,
                output_tokens=200,
            )
        raise ValueError(f"Unexpected: {response_model}")


def _make_extractions() -> dict[str, PaperExtraction]:
    return {
        "p1": PaperExtraction(
            paper_id="p1",
            key_findings=[
                Finding(claim="Claim A", evidence_strength=EvidenceStrength.STRONG, paper_id="p1")
            ],
            methods_summary="Methods A",
            limitations="Limits A",
        ),
        "p2": PaperExtraction(
            paper_id="p2",
            key_findings=[
                Finding(claim="Claim B", evidence_strength=EvidenceStrength.MODERATE, paper_id="p2")
            ],
            methods_summary="Methods B",
            limitations="Limits B",
        ),
        "p3": PaperExtraction(
            paper_id="p3",
            key_findings=[
                Finding(claim="Claim C", evidence_strength=EvidenceStrength.WEAK, paper_id="p3")
            ],
            methods_summary="Methods C",
            limitations="Limits C",
        ),
    }


class TestFormatFindings:
    def test_formats_all_papers(self):
        extractions = _make_extractions()
        text = _format_findings_for_clustering(extractions)
        assert "p1" in text
        assert "Claim A" in text
        assert "strong" in text.lower()


class TestThematicClusterer:
    async def test_cluster(self):
        llm = MockClusteringLLM()
        clusterer = ThematicClusterer(llm)
        themes = await clusterer.cluster(_make_extractions(), "Test scope")
        assert len(themes) == 2
        assert themes[0].name == "Gut-Brain Axis"
        assert len(themes[0].sub_themes) == 1

    async def test_detect_contradictions(self):
        llm = MockClusteringLLM()
        clusterer = ThematicClusterer(llm)
        themes = [
            Theme(name="Test Theme", description="Desc", paper_ids=["p1", "p2"]),
        ]
        consensus, contradictions = await clusterer.detect_contradictions(
            themes, _make_extractions()
        )
        assert len(consensus) == 1
        assert consensus[0].claim == "Gut dysbiosis is common in PD"
        assert len(contradictions) == 1
        assert contradictions[0].theme == "Test Theme"

    async def test_build_evidence_map(self):
        llm = MockClusteringLLM()
        clusterer = ThematicClusterer(llm)
        em = await clusterer.build_evidence_map(_make_extractions(), "Test scope")
        assert isinstance(em, EvidenceMap)
        assert len(em.themes) == 2
        assert len(em.consensus_claims) >= 1
        assert "p1" in em.paper_theme_mapping


class TestGapDetector:
    async def test_detect_gaps(self):
        llm = MockClusteringLLM()
        detector = GapDetector(llm)
        themes = [Theme(name="Theme A", description="Desc", paper_ids=["p1"])]
        gaps, score = await detector.detect_gaps(themes, "Test scope")
        assert len(gaps) == 1
        assert gaps[0].expected_topic == "Viral microbiome"
        assert gaps[0].severity == "major"
        assert score == 0.75
