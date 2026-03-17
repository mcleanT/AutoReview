"""Integration test: verify that depth levels produce different word allocations."""

from unittest.mock import MagicMock

from autoreview.analysis.evidence_map import EvidenceMap
from autoreview.config.depth import EvidenceWeightedAllocator, get_depth_profile
from autoreview.config.models import DepthLevel
from autoreview.llm.prompts.outline import OutlineSection, ReviewOutline


def _make_realistic_outline() -> ReviewOutline:
    return ReviewOutline(
        title="Test Review",
        sections=[
            OutlineSection(
                id="s1",
                title="Introduction",
                description="Intro",
                paper_ids=["p1", "p2"],
                theme_refs=[],
            ),
            OutlineSection(
                id="s2",
                title="Methods of Review",
                description="Search strategy",
                paper_ids=["p1"],
                theme_refs=[],
            ),
            OutlineSection(
                id="s3",
                title="Deep Learning Architectures",
                description="DL overview",
                paper_ids=["p1", "p2", "p3", "p4"],
                theme_refs=[],
            ),
            OutlineSection(
                id="s4",
                title="Training Strategies",
                description="Training",
                paper_ids=["p2", "p3", "p5"],
                theme_refs=[],
            ),
            OutlineSection(
                id="s5",
                title="Applications and Results",
                description="Applications",
                paper_ids=["p1", "p3", "p4", "p5", "p6"],
                theme_refs=[],
            ),
            OutlineSection(
                id="s6",
                title="Conclusion",
                description="Summary",
                paper_ids=["p1", "p2"],
                theme_refs=[],
            ),
        ],
    )


def _make_realistic_extractions() -> dict:
    extractions = {}
    findings_per_paper = {"p1": 3, "p2": 4, "p3": 5, "p4": 2, "p5": 3, "p6": 4}
    for pid, n in findings_per_paper.items():
        mock = MagicMock()
        mock.key_findings = [f"finding_{i}" for i in range(n)]
        extractions[pid] = mock
    return extractions


def _make_realistic_evidence_map() -> EvidenceMap:
    em = EvidenceMap(themes=[], consensus_claims=[], contradictions=[], gaps=[])
    em.evidence_chains = [
        {"paper_ids": ["p1", "p2", "p3"], "chain_id": "c1"},
        {"paper_ids": ["p3", "p4", "p5"], "chain_id": "c2"},
        {"paper_ids": ["p5", "p6"], "chain_id": "c3"},
    ]
    return em


def test_depth_levels_produce_increasing_total_words():
    """low < medium < deep total word count."""
    extractions = _make_realistic_extractions()
    evidence_map = _make_realistic_evidence_map()
    totals = {}

    for level in DepthLevel:
        outline = _make_realistic_outline()
        profile = get_depth_profile(level)
        allocator = EvidenceWeightedAllocator(profile)
        allocator.allocate(outline, evidence_map, extractions)
        totals[level] = sum(s.estimated_word_count for s in outline.sections)

    assert totals[DepthLevel.LOW] < totals[DepthLevel.MEDIUM]
    assert totals[DepthLevel.MEDIUM] < totals[DepthLevel.DEEP]


def test_introduction_dampened_relative_to_body():
    """Introduction gets fewer words than a body section with same evidence."""
    outline = _make_realistic_outline()
    extractions = _make_realistic_extractions()
    evidence_map = _make_realistic_evidence_map()
    profile = get_depth_profile(DepthLevel.MEDIUM)

    allocator = EvidenceWeightedAllocator(profile)
    allocator.allocate(outline, evidence_map, extractions)

    intro = outline.sections[0]  # Introduction
    body_sections = [
        s
        for s in outline.sections
        if s.title not in ("Introduction", "Methods of Review", "Conclusion")
    ]
    assert any(s.estimated_word_count > intro.estimated_word_count for s in body_sections)


def test_all_sections_above_floor():
    """Every section meets the minimum word count for its depth level."""
    for level in DepthLevel:
        outline = _make_realistic_outline()
        extractions = _make_realistic_extractions()
        evidence_map = _make_realistic_evidence_map()
        profile = get_depth_profile(level)

        allocator = EvidenceWeightedAllocator(profile)
        allocator.allocate(outline, evidence_map, extractions)

        for section in outline.sections:
            if section.paper_ids:
                assert section.estimated_word_count >= profile.min_section_words, (
                    f"{level}: {section.title} has {section.estimated_word_count} words, "
                    f"below floor of {profile.min_section_words}"
                )


def test_budget_overflow_floors_preserved():
    """When many sections hit the floor, floors are never violated (budget is soft target)."""
    sections = [
        OutlineSection(
            id=f"s{i}", title=f"Topic {i}", description="Desc", paper_ids=[f"p{i}"], theme_refs=[]
        )
        for i in range(10)
    ]
    outline = ReviewOutline(title="Test Review", sections=sections)

    extractions = {}
    for i in range(10):
        mock = MagicMock()
        mock.key_findings = ["finding_0"]
        extractions[f"p{i}"] = mock

    evidence_map = EvidenceMap(themes=[], consensus_claims=[], contradictions=[], gaps=[])
    evidence_map.evidence_chains = []

    profile = get_depth_profile(DepthLevel.LOW)
    allocator = EvidenceWeightedAllocator(profile)
    allocator.allocate(outline, evidence_map, extractions)

    for section in outline.sections:
        assert section.estimated_word_count >= profile.min_section_words
