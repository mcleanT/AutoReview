from unittest.mock import MagicMock

from autoreview.analysis.evidence_map import EvidenceMap
from autoreview.config.depth import (
    DepthProfile,
    EvidenceWeightedAllocator,
    classify_section_type,
    get_depth_instructions,
    get_depth_profile,
)
from autoreview.config.models import DepthLevel, WritingConfig
from autoreview.llm.prompts.outline import OutlineSection, ReviewOutline


def test_depth_level_values():
    assert DepthLevel.LOW == "low"
    assert DepthLevel.MEDIUM == "medium"
    assert DepthLevel.DEEP == "deep"


def test_writing_config_default_depth():
    config = WritingConfig()
    assert config.depth == DepthLevel.MEDIUM


def test_writing_config_accepts_depth():
    config = WritingConfig(depth=DepthLevel.DEEP)
    assert config.depth == DepthLevel.DEEP


def test_writing_config_depth_serialization():
    """Verify DepthLevel survives Pydantic model_dump/model_validate roundtrip (extra='forbid')."""
    config = WritingConfig(depth=DepthLevel.LOW)
    dumped = config.model_dump()
    assert dumped["depth"] == "low"
    restored = WritingConfig.model_validate(dumped)
    assert restored.depth == DepthLevel.LOW


def test_get_depth_profile_returns_profile():
    profile = get_depth_profile(DepthLevel.LOW)
    assert isinstance(profile, DepthProfile)


def test_low_profile_values():
    p = get_depth_profile(DepthLevel.LOW)
    assert p.base_word_multiplier == 0.6
    assert p.key_insights_range == (2, 3)
    assert p.evidence_chain_detail == "critical_only"
    assert p.total_word_budget == 4000
    assert p.min_section_words == 200
    assert p.max_tokens_override is None


def test_medium_profile_values():
    p = get_depth_profile(DepthLevel.MEDIUM)
    assert p.base_word_multiplier == 1.0
    assert p.key_insights_range == (3, 5)
    assert p.evidence_chain_detail == "standard"
    assert p.total_word_budget == 8000
    assert p.min_section_words == 400
    assert p.max_tokens_override is None


def test_deep_profile_values():
    p = get_depth_profile(DepthLevel.DEEP)
    assert p.base_word_multiplier == 2.5
    assert p.key_insights_range == (7, 10)
    assert p.evidence_chain_detail == "exhaustive"
    assert p.total_word_budget == 25000
    assert p.min_section_words == 600
    assert p.max_tokens_override == 16384


def test_deep_profile_has_higher_dampening_than_low():
    low = get_depth_profile(DepthLevel.LOW)
    deep = get_depth_profile(DepthLevel.DEEP)
    assert deep.section_type_dampening["introduction"] > low.section_type_dampening["introduction"]
    assert deep.section_type_dampening["conclusion"] > low.section_type_dampening["conclusion"]


def test_all_profiles_have_body_dampening_of_one():
    for level in DepthLevel:
        p = get_depth_profile(level)
        assert p.section_type_dampening["body"] == 1.0


def test_depth_instructions_low():
    text = get_depth_instructions(DepthLevel.LOW, 300)
    assert "critical findings" in text.lower()
    assert "300" in text


def test_depth_instructions_medium():
    text = get_depth_instructions(DepthLevel.MEDIUM, 800)
    assert "thoroughness" in text.lower() or "readability" in text.lower()
    assert "800" in text


def test_depth_instructions_deep():
    text = get_depth_instructions(DepthLevel.DEEP, 2000)
    assert "exhaustive" in text.lower()
    assert "2000" in text


def test_classify_introduction():
    assert classify_section_type("Introduction") == "introduction"
    assert classify_section_type("Background and Introduction") == "introduction"
    assert classify_section_type("1. Background") == "introduction"


def test_classify_conclusion():
    assert classify_section_type("Conclusion") == "conclusion"
    assert classify_section_type("Concluding Remarks") == "conclusion"
    assert classify_section_type("Summary and Conclusions") == "conclusion"


def test_classify_methods():
    assert classify_section_type("Methods of Review") == "methods"
    assert classify_section_type("Search Strategy") == "methods"
    assert classify_section_type("Review Methodology") == "methods"


def test_classify_body_default():
    assert classify_section_type("Deep Learning Architectures") == "body"
    assert classify_section_type("Results and Discussion") == "body"
    assert classify_section_type("Future Directions") == "body"


# ---------------------------------------------------------------------------
# EvidenceWeightedAllocator tests
# ---------------------------------------------------------------------------


def _make_outline(sections: list[dict]) -> ReviewOutline:
    outline_sections = []
    for s in sections:
        outline_sections.append(
            OutlineSection(
                id=s["id"],
                title=s["title"],
                description=s.get("description", "Test section"),
                paper_ids=s.get("paper_ids", []),
                theme_refs=s.get("theme_refs", []),
            )
        )
    return ReviewOutline(title="Test Review", sections=outline_sections)


def _make_evidence_map(chains: list[dict] | None = None) -> EvidenceMap:
    em = EvidenceMap(themes=[], consensus_claims=[], contradictions=[], gaps=[])
    em.evidence_chains = chains or []
    return em


def _make_extractions(paper_findings: dict[str, int]) -> dict:
    extractions = {}
    for pid, n_findings in paper_findings.items():
        mock = MagicMock()
        mock.key_findings = [f"finding_{i}" for i in range(n_findings)]
        extractions[pid] = mock
    return extractions


def test_allocator_basic_proportional():
    outline = _make_outline(
        [
            {"id": "s1", "title": "Topic A", "paper_ids": ["p1"]},
            {"id": "s2", "title": "Topic B", "paper_ids": ["p1", "p2", "p3"]},
        ]
    )
    extractions = _make_extractions({"p1": 2, "p2": 3, "p3": 1})
    evidence_map = _make_evidence_map()
    profile = get_depth_profile(DepthLevel.MEDIUM)
    allocator = EvidenceWeightedAllocator(profile)
    allocator.allocate(outline, evidence_map, extractions)
    assert outline.sections[1].estimated_word_count > outline.sections[0].estimated_word_count


def test_allocator_respects_floor():
    outline = _make_outline(
        [
            {"id": "s1", "title": "Tiny Topic", "paper_ids": ["p1"]},
            {"id": "s2", "title": "Big Topic", "paper_ids": ["p1", "p2", "p3", "p4", "p5"]},
        ]
    )
    extractions = _make_extractions({"p1": 1, "p2": 5, "p3": 5, "p4": 5, "p5": 5})
    evidence_map = _make_evidence_map()
    profile = get_depth_profile(DepthLevel.MEDIUM)
    allocator = EvidenceWeightedAllocator(profile)
    allocator.allocate(outline, evidence_map, extractions)
    assert outline.sections[0].estimated_word_count >= profile.min_section_words


def test_allocator_dampens_introduction():
    outline = _make_outline(
        [
            {"id": "s1", "title": "Introduction", "paper_ids": ["p1", "p2"]},
            {"id": "s2", "title": "Core Topic", "paper_ids": ["p1", "p2"]},
        ]
    )
    extractions = _make_extractions({"p1": 3, "p2": 3})
    evidence_map = _make_evidence_map()
    profile = get_depth_profile(DepthLevel.MEDIUM)
    allocator = EvidenceWeightedAllocator(profile)
    allocator.allocate(outline, evidence_map, extractions)
    assert outline.sections[0].estimated_word_count < outline.sections[1].estimated_word_count


def test_allocator_zero_evidence_section_gets_fixed_allocation():
    extractions = _make_extractions({"p1": 3, "p2": 3})
    evidence_map = _make_evidence_map()
    for level in DepthLevel:
        outline_copy = _make_outline(
            [
                {"id": "s1", "title": "Core Topic", "paper_ids": ["p1", "p2"]},
                {"id": "s2", "title": "Future Directions", "paper_ids": []},
            ]
        )
        profile = get_depth_profile(level)
        allocator = EvidenceWeightedAllocator(profile)
        allocator.allocate(outline_copy, evidence_map, extractions)
        expected_fixed = int(profile.base_word_multiplier * 500)
        assert outline_copy.sections[1].estimated_word_count == expected_fixed


def test_allocator_evidence_chains_increase_density():
    outline = _make_outline(
        [
            {"id": "s1", "title": "Topic A", "paper_ids": ["p1"]},
            {"id": "s2", "title": "Topic B", "paper_ids": ["p2"]},
        ]
    )
    extractions = _make_extractions({"p1": 2, "p2": 2})
    chains = [{"paper_ids": ["p2", "p3"], "chain_id": "c1"}]
    evidence_map = _make_evidence_map(chains=chains)
    profile = get_depth_profile(DepthLevel.MEDIUM)
    allocator = EvidenceWeightedAllocator(profile)
    allocator.allocate(outline, evidence_map, extractions)
    assert outline.sections[1].estimated_word_count > outline.sections[0].estimated_word_count


def test_allocator_depth_scales_output():
    outline_low = _make_outline([{"id": "s1", "title": "Topic", "paper_ids": ["p1", "p2"]}])
    outline_deep = _make_outline([{"id": "s1", "title": "Topic", "paper_ids": ["p1", "p2"]}])
    extractions = _make_extractions({"p1": 3, "p2": 3})
    evidence_map = _make_evidence_map()
    EvidenceWeightedAllocator(get_depth_profile(DepthLevel.LOW)).allocate(
        outline_low, evidence_map, extractions
    )
    EvidenceWeightedAllocator(get_depth_profile(DepthLevel.DEEP)).allocate(
        outline_deep, evidence_map, extractions
    )
    assert (
        outline_deep.sections[0].estimated_word_count > outline_low.sections[0].estimated_word_count
    )


# ---------------------------------------------------------------------------
# EXHAUSTIVE depth level and citation field tests
# ---------------------------------------------------------------------------


def test_exhaustive_depth_level():
    from autoreview.config.models import DepthLevel

    assert DepthLevel.EXHAUSTIVE == "exhaustive"


def test_exhaustive_depth_profile():
    from autoreview.config.depth import get_depth_profile
    from autoreview.config.models import DepthLevel

    profile = get_depth_profile(DepthLevel.EXHAUSTIVE)
    assert profile.total_word_budget == 40000
    assert profile.citation_density == "exhaustive"
    assert profile.target_citations_per_1k_words == 16.0
    assert profile.min_total_citations == 300


def test_medium_depth_citation_fields():
    from autoreview.config.depth import get_depth_profile
    from autoreview.config.models import DepthLevel

    profile = get_depth_profile(DepthLevel.MEDIUM)
    assert profile.citation_density == "standard"
    assert profile.target_citations_per_1k_words == 9.0
    assert profile.min_total_citations == 75


def test_exhaustive_depth_instructions():
    from autoreview.config.depth import get_depth_instructions
    from autoreview.config.models import DepthLevel

    result = get_depth_instructions(DepthLevel.EXHAUSTIVE, 40000)
    assert "40000" in result
