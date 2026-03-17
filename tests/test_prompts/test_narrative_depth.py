from autoreview.analysis.evidence_map import EvidenceMap
from autoreview.config.models import DepthLevel
from autoreview.llm.prompts.narrative import build_narrative_planning_prompt
from autoreview.llm.prompts.outline import OutlineSection, ReviewOutline


def _make_test_outline() -> ReviewOutline:
    return ReviewOutline(
        title="Test",
        sections=[
            OutlineSection(
                id="s1",
                title="Topic",
                description="Desc",
                paper_ids=["p1"],
                theme_refs=[],
            ),
        ],
    )


def _make_test_evidence_map() -> EvidenceMap:
    return EvidenceMap(
        themes=[],
        consensus_claims=[],
        contradictions=[],
        gaps=[],
    )


def test_narrative_prompt_low_depth_key_insights_range():
    prompt = build_narrative_planning_prompt(
        outline=_make_test_outline(),
        evidence_map=_make_test_evidence_map(),
        scope_document="Test scope",
        depth=DepthLevel.LOW,
    )
    # Low depth: key_insights_range is (2, 3)
    assert "2\u20133" in prompt or "2-3" in prompt


def test_narrative_prompt_deep_depth_key_insights_range():
    prompt = build_narrative_planning_prompt(
        outline=_make_test_outline(),
        evidence_map=_make_test_evidence_map(),
        scope_document="Test scope",
        depth=DepthLevel.DEEP,
    )
    # Deep depth: key_insights_range is (7, 10)
    assert "7\u201310" in prompt or "7-10" in prompt


def test_narrative_prompt_no_depth_keeps_default():
    prompt = build_narrative_planning_prompt(
        outline=_make_test_outline(),
        evidence_map=_make_test_evidence_map(),
        scope_document="Test scope",
    )
    # Default: "3–5" (current behavior)
    assert "3\u20135" in prompt or "3-5" in prompt
