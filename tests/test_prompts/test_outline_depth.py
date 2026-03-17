from autoreview.config.models import DepthLevel
from autoreview.llm.prompts.outline import build_outline_prompt


def test_outline_prompt_low_depth():
    prompt = build_outline_prompt(
        scope_document="Test scope",
        evidence_summary="Test evidence",
        required_sections=["Introduction", "Discussion"],
        depth=DepthLevel.LOW,
    )
    assert "DEPTH GUIDANCE" in prompt
    assert "critical" in prompt.lower() or "key findings" in prompt.lower()


def test_outline_prompt_deep_depth():
    prompt = build_outline_prompt(
        scope_document="Test scope",
        evidence_summary="Test evidence",
        required_sections=["Introduction", "Discussion"],
        depth=DepthLevel.DEEP,
    )
    assert "DEPTH GUIDANCE" in prompt
    assert "exhaustive" in prompt.lower() or "methodological" in prompt.lower()


def test_outline_prompt_no_depth_backwards_compatible():
    prompt = build_outline_prompt(
        scope_document="Test scope",
        evidence_summary="Test evidence",
        required_sections=["Introduction", "Discussion"],
    )
    assert "DEPTH GUIDANCE" not in prompt
