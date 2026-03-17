from autoreview.config.depth import DepthProfile, get_depth_instructions, get_depth_profile
from autoreview.config.models import DepthLevel, WritingConfig


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
