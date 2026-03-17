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
