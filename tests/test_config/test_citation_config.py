from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_citation_config_defaults():
    from autoreview.config.citation import CitationConfig

    cfg = CitationConfig()
    assert cfg.citation_density == "standard"
    assert cfg.selection_strategy == "balanced"
    assert cfg.paper_tiers_enabled is True
    assert cfg.min_citations_per_section == 8
    assert cfg.w_evidence_strength == 0.30


def test_citation_config_weights_must_sum_to_one():
    from autoreview.config.citation import CitationConfig

    with pytest.raises(ValidationError, match="must sum to 1.0"):
        CitationConfig(w_evidence_strength=0.5, w_recency=0.5, w_relevance_score=0.5)


def test_citation_config_weights_valid():
    from autoreview.config.citation import CitationConfig

    cfg = CitationConfig(
        w_evidence_strength=0.40,
        w_recency=0.20,
        w_relevance_score=0.20,
        w_uniqueness=0.10,
        w_source_diversity=0.10,
    )
    assert cfg.w_evidence_strength == 0.40


def test_citation_config_extra_forbid():
    from autoreview.config.citation import CitationConfig

    with pytest.raises(ValidationError):
        CitationConfig(nonexistent_field="bad")


def test_reference_match_config_defaults():
    from autoreview.config.citation import ReferenceMatchConfig

    cfg = ReferenceMatchConfig()
    assert cfg.enabled is False
    assert cfg.reference_path is None
    assert cfg.word_count_tolerance == 0.15


def test_reference_match_config_extra_forbid():
    from autoreview.config.citation import ReferenceMatchConfig

    with pytest.raises(ValidationError):
        ReferenceMatchConfig(nonexistent_field="bad")


def test_writing_config_has_citation():
    from autoreview.config.models import WritingConfig

    wc = WritingConfig()
    assert hasattr(wc, "citation")
    assert wc.citation.citation_density == "standard"


def test_outline_config_has_model_fields():
    from autoreview.config.models import OutlineConfig

    oc = OutlineConfig()
    assert oc.draft_model == "haiku"
    assert oc.final_model == "sonnet"


def test_critique_config_has_corpus_utilization():
    from autoreview.config.models import CritiqueConfig

    cc = CritiqueConfig()
    assert cc.target_corpus_utilization == 0.25
