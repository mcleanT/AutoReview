"""Test evidence_chains config flag."""

from __future__ import annotations

from autoreview.config.models import WritingConfig


def test_evidence_chains_default_true() -> None:
    config = WritingConfig()
    assert config.evidence_chains is True


def test_evidence_chains_can_disable() -> None:
    config = WritingConfig(evidence_chains=False)
    assert config.evidence_chains is False
