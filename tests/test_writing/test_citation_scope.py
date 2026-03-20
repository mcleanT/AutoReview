from __future__ import annotations

import pytest

from autoreview.writing.citation_scope import validate_citation_scope


def test_all_citations_in_scope():
    result = validate_citation_scope(
        cited_ids=["p1", "p2"],
        assigned_ids=["p1", "p2", "p3"],
    )
    assert result.all_in_scope is True
    assert result.out_of_scope == []


def test_out_of_scope_citations_detected():
    result = validate_citation_scope(
        cited_ids=["p1", "p99"],
        assigned_ids=["p1", "p2"],
    )
    assert result.all_in_scope is False
    assert "p99" in result.out_of_scope


def test_empty_citations_passes():
    result = validate_citation_scope(
        cited_ids=[],
        assigned_ids=["p1", "p2"],
    )
    assert result.all_in_scope is True
    assert result.out_of_scope == []


def test_uncited_assigned_papers_reported():
    result = validate_citation_scope(
        cited_ids=["p1"],
        assigned_ids=["p1", "p2", "p3"],
    )
    assert "p2" in result.uncited_assigned
    assert "p3" in result.uncited_assigned
    assert "p1" not in result.uncited_assigned


def test_citation_utilization_rate():
    result = validate_citation_scope(
        cited_ids=["p1", "p2"],
        assigned_ids=["p1", "p2", "p3", "p4"],
    )
    assert result.utilization_rate == pytest.approx(0.5)
