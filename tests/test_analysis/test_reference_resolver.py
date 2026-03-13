from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from autoreview.models.paper import CandidatePaper
from paper.analysis.reference_resolver import (
    ResolutionConfidence,
    resolve_reference,
)


def _make_paper(title: str = "Test Paper", doi: str = "10.1234/test") -> CandidatePaper:
    return CandidatePaper(
        title=title,
        authors=["Smith J"],
        year=2020,
        source_database="semantic_scholar",
        doi=doi,
    )


class TestResolveReference:
    @pytest.mark.asyncio
    async def test_resolves_by_doi(self):
        mock_s2 = AsyncMock()
        mock_s2.get_paper_details = AsyncMock(return_value=_make_paper())

        result = await resolve_reference(
            parsed={"raw": "[1] ...", "doi": "10.1234/test", "title": "Test Paper"},
            search_sources=[mock_s2],
        )
        assert result is not None
        assert result.confidence == ResolutionConfidence.HIGH
        assert result.paper.doi == "10.1234/test"
        mock_s2.get_paper_details.assert_called_once_with("DOI:10.1234/test")

    @pytest.mark.asyncio
    async def test_falls_back_to_title_search(self):
        mock_s2 = AsyncMock()
        mock_s2.get_paper_details = AsyncMock(return_value=None)
        mock_s2.source_name = "semantic_scholar"
        mock_s2.search = AsyncMock(return_value=[_make_paper(title="Test Paper Exact")])

        result = await resolve_reference(
            parsed={"raw": "[1] ...", "doi": None, "title": "Test Paper Exact"},
            search_sources=[mock_s2],
        )
        assert result is not None
        assert result.confidence in (ResolutionConfidence.MEDIUM, ResolutionConfidence.HIGH)

    @pytest.mark.asyncio
    async def test_returns_none_when_unresolvable(self):
        mock_s2 = AsyncMock()
        mock_s2.get_paper_details = AsyncMock(return_value=None)
        mock_s2.source_name = "semantic_scholar"
        mock_s2.search = AsyncMock(return_value=[])

        result = await resolve_reference(
            parsed={"raw": "[1] ...", "doi": None, "title": "Nonexistent Paper"},
            search_sources=[mock_s2],
        )
        assert result is None


class TestResolutionConfidence:
    def test_high_for_doi_match(self):
        assert ResolutionConfidence.HIGH.value == "high"

    def test_medium_for_fuzzy_match(self):
        assert ResolutionConfidence.MEDIUM.value == "medium"

    def test_low_for_weak_match(self):
        assert ResolutionConfidence.LOW.value == "low"
