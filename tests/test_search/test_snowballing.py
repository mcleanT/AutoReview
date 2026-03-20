"""Tests for backward/forward citation snowballing."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from autoreview.models.paper import CandidatePaper
from autoreview.search.snowballing import (
    SnowballConfig,
    _normalize_title,
    _s2_paper_to_candidate,
    snowball_papers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PAPER_FIELDS = "paperId,title,abstract,year,authors,journal,externalIds,citationCount"


def _make_s2_paper(
    paper_id: str = "s2abc",
    title: str = "Test Paper",
    doi: str | None = "10.1/test",
    year: int = 2023,
    authors: list[str] | None = None,
) -> dict:
    if authors is None:
        authors = ["Alice A", "Bob B"]
    return {
        "paperId": paper_id,
        "title": title,
        "abstract": "An abstract.",
        "year": year,
        "authors": [{"name": n} for n in authors],
        "journal": {"name": "Nature"},
        "externalIds": {"DOI": doi} if doi else {},
        "citationCount": 5,
    }


def _make_seed(
    title: str = "Seed Paper",
    doi: str | None = "10.1/seed",
    s2_id: str | None = "seedS2",
) -> CandidatePaper:
    external_ids: dict[str, str] = {}
    if s2_id:
        external_ids["s2_id"] = s2_id
    return CandidatePaper(
        title=title,
        authors=["Author A"],
        year=2020,
        doi=doi,
        source_database="semantic_scholar",
        external_ids=external_ids,
    )


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


class TestNormalizeTitle:
    def test_lowercases(self):
        assert _normalize_title("Hello World") == "hello world"

    def test_strips_punctuation(self):
        assert _normalize_title("Hello, World!") == "hello world"

    def test_collapses_whitespace(self):
        assert _normalize_title("  lots   of   spaces  ") == "lots of spaces"

    def test_strips_accents(self):
        # é → e after NFKD + ascii
        assert _normalize_title("Café") == "cafe"

    def test_identical_after_normalization(self):
        a = _normalize_title("Machine Learning: A Review")
        b = _normalize_title("machine learning a review")
        assert a == b


class TestS2PaperToCandidate:
    def test_valid_paper(self):
        raw = _make_s2_paper(doi="10.1/x")
        paper = _s2_paper_to_candidate(raw, "snowball_backward")
        assert paper is not None
        assert paper.title == "Test Paper"
        assert paper.source_database == "snowball_backward"
        assert paper.doi == "10.1/x"
        assert paper.external_ids["s2_id"] == "s2abc"
        assert paper.citation_count == 5

    def test_missing_title_returns_none(self):
        raw = _make_s2_paper()
        raw["title"] = None
        assert _s2_paper_to_candidate(raw, "snowball_forward") is None

    def test_no_doi(self):
        raw = _make_s2_paper(doi=None)
        paper = _s2_paper_to_candidate(raw, "snowball_backward")
        assert paper is not None
        assert paper.doi is None

    def test_source_label_forward(self):
        raw = _make_s2_paper()
        paper = _s2_paper_to_candidate(raw, "snowball_forward")
        assert paper is not None
        assert paper.source_database == "snowball_forward"


# ---------------------------------------------------------------------------
# Integration-style tests for snowball_papers (mock _fetch_related)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSnowballPapers:
    async def test_returns_new_candidates(self):
        """snowball_papers returns CandidatePapers from both backward and forward passes."""
        ref_paper = _make_s2_paper(paper_id="ref1", title="Referenced Paper", doi="10.1/ref1")
        cit_paper = _make_s2_paper(paper_id="cit1", title="Citing Paper", doi="10.1/cit1")

        # _fetch_related is called twice per seed: once for references, once for citations
        async def fake_fetch(paper_id, endpoint, limit, client, limiter):
            if endpoint == "references":
                return [ref_paper]
            elif endpoint == "citations":
                return [cit_paper]
            return []

        seed = _make_seed()
        config = SnowballConfig(max_seed_papers=5, enable_backward=True, enable_forward=True)

        with patch(
            "autoreview.search.snowballing._fetch_related",
            side_effect=fake_fetch,
        ):
            results = await snowball_papers([seed], config=config)

        titles = {p.title for p in results}
        assert "Referenced Paper" in titles
        assert "Citing Paper" in titles
        assert len(results) == 2

    async def test_deduplicates_against_existing(self):
        """Papers whose DOIs are in existing_dois are excluded from results."""
        ref_paper = _make_s2_paper(paper_id="ref1", title="Referenced Paper", doi="10.1/ref1")

        async def fake_fetch(paper_id, endpoint, limit, client, limiter):
            if endpoint == "references":
                return [ref_paper]
            return []

        seed = _make_seed()
        config = SnowballConfig(enable_backward=True, enable_forward=False)

        with patch(
            "autoreview.search.snowballing._fetch_related",
            side_effect=fake_fetch,
        ):
            results = await snowball_papers(
                [seed],
                config=config,
                existing_dois={"10.1/ref1"},
            )

        assert results == [], "Already-known DOI should be filtered out"

    async def test_deduplicates_by_title_when_no_doi(self):
        """Papers without DOIs are deduped by normalized title."""
        ref_paper = _make_s2_paper(paper_id="ref1", title="No DOI Paper", doi=None)

        async def fake_fetch(paper_id, endpoint, limit, client, limiter):
            if endpoint == "references":
                return [ref_paper]
            return []

        seed = _make_seed()
        config = SnowballConfig(enable_backward=True, enable_forward=False)

        with patch(
            "autoreview.search.snowballing._fetch_related",
            side_effect=fake_fetch,
        ):
            results = await snowball_papers(
                [seed],
                config=config,
                existing_titles={"no doi paper"},
            )

        assert results == [], "Already-known title should be filtered out"

    async def test_limits_seed_papers(self):
        """Only max_seed_papers seeds are processed."""
        call_count = 0

        async def fake_fetch(paper_id, endpoint, limit, client, limiter):
            nonlocal call_count
            call_count += 1
            return []

        seeds = [_make_seed(title=f"Seed {i}", doi=f"10.1/s{i}") for i in range(10)]
        config = SnowballConfig(max_seed_papers=3, enable_backward=True, enable_forward=True)

        with patch(
            "autoreview.search.snowballing._fetch_related",
            side_effect=fake_fetch,
        ):
            await snowball_papers(seeds, config=config)

        # 3 seeds × 2 calls (references + citations) = 6
        assert call_count == 6, f"Expected 6 _fetch_related calls, got {call_count}"

    async def test_skips_seeds_without_identifier(self):
        """Seeds without an s2_id or DOI are skipped gracefully."""
        call_count = 0

        async def fake_fetch(paper_id, endpoint, limit, client, limiter):
            nonlocal call_count
            call_count += 1
            return []

        # Seed with no doi and no s2_id
        seed_no_id = CandidatePaper(
            title="Unknown Paper",
            authors=["X"],
            source_database="semantic_scholar",
            external_ids={},
        )
        config = SnowballConfig(enable_backward=True, enable_forward=True)

        with patch(
            "autoreview.search.snowballing._fetch_related",
            side_effect=fake_fetch,
        ):
            results = await snowball_papers([seed_no_id], config=config)

        assert results == []
        assert call_count == 0

    async def test_deduplicates_within_snowball_run(self):
        """Same paper appearing in refs of two seeds is returned only once."""
        shared_paper = _make_s2_paper(paper_id="shared", title="Shared Paper", doi="10.1/shared")

        async def fake_fetch(paper_id, endpoint, limit, client, limiter):
            if endpoint == "references":
                return [shared_paper]
            return []

        seeds = [
            _make_seed(title="Seed A", doi="10.1/a", s2_id="sA"),
            _make_seed(title="Seed B", doi="10.1/b", s2_id="sB"),
        ]
        config = SnowballConfig(max_seed_papers=5, enable_backward=True, enable_forward=False)

        with patch(
            "autoreview.search.snowballing._fetch_related",
            side_effect=fake_fetch,
        ):
            results = await snowball_papers(seeds, config=config)

        assert len(results) == 1
        assert results[0].title == "Shared Paper"

    async def test_backward_only_config(self):
        """When enable_forward=False, only references are fetched."""
        endpoints_called: list[str] = []

        async def fake_fetch(paper_id, endpoint, limit, client, limiter):
            endpoints_called.append(endpoint)
            return []

        seed = _make_seed()
        config = SnowballConfig(enable_backward=True, enable_forward=False)

        with patch(
            "autoreview.search.snowballing._fetch_related",
            side_effect=fake_fetch,
        ):
            await snowball_papers([seed], config=config)

        assert endpoints_called == ["references"]

    async def test_forward_only_config(self):
        """When enable_backward=False, only citations are fetched."""
        endpoints_called: list[str] = []

        async def fake_fetch(paper_id, endpoint, limit, client, limiter):
            endpoints_called.append(endpoint)
            return []

        seed = _make_seed()
        config = SnowballConfig(enable_backward=False, enable_forward=True)

        with patch(
            "autoreview.search.snowballing._fetch_related",
            side_effect=fake_fetch,
        ):
            await snowball_papers([seed], config=config)

        assert endpoints_called == ["citations"]
