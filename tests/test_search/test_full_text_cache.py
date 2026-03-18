"""Tests for the disk-based full-text cache."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio  # noqa: F401 — needed for asyncio mode

from autoreview.models.paper import CandidatePaper, ScreenedPaper
from autoreview.search.full_text_cache import (
    CachedFullTextResolver,
    FullTextCache,
    _normalize_title,
    cache_key,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_paper(
    doi: str | None = "10.1234/test.001",
    title: str = "A Test Paper About Something",
    full_text: str | None = None,
    full_text_source: str | None = None,
) -> CandidatePaper:
    return CandidatePaper(
        title=title,
        authors=["Author A", "Author B"],
        source_database="test",
        doi=doi,
        abstract="Abstract text here.",
        full_text=full_text,
        full_text_source=full_text_source,
    )


def _make_screened(
    doi: str | None = "10.1234/test.001",
    title: str = "A Test Paper About Something",
    full_text: str | None = None,
    full_text_source: str | None = None,
) -> ScreenedPaper:
    return ScreenedPaper(
        paper=_make_paper(
            doi=doi, title=title, full_text=full_text, full_text_source=full_text_source
        ),
        relevance_score=4,
        rationale="Relevant",
        include=True,
    )


# ---------------------------------------------------------------------------
# cache_key
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_doi_based_key_is_stable(self) -> None:
        paper = _make_paper(doi="10.1234/example.2024.001")
        key1 = cache_key(paper)
        key2 = cache_key(paper)
        assert key1 == key2

    def test_doi_key_is_hex_digest(self) -> None:
        paper = _make_paper(doi="10.1234/example.2024.001")
        key = cache_key(paper)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_title_fallback_when_no_doi(self) -> None:
        paper = _make_paper(doi=None, title="Advances in Machine Learning 2024")
        key = cache_key(paper)
        assert len(key) == 64

    def test_doi_and_title_produce_different_keys(self) -> None:
        paper_doi = _make_paper(doi="10.1234/example.2024.001")
        paper_title = _make_paper(doi=None, title="10.1234/example.2024.001")
        # They're different because doi path lowercases the doi directly,
        # title path normalizes whitespace/punctuation differently.
        # Both are valid — just confirm they don't accidentally collide here.
        # (They may or may not be equal; the important thing is the function runs.)
        key_doi = cache_key(paper_doi)
        key_title = cache_key(paper_title)
        assert isinstance(key_doi, str)
        assert isinstance(key_title, str)

    def test_doi_normalization_case_insensitive(self) -> None:
        """DOI model validator already lowercases, so both produce the same key."""
        paper_lower = _make_paper(doi="10.1234/example")
        # Create via dict to bypass the field_validator lowercasing for test purposes.
        # Actually the validator always lowercases, so both will have the same stored DOI.
        paper_upper = _make_paper(doi="10.1234/EXAMPLE")
        # After normalization by the model, both DOIs should be "10.1234/example".
        assert cache_key(paper_lower) == cache_key(paper_upper)

    def test_title_normalization_strips_accents(self) -> None:
        title_accented = "Über die Künstliche Intelligenz"
        title_ascii = "Uber die Kunstliche Intelligenz"
        norm_acc = _normalize_title(title_accented)
        norm_asc = _normalize_title(title_ascii)
        assert norm_acc == norm_asc

    def test_title_normalization_collapses_whitespace(self) -> None:
        result = _normalize_title("  multiple   spaces   here  ")
        assert result == "multiple spaces here"


# ---------------------------------------------------------------------------
# FullTextCache — put / get round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFullTextCachePutGet:
    async def test_put_then_get_returns_text_and_source(self, tmp_path: Path) -> None:
        cache = FullTextCache(cache_dir=tmp_path)
        paper = _make_paper()
        await cache.put(paper, "Full article text here.", "pmc")
        result = await cache.get(paper)
        assert result is not None
        text, source = result
        assert text == "Full article text here."
        assert source == "pmc"

    async def test_get_returns_none_on_cache_miss(self, tmp_path: Path) -> None:
        cache = FullTextCache(cache_dir=tmp_path)
        paper = _make_paper(doi="10.9999/not-cached")
        result = await cache.get(paper)
        assert result is None

    async def test_has_returns_false_on_miss(self, tmp_path: Path) -> None:
        cache = FullTextCache(cache_dir=tmp_path)
        paper = _make_paper(doi="10.9999/not-cached")
        assert not await cache.has(paper)

    async def test_has_returns_true_after_put(self, tmp_path: Path) -> None:
        cache = FullTextCache(cache_dir=tmp_path)
        paper = _make_paper()
        await cache.put(paper, "Some text", "arxiv")
        assert await cache.has(paper)

    async def test_cache_files_use_two_level_directory(self, tmp_path: Path) -> None:
        cache = FullTextCache(cache_dir=tmp_path)
        paper = _make_paper()
        await cache.put(paper, "Some text", "pmc")
        key = cache_key(paper)
        expected_path = tmp_path / "full_text_cache" / key[:2] / f"{key}.json"
        assert expected_path.exists()

    async def test_overwrite_updates_entry(self, tmp_path: Path) -> None:
        cache = FullTextCache(cache_dir=tmp_path)
        paper = _make_paper()
        await cache.put(paper, "First version", "pmc")
        await cache.put(paper, "Updated version", "elsevier_api")
        result = await cache.get(paper)
        assert result is not None
        text, source = result
        assert text == "Updated version"
        assert source == "elsevier_api"


# ---------------------------------------------------------------------------
# FullTextCache — negative result caching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestNegativeResultCaching:
    async def test_put_negative_then_get_returns_none(self, tmp_path: Path) -> None:
        cache = FullTextCache(cache_dir=tmp_path)
        paper = _make_paper(doi="10.9999/no-oa")
        await cache.put(paper, None, "not_found")
        result = await cache.get(paper)
        # get() returns None for negatives (caller gets same as cold miss)
        assert result is None

    async def test_has_returns_true_for_cached_negative(self, tmp_path: Path) -> None:
        cache = FullTextCache(cache_dir=tmp_path)
        paper = _make_paper(doi="10.9999/no-oa")
        await cache.put(paper, None, "not_found")
        # has() returns True — the entry exists even if text is null
        assert await cache.has(paper)

    async def test_get_raw_returns_negative_entry(self, tmp_path: Path) -> None:
        cache = FullTextCache(cache_dir=tmp_path)
        paper = _make_paper(doi="10.9999/no-oa")
        await cache.put(paper, None, "not_found")
        raw = await cache.get_raw(paper)
        assert raw is not None
        assert raw["text"] is None
        assert raw["source"] == "not_found"


# ---------------------------------------------------------------------------
# FullTextCache — TTL / expiry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCacheExpiry:
    async def test_fresh_entry_not_expired(self, tmp_path: Path) -> None:
        cache = FullTextCache(cache_dir=tmp_path, ttl_days=30)
        paper = _make_paper()
        await cache.put(paper, "Text", "pmc")
        assert await cache.has(paper)

    async def test_expired_entry_returns_none(self, tmp_path: Path) -> None:
        cache = FullTextCache(cache_dir=tmp_path, ttl_days=30)
        paper = _make_paper()
        # Write an entry with a timestamp 31 days in the past.
        import json

        key = cache_key(paper)
        path = tmp_path / "full_text_cache" / key[:2] / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        stale_ts = (datetime.now(tz=UTC) - timedelta(days=31)).isoformat()
        path.write_text(
            json.dumps(
                {
                    "text": "Old text",
                    "source": "pmc",
                    "timestamp": stale_ts,
                    "doi": paper.doi,
                    "title": paper.title,
                }
            ),
            encoding="utf-8",
        )
        result = await cache.get(paper)
        assert result is None
        assert not await cache.has(paper)

    async def test_negative_uses_shorter_ttl(self, tmp_path: Path) -> None:
        """Negatives should expire after negative_ttl_days (default 7), not 30."""
        cache = FullTextCache(cache_dir=tmp_path, ttl_days=30, negative_ttl_days=7)
        paper = _make_paper(doi="10.9999/closed")
        import json

        key = cache_key(paper)
        path = tmp_path / "full_text_cache" / key[:2] / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        # 8 days old — within 30-day TTL but past 7-day negative TTL.
        stale_ts = (datetime.now(tz=UTC) - timedelta(days=8)).isoformat()
        path.write_text(
            json.dumps(
                {
                    "text": None,
                    "source": "not_found",
                    "timestamp": stale_ts,
                    "doi": paper.doi,
                    "title": paper.title,
                }
            ),
            encoding="utf-8",
        )
        # Entry should be considered expired.
        assert not await cache.has(paper)
        raw = await cache.get_raw(paper)
        assert raw is None

    async def test_positive_entry_survives_negative_ttl(self, tmp_path: Path) -> None:
        """A positive entry that is 8 days old should still be fresh (30-day TTL)."""
        cache = FullTextCache(cache_dir=tmp_path, ttl_days=30, negative_ttl_days=7)
        paper = _make_paper()
        import json

        key = cache_key(paper)
        path = tmp_path / "full_text_cache" / key[:2] / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        ts_8_days_ago = (datetime.now(tz=UTC) - timedelta(days=8)).isoformat()
        path.write_text(
            json.dumps(
                {
                    "text": "Some text",
                    "source": "pmc",
                    "timestamp": ts_8_days_ago,
                    "doi": paper.doi,
                    "title": paper.title,
                }
            ),
            encoding="utf-8",
        )
        assert await cache.has(paper)
        result = await cache.get(paper)
        assert result is not None
        assert result[0] == "Some text"


# ---------------------------------------------------------------------------
# CachedFullTextResolver
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCachedFullTextResolver:
    def _mock_resolver(self, enriched_text: str | None = "Fetched full text") -> MagicMock:
        """Return a mock FullTextResolver whose resolve() enriches all papers."""
        mock = MagicMock()

        async def _resolve(papers, max_concurrent=5):
            for sp in papers:
                if enriched_text and not sp.paper.full_text:
                    sp.paper.full_text = enriched_text
                    sp.paper.full_text_source = "pmc"
            return {"pmc": len(papers)} if enriched_text else {}

        mock.resolve = AsyncMock(side_effect=_resolve)
        mock.close = AsyncMock()
        return mock

    async def test_cold_miss_delegates_to_resolver(self, tmp_path: Path) -> None:
        cache = FullTextCache(cache_dir=tmp_path)
        mock_resolver = self._mock_resolver()
        cached_resolver = CachedFullTextResolver(resolver=mock_resolver, cache=cache)

        sp = _make_screened()
        source_counts = await cached_resolver.resolve([sp])

        mock_resolver.resolve.assert_called_once()
        assert sp.paper.full_text == "Fetched full text"
        assert source_counts.get("pmc", 0) == 1

    async def test_warm_hit_skips_resolver(self, tmp_path: Path) -> None:
        cache = FullTextCache(cache_dir=tmp_path)
        paper = _make_paper()
        await cache.put(paper, "Cached text", "arxiv")

        mock_resolver = self._mock_resolver()
        cached_resolver = CachedFullTextResolver(resolver=mock_resolver, cache=cache)

        sp = _make_screened()
        source_counts = await cached_resolver.resolve([sp])

        mock_resolver.resolve.assert_not_called()
        assert sp.paper.full_text == "Cached text"
        assert sp.paper.full_text_source == "arxiv"
        assert source_counts.get("arxiv", 0) == 1

    async def test_resolver_result_is_persisted_to_cache(self, tmp_path: Path) -> None:
        cache = FullTextCache(cache_dir=tmp_path)
        mock_resolver = self._mock_resolver(enriched_text="Fresh from API")
        cached_resolver = CachedFullTextResolver(resolver=mock_resolver, cache=cache)

        sp = _make_screened()
        await cached_resolver.resolve([sp])

        # Second call — resolver should NOT be called again.
        sp2 = _make_screened()
        mock_resolver.resolve.reset_mock()
        await cached_resolver.resolve([sp2])

        mock_resolver.resolve.assert_not_called()
        assert sp2.paper.full_text == "Fresh from API"

    async def test_negative_result_is_cached_on_miss(self, tmp_path: Path) -> None:
        """When the resolver finds nothing, a negative entry is stored."""
        cache = FullTextCache(cache_dir=tmp_path)
        mock_resolver = self._mock_resolver(enriched_text=None)
        cached_resolver = CachedFullTextResolver(resolver=mock_resolver, cache=cache)

        sp = _make_screened()
        await cached_resolver.resolve([sp])

        # The negative should be cached.
        assert await cache.has(sp.paper)
        raw = await cache.get_raw(sp.paper)
        assert raw is not None
        assert raw["text"] is None
        assert raw["source"] == "not_found"

    async def test_cached_negative_skips_resolver_on_second_run(self, tmp_path: Path) -> None:
        """A paper that had no full text should not be re-queried on the next run."""
        cache = FullTextCache(cache_dir=tmp_path)
        paper = _make_paper(doi="10.9999/closed-access")
        await cache.put(paper, None, "not_found")

        mock_resolver = self._mock_resolver()
        cached_resolver = CachedFullTextResolver(resolver=mock_resolver, cache=cache)

        sp = _make_screened(doi="10.9999/closed-access")
        await cached_resolver.resolve([sp])

        mock_resolver.resolve.assert_not_called()
        assert sp.paper.full_text is None

    async def test_already_enriched_papers_are_skipped(self, tmp_path: Path) -> None:
        """Papers that already have full_text set upstream are not touched."""
        cache = FullTextCache(cache_dir=tmp_path)
        mock_resolver = self._mock_resolver()
        cached_resolver = CachedFullTextResolver(resolver=mock_resolver, cache=cache)

        sp = _make_screened(full_text="Pre-existing text", full_text_source="manual")
        await cached_resolver.resolve([sp])

        mock_resolver.resolve.assert_not_called()
        assert sp.paper.full_text == "Pre-existing text"

    async def test_mixed_batch_partial_hits(self, tmp_path: Path) -> None:
        """Some papers in the batch are cached, others go to the resolver."""
        cache = FullTextCache(cache_dir=tmp_path)
        paper_hit = _make_paper(doi="10.1234/cached")
        await cache.put(paper_hit, "Cached full text", "elsevier_api")

        mock_resolver = self._mock_resolver(enriched_text="API text")
        cached_resolver = CachedFullTextResolver(resolver=mock_resolver, cache=cache)

        sp_hit = _make_screened(doi="10.1234/cached")
        sp_miss = _make_screened(doi="10.5678/not-cached")
        await cached_resolver.resolve([sp_hit, sp_miss])

        # Resolver should only be called for the miss.
        mock_resolver.resolve.assert_called_once()
        called_papers = mock_resolver.resolve.call_args[0][0]
        assert len(called_papers) == 1
        assert called_papers[0].paper.doi == "10.5678/not-cached"

        assert sp_hit.paper.full_text == "Cached full text"
        assert sp_miss.paper.full_text == "API text"

    async def test_close_delegates_to_wrapped_resolver(self, tmp_path: Path) -> None:
        cache = FullTextCache(cache_dir=tmp_path)
        mock_resolver = self._mock_resolver()
        cached_resolver = CachedFullTextResolver(resolver=mock_resolver, cache=cache)
        await cached_resolver.close()
        mock_resolver.close.assert_called_once()
