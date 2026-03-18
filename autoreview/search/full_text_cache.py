"""Disk-based cache for full-text retrieval results.

Cache key  : SHA-256 hash of lowercase DOI, falling back to normalized title.
Cache value: JSON file with {text, source, timestamp, doi, negative}.
Cache path : {cache_dir}/full_text_cache/{hash[:2]}/{hash}.json

Two-level directory structure keeps individual directories small even with
thousands of cached papers.

Negative results (paper attempted but no full text found) are also cached
so that repeated pipeline runs do not re-query external APIs for known
failures.  Negatives use a shorter default TTL (7 days vs 30 days).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import unicodedata
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import structlog

if TYPE_CHECKING:
    from autoreview.models.paper import CandidatePaper, ScreenedPaper
    from autoreview.search.full_text import FullTextResolver

logger = structlog.get_logger()

_DEFAULT_TTL_DAYS = 30
_NEGATIVE_TTL_DAYS = 7


# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------


def _normalize_title(title: str) -> str:
    """Lowercase, strip accents, collapse whitespace, remove punctuation."""
    title = unicodedata.normalize("NFD", title)
    title = "".join(c for c in title if unicodedata.category(c) != "Mn")
    title = title.lower()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def cache_key(paper: CandidatePaper) -> str:
    """Return a stable SHA-256 hex digest for *paper*.

    Priority:
      1. Normalised DOI (already lowercased by the model validator).
      2. Normalised title as a fallback.
    """
    raw = paper.doi.lower().strip() if paper.doi else _normalize_title(paper.title)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# FullTextCache
# ---------------------------------------------------------------------------


class FullTextCache:
    """Disk-based cache for full-text retrieval results.

    Cache key: SHA256 hash of DOI (or title if no DOI)
    Cache value: JSON with {text: str | null, source: str, timestamp: str, doi: str}
    Cache location: {cache_dir}/full_text_cache/{hash[:2]}/{hash}.json

    Two-level directory structure avoids too many files in one directory.
    Thread-safe via per-key asyncio locks.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        ttl_days: int = _DEFAULT_TTL_DAYS,
        negative_ttl_days: int = _NEGATIVE_TTL_DAYS,
    ) -> None:
        self._root = Path(cache_dir) / "full_text_cache"
        self._ttl = timedelta(days=ttl_days)
        self._negative_ttl = timedelta(days=negative_ttl_days)
        # Per-key locks to prevent concurrent reads/writes of the same entry.
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_meta: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _lock_for(self, key: str) -> asyncio.Lock:
        async with self._locks_meta:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            return self._locks[key]

    def _path(self, key: str) -> Path:
        return self._root / key[:2] / f"{key}.json"

    def _is_fresh(self, entry: dict[str, Any]) -> bool:
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
            is_negative = entry.get("text") is None
            ttl = self._negative_ttl if is_negative else self._ttl
            return datetime.now(tz=UTC) - ts < ttl
        except (KeyError, ValueError):
            return False

    def _read_entry(self, path: Path) -> dict[str, Any] | None:
        try:
            return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_entry(self, path: Path, entry: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def has(self, paper: CandidatePaper) -> bool:
        """Return True if a fresh (non-expired) entry exists for *paper*."""
        key = cache_key(paper)
        lock = await self._lock_for(key)
        async with lock:
            entry = self._read_entry(self._path(key))
            return entry is not None and self._is_fresh(entry)

    async def get(self, paper: CandidatePaper) -> tuple[str, str] | None:
        """Return (text, source) for a cache hit, or None on miss / expiry.

        For cached negative results (text is null), returns None so that
        callers do not need to special-case the sentinel — they just get
        no text, same as a cold miss.  The difference is that a subsequent
        ``has()`` call will still return True (the entry exists), so the
        resolver knows to skip the external API call.

        If callers need to distinguish "not yet tried" from "tried and
        failed", they should call ``get_raw()`` directly.
        """
        key = cache_key(paper)
        lock = await self._lock_for(key)
        async with lock:
            path = self._path(key)
            entry = self._read_entry(path)
            if entry is None or not self._is_fresh(entry):
                return None
            text = entry.get("text")
            source = entry.get("source", "")
            if text is None:
                # Negative result — caller should treat as "no text available"
                return None
            return text, source

    async def get_raw(self, paper: CandidatePaper) -> dict[str, Any] | None:
        """Return the raw cache entry dict, or None on miss / expiry.

        Includes negative entries (text=null, source='not_found').
        """
        key = cache_key(paper)
        lock = await self._lock_for(key)
        async with lock:
            entry = self._read_entry(self._path(key))
            if entry is None or not self._is_fresh(entry):
                return None
            return entry

    async def put(
        self,
        paper: CandidatePaper,
        text: str | None,
        source: str,
    ) -> None:
        """Store (text, source) for *paper*.

        Pass ``text=None`` and ``source='not_found'`` to cache a negative
        result so that future runs skip the expensive API calls.
        """
        key = cache_key(paper)
        lock = await self._lock_for(key)
        async with lock:
            entry = {
                "text": text,
                "source": source,
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "doi": paper.doi,
                "title": paper.title,
            }
            self._write_entry(self._path(key), entry)

    async def stats(self) -> dict[str, int]:
        """Return {total, fresh, negative} counts for monitoring."""
        total = fresh = negative = 0
        if not self._root.exists():
            return {"total": 0, "fresh": 0, "negative": 0}
        for path in self._root.rglob("*.json"):
            total += 1
            entry = self._read_entry(path)
            if entry and self._is_fresh(entry):
                fresh += 1
                if entry.get("text") is None:
                    negative += 1
        return {"total": total, "fresh": fresh, "negative": negative}


# ---------------------------------------------------------------------------
# CachedFullTextResolver
# ---------------------------------------------------------------------------


class CachedFullTextResolver:
    """Wraps :class:`FullTextResolver` with a disk-based cache layer.

    On ``resolve()``, each paper is checked against the cache first.
    Cache hits set ``paper.full_text`` / ``paper.full_text_source`` directly
    without touching the network.  Cache misses are delegated to the wrapped
    resolver and the result is then stored in the cache (including negatives).
    """

    def __init__(
        self,
        resolver: FullTextResolver,
        cache: FullTextCache,
    ) -> None:
        self._resolver = resolver
        self._cache = cache

    async def resolve(
        self,
        papers: list[ScreenedPaper],
        max_concurrent: int = 5,
    ) -> dict[str, int]:
        """Enrich papers with full text, consulting the cache first.

        Returns ``{source_name: count}`` for audit logging (cache hits are
        attributed to their original source, not a synthetic 'cache' source).
        """
        cache_hits: list[ScreenedPaper] = []
        cache_misses: list[ScreenedPaper] = []

        # Phase 1: classify papers as hits or misses.
        for sp in papers:
            if sp.paper.full_text:
                continue  # already enriched upstream

            raw = await self._cache.get_raw(sp.paper)
            if raw is not None:
                # Fresh cache entry found.
                text = raw.get("text")
                source = raw.get("source", "")
                if text:
                    sp.paper.full_text = text
                    sp.paper.full_text_source = source
                # If text is None this is a cached negative — skip re-querying.
                cache_hits.append(sp)
            else:
                cache_misses.append(sp)

        hit_count = sum(1 for sp in cache_hits if sp.paper.full_text)
        negative_count = len(cache_hits) - hit_count

        logger.info(
            "full_text_cache.classify",
            total=len(papers),
            cache_hits=hit_count,
            cached_negatives=negative_count,
            cache_misses=len(cache_misses),
        )

        if not cache_misses:
            # Build source_counts from cache hits only.
            source_counts: dict[str, int] = {}
            for sp in cache_hits:
                if sp.paper.full_text_source:
                    source_counts[sp.paper.full_text_source] = (
                        source_counts.get(sp.paper.full_text_source, 0) + 1
                    )
            return source_counts

        # Phase 2: snapshot state before delegating so we can detect what the
        # resolver actually filled in.
        pre_state: dict[int, tuple[str | None, str | None]] = {
            id(sp): (sp.paper.full_text, sp.paper.full_text_source) for sp in cache_misses
        }

        source_counts = await self._resolver.resolve(cache_misses, max_concurrent)

        # Phase 3: persist resolver results to cache.
        for sp in cache_misses:
            old_text, _ = pre_state[id(sp)]
            if old_text is not None:
                continue  # already had text before — shouldn't happen but guard

            new_text = sp.paper.full_text
            new_source = sp.paper.full_text_source

            if new_text:
                await self._cache.put(sp.paper, new_text, new_source or "unknown")
            else:
                # Cache the negative so we don't retry on the next run.
                await self._cache.put(sp.paper, None, "not_found")

        # Merge cache-hit source counts with resolver source counts.
        for sp in cache_hits:
            src = sp.paper.full_text_source
            if src:
                source_counts[src] = source_counts.get(src, 0) + 1

        total_hits = hit_count + sum(1 for sp in cache_misses if sp.paper.full_text)
        logger.info(
            "full_text_cache.resolve_complete",
            total=len(papers),
            cache_hits=hit_count,
            cached_negatives=negative_count,
            resolver_hits=sum(1 for sp in cache_misses if sp.paper.full_text),
            total_enriched=total_hits,
            by_source=source_counts,
        )

        return source_counts

    async def close(self) -> None:
        """Close the underlying resolver's HTTP clients."""
        await self._resolver.close()
