"""Factory for building search source instances from a list of database names.

This module centralises search source instantiation so every pipeline node
uses a consistent, DRY implementation.  Adding a new backend only requires a
single change here.
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Intentionally excluded — unreliable and not worth the noise
_EXCLUDED_SOURCES = frozenset({"core"})

# All sources we know how to build
_KNOWN_SOURCES = frozenset({"pubmed", "semantic_scholar", "openalex", "crossref", "europe_pmc"})


def build_search_sources(db_names: list[str]) -> list[Any]:
    """Return a list of search source instances for the requested database names.

    Unknown names emit a warning and are skipped.  ``"core"`` is silently
    skipped with an info-level log (intentional exclusion).  Any instantiation
    error is caught, logged, and skipped so the pipeline degrades gracefully.

    Args:
        db_names: Ordered list of database identifiers, e.g.
            ``["pubmed", "semantic_scholar", "openalex", "crossref",
            "europe_pmc"]``.

    Returns:
        List of instantiated search source objects (may be shorter than
        ``db_names`` if some sources failed or were skipped).
    """
    sources: list[Any] = []

    for db in db_names:
        if db in _EXCLUDED_SOURCES:
            logger.info(
                "search_factory.source_excluded", source=db, reason="intentionally excluded"
            )
            continue

        if db not in _KNOWN_SOURCES:
            logger.warning("search_factory.unknown_source", source=db)
            continue

        try:
            if db == "pubmed":
                from autoreview.search.pubmed import PubMedSearch

                sources.append(PubMedSearch())
            elif db == "semantic_scholar":
                from autoreview.search.semantic_scholar import SemanticScholarSearch

                sources.append(SemanticScholarSearch())
            elif db == "openalex":
                from autoreview.search.openalex import OpenAlexSearch

                sources.append(OpenAlexSearch())
            elif db == "crossref":
                from autoreview.search.crossref import CrossRefSearch

                sources.append(CrossRefSearch(email=None))
            elif db == "europe_pmc":
                from autoreview.search.europe_pmc import EuropePMCSearch

                sources.append(EuropePMCSearch())
        except Exception as e:
            logger.warning("search_factory.source_init_failed", source=db, error=str(e))

    return sources
