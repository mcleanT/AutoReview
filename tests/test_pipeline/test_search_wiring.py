"""Tests for search_factory.build_search_sources().

Verifies that CrossRef and Europe PMC are properly wired in, that unknown
sources are skipped with a warning, and that "core" is intentionally excluded.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _source_names(sources: list) -> list[str]:
    """Return the source_name attribute of each source object."""
    return [s.source_name for s in sources]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("db", ["crossref", "europe_pmc"])
def test_source_instantiated_for_declared_db(db: str) -> None:
    """CrossRef and Europe PMC are instantiated when declared in db_names."""
    from autoreview.pipeline.search_factory import build_search_sources

    sources = build_search_sources([db])
    names = _source_names(sources)
    assert db in names, f"Expected '{db}' in sources, got {names}"


def test_all_five_sources_instantiated() -> None:
    """All five supported backends are returned when all are requested."""
    from autoreview.pipeline.search_factory import build_search_sources

    all_dbs = ["pubmed", "semantic_scholar", "openalex", "crossref", "europe_pmc"]
    sources = build_search_sources(all_dbs)
    names = _source_names(sources)
    assert len(sources) == 5, f"Expected 5 sources, got {len(sources)}: {names}"
    for db in all_dbs:
        assert db in names, f"Missing source '{db}' in {names}"


def test_unknown_source_skipped_with_warning(capsys: pytest.CaptureFixture) -> None:
    """An unrecognised database name is skipped and a warning is emitted.

    structlog writes to stdout by default; we capture that instead of caplog.
    """
    from autoreview.pipeline.search_factory import build_search_sources

    sources = build_search_sources(["unknown_db_xyz"])
    captured = capsys.readouterr()

    assert len(sources) == 0
    assert "unknown_db_xyz" in captured.out or "unknown" in captured.out, (
        f"Expected warning about unknown source in stdout, got: {captured.out!r}"
    )


def test_core_skipped_intentionally() -> None:
    """'core' is intentionally excluded and must not appear in results."""
    from autoreview.pipeline.search_factory import build_search_sources

    sources = build_search_sources(["core"])
    assert len(sources) == 0, "CORE should be excluded from search sources"


def test_core_does_not_emit_warning(capsys: pytest.CaptureFixture) -> None:
    """'core' exclusion does not emit a 'warning' log event (it's intentional).

    structlog writes to stdout; we verify the word 'warning' doesn't appear
    alongside 'core' in the output (only 'info' level should fire).
    """
    from autoreview.pipeline.search_factory import build_search_sources

    build_search_sources(["core"])
    captured = capsys.readouterr()

    # The structlog output for an info event contains [info ]; a warning event
    # would contain [warning ].  Verify no warning-level line mentions "core".
    warning_lines = [
        line for line in captured.out.splitlines() if "warning" in line.lower() and "core" in line
    ]
    assert len(warning_lines) == 0, (
        f"'core' exclusion should not emit WARNING, got: {warning_lines}"
    )


def test_failed_instantiation_skipped_gracefully() -> None:
    """If a source raises on __init__, build_search_sources skips it and continues."""
    from unittest.mock import patch

    from autoreview.pipeline.search_factory import build_search_sources

    with patch(
        "autoreview.search.pubmed.PubMedSearch.__init__",
        side_effect=RuntimeError("injected failure"),
    ):
        sources = build_search_sources(["pubmed", "openalex"])

    names = _source_names(sources)
    assert "openalex" in names, "openalex should still be built after pubmed failure"
    assert "pubmed" not in names, "pubmed should be skipped after init failure"


def test_nodes_uses_factory_for_all_dbs() -> None:
    """nodes.py must not contain inline if/elif blocks for search sources."""
    from pathlib import Path

    nodes_path = Path(__file__).parent.parent.parent / "autoreview" / "pipeline" / "nodes.py"
    content = nodes_path.read_text()

    # The old pattern: instantiating PubMedSearch inside a for-db-in loop
    # After refactoring, direct instantiation inside for-loops should be gone
    # We allow imports at module level but not the repeated elif pattern
    old_patterns = [
        'elif db == "pubmed"',
        'elif db == "semantic_scholar"',
        'elif db == "openalex"',
    ]
    for pattern in old_patterns:
        assert pattern not in content, (
            f"nodes.py still contains inline search if/elif: {pattern!r}. "
            "Use build_search_sources() instead."
        )


def test_remediation_uses_factory() -> None:
    """remediation.py must not contain inline if/elif blocks for search sources."""
    from pathlib import Path

    remediation_path = (
        Path(__file__).parent.parent.parent / "autoreview" / "pipeline" / "remediation.py"
    )
    content = remediation_path.read_text()

    old_patterns = [
        'elif db == "pubmed"',
        'elif db == "semantic_scholar"',
        'elif db == "openalex"',
    ]
    for pattern in old_patterns:
        assert pattern not in content, (
            f"remediation.py still contains inline search if/elif: {pattern!r}. "
            "Use build_search_sources() instead."
        )
