"""Verify date_range is passed at all SearchAggregator instantiation sites in nodes.py."""

from __future__ import annotations

from pathlib import Path


def test_all_aggregator_sites_pass_date_range():
    """Every SearchAggregator() call in nodes.py must include date_range=."""
    nodes_path = Path(__file__).parent.parent.parent / "autoreview" / "pipeline" / "nodes.py"
    content = nodes_path.read_text()

    import re

    sites = [m.start() for m in re.finditer(r"SearchAggregator\(", content)]
    assert len(sites) >= 5, f"Expected >=5 SearchAggregator sites, found {len(sites)}"

    for pos in sites:
        snippet = content[pos : pos + 200]
        assert "date_range" in snippet, (
            f"SearchAggregator at char {pos} missing date_range=. Snippet: {snippet[:100]!r}"
        )
