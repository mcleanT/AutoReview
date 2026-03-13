"""Tests for SemanticScholarSearch citation snowballing."""

from __future__ import annotations

import httpx
import respx

from autoreview.search.semantic_scholar import SemanticScholarSearch

S2_BASE = "https://api.semanticscholar.org/graph/v1"


class TestGetReferences:
    @respx.mock
    async def test_get_references_returns_papers(self):
        paper_id = "abc123"
        respx.get(f"{S2_BASE}/paper/{paper_id}/references").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "citedPaper": {
                                "paperId": "ref1",
                                "title": "Reference Paper One",
                                "abstract": "About senescence",
                                "year": 2022,
                                "authors": [{"name": "Alice A"}],
                                "journal": {"name": "Nature"},
                                "externalIds": {"DOI": "10.1/ref1"},
                                "citationCount": 10,
                            }
                        },
                        {
                            "citedPaper": {
                                "paperId": "ref2",
                                "title": None,  # invalid — should be skipped
                                "abstract": None,
                                "year": 2021,
                                "authors": [],
                                "journal": None,
                                "externalIds": {},
                                "citationCount": 0,
                            }
                        },
                    ]
                },
            )
        )
        s2 = SemanticScholarSearch()
        papers = await s2.get_references(paper_id, limit=50)
        assert len(papers) == 1
        assert papers[0].title == "Reference Paper One"
        assert papers[0].source_database == "semantic_scholar"

    @respx.mock
    async def test_get_references_handles_http_error(self):
        paper_id = "bad_id"
        respx.get(f"{S2_BASE}/paper/{paper_id}/references").mock(return_value=httpx.Response(404))
        s2 = SemanticScholarSearch()
        papers = await s2.get_references(paper_id, limit=50)
        assert papers == []

    @respx.mock
    async def test_snowball_references_across_papers(self):
        """snowball_references should collect references from multiple papers."""
        respx.get(f"{S2_BASE}/paper/s2_p1/references").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "citedPaper": {
                                "paperId": "r1",
                                "title": "Ref A",
                                "abstract": "abstract a",
                                "year": 2020,
                                "authors": [{"name": "Bob B"}],
                                "journal": {"name": "Cell"},
                                "externalIds": {"DOI": "10.1/a"},
                                "citationCount": 5,
                            }
                        }
                    ]
                },
            )
        )
        respx.get(f"{S2_BASE}/paper/s2_p2/references").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "citedPaper": {
                                "paperId": "r2",
                                "title": "Ref B",
                                "abstract": "abstract b",
                                "year": 2021,
                                "authors": [{"name": "Carol C"}],
                                "journal": {"name": "Science"},
                                "externalIds": {"DOI": "10.1/b"},
                                "citationCount": 3,
                            }
                        }
                    ]
                },
            )
        )
        s2 = SemanticScholarSearch()
        papers = await s2.snowball_references(["s2_p1", "s2_p2"], limit_per_paper=20)
        assert len(papers) == 2
        titles = {p.title for p in papers}
        assert "Ref A" in titles
        assert "Ref B" in titles
