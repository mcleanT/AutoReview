"""MCP server exposing AutoReview search APIs as tools.

Uses FastMCP from the `mcp` package to serve PubMed, Semantic Scholar,
and OpenAlex search capabilities over the MCP stdio transport.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("autoreview-search")


def _papers_to_dicts(papers: list[Any]) -> list[dict[str, Any]]:
    """Convert CandidatePaper objects to simplified dictionaries."""
    results = []
    for p in papers:
        results.append(
            {
                "title": p.title,
                "authors": p.authors,
                "year": p.year,
                "doi": p.doi,
                "abstract": p.abstract,
                "source": p.source_database,
            }
        )
    return results


@mcp.tool()
async def search_pubmed(query: str, max_results: int = 20) -> str:
    """Search PubMed for papers matching the query.

    Uses NCBI Entrez API via biopython. Requires ENTREZ_EMAIL and
    optionally NCBI_API_KEY environment variables.

    Args:
        query: Search query string (supports PubMed Boolean syntax).
        max_results: Maximum number of papers to return (default 20).

    Returns:
        JSON string containing a list of paper objects with title, authors,
        year, doi, abstract, and source fields.
    """
    try:
        from autoreview.search.pubmed import PubMedSearch

        searcher = PubMedSearch()
        papers = await searcher.search(queries=[query], max_results=max_results)
        return json.dumps(_papers_to_dicts(papers), indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def search_semantic_scholar(query: str, max_results: int = 20) -> str:
    """Search Semantic Scholar for papers matching the query.

    Uses the Semantic Scholar REST API. Optionally uses S2_API_KEY
    environment variable for higher rate limits.

    Args:
        query: Search query string (natural language or keywords).
        max_results: Maximum number of papers to return (default 20).

    Returns:
        JSON string containing a list of paper objects with title, authors,
        year, doi, abstract, and source fields.
    """
    try:
        from autoreview.search.semantic_scholar import SemanticScholarSearch

        searcher = SemanticScholarSearch()
        papers = await searcher.search(queries=[query], max_results=max_results)
        return json.dumps(_papers_to_dicts(papers), indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def search_openalex(query: str, max_results: int = 20) -> str:
    """Search OpenAlex for papers matching the query.

    Uses the OpenAlex API via pyalex. Optionally uses OPENALEX_EMAIL
    environment variable for polite pool access.

    Args:
        query: Search query string (natural language or keywords).
        max_results: Maximum number of papers to return (default 20).

    Returns:
        JSON string containing a list of paper objects with title, authors,
        year, doi, abstract, and source fields.
    """
    try:
        from autoreview.search.openalex import OpenAlexSearch

        searcher = OpenAlexSearch()
        papers = await searcher.search(queries=[query], max_results=max_results)
        return json.dumps(_papers_to_dicts(papers), indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run()
