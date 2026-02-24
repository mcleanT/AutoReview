from __future__ import annotations

SYSTEM_PROMPT = """\
You are an expert research librarian and systematic review specialist. Your task is to \
generate comprehensive search queries for a scientific literature review.

You must produce:
1. Boolean queries optimized for PubMed (using MeSH terms where appropriate)
2. Semantic search queries for Semantic Scholar (natural language, concept-focused)
3. General academic queries for OpenAlex
4. Natural language discovery questions for Perplexity (to find recent or under-indexed work)
5. A scope document that defines the review's boundaries, expected sub-topics, and exclusion criteria
"""


def build_query_expansion_prompt(topic: str, domain: str, date_range: str = "2015-2025") -> str:
    """Build the prompt for query expansion."""
    return f"""\
Generate search queries and a scope document for a scientific literature review on the following topic:

**Topic:** {topic}
**Domain:** {domain}
**Date range:** {date_range}

Produce structured output with:
- pubmed_queries: list of Boolean/MeSH queries for PubMed (3-5 queries)
- semantic_scholar_queries: list of semantic search queries (3-5 queries)
- openalex_queries: list of general academic queries (3-5 queries)
- perplexity_questions: list of natural language discovery questions (2-3 questions)
- scope_document: a detailed paragraph defining what this review should cover, its boundaries, expected sub-topics (at least 5), and what is explicitly out of scope
"""
