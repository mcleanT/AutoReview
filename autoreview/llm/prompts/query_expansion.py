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


TARGETED_EXPANSION_SYSTEM_PROMPT = """\
You are an expert research librarian. You previously generated search queries for a \
literature review, but some sub-topics were not adequately covered. Generate additional \
targeted queries to fill these specific gaps. Use different terminology, synonyms, and \
broader formulations than your initial queries.
"""


def build_targeted_query_expansion_prompt(
    uncovered_topics: list[str],
    domain: str,
    date_range: str = "2015-2025",
) -> str:
    """Build a prompt to generate queries for specific uncovered sub-topics."""
    topics_list = "\n".join(f"- {t}" for t in uncovered_topics)
    return f"""\
The following sub-topics were NOT adequately covered by the initial search queries \
and need additional targeted queries:

**Uncovered Sub-Topics:**
{topics_list}

**Domain:** {domain}
**Date range:** {date_range}

For EACH uncovered sub-topic, generate:
- pubmed_queries: 1-2 Boolean/MeSH queries targeting this specific sub-topic
- semantic_scholar_queries: 1-2 semantic search queries
- openalex_queries: 1-2 general academic queries

Use different terminology than typical queries for these topics — try synonyms, \
related concepts, and broader formulations to maximize discovery.
"""
