"""Prompts for comprehensiveness validation checks."""
from __future__ import annotations

from pydantic import Field

from autoreview.models.base import AutoReviewModel


class SubTopicCoverage(AutoReviewModel):
    """Assessment of whether a sub-topic is covered by queries."""
    sub_topic: str
    covered: bool
    matching_queries: list[str] = Field(default_factory=list)


class QueryCoverageResult(AutoReviewModel):
    """Result of query coverage assessment."""
    sub_topic_assessments: list[SubTopicCoverage]
    overall_coverage_score: float = 0.0


QUERY_COVERAGE_SYSTEM_PROMPT = """\
You are an expert research librarian evaluating whether a set of search queries \
adequately covers all expected sub-topics for a literature review. For each sub-topic \
identified in the scope document, determine whether at least one query would likely \
retrieve relevant papers on that sub-topic.
"""


def build_query_coverage_prompt(
    scope_document: str,
    queries_by_source: dict[str, list[str]],
) -> str:
    query_blocks = []
    for source, queries in queries_by_source.items():
        query_blocks.append(f"**{source}:**")
        for q in queries:
            query_blocks.append(f"  - {q}")
    queries_text = "\n".join(query_blocks)

    return f"""\
## Review Scope Document
{scope_document}

## Generated Search Queries
{queries_text}

Evaluate whether these queries cover all sub-topics expected by the scope document.
For each sub-topic in the scope:
1. Identify the sub-topic
2. Determine if at least one query would retrieve papers on it (covered=true/false)
3. List matching queries if covered

Provide an overall_coverage_score from 0.0 (no sub-topics covered) to 1.0 (all covered).
"""
