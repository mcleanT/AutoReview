from __future__ import annotations

from pydantic import Field

from autoreview.models.base import AutoReviewModel


class ThemeCluster(AutoReviewModel):
    """A theme identified from clustering findings."""
    name: str
    description: str
    sub_themes: list[SubThemeCluster] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)


class SubThemeCluster(AutoReviewModel):
    """A sub-theme within a broader theme."""
    name: str
    description: str
    paper_ids: list[str] = Field(default_factory=list)
    finding_claims: list[str] = Field(default_factory=list)


# Fix forward reference
ThemeCluster.model_rebuild()


class ClusteringResult(AutoReviewModel):
    """Result of thematic clustering."""
    themes: list[ThemeCluster]


class ContradictionResult(AutoReviewModel):
    """Contradictions identified within a theme."""
    consensus_claims: list[ConsensusClaimResult] = Field(default_factory=list)
    contradictions: list[ContradictionItem] = Field(default_factory=list)


class ConsensusClaimResult(AutoReviewModel):
    """A consensus claim identified in the literature."""
    claim: str
    supporting_paper_ids: list[str]
    strength: str
    evidence_count: int


class ContradictionItem(AutoReviewModel):
    """A contradiction identified in the literature."""
    claim_a: str
    claim_b: str
    paper_ids_a: list[str]
    paper_ids_b: list[str]
    possible_explanation: str | None = None


# Fix forward references
ContradictionResult.model_rebuild()


class GapAnalysisResult(AutoReviewModel):
    """Result of gap analysis."""
    gaps: list[GapItem] = Field(default_factory=list)
    coverage_score: float = 0.0


class GapItem(AutoReviewModel):
    """An identified gap."""
    expected_topic: str
    current_coverage: str
    severity: str  # "major" or "minor"
    suggested_queries: list[str] = Field(default_factory=list)


GapAnalysisResult.model_rebuild()


CLUSTERING_SYSTEM_PROMPT = """\
You are an expert research analyst performing thematic clustering of scientific findings. \
Group related findings into coherent themes and sub-themes that will form the structure \
of a review paper. Focus on conceptual relationships, not surface-level keyword similarity.
"""

CONTRADICTION_SYSTEM_PROMPT = """\
You are an expert research analyst identifying consensus and contradictions in scientific \
literature. For a given theme, identify claims supported by multiple papers (consensus) \
and claims where papers disagree (contradictions). For contradictions, hypothesize why \
the disagreement exists (different methods, populations, time periods, etc.).
"""

GAP_ANALYSIS_SYSTEM_PROMPT = """\
You are an expert research analyst assessing the completeness of evidence coverage. \
Compare what a review scope document expects to cover against the themes and findings \
actually present in the evidence. Identify gaps where expected sub-topics have insufficient \
or no coverage.
"""


def build_clustering_prompt(
    scope_document: str,
    findings_text: str,
) -> str:
    return f"""\
## Review Scope
{scope_document}

## Extracted Findings
{findings_text}

Group these findings into coherent themes and sub-themes. Each theme should represent \
a major aspect of the review topic. Assign paper IDs to each theme/sub-theme.
"""


def build_contradiction_prompt(theme_name: str, findings_text: str) -> str:
    return f"""\
## Theme: {theme_name}

## Findings in This Theme
{findings_text}

Within this theme, identify:
1. Consensus claims: findings supported by multiple papers
2. Contradictions: findings where papers disagree, with hypotheses about why
"""


def build_gap_analysis_prompt(scope_document: str, themes_text: str) -> str:
    return f"""\
## Review Scope
{scope_document}

## Current Evidence Coverage
{themes_text}

Compare the expected scope against actual coverage. Identify gaps where expected \
sub-topics have insufficient evidence. Rate each gap as "major" (critical to the review) \
or "minor" (nice to have). Suggest targeted search queries to fill each gap. \
Provide a coverage_score from 0.0 (no coverage) to 1.0 (complete coverage).
"""


RETRY_GAP_SEARCH_SYSTEM_PROMPT = """\
You are an expert research librarian. A literature review pipeline identified gaps in \
evidence coverage and searched for papers to fill them, but some gaps remain unfilled. \
The previous queries did not return sufficient results. Generate NEW queries using \
alternative terminology, synonyms, broader formulations, and related concepts.
"""


def build_retry_gap_queries_prompt(
    remaining_gaps: list[dict],
    previous_queries: list[str],
) -> str:
    """Build a prompt to generate retry queries for unfilled gaps."""
    gaps_text = []
    for gap in remaining_gaps:
        topic = gap.get("expected_topic", "Unknown")
        coverage = gap.get("current_coverage", "None")
        gaps_text.append(f"- **{topic}**: {coverage}")
    gaps_block = "\n".join(gaps_text)

    prev_block = "\n".join(f"- {q}" for q in previous_queries)

    return f"""\
The following gaps in evidence coverage remain unfilled after an initial search attempt:

**Remaining Gaps:**
{gaps_block}

**Previous Queries That Were Insufficient:**
{prev_block}

Generate NEW search queries for each remaining gap. You MUST use different and alternative \
terminology from the previous queries listed above — try synonyms, related concepts, \
broader or narrower formulations, and cross-disciplinary terms. For each gap, produce \
3-5 queries suitable for academic search engines.
"""
