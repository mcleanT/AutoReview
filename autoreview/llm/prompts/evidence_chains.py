"""Prompts for evidence chain building and contradiction enrichment."""

from __future__ import annotations


EVIDENCE_CHAIN_SYSTEM_PROMPT = """\
You are an expert research analyst identifying evidence chains across papers. \
An evidence chain traces how a line of inquiry developed: early exploratory work, \
replication, methodological refinement, contradictory findings, and resolution.

Identify chains of connected findings that span multiple papers. Each chain should \
represent one of these patterns:
- **progressive**: Findings that build on each other, each extending the previous
- **replication**: Independent studies that confirm the same result
- **methodological_escalation**: Same question studied with increasingly rigorous methods
- **contradiction_resolution**: Conflicting results that are eventually explained or reconciled
"""

CONTRADICTION_ENRICHMENT_SYSTEM_PROMPT = """\
You are an expert methodologist analyzing why two studies reached conflicting conclusions. \
Focus on concrete methodological differences (sample size, population, measurement approach, \
statistical methods, study design) rather than vague explanations. Suggest a framing strategy \
the review writer can use to present this contradiction constructively.
"""


def build_evidence_chain_prompt(
    theme_name: str,
    findings_with_metadata: list[dict],
    relationships: list[dict],
) -> str:
    """Build prompt for evidence chain identification within a theme.

    Args:
        theme_name: Name of the theme being analyzed.
        findings_with_metadata: List of dicts with paper_id, claim, evidence_strength, year.
        relationships: List of dicts with source_paper_id, target_paper_id, relationship_type, description.
    """
    findings_block = "\n".join(
        f"- [@{f['paper_id']}] ({f.get('year', '?')}, {f.get('evidence_strength', '?')}): {f['claim']}"
        for f in findings_with_metadata
    )
    rel_block = "\n".join(
        f"- [@{r['source_paper_id']}] → [@{r['target_paper_id']}] ({r['relationship_type']}): {r['description']}"
        for r in relationships
    ) if relationships else "(No explicit relationships recorded)"

    return f"""\
## Theme: {theme_name}

## Findings
{findings_block}

## Known Relationships Between Papers
{rel_block}

Identify evidence chains within this theme. For each chain, provide:
1. A descriptive label
2. The chain type (progressive / replication / methodological_escalation / contradiction_resolution)
3. The ordered sequence of paper_ids in the chain
4. A brief description of how the chain develops
"""


def build_contradiction_enrichment_prompt(
    claim_a: str,
    claim_b: str,
    papers_a_methods: list[str],
    papers_b_methods: list[str],
) -> str:
    """Build prompt for enriching a contradiction with methodology differences.

    Args:
        claim_a: First conflicting claim.
        claim_b: Second conflicting claim.
        papers_a_methods: Methods summaries for papers supporting claim A.
        papers_b_methods: Methods summaries for papers supporting claim B.
    """
    methods_a = "\n".join(f"- {m}" for m in papers_a_methods) or "(No methods available)"
    methods_b = "\n".join(f"- {m}" for m in papers_b_methods) or "(No methods available)"

    return f"""\
## Conflicting Claims

**Claim A:** {claim_a}
**Methods of supporting studies:**
{methods_a}

**Claim B:** {claim_b}
**Methods of supporting studies:**
{methods_b}

Analyze:
1. What specific methodological differences could explain the conflicting results?
2. Suggest a framing strategy for presenting this contradiction in a review paper \
(e.g., "The discrepancy likely reflects differences in X, suggesting that Y").
"""
