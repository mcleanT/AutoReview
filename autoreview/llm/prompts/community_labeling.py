"""Prompt models and builders for LLM-based community subfield labeling."""

from __future__ import annotations

from autoreview.models.base import AutoReviewModel

COMMUNITY_LABELING_SYSTEM_PROMPT = """\
You are an expert scientific domain classifier. Given clusters of related scientific claims \
from a knowledge graph, assign each cluster a concise scientific subfield label (2-6 words) \
that captures what topic or research area the cluster represents.

Guidelines:
- Labels should be recognizable scientific subfields or research topics
- Be specific enough to distinguish between clusters (not just "biology")
- Use standard scientific terminology
- Examples of good labels: "Wnt Signaling in Axis Patterning", "Mesoderm Specification", \
"Somitogenesis Clock Genes", "BMP-mediated Dorsoventral Patterning", "ECM and Morphogenesis"
- If a cluster is too small or generic to classify, use "Miscellaneous" as the label
"""


class CommunitySubfieldLabel(AutoReviewModel):
    community_id: int
    subfield: str  # 2-6 word scientific subfield label


class CommunitySubfieldBatch(AutoReviewModel):
    labels: list[CommunitySubfieldLabel]


def build_community_labeling_prompt(
    community_summaries: list[dict],
    domain_context: str = "",
) -> str:
    """Build a prompt for batch community subfield labeling.

    Args:
        community_summaries: List of dicts with keys: community_id, claim_count,
            top_entities, top_entity_types, top_predicates, example_claims.
        domain_context: Optional domain hint (e.g. "gastruloid biology and developmental biology").

    Returns:
        Prompt string for the LLM.
    """
    domain_section = ""
    if domain_context:
        domain_section = f"\n## Domain Context\nThis knowledge graph covers: {domain_context}\n"

    community_blocks: list[str] = []
    for summary in community_summaries:
        community_id = summary.get("community_id", "?")
        claim_count = summary.get("claim_count", 0)

        # Top 5 entities with types
        top_entities = summary.get("top_entities", [])[:5]
        top_entity_types = summary.get("top_entity_types", [])
        # Build entity+type pairs where possible
        entity_lines: list[str] = []
        for i, entity in enumerate(top_entities):
            etype = top_entity_types[i] if i < len(top_entity_types) else "other"
            entity_lines.append(f"  - {entity} ({etype})")
        entities_text = "\n".join(entity_lines) if entity_lines else "  (none)"

        # Top 3 predicates
        top_predicates = summary.get("top_predicates", [])[:3]
        predicates_text = ", ".join(top_predicates) if top_predicates else "(none)"

        # 3 example claims
        example_claims = summary.get("example_claims", [])[:3]
        claims_lines = (
            "\n".join(f"  - {c}" for c in example_claims) if example_claims else "  (none)"
        )

        community_blocks.append(
            f"### Community {community_id} ({claim_count} claims)\n"
            f"**Top entities:**\n{entities_text}\n"
            f"**Top predicates:** {predicates_text}\n"
            f"**Example claims:**\n{claims_lines}"
        )

    communities_text = "\n\n".join(community_blocks)

    return f"""\
Assign a concise scientific subfield label (2-6 words) to each of the following \
knowledge graph communities.{domain_section}

## Communities to Label

{communities_text}

Return a CommunitySubfieldBatch with one CommunitySubfieldLabel per community. \
Each label must have the community_id (integer) and a subfield string (2-6 words). \
Use "Miscellaneous" for clusters that are too small or generic to classify.
"""
