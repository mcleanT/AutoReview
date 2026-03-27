"""LLM-based subfield labeling for knowledge graph communities."""

from __future__ import annotations

from collections import Counter
from typing import Any

import networkx as nx
import structlog

from autoreview.knowledge_graph.contradiction_viz import (
    CommunityLabel,
    ContradictionVizData,
)
from autoreview.llm.prompts.community_labeling import (
    COMMUNITY_LABELING_SYSTEM_PROMPT,
    CommunitySubfieldBatch,
    build_community_labeling_prompt,
)
from autoreview.llm.provider import LLMProvider

log = structlog.get_logger(__name__)


def _build_community_summary(
    community_id: int,
    claim_ids: set[str],
    graph: nx.MultiDiGraph,
) -> dict[str, Any]:
    """Build a summary dict for a single community.

    Collects top entities (with types), top predicates, and example claims.
    """
    predicate_counter: Counter[str] = Counter()
    entity_counter: Counter[str] = Counter()
    entity_type_map: dict[str, str] = {}
    example_claims: list[str] = []

    for claim_id in claim_ids:
        parts = claim_id.split("__")
        if len(parts) < 4:
            continue
        u, predicate, v = parts[0], parts[1], parts[2]

        predicate_counter[predicate] += 1

        u_name = graph.nodes[u].get("canonical_name", u) if graph.has_node(u) else u
        v_name = graph.nodes[v].get("canonical_name", v) if graph.has_node(v) else v
        entity_counter[u_name] += 1
        entity_counter[v_name] += 1

        if graph.has_node(u):
            u_type: str = graph.nodes[u].get("entity_type", "other") or "other"
            entity_type_map.setdefault(u_name, u_type)
        if graph.has_node(v):
            v_type: str = graph.nodes[v].get("entity_type", "other") or "other"
            entity_type_map.setdefault(v_name, v_type)

        if len(example_claims) < 3:
            example_claims.append(f"{u_name} {predicate} {v_name}")

    top_entities = [e for e, _ in entity_counter.most_common(5)]
    top_entity_types = [entity_type_map.get(e, "other") for e in top_entities]
    top_predicates = [p for p, _ in predicate_counter.most_common(3)]

    return {
        "community_id": community_id,
        "claim_count": len(claim_ids),
        "top_entities": top_entities,
        "top_entity_types": top_entity_types,
        "top_predicates": top_predicates,
        "example_claims": example_claims,
    }


async def enrich_community_subfields(
    viz_data: ContradictionVizData,
    graph: nx.MultiDiGraph,
    communities: list[set[str]],
    llm: LLMProvider,
    domain_context: str = "",
    min_claims: int = 5,
) -> ContradictionVizData:
    """Enrich community labels with LLM-generated subfield names.

    Labels communities with min_claims+ claims or involved in disagreements.
    Returns a new ContradictionVizData with updated community_labels.

    Args:
        viz_data: Existing ContradictionVizData with community_labels.
        graph: The knowledge graph MultiDiGraph with node attributes.
        communities: List of sets of claim IDs, one set per community.
        llm: LLMProvider for structured generation.
        domain_context: Optional domain hint for the labeling prompt.
        min_claims: Minimum claim count to include a community for labeling.

    Returns:
        A new ContradictionVizData instance with updated community_labels.
    """
    # Collect community IDs that appear in any disagreement
    disagreement_community_ids: set[int] = set()
    for disagreement in viz_data.community_disagreements:
        disagreement_community_ids.add(disagreement.comm_a_id)
        disagreement_community_ids.add(disagreement.comm_b_id)

    # Determine which communities are significant
    significant_ids: list[int] = []
    for community_id, claim_ids in enumerate(communities):
        if len(claim_ids) >= min_claims or community_id in disagreement_community_ids:
            significant_ids.append(community_id)

    if not significant_ids:
        log.info(
            "enrich_community_subfields.no_significant_communities",
            min_claims=min_claims,
            total_communities=len(communities),
        )
        return viz_data

    log.info(
        "enrich_community_subfields.start",
        significant_communities=len(significant_ids),
        total_communities=len(communities),
        domain_context=domain_context or "(none)",
    )

    # Build summaries for all significant communities
    summaries: list[dict[str, Any]] = []
    for community_id in significant_ids:
        claim_ids = communities[community_id]
        summaries.append(_build_community_summary(community_id, claim_ids, graph))

    # Single batched LLM call
    try:
        response = await llm.generate_structured(
            prompt=build_community_labeling_prompt(summaries, domain_context),
            response_model=CommunitySubfieldBatch,
            system=COMMUNITY_LABELING_SYSTEM_PROMPT,
            max_tokens=4096,
            temperature=0.1,
        )
        batch: CommunitySubfieldBatch = response.parsed  # type: ignore[assignment]

        log.info(
            "enrich_community_subfields.llm_done",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            labels_returned=len(batch.labels),
        )
    except Exception as exc:
        log.warning(
            "enrich_community_subfields.llm_failed",
            error=str(exc),
            significant_communities=len(significant_ids),
        )
        return viz_data

    # Build updated community_labels dict from the existing one
    updated_labels: dict[int, CommunityLabel] = dict(viz_data.community_labels)
    enriched_count = 0

    for result in batch.labels:
        community_id = result.community_id
        if community_id in updated_labels:
            existing = updated_labels[community_id]
            updated_labels[community_id] = existing.model_copy(update={"subfield": result.subfield})
            enriched_count += 1
        else:
            log.warning(
                "enrich_community_subfields.unknown_community_id",
                community_id=community_id,
            )

    log.info(
        "enrich_community_subfields.done",
        enriched_communities=enriched_count,
        total_input_tokens=response.input_tokens,
        total_output_tokens=response.output_tokens,
    )

    return viz_data.model_copy(update={"community_labels": updated_labels})
