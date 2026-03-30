"""Entity deduplication, predicate normalization, and assertion merging for the knowledge graph.

Three-pass entity resolution:
1. Ontology ID match — entities sharing an ontology_id merge regardless of name.
2. Canonical name normalization — same lowercased/stripped name merges.
3. Fuzzy similarity — within same entity_type, merge at rapidfuzz ratio >= 85.
   For 'other' type, token-based blocking avoids O(n^2) comparisons.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import structlog
from rapidfuzz import fuzz

from autoreview.knowledge_graph.models import EntityType, KGEntity

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Entity Registry
# ---------------------------------------------------------------------------


@dataclass
class EntityRegistry:
    """Result of entity deduplication."""

    entities: dict[str, KGEntity] = field(default_factory=dict)
    """entity_id → KGEntity"""

    surface_to_id: dict[str, str] = field(default_factory=dict)
    """lowercased surface form (canonical_name + aliases) → entity_id"""

    merge_log: list[dict[str, Any]] = field(default_factory=list)
    """Audit trail: each entry has 'pass', 'merged_entities', 'reason'."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FUZZY_MERGE_THRESHOLD = 85
_FUZZY_MERGE_THRESHOLD_BLOCKED = 82  # lower threshold for token-blocked 'other' entities
_FUZZY_FLAG_THRESHOLD = 70


def _make_entity_id(canonical_name: str, entity_type: str) -> str:
    """Deterministic hash of (canonical_name, entity_type)."""
    key = f"{canonical_name.lower().strip()}|{entity_type}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]  # noqa: S324


def _normalize_name(name: str) -> str:
    """Lowercase + strip for Pass 2 comparison."""
    return name.lower().strip()


def _merge_entity_group(group: list[dict[str, Any]]) -> KGEntity:
    """Merge a list of raw entity dicts into a single KGEntity.

    The entity with an ontology_id is preferred as canonical.
    Aliases accumulate; paper_ids are unioned.
    """
    # Sort so that entities WITH an ontology_id come first → they win canonical.
    ordered = sorted(group, key=lambda e: (e.get("ontology_id") is None, e["canonical_name"]))
    canonical = ordered[0]

    all_aliases: list[str] = []
    all_paper_ids: list[str] = []
    for ent in ordered:
        # Add non-canonical names as aliases.
        if ent["canonical_name"] != canonical["canonical_name"]:
            all_aliases.append(ent["canonical_name"])
        all_aliases.extend(ent.get("aliases") or [])
        all_paper_ids.extend(ent.get("paper_ids") or [])

    # Deduplicate aliases while preserving order.
    seen: set[str] = set()
    deduped_aliases: list[str] = []
    for a in all_aliases:
        if a not in seen:
            seen.add(a)
            deduped_aliases.append(a)

    unique_paper_ids = list(dict.fromkeys(all_paper_ids))

    entity_id = _make_entity_id(canonical["canonical_name"], canonical["entity_type"])
    return KGEntity(
        entity_id=entity_id,
        canonical_name=canonical["canonical_name"],
        entity_type=EntityType(canonical["entity_type"]),
        ontology_id=canonical.get("ontology_id"),
        ontology_source=canonical.get("ontology_source"),
        aliases=deduped_aliases,
        paper_count=len(unique_paper_ids),
        source_paper_ids=unique_paper_ids,
    )


def _register_entity(registry: EntityRegistry, entity: KGEntity) -> None:
    """Add entity to registry and update surface_to_id index."""
    registry.entities[entity.entity_id] = entity
    registry.surface_to_id[_normalize_name(entity.canonical_name)] = entity.entity_id
    for alias in entity.aliases:
        registry.surface_to_id[_normalize_name(alias)] = entity.entity_id


# ---------------------------------------------------------------------------
# Pass 1: ontology_id grouping
# ---------------------------------------------------------------------------


def _pass1_ontology(
    raw_entities: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group by ontology_id. Returns (merged_raws, merge_log_entries)."""
    has_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    no_id: list[dict[str, Any]] = []

    for ent in raw_entities:
        oid = ent.get("ontology_id")
        if oid:
            has_id[oid].append(ent)
        else:
            no_id.append(ent)

    merged: list[dict[str, Any]] = []
    log_entries: list[dict[str, Any]] = []

    for oid, group in has_id.items():
        if len(group) > 1:
            log_entries.append(
                {
                    "pass": "ontology_id",
                    "merged_entities": [e["canonical_name"] for e in group],
                    "reason": f"shared ontology_id={oid}",
                }
            )
        # Produce a single merged raw dict to carry forward.
        merged_entity = _merge_entity_group(group)
        merged.append(
            {
                "canonical_name": merged_entity.canonical_name,
                "entity_type": str(merged_entity.entity_type),
                "ontology_id": merged_entity.ontology_id,
                "ontology_source": merged_entity.ontology_source,
                "aliases": merged_entity.aliases,
                "paper_ids": merged_entity.source_paper_ids,
            }
        )

    return merged + no_id, log_entries


# ---------------------------------------------------------------------------
# Pass 2: canonical name normalization
# ---------------------------------------------------------------------------


def _pass2_name(
    raw_entities: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group by (entity_type, normalized_name). Returns (merged_raws, log_entries)."""
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for ent in raw_entities:
        key = (ent["entity_type"], _normalize_name(ent["canonical_name"]))
        buckets[key].append(ent)

    merged: list[dict[str, Any]] = []
    log_entries: list[dict[str, Any]] = []

    for (entity_type, norm_name), group in buckets.items():
        if len(group) > 1:
            log_entries.append(
                {
                    "pass": "canonical_name",
                    "merged_entities": [e["canonical_name"] for e in group],
                    "reason": f"same normalized name='{norm_name}' entity_type={entity_type}",
                }
            )
        merged_entity = _merge_entity_group(group)
        merged.append(
            {
                "canonical_name": merged_entity.canonical_name,
                "entity_type": str(merged_entity.entity_type),
                "ontology_id": merged_entity.ontology_id,
                "ontology_source": merged_entity.ontology_source,
                "aliases": merged_entity.aliases,
                "paper_ids": merged_entity.source_paper_ids,
            }
        )

    return merged, log_entries


# ---------------------------------------------------------------------------
# Pass 3: fuzzy matching
# ---------------------------------------------------------------------------


def _significant_tokens(name: str) -> set[str]:
    """Return lowercase word tokens longer than 3 chars (stop-word-lite filter)."""
    return {t for t in name.lower().split() if len(t) > 3}


def _build_token_blocks(group: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group entities by shared significant tokens (for 'other' type blocking)."""
    # Union-Find approach for overlapping token sets.
    n = len(group)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    token_sets = [_significant_tokens(e["canonical_name"]) for e in group]

    for i in range(n):
        for j in range(i + 1, n):
            if token_sets[i] & token_sets[j]:  # shared token → same block
                union(i, j)

    blocks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i, ent in enumerate(group):
        blocks[find(i)].append(ent)

    return list(blocks.values())


def _pass3_fuzzy(
    raw_entities: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fuzzy-merge within each entity_type. Returns (merged_raws, log_entries)."""
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ent in raw_entities:
        by_type[ent["entity_type"]].append(ent)

    all_merged: list[dict[str, Any]] = []
    log_entries: list[dict[str, Any]] = []

    for entity_type, type_group in by_type.items():
        # For 'other' type, use token-based blocks to avoid O(n^2).
        if entity_type == EntityType.other:
            blocks = _build_token_blocks(type_group)
            threshold = _FUZZY_MERGE_THRESHOLD_BLOCKED
        else:
            blocks = [type_group]
            threshold = _FUZZY_MERGE_THRESHOLD

        for block in blocks:
            merged_in_block, block_log = _fuzzy_merge_block(block, threshold=threshold)
            all_merged.extend(merged_in_block)
            log_entries.extend(block_log)

    return all_merged, log_entries


def _fuzzy_merge_block(
    block: list[dict[str, Any]],
    threshold: int = _FUZZY_MERGE_THRESHOLD,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Union-Find fuzzy merge within a single block.

    Args:
        block: List of raw entity dicts to compare pairwise.
        threshold: Fuzzy ratio threshold for merging (default 85; use lower for
                   token-blocked 'other' entities where blocking already pre-filters).
    """
    n = len(block)
    if n == 1:
        return block, []

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    log_entries: list[dict[str, Any]] = []

    for i in range(n):
        for j in range(i + 1, n):
            ratio = fuzz.ratio(block[i]["canonical_name"], block[j]["canonical_name"])
            if ratio >= threshold:
                union(i, j)
                log_entries.append(
                    {
                        "pass": "fuzzy",
                        "merged_entities": [block[i]["canonical_name"], block[j]["canonical_name"]],
                        "reason": f"fuzzy ratio={ratio:.1f}",
                        "flagged_ambiguous": False,
                    }
                )
            elif ratio >= _FUZZY_FLAG_THRESHOLD:
                log_entries.append(
                    {
                        "pass": "fuzzy",
                        "merged_entities": [block[i]["canonical_name"], block[j]["canonical_name"]],
                        "reason": f"fuzzy ratio={ratio:.1f} (ambiguous zone, NOT merged)",
                        "flagged_ambiguous": True,
                    }
                )

    # Collect groups.
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i, ent in enumerate(block):
        groups[find(i)].append(ent)

    merged: list[dict[str, Any]] = []
    for group in groups.values():
        merged_entity = _merge_entity_group(group)
        merged.append(
            {
                "canonical_name": merged_entity.canonical_name,
                "entity_type": str(merged_entity.entity_type),
                "ontology_id": merged_entity.ontology_id,
                "ontology_source": merged_entity.ontology_source,
                "aliases": merged_entity.aliases,
                "paper_ids": merged_entity.source_paper_ids,
            }
        )

    return merged, log_entries


# ---------------------------------------------------------------------------
# Public API: deduplicate_entities
# ---------------------------------------------------------------------------


def deduplicate_entities(entities: list[dict[str, Any]]) -> EntityRegistry:
    """Run three-pass entity deduplication and return an EntityRegistry.

    Pass 1: Group by ontology_id (skip None).
    Pass 2: Group by normalized (lowercase, stripped) name within entity_type.
    Pass 3: Fuzzy similarity within entity_type (rapidfuzz ratio >= 85).
             'other' type uses token-based blocking.

    Args:
        entities: List of raw entity dicts with keys canonical_name, entity_type,
                  ontology_id, ontology_source, aliases, paper_ids.

    Returns:
        EntityRegistry with deduplicated entities and audit merge_log.
    """
    registry = EntityRegistry()

    raw, log1 = _pass1_ontology(entities)
    registry.merge_log.extend(log1)

    raw, log2 = _pass2_name(raw)
    registry.merge_log.extend(log2)

    raw, log3 = _pass3_fuzzy(raw)
    registry.merge_log.extend(log3)

    for raw_ent in raw:
        entity = _merge_entity_group([raw_ent])
        _register_entity(registry, entity)

    logger.info(
        "entity_dedup_complete",
        input_count=len(entities),
        output_count=len(registry.entities),
        merges=len(registry.merge_log),
    )
    return registry


# ---------------------------------------------------------------------------
# Predicate Normalization
# ---------------------------------------------------------------------------


class PredicateNormalizer:
    """Normalizes predicate synonyms to canonical forms.

    Uses an exact lookup table. Unknown predicates are kept as-is.
    All normalizations are logged for auditability.
    """

    SYNONYM_TABLE: dict[str, str] = {
        # is_required_for
        "is_necessary_for": "is_required_for",
        "is_essential_for": "is_required_for",
        "is_critical_for": "is_required_for",
        "is_needed_for": "is_required_for",
        "enables": "is_required_for",
        # induces
        "activates": "induces",
        "triggers": "induces",
        "initiates": "induces",
        "promotes": "induces",
        "stimulates": "induces",
        "upregulates": "induces",
        "enhances": "induces",
        # inhibits
        "suppresses": "inhibits",
        "blocks": "inhibits",
        "represses": "inhibits",
        "downregulates": "inhibits",
        "prevents": "inhibits",
        "attenuates": "inhibits",
        "reduces": "inhibits",
        # expresses
        "produces": "expresses",
        "encodes": "expresses",
        "transcribes": "expresses",
        # regulates
        "modulates": "regulates",
        "controls": "regulates",
        "mediates": "regulates",
        # differentiates_into
        "gives_rise_to": "differentiates_into",
        "develops_into": "differentiates_into",
        "matures_into": "differentiates_into",
        "becomes": "differentiates_into",
        # interacts_with
        "binds_to": "interacts_with",
        "associates_with": "interacts_with",
        "complexes_with": "interacts_with",
        # is_marker_for
        "marks": "is_marker_for",
        "identifies": "is_marker_for",
        "labels": "is_marker_for",
        "characterizes": "is_marker_for",
        # contains
        "comprises": "contains",
        "includes": "contains",
        "consists_of": "contains",
        # is_located_in
        "localizes_to": "is_located_in",
        "is_expressed_in": "is_located_in",
        "is_found_in": "is_located_in",
    }

    def __init__(self) -> None:
        self.log: list[dict[str, str]] = []

    def normalize(self, predicate: str) -> str:
        """Normalize a predicate to its canonical form.

        Args:
            predicate: Raw predicate string.

        Returns:
            Canonical predicate string (unchanged if not in synonym table).
        """
        canonical = self.SYNONYM_TABLE.get(predicate, predicate)
        self.log.append(
            {
                "original": predicate,
                "canonical": canonical,
                "method": "exact_lookup" if canonical != predicate else "passthrough",
            }
        )
        return canonical


# Module-level singleton for convenience function.
_module_normalizer = PredicateNormalizer()


def normalize_predicate(predicate: str) -> str:
    """Normalize a predicate using the module-level PredicateNormalizer.

    Args:
        predicate: Raw predicate string.

    Returns:
        Canonical predicate string.
    """
    return _module_normalizer.normalize(predicate)


# ---------------------------------------------------------------------------
# Assertion Merging
# ---------------------------------------------------------------------------


@dataclass
class MergeResult:
    """Result of assertion merging."""

    assertions: list[dict[str, Any]] = field(default_factory=list)
    merge_log: list[dict[str, Any]] = field(default_factory=list)


def merge_assertions(assertions: list[dict[str, Any]]) -> MergeResult:
    """Merge assertions sharing the same (subject_id, predicate, object_id) triple.

    For each group:
    - source_assertions: list of draft_ids
    - evidence_unit_ids: union of all evidence unit ids
    - direction: unanimous value or None if conflicted (direction_conflict=True)
    - assertion_type: majority vote
    - publication_date: earliest date (lexicographic ISO-8601 string comparison)

    Self-loops (subject_id == object_id) are valid and not filtered.

    Args:
        assertions: List of raw assertion dicts.

    Returns:
        MergeResult with merged assertions and audit merge_log.
    """
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for assertion in assertions:
        key = (assertion["subject_id"], assertion["predicate"], assertion["object_id"])
        groups[key].append(assertion)

    result = MergeResult()

    for (subject_id, predicate, object_id), group in groups.items():
        draft_ids = [a["draft_id"] for a in group]
        paper_ids = [a["paper_id"] for a in group]

        # Accumulate evidence_unit_ids (flatten + deduplicate).
        all_evidence: list[str] = []
        for a in group:
            all_evidence.extend(a.get("evidence_unit_ids") or [])
        unique_evidence = list(dict.fromkeys(all_evidence))

        # Direction: unanimous → keep; conflict → None.
        directions = {a.get("direction") for a in group}
        if len(directions) == 1:
            direction = directions.pop()
            direction_conflict = False
        else:
            direction = None
            direction_conflict = True

        # Assertion type: majority vote.
        type_counts: dict[str, int] = defaultdict(int)
        for a in group:
            type_counts[a.get("assertion_type", "mechanistic_causal")] += 1
        assertion_type = max(type_counts, key=lambda t: type_counts[t])

        # Publication date: earliest (ISO-8601 strings compare lexicographically).
        dates = [a["publication_date"] for a in group if a.get("publication_date")]
        earliest_date = min(dates) if dates else None

        merged = {
            "subject_id": subject_id,
            "predicate": predicate,
            "object_id": object_id,
            "direction": direction,
            "direction_conflict": direction_conflict,
            "assertion_type": assertion_type,
            "evidence_unit_ids": unique_evidence,
            "source_assertions": draft_ids,
            "publication_date": earliest_date,
        }
        result.assertions.append(merged)

        if len(group) > 1:
            result.merge_log.append(
                {
                    "merged_draft_ids": draft_ids,
                    "papers": paper_ids,
                    "direction_conflict": direction_conflict,
                    "triple": (subject_id, predicate, object_id),
                }
            )

    logger.info(
        "assertion_merge_complete",
        input_count=len(assertions),
        output_count=len(result.assertions),
        merges=len(result.merge_log),
    )
    return result
