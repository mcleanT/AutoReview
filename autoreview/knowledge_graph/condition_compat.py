"""Condition compatibility scoring for MRF propagation gating.

Scores how comparable two claims' experimental contexts are and returns a
coupling strength [0, 1].  Claims sharing similar conditions influence each
other more; claims from different experimental contexts influence each other
less.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Species groupings
# ---------------------------------------------------------------------------

_SPECIES_GROUPS: dict[str, str] = {
    "Mus musculus": "rodent",
    "Rattus norvegicus": "rodent",
    "Homo sapiens": "primate",
    "Macaca fascicularis": "primate",
    "Danio rerio": "fish",
    "Xenopus laevis": "amphibian",
    "Xenopus tropicalis": "amphibian",
    "Gallus gallus": "bird",
    "Drosophila melanogaster": "insect",
    "Caenorhabditis elegans": "nematode",
}


# ---------------------------------------------------------------------------
# ConditionVector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionVector:
    """Compact summary of experimental conditions attached to a KGEdge."""

    organism: str | None = None
    model_system: str | None = None
    in_vitro: bool | None = None
    cell_types: tuple[str, ...] = ()
    tissue: str | None = None
    developmental_stage: str | None = None

    @classmethod
    def from_edge_data(cls, data: dict[str, Any]) -> ConditionVector:
        """Extract from a KGEdge's attribute dict (or a plain dict with the same fields)."""
        conditions: dict[str, Any] = data.get("conditions") or {}
        cell_types = conditions.get("cell_type") or []
        tissue_list = conditions.get("tissue") or []
        return cls(
            organism=data.get("organism"),
            model_system=data.get("model_system"),
            in_vitro=data.get("in_vitro"),
            cell_types=tuple(cell_types),
            tissue=tissue_list[0] if tissue_list else None,
            developmental_stage=conditions.get("developmental_stage"),
        )


# ---------------------------------------------------------------------------
# Primitive scorers
# ---------------------------------------------------------------------------


def _species_score(org_a: str | None, org_b: str | None) -> float:
    """Return a species-level similarity score.

    - 1.0  — same species (exact string match)
    - 0.6  — same phylogenetic group (both rodents, both primates, …)
    - 0.3  — different groups
    - 0.5  — at least one is unknown
    """
    if org_a is None or org_b is None:
        return 0.5
    if org_a == org_b:
        return 1.0
    group_a = _SPECIES_GROUPS.get(org_a)
    group_b = _SPECIES_GROUPS.get(org_b)
    if group_a is not None and group_b is not None and group_a == group_b:
        return 0.6
    return 0.3


def _system_score(sys_a: str | None, sys_b: str | None) -> float:
    """Return a model-system similarity score.

    - 1.0  — same system (case-insensitive)
    - 0.5  — different or unknown
    """
    if sys_a is None or sys_b is None:
        return 0.5
    if sys_a.strip().lower() == sys_b.strip().lower():
        return 1.0
    return 0.5


def _vitro_score(a: bool | None, b: bool | None) -> float:
    """Return an in-vitro vs. in-vivo compatibility score.

    - 1.0  — same modality (both in vitro or both in vivo)
    - 0.6  — different modalities (one in vitro, one in vivo)
    - 0.5  — at least one is unknown
    """
    if a is None or b is None:
        return 0.5
    if a == b:
        return 1.0
    return 0.6


# ---------------------------------------------------------------------------
# Main coupling function
# ---------------------------------------------------------------------------


def condition_coupling(a: ConditionVector, b: ConditionVector) -> float:
    """Compute weighted condition compatibility score [0, 1].

    Weights:
        species  → 0.5
        system   → 0.3
        in_vitro → 0.2

    Returns:
        A float in [0, 1] where 1.0 means identical conditions and lower
        values indicate greater experimental divergence.
    """
    sp = _species_score(a.organism, b.organism)
    sy = _system_score(a.model_system, b.model_system)
    iv = _vitro_score(a.in_vitro, b.in_vitro)
    return 0.5 * sp + 0.3 * sy + 0.2 * iv
