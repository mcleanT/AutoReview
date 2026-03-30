"""Unit tests for autoreview.knowledge_graph.predicate_algebra.

Tests are organized into four classes:
- TestCanonicalization     — get_canonical_predicate
- TestOpposition           — are_opposing
- TestComposition          — compose_predicates
- TestCompositionStrengths — strength invariants on COMPOSITION_TABLE
"""

from __future__ import annotations

from autoreview.knowledge_graph.predicate_algebra import (
    COMPOSITION_TABLE,
    OPPOSITION_PAIRS,
    CompositionResult,
    are_opposing,
    compose_predicates,
    get_canonical_predicate,
)

# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


class TestCanonicalization:
    def test_activates_maps_to_induces(self) -> None:
        assert get_canonical_predicate("activates") == "induces"

    def test_suppresses_maps_to_inhibits(self) -> None:
        assert get_canonical_predicate("suppresses") == "inhibits"

    def test_triggers_maps_to_induces(self) -> None:
        assert get_canonical_predicate("triggers") == "induces"

    def test_blocks_maps_to_inhibits(self) -> None:
        assert get_canonical_predicate("blocks") == "inhibits"

    def test_is_necessary_for_maps_to_is_required_for(self) -> None:
        assert get_canonical_predicate("is_necessary_for") == "is_required_for"

    def test_is_essential_for_maps_to_is_required_for(self) -> None:
        assert get_canonical_predicate("is_essential_for") == "is_required_for"

    def test_is_critical_for_maps_to_is_required_for(self) -> None:
        assert get_canonical_predicate("is_critical_for") == "is_required_for"

    def test_localizes_to_maps_to_is_located_in(self) -> None:
        assert get_canonical_predicate("localizes_to") == "is_located_in"

    def test_is_expressed_in_maps_to_is_located_in(self) -> None:
        assert get_canonical_predicate("is_expressed_in") == "is_located_in"

    def test_binds_to_maps_to_interacts_with(self) -> None:
        assert get_canonical_predicate("binds_to") == "interacts_with"

    def test_associates_with_maps_to_interacts_with(self) -> None:
        assert get_canonical_predicate("associates_with") == "interacts_with"

    def test_unknown_predicate_returned_unchanged(self) -> None:
        assert get_canonical_predicate("some_novel_predicate") == "some_novel_predicate"

    def test_canonical_predicate_returned_unchanged(self) -> None:
        # Canonical predicates are their own canonical form
        for pred in ("induces", "inhibits", "is_required_for", "regulates"):
            assert get_canonical_predicate(pred) == pred

    def test_all_induction_synonyms(self) -> None:
        synonyms = ["activates", "triggers", "initiates", "promotes", "stimulates", "upregulates"]
        for syn in synonyms:
            assert get_canonical_predicate(syn) == "induces", f"Failed for {syn!r}"

    def test_all_inhibition_synonyms(self) -> None:
        synonyms = ["suppresses", "blocks", "represses", "downregulates", "prevents", "attenuates"]
        for syn in synonyms:
            assert get_canonical_predicate(syn) == "inhibits", f"Failed for {syn!r}"

    def test_enhances_maps_to_induces(self) -> None:
        assert get_canonical_predicate("enhances") == "induces"

    def test_reduces_maps_to_inhibits(self) -> None:
        assert get_canonical_predicate("reduces") == "inhibits"

    def test_enables_maps_to_is_required_for(self) -> None:
        assert get_canonical_predicate("enables") == "is_required_for"


# ---------------------------------------------------------------------------
# Opposition
# ---------------------------------------------------------------------------


class TestOpposition:
    def test_induces_opposes_inhibits(self) -> None:
        assert are_opposing("induces", "inhibits") is True

    def test_inhibits_opposes_induces(self) -> None:
        # Symmetry
        assert are_opposing("inhibits", "induces") is True

    def test_activates_opposes_inhibits_via_canonicalization(self) -> None:
        # "activates" → "induces"; "inhibits" stays "inhibits"
        assert are_opposing("activates", "inhibits") is True

    def test_activates_opposes_suppresses_via_canonicalization(self) -> None:
        # Both sides are synonyms
        assert are_opposing("activates", "suppresses") is True

    def test_same_predicate_not_opposing(self) -> None:
        assert are_opposing("induces", "induces") is False

    def test_same_synonym_not_opposing(self) -> None:
        assert are_opposing("activates", "promotes") is False

    def test_unrelated_predicates_not_opposing(self) -> None:
        assert are_opposing("induces", "is_required_for") is False
        assert are_opposing("correlates_with", "is_located_in") is False

    def test_is_required_for_opposes_is_not_required_for(self) -> None:
        assert are_opposing("is_required_for", "is_not_required_for") is True

    def test_contains_opposes_does_not_contain(self) -> None:
        assert are_opposing("contains", "does_not_contain") is True

    def test_regulates_opposes_does_not_regulate(self) -> None:
        assert are_opposing("regulates", "does_not_regulate") is True

    def test_differentiates_into_opposes_does_not_generate(self) -> None:
        assert are_opposing("differentiates_into", "does_not_generate") is True

    def test_affects_opposes_does_not_affect(self) -> None:
        assert are_opposing("affects", "does_not_affect") is True

    def test_is_located_in_opposes_is_not_located_in(self) -> None:
        assert are_opposing("is_located_in", "is_not_located_in") is True

    def test_correlates_with_opposes_does_not_correlate_with(self) -> None:
        assert are_opposing("correlates_with", "does_not_correlate_with") is True

    def test_interacts_with_opposes_does_not_interact_with(self) -> None:
        assert are_opposing("interacts_with", "does_not_interact_with") is True

    def test_opposition_symmetry_for_all_pairs(self) -> None:
        """Every pair in OPPOSITION_PAIRS must be symmetric."""
        for a, b in OPPOSITION_PAIRS:
            assert are_opposing(a, b) is True, f"Expected {a!r} opposing {b!r}"
            assert are_opposing(b, a) is True, f"Expected {b!r} opposing {a!r}"

    def test_unknown_predicates_not_opposing(self) -> None:
        assert are_opposing("foo_predicate", "bar_predicate") is False

    def test_degrades_opposes_stabilizes(self) -> None:
        assert are_opposing("degrades", "stabilizes") is True

    def test_stabilizes_opposes_degrades(self) -> None:
        # Symmetry
        assert are_opposing("stabilizes", "degrades") is True

    def test_enhances_opposes_inhibits_via_canonicalization(self) -> None:
        # "enhances" → "induces"; "inhibits" stays "inhibits"
        assert are_opposing("enhances", "inhibits") is True

    def test_reduces_opposes_induces_via_canonicalization(self) -> None:
        # "reduces" → "inhibits"; "induces" stays "induces"
        assert are_opposing("reduces", "induces") is True


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class TestComposition:
    def test_induces_induces_gives_induces(self) -> None:
        result = compose_predicates("induces", "induces")
        assert result is not None
        assert result.composed_predicate == "induces"

    def test_induces_inhibits_gives_inhibits(self) -> None:
        result = compose_predicates("induces", "inhibits")
        assert result is not None
        assert result.composed_predicate == "inhibits"

    def test_double_negative_inhibits_inhibits_gives_induces(self) -> None:
        result = compose_predicates("inhibits", "inhibits")
        assert result is not None
        assert result.composed_predicate == "induces"

    def test_inhibits_induces_gives_inhibits(self) -> None:
        result = compose_predicates("inhibits", "induces")
        assert result is not None
        assert result.composed_predicate == "inhibits"

    def test_is_required_for_chain(self) -> None:
        result = compose_predicates("is_required_for", "is_required_for")
        assert result is not None
        assert result.composed_predicate == "is_required_for"

    def test_is_required_for_induces(self) -> None:
        result = compose_predicates("is_required_for", "induces")
        assert result is not None
        assert result.composed_predicate == "induces"

    def test_is_required_for_inhibits(self) -> None:
        result = compose_predicates("is_required_for", "inhibits")
        assert result is not None
        assert result.composed_predicate == "inhibits"

    def test_regulates_induces_gives_regulates(self) -> None:
        result = compose_predicates("regulates", "induces")
        assert result is not None
        assert result.composed_predicate == "regulates"

    def test_regulates_inhibits_gives_regulates(self) -> None:
        result = compose_predicates("regulates", "inhibits")
        assert result is not None
        assert result.composed_predicate == "regulates"

    def test_induces_is_located_in_gives_regulates(self) -> None:
        result = compose_predicates("induces", "is_located_in")
        assert result is not None
        assert result.composed_predicate == "regulates"

    def test_induces_differentiates_into_gives_induces(self) -> None:
        result = compose_predicates("induces", "differentiates_into")
        assert result is not None
        assert result.composed_predicate == "induces"

    def test_is_required_for_differentiates_into(self) -> None:
        result = compose_predicates("is_required_for", "differentiates_into")
        assert result is not None
        assert result.composed_predicate == "is_required_for"

    def test_inhibits_differentiates_into_gives_inhibits(self) -> None:
        result = compose_predicates("inhibits", "differentiates_into")
        assert result is not None
        assert result.composed_predicate == "inhibits"

    def test_synonym_composition_activates_suppresses(self) -> None:
        # "activates" → "induces"; "suppresses" → "inhibits" → induces+inhibits = inhibits
        result = compose_predicates("activates", "suppresses")
        assert result is not None
        assert result.composed_predicate == "inhibits"

    def test_synonym_composition_activates_activates(self) -> None:
        # "activates" → "induces"; "promotes" → "induces" → induces+induces = induces
        result = compose_predicates("activates", "promotes")
        assert result is not None
        assert result.composed_predicate == "induces"

    def test_synonym_composition_blocks_suppresses_double_negative(self) -> None:
        # "blocks" → "inhibits"; "suppresses" → "inhibits" → inhibits+inhibits = induces
        result = compose_predicates("blocks", "suppresses")
        assert result is not None
        assert result.composed_predicate == "induces"

    def test_no_composition_for_unrelated_pair(self) -> None:
        result = compose_predicates("correlates_with", "correlates_with")
        assert result is None

    def test_no_composition_for_unknown_predicates(self) -> None:
        result = compose_predicates("foo", "bar")
        assert result is None

    def test_result_is_composition_result_instance(self) -> None:
        result = compose_predicates("induces", "induces")
        assert isinstance(result, CompositionResult)

    def test_result_has_non_empty_rule(self) -> None:
        result = compose_predicates("induces", "inhibits")
        assert result is not None
        assert isinstance(result.rule, str)
        assert len(result.rule) > 0

    def test_synonym_localizes_to_composes(self) -> None:
        # "localizes_to" → "is_located_in"; induces + is_located_in = regulates
        result = compose_predicates("induces", "localizes_to")
        assert result is not None
        assert result.composed_predicate == "regulates"


# ---------------------------------------------------------------------------
# Composition strength invariants
# ---------------------------------------------------------------------------


class TestCompositionStrengths:
    def test_all_strengths_in_range(self) -> None:
        """Every composition strength must be in (0, 1]."""
        for key, cr in COMPOSITION_TABLE.items():
            assert 0.0 < cr.strength <= 1.0, (
                f"Strength {cr.strength} out of range (0, 1] for key {key}"
            )

    def test_direct_activation_stronger_than_required_for_activation(self) -> None:
        """induces+induces (0.7) should be stronger than is_required_for+induces (0.4)."""
        direct = COMPOSITION_TABLE[("induces", "induces")]
        indirect = COMPOSITION_TABLE[("is_required_for", "induces")]
        assert direct.strength > indirect.strength

    def test_double_negative_weaker_than_direct_activation(self) -> None:
        """inhibits+inhibits (0.5) should be weaker than direct induces+induces (0.7)."""
        double_neg = COMPOSITION_TABLE[("inhibits", "inhibits")]
        direct = COMPOSITION_TABLE[("induces", "induces")]
        assert double_neg.strength < direct.strength

    def test_regulates_compositions_have_lowest_strength(self) -> None:
        """Compositions that produce 'regulates' (direction unknown) should be weakest."""
        regulates_strengths = [
            cr.strength for cr in COMPOSITION_TABLE.values() if cr.composed_predicate == "regulates"
        ]
        other_strengths = [
            cr.strength for cr in COMPOSITION_TABLE.values() if cr.composed_predicate != "regulates"
        ]
        assert max(regulates_strengths) <= min(other_strengths)
