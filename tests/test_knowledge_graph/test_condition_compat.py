"""Tests for autoreview.knowledge_graph.condition_compat."""

from __future__ import annotations

import pytest

from autoreview.knowledge_graph.condition_compat import (
    ConditionVector,
    _species_score,
    _system_score,
    _vitro_score,
    condition_coupling,
)

# ---------------------------------------------------------------------------
# ConditionVector.from_edge_data
# ---------------------------------------------------------------------------


class TestFromEdgeData:
    def test_full_dict(self):
        data = {
            "organism": "Mus musculus",
            "model_system": "Mouse ESC-derived gastruloids",
            "in_vitro": True,
            "conditions": {
                "cell_type": ["ESC", "mesoderm"],
                "tissue": ["posterior mesoderm"],
                "developmental_stage": "E7.5",
            },
        }
        cv = ConditionVector.from_edge_data(data)
        assert cv.organism == "Mus musculus"
        assert cv.model_system == "Mouse ESC-derived gastruloids"
        assert cv.in_vitro is True
        assert cv.cell_types == ("ESC", "mesoderm")
        assert cv.tissue == "posterior mesoderm"
        assert cv.developmental_stage == "E7.5"

    def test_empty_dict_gives_all_none(self):
        cv = ConditionVector.from_edge_data({})
        assert cv.organism is None
        assert cv.model_system is None
        assert cv.in_vitro is None
        assert cv.cell_types == ()
        assert cv.tissue is None
        assert cv.developmental_stage is None

    def test_conditions_none_handled(self):
        data = {"organism": "Homo sapiens", "conditions": None}
        cv = ConditionVector.from_edge_data(data)
        assert cv.organism == "Homo sapiens"
        assert cv.cell_types == ()
        assert cv.tissue is None

    def test_tissue_list_takes_first_element(self):
        data = {"conditions": {"tissue": ["liver", "kidney"]}}
        cv = ConditionVector.from_edge_data(data)
        assert cv.tissue == "liver"

    def test_empty_tissue_list_gives_none(self):
        data = {"conditions": {"tissue": []}}
        cv = ConditionVector.from_edge_data(data)
        assert cv.tissue is None

    def test_in_vitro_false(self):
        data = {"in_vitro": False}
        cv = ConditionVector.from_edge_data(data)
        assert cv.in_vitro is False


# ---------------------------------------------------------------------------
# _species_score
# ---------------------------------------------------------------------------


class TestSpeciesScore:
    def test_identical_species(self):
        assert _species_score("Mus musculus", "Mus musculus") == 1.0

    def test_same_group_rodents(self):
        assert _species_score("Mus musculus", "Rattus norvegicus") == 0.6

    def test_same_group_primates(self):
        assert _species_score("Homo sapiens", "Macaca fascicularis") == 0.6

    def test_same_group_amphibians(self):
        assert _species_score("Xenopus laevis", "Xenopus tropicalis") == 0.6

    def test_different_groups(self):
        assert _species_score("Mus musculus", "Homo sapiens") == 0.3

    def test_different_groups_fish_insect(self):
        assert _species_score("Danio rerio", "Drosophila melanogaster") == 0.3

    def test_unknown_first(self):
        assert _species_score(None, "Mus musculus") == 0.5

    def test_unknown_second(self):
        assert _species_score("Mus musculus", None) == 0.5

    def test_both_unknown(self):
        assert _species_score(None, None) == 0.5

    def test_unlisted_species_vs_known(self):
        # Unlisted species has no group → different group → 0.3
        assert _species_score("Some rattus", "Mus musculus") == 0.3

    def test_both_unlisted_species(self):
        # Both unlisted → group lookups return None.
        # None == None → 0.6 (same group).
        # group_a = _SPECIES_GROUPS.get("Unknown A") → None
        # group_b = _SPECIES_GROUPS.get("Unknown B") → None
        # group_a is not None → False → falls through to 0.3
        assert _species_score("Unknown species A", "Unknown species B") == 0.3


# ---------------------------------------------------------------------------
# _system_score
# ---------------------------------------------------------------------------


class TestSystemScore:
    def test_identical(self):
        assert _system_score("Mouse ESC gastruloids", "Mouse ESC gastruloids") == 1.0

    def test_case_insensitive(self):
        assert _system_score("Mouse ESC", "mouse esc") == 1.0

    def test_whitespace_stripped(self):
        assert _system_score("  HeLa cells  ", "HeLa cells") == 1.0

    def test_different_systems(self):
        assert _system_score("Mouse ESC gastruloids", "HeLa cells") == 0.5

    def test_first_none(self):
        assert _system_score(None, "HeLa cells") == 0.5

    def test_second_none(self):
        assert _system_score("Mouse ESC gastruloids", None) == 0.5

    def test_both_none(self):
        assert _system_score(None, None) == 0.5


# ---------------------------------------------------------------------------
# _vitro_score
# ---------------------------------------------------------------------------


class TestVitroScore:
    def test_both_in_vitro(self):
        assert _vitro_score(True, True) == 1.0

    def test_both_in_vivo(self):
        assert _vitro_score(False, False) == 1.0

    def test_mixed(self):
        assert _vitro_score(True, False) == 0.6

    def test_mixed_reverse(self):
        assert _vitro_score(False, True) == 0.6

    def test_first_none(self):
        assert _vitro_score(None, True) == 0.5

    def test_second_none(self):
        assert _vitro_score(False, None) == 0.5

    def test_both_none(self):
        assert _vitro_score(None, None) == 0.5


# ---------------------------------------------------------------------------
# condition_coupling — integration tests
# ---------------------------------------------------------------------------


class TestConditionCoupling:
    def _make(self, **kwargs) -> ConditionVector:
        return ConditionVector(**kwargs)

    def test_identical_conditions_returns_one(self):
        cv = self._make(
            organism="Mus musculus",
            model_system="Mouse ESC gastruloids",
            in_vitro=True,
        )
        assert condition_coupling(cv, cv) == pytest.approx(1.0)

    def test_completely_unknown_returns_half(self):
        cv = ConditionVector()
        # All scorers return 0.5 → weighted sum = 0.5
        assert condition_coupling(cv, cv) == pytest.approx(0.5)

    def test_same_species_different_system(self):
        a = self._make(organism="Mus musculus", model_system="Gastruloids")
        b = self._make(organism="Mus musculus", model_system="MEF cells")
        score = condition_coupling(a, b)
        # sp=1.0, sy=0.5, iv=0.5 → 0.5 + 0.15 + 0.1 = 0.75
        assert 0.3 < score < 0.95
        assert score == pytest.approx(0.75)

    def test_different_species_same_group(self):
        a = self._make(organism="Mus musculus")
        b = self._make(organism="Rattus norvegicus")
        score = condition_coupling(a, b)
        # sp=0.6, sy=0.5, iv=0.5 → 0.3 + 0.15 + 0.1 = 0.55
        assert score < 0.7
        assert score == pytest.approx(0.55)

    def test_different_species_different_group(self):
        a = self._make(organism="Mus musculus", in_vitro=True)
        b = self._make(organism="Homo sapiens", in_vitro=True)
        score = condition_coupling(a, b)
        # sp=0.3, sy=0.5, iv=1.0 → 0.15 + 0.15 + 0.2 = 0.5
        assert score < 0.6
        assert score == pytest.approx(0.5)

    def test_very_different_conditions_low_score(self):
        a = self._make(organism="Mus musculus", model_system="Gastruloids", in_vitro=True)
        b = self._make(organism="Drosophila melanogaster", model_system="S2 cells", in_vitro=False)
        score = condition_coupling(a, b)
        # sp=0.3, sy=0.5, iv=0.6 → 0.15 + 0.15 + 0.12 = 0.42
        assert score < 0.5

    def test_missing_conditions_moderate_coupling(self):
        a = self._make(organism="Mus musculus", model_system="Gastruloids")
        b = ConditionVector()  # all unknown
        score = condition_coupling(a, b)
        # sp=0.5, sy=0.5, iv=0.5 → all 0.5 → 0.5
        assert 0.4 < score <= 1.0

    def test_symmetry(self):
        a = self._make(organism="Mus musculus", model_system="Gastruloids", in_vitro=True)
        b = self._make(organism="Homo sapiens", model_system="HeLa cells", in_vitro=False)
        assert condition_coupling(a, b) == pytest.approx(condition_coupling(b, a))

    def test_symmetry_with_unknowns(self):
        a = self._make(organism="Mus musculus")
        b = ConditionVector()
        assert condition_coupling(a, b) == pytest.approx(condition_coupling(b, a))

    def test_result_bounded_zero_one(self):
        """Coupling is always in [0, 1] across a variety of configurations."""
        configs = [
            (ConditionVector(), ConditionVector()),
            (
                self._make(organism="Mus musculus", in_vitro=True),
                self._make(organism="Drosophila melanogaster", in_vitro=False),
            ),
            (
                self._make(organism="Homo sapiens", model_system="Primary neurons", in_vitro=False),
                self._make(organism="Homo sapiens", model_system="Primary neurons", in_vitro=False),
            ),
        ]
        for a, b in configs:
            score = condition_coupling(a, b)
            assert 0.0 <= score <= 1.0, f"Out of bounds: {score} for {a}, {b}"

    def test_weighted_formula_correct(self):
        """Verify the exact arithmetic of the weighted formula."""
        a = self._make(organism="Mus musculus", model_system="Gastruloids", in_vitro=True)
        b = self._make(organism="Rattus norvegicus", model_system="Gastruloids", in_vitro=False)
        # sp=0.6 (same rodent group), sy=1.0 (same system), iv=0.6 (different modality)
        expected = 0.5 * 0.6 + 0.3 * 1.0 + 0.2 * 0.6
        assert condition_coupling(a, b) == pytest.approx(expected)

    def test_from_edge_data_round_trip(self):
        """ConditionVector built from a dict should produce consistent coupling."""
        data_a = {
            "organism": "Mus musculus",
            "model_system": "Mouse ESC gastruloids",
            "in_vitro": True,
            "conditions": {"cell_type": ["ESC"], "tissue": []},
        }
        data_b = {
            "organism": "Mus musculus",
            "model_system": "Mouse ESC gastruloids",
            "in_vitro": True,
            "conditions": {"cell_type": ["ESC"], "tissue": []},
        }
        cv_a = ConditionVector.from_edge_data(data_a)
        cv_b = ConditionVector.from_edge_data(data_b)
        assert condition_coupling(cv_a, cv_b) == pytest.approx(1.0)
