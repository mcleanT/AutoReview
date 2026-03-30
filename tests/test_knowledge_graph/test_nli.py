"""Unit tests for the cross-claim NLI pipeline.

Tests the deterministic logic in nli.py without loading the real DeBERTa model.
All torch/transformers usage is mocked.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from autoreview.knowledge_graph.models import (
    AssertionType,
    BetaPosterior,
    KGEdge,
    KGEvidenceLink,
)


def _make_kg_edge(
    subject_id: str,
    object_id: str,
    predicate: str,
    evidence_strengths: list[str] | None = None,
) -> KGEdge:
    """Build a KGEdge with evidence links for testing."""
    evidence_links = []
    for i, strength in enumerate(evidence_strengths or []):
        evidence_links.append(
            KGEvidenceLink(
                evidence_id=f"ev_{subject_id}_{object_id}_{i}",
                paper_id=f"paper_{i}",
                evidence_strength=strength,
                evidence_direction="supports",
                experiment_summary=f"Test evidence {i}",
                model_system=None,
                sample_size=None,
                key_figure=None,
                publication_date=None,
            )
        )
    return KGEdge(
        edge_id=f"{subject_id}_{predicate}_{object_id}",
        subject_id=subject_id,
        object_id=object_id,
        predicate=predicate,
        direction="positive",
        assertion_type=AssertionType.mechanistic_causal,
        confidence=BetaPosterior(),
        evidence_links=evidence_links,
        source_assertions=[],
        publication_date=None,
    )


# ---------------------------------------------------------------------------
# Helpers to build minimal claim dicts and graphs
# ---------------------------------------------------------------------------


def _make_claim(
    edge_key: str,
    subj_id: str,
    obj_id: str,
    predicate: str,
    subject_name: str | None = None,
    object_name: str | None = None,
    edge_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "edge_key": edge_key,
        "subj_id": subj_id,
        "obj_id": obj_id,
        "key": 0,
        "predicate": predicate,
        "text": f"{subject_name or subj_id} {predicate} {object_name or obj_id}",
        "subject_name": subject_name or subj_id,
        "object_name": object_name or obj_id,
        "edge_data": edge_data or {},
    }


def _build_simple_graph() -> nx.MultiDiGraph:
    """Three-node, two-edge graph with _kg_edge attributes and canonical_name."""
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    g.add_node("A", canonical_name="GeneA")
    g.add_node("B", canonical_name="ProteinB")
    g.add_node("C", canonical_name="PathwayC")
    g.add_edge("A", "B", predicate="activates", evidence_links=[])
    g.add_edge("B", "C", predicate="inhibits", evidence_links=[])
    return g


# ---------------------------------------------------------------------------
# 1. _is_parallel_assertion
# ---------------------------------------------------------------------------


class TestIsParallelAssertion:
    def test_same_subject_same_predicate_different_object_returns_true(self):
        from autoreview.knowledge_graph.nli import _is_parallel_assertion

        a = _make_claim("k1", subj_id="A", obj_id="B", predicate="activates")
        b = _make_claim("k2", subj_id="A", obj_id="C", predicate="activates")
        assert _is_parallel_assertion(a, b) is True

    def test_same_object_same_predicate_different_subject_returns_true(self):
        from autoreview.knowledge_graph.nli import _is_parallel_assertion

        a = _make_claim("k1", subj_id="A", obj_id="C", predicate="inhibits")
        b = _make_claim("k2", subj_id="B", obj_id="C", predicate="inhibits")
        assert _is_parallel_assertion(a, b) is True

    def test_different_subject_and_object_returns_false(self):
        from autoreview.knowledge_graph.nli import _is_parallel_assertion

        a = _make_claim("k1", subj_id="A", obj_id="B", predicate="activates")
        b = _make_claim("k2", subj_id="C", obj_id="D", predicate="activates")
        assert _is_parallel_assertion(a, b) is False

    def test_identical_claims_returns_false(self):
        """Same subject, predicate AND object — not a parallel assertion."""
        from autoreview.knowledge_graph.nli import _is_parallel_assertion

        a = _make_claim("k1", subj_id="A", obj_id="B", predicate="activates")
        b = _make_claim("k2", subj_id="A", obj_id="B", predicate="activates")
        assert _is_parallel_assertion(a, b) is False

    def test_different_predicates_returns_false(self):
        from autoreview.knowledge_graph.nli import _is_parallel_assertion

        a = _make_claim("k1", subj_id="A", obj_id="B", predicate="activates")
        b = _make_claim("k2", subj_id="A", obj_id="C", predicate="inhibits")
        assert _is_parallel_assertion(a, b) is False


# ---------------------------------------------------------------------------
# 2. _predicates_oppose
# ---------------------------------------------------------------------------


class TestPredicatesOppose:
    def test_activates_vs_inhibits_returns_1(self):
        from autoreview.knowledge_graph.nli import _predicates_oppose

        assert _predicates_oppose("activates", "inhibits") == pytest.approx(1.0)

    def test_inhibits_vs_activates_symmetric(self):
        from autoreview.knowledge_graph.nli import _predicates_oppose

        assert _predicates_oppose("inhibits", "activates") == pytest.approx(1.0)

    def test_induces_vs_suppresses_returns_1(self):
        from autoreview.knowledge_graph.nli import _predicates_oppose

        assert _predicates_oppose("induces", "suppresses") == pytest.approx(1.0)

    def test_is_required_for_vs_is_not_required_for_returns_1(self):
        from autoreview.knowledge_graph.nli import _predicates_oppose

        assert _predicates_oppose("is_required_for", "is_not_required_for") == pytest.approx(1.0)

    def test_same_predicate_returns_none(self):
        from autoreview.knowledge_graph.nli import _predicates_oppose

        assert _predicates_oppose("activates", "activates") is None

    def test_unrelated_predicates_returns_none(self):
        from autoreview.knowledge_graph.nli import _predicates_oppose

        assert _predicates_oppose("activates", "binds_to") is None

    def test_case_insensitive_match(self):
        """Predicate strings from extractions may arrive in any case."""
        from autoreview.knowledge_graph.nli import _predicates_oppose

        # The _OPPOSING_PREDICATES sets store lowercase; the function does NOT
        # lowercase internally — so uppercase will NOT match.  This test
        # documents the current (case-sensitive) behaviour.
        result = _predicates_oppose("ACTIVATES", "inhibits")
        assert result is None  # uppercase "ACTIVATES" is not in the frozenset

    def test_upregulates_vs_downregulates_returns_1(self):
        from autoreview.knowledge_graph.nli import _predicates_oppose

        assert _predicates_oppose("upregulates", "downregulates") == pytest.approx(1.0)

    def test_contains_vs_lacks_returns_1(self):
        from autoreview.knowledge_graph.nli import _predicates_oppose

        assert _predicates_oppose("contains", "lacks") == pytest.approx(1.0)

    def test_promotes_vs_prevents_returns_1(self):
        from autoreview.knowledge_graph.nli import _predicates_oppose

        assert _predicates_oppose("promotes", "prevents") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. _build_claims
# ---------------------------------------------------------------------------


class TestBuildClaims:
    def test_claims_structure_from_simple_graph(self):
        from autoreview.knowledge_graph.nli import _build_claims

        g = _build_simple_graph()
        claims, entity_to_claims = _build_claims(g)

        # Two edges → two claims
        assert len(claims) == 2

        # Every claim dict has all required keys
        required_keys = {
            "edge_key",
            "subj_id",
            "obj_id",
            "key",
            "predicate",
            "text",
            "subject_name",
            "object_name",
            "edge_data",
        }
        for claim in claims.values():
            assert required_keys.issubset(claim.keys())

    def test_claim_text_uses_canonical_names(self):
        from autoreview.knowledge_graph.nli import _build_claims

        g = _build_simple_graph()
        claims, _ = _build_claims(g)

        texts = {c["text"] for c in claims.values()}
        assert "GeneA activates ProteinB" in texts
        assert "ProteinB inhibits PathwayC" in texts

    def test_edge_key_format(self):
        from autoreview.knowledge_graph.nli import _build_claims

        g = _build_simple_graph()
        claims, _ = _build_claims(g)

        for key in claims:
            # Format: "{subj}__{predicate}__{obj}__{edge_int}"
            parts = key.split("__")
            assert len(parts) == 4, f"Unexpected key format: {key!r}"

    def test_entity_to_claims_index_correct(self):
        from autoreview.knowledge_graph.nli import _build_claims

        g = _build_simple_graph()
        claims, entity_to_claims = _build_claims(g)

        # ProteinB appears in both edges
        assert len(entity_to_claims["ProteinB"]) == 2
        # GeneA appears only in the first edge
        assert len(entity_to_claims["GeneA"]) == 1
        # PathwayC appears only in the second edge
        assert len(entity_to_claims["PathwayC"]) == 1

    def test_empty_graph_returns_empty_dicts(self):
        from autoreview.knowledge_graph.nli import _build_claims

        g: nx.MultiDiGraph = nx.MultiDiGraph()
        claims, entity_to_claims = _build_claims(g)
        assert claims == {}
        assert entity_to_claims == {}

    def test_node_without_canonical_name_falls_back_to_node_id(self):
        from autoreview.knowledge_graph.nli import _build_claims

        g: nx.MultiDiGraph = nx.MultiDiGraph()
        g.add_node("X")  # no canonical_name
        g.add_node("Y")  # no canonical_name
        g.add_edge("X", "Y", predicate="activates")
        claims, _ = _build_claims(g)
        texts = [c["text"] for c in claims.values()]
        assert texts[0] == "X activates Y"


# ---------------------------------------------------------------------------
# 4. _find_shared_entity_pairs
# ---------------------------------------------------------------------------


class TestFindSharedEntityPairs:
    def _make_entity_to_claims(self) -> dict[str, set[str]]:
        return {
            "GeneA": {"claim_1", "claim_2"},
            "ProteinB": {"claim_2", "claim_3"},
            "PathwayC": {"claim_4"},  # singleton — no pairs
        }

    def test_shared_entity_pairs_generated_correctly(self):
        from autoreview.knowledge_graph.nli import _find_shared_entity_pairs

        e2c = self._make_entity_to_claims()
        pairs = _find_shared_entity_pairs(e2c, max_entities=5)

        # Pairs that share GeneA: (claim_1, claim_2)
        # Pairs that share ProteinB: (claim_2, claim_3)
        assert ("claim_1", "claim_2") in pairs
        assert ("claim_2", "claim_3") in pairs

    def test_singleton_entities_produce_no_pairs(self):
        from autoreview.knowledge_graph.nli import _find_shared_entity_pairs

        e2c = self._make_entity_to_claims()
        pairs = _find_shared_entity_pairs(e2c, max_entities=5)

        # PathwayC has only claim_4; nothing else maps to it
        for key in pairs:
            assert "claim_4" not in key

    def test_pairs_are_alphabetically_sorted(self):
        from autoreview.knowledge_graph.nli import _find_shared_entity_pairs

        e2c = {"EntityX": {"z_claim", "a_claim"}}
        pairs = _find_shared_entity_pairs(e2c, max_entities=5)
        keys = list(pairs.keys())
        assert len(keys) == 1
        a, b = keys[0]
        assert a < b, "Pair keys must be alphabetically ordered"

    def test_max_entities_cap_respected(self):
        from autoreview.knowledge_graph.nli import _find_shared_entity_pairs

        # Two claims share 5 different entities; cap is 3
        e2c = {f"Entity{i}": {"claim_1", "claim_2"} for i in range(5)}
        pairs = _find_shared_entity_pairs(e2c, max_entities=3)
        # Should have exactly one pair
        assert len(pairs) == 1
        shared = list(pairs.values())[0]
        assert len(shared) == 3

    def test_no_pairs_when_all_claims_unique(self):
        from autoreview.knowledge_graph.nli import _find_shared_entity_pairs

        e2c = {
            "GeneA": {"claim_1"},
            "ProteinB": {"claim_2"},
            "PathwayC": {"claim_3"},
        }
        pairs = _find_shared_entity_pairs(e2c, max_entities=5)
        assert pairs == {}


# ---------------------------------------------------------------------------
# 5. classify_cross_claims — integration test with mocked model
# ---------------------------------------------------------------------------


def _make_contradiction_graph() -> nx.MultiDiGraph:
    """Graph with a structural opposition pair and a parallel assertion pair."""
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    g.add_node("A", canonical_name="GeneA")
    g.add_node("B", canonical_name="ProteinB")
    g.add_node("C", canonical_name="PathwayC")

    # Edge 1: A activates B
    g.add_edge("A", "B", predicate="activates", evidence_links=[])
    # Edge 2: A inhibits B  ← opposes edge 1 (same subject/object, opposing predicates)
    g.add_edge("A", "B", predicate="inhibits", evidence_links=[])
    # Edge 3: A activates C  ← parallel assertion to edge 1 (same subject+predicate, diff object)
    g.add_edge("A", "C", predicate="activates", evidence_links=[])
    return g


class TestClassifyCrossClaims:
    def test_parallel_assertions_are_skipped(self):
        from autoreview.knowledge_graph.nli import NLIConfig, classify_cross_claims

        g = _make_contradiction_graph()
        config = NLIConfig(
            filter_parallel_assertions=True,
            use_predicate_opposition=True,
            contradiction_threshold=0.3,
            device="cpu",
        )

        # Patch _load_model so no actual model is loaded (NLI path shouldn't be
        # reached for this graph, but we patch defensively)
        with patch("autoreview.knowledge_graph.nli._load_model") as mock_load:
            fake_model = MagicMock()
            fake_tokenizer = MagicMock()
            fake_device = MagicMock()
            mock_load.return_value = (fake_model, fake_tokenizer, fake_device)

            result = classify_cross_claims(g, config)

        assert result.parallel_skipped >= 1

    def test_predicate_oppositions_are_resolved_structurally(self):
        from autoreview.knowledge_graph.nli import NLIConfig, classify_cross_claims

        g = _make_contradiction_graph()
        config = NLIConfig(
            filter_parallel_assertions=True,
            use_predicate_opposition=True,
            contradiction_threshold=0.3,
            device="cpu",
        )

        with patch("autoreview.knowledge_graph.nli._load_model") as mock_load:
            fake_model = MagicMock()
            mock_load.return_value = (fake_model, MagicMock(), MagicMock())

            result = classify_cross_claims(g, config)

        assert result.structural_resolved >= 1

    def test_result_counts_are_consistent(self):
        from autoreview.knowledge_graph.nli import NLIConfig, classify_cross_claims

        g = _make_contradiction_graph()
        config = NLIConfig(
            filter_parallel_assertions=True,
            use_predicate_opposition=True,
            contradiction_threshold=0.3,
            device="cpu",
        )

        with patch("autoreview.knowledge_graph.nli._load_model") as mock_load:
            mock_load.return_value = (MagicMock(), MagicMock(), MagicMock())
            result = classify_cross_claims(g, config)

        assert (
            result.total_pairs
            == result.parallel_skipped + result.structural_resolved + result.nli_classified
        )

    def test_structural_contradiction_updates_graph_posteriors(self):
        """Structural opposition (p_contra=1.0) should write confidence_mean to edges."""
        from autoreview.knowledge_graph.nli import NLIConfig, classify_cross_claims

        g: nx.MultiDiGraph = nx.MultiDiGraph()
        g.add_node("A", canonical_name="GeneA")
        g.add_node("B", canonical_name="ProteinB")
        kg_edge_a = _make_kg_edge("A", "B", "activates", ["direct_experimental"])
        kg_edge_b = _make_kg_edge("A", "B", "inhibits", ["direct_experimental"])
        g.add_edge("A", "B", predicate="activates", _kg_edge=kg_edge_a)
        g.add_edge("A", "B", predicate="inhibits", _kg_edge=kg_edge_b)

        config = NLIConfig(
            filter_parallel_assertions=False,
            use_predicate_opposition=True,
            contradiction_threshold=0.3,
            device="cpu",
        )

        with patch("autoreview.knowledge_graph.nli._load_model") as mock_load:
            mock_load.return_value = (MagicMock(), MagicMock(), MagicMock())
            result = classify_cross_claims(g, config)

        assert result.claims_updated >= 1
        # At least one edge should now carry confidence_mean
        updated_edges = [data for _, _, data in g.edges(data=True) if "confidence_mean" in data]
        assert len(updated_edges) >= 1

    def test_empty_graph_returns_zero_result(self):
        from autoreview.knowledge_graph.nli import NLIConfig, classify_cross_claims

        g: nx.MultiDiGraph = nx.MultiDiGraph()
        result = classify_cross_claims(g, NLIConfig())

        assert result.total_pairs == 0
        assert result.parallel_skipped == 0
        assert result.claims_updated == 0
        assert result.pair_results == []

    def test_mocked_nli_model_scores_flow_through(self):
        """When NLI is invoked, mocked logits produce expected p_contra on result."""
        import torch

        from autoreview.knowledge_graph.nli import NLIConfig, classify_cross_claims

        # Graph where two claims share an entity but DON'T have opposing predicates
        # and are NOT parallel assertions → should reach NLI path
        g: nx.MultiDiGraph = nx.MultiDiGraph()
        g.add_node("A", canonical_name="GeneA")
        g.add_node("B", canonical_name="ProteinB")
        g.add_node("C", canonical_name="PathwayC")
        # A activates B and A binds_to B — different predicates, same subj+obj
        # These are NOT parallel (different predicates) and not structurally opposing
        kg_edge_a = _make_kg_edge("A", "B", "activates", [])
        kg_edge_b = _make_kg_edge("A", "B", "binds_to", [])
        g.add_edge("A", "B", predicate="activates", _kg_edge=kg_edge_a)
        g.add_edge("A", "B", predicate="binds_to", _kg_edge=kg_edge_b)

        config = NLIConfig(
            filter_parallel_assertions=True,
            use_predicate_opposition=True,
            contradiction_threshold=0.3,
            device="cpu",
        )

        # Build fake logits: high contradiction score [10, 0, 0] → softmax ≈ [1, 0, 0]
        fake_logits = torch.tensor([[10.0, 0.0, 0.0]])
        fake_probs = torch.softmax(fake_logits, dim=-1)

        fake_model_output = MagicMock()
        fake_model_output.logits = fake_logits

        fake_model = MagicMock()
        fake_model.return_value = fake_model_output

        fake_tokenizer = MagicMock()
        # tokenizer(...) returns a dict-like; we need .to(device) to work
        fake_encoding = {"input_ids": torch.zeros(1, 5, dtype=torch.long)}
        fake_tokenizer.return_value = fake_encoding

        # Use a real torch.device so tensor.to(device) works correctly
        import torch as _torch

        fake_device = _torch.device("cpu")

        with (
            patch("autoreview.knowledge_graph.nli._load_model") as mock_load,
            patch("autoreview.knowledge_graph.nli._resolve_device") as mock_resolve,
        ):
            mock_load.return_value = (fake_model, fake_tokenizer, fake_device)
            mock_resolve.return_value = fake_device

            result = classify_cross_claims(g, config)

        assert result.nli_classified >= 1
        assert result.contradictions_p05 >= 1
        # At least one pair_result with high p_contradiction
        assert any(pr.p_contradiction >= 0.5 for pr in result.pair_results)


# ---------------------------------------------------------------------------
# 6. _update_graph_posteriors
# ---------------------------------------------------------------------------


class TestUpdateGraphPosteriors:
    def _make_graph_with_evidence(self) -> tuple[nx.MultiDiGraph, dict[str, Any]]:
        """Create a two-edge graph where each edge has one evidence link."""
        g: nx.MultiDiGraph = nx.MultiDiGraph()
        g.add_node("A", canonical_name="GeneA")
        g.add_node("B", canonical_name="ProteinB")

        kg_edge_a = _make_kg_edge("A", "B", "activates", ["direct_experimental"])  # weight = 1.0
        kg_edge_b = _make_kg_edge("A", "B", "inhibits", ["observational"])  # weight = 0.5

        g.add_edge("A", "B", predicate="activates", _kg_edge=kg_edge_a)
        g.add_edge("A", "B", predicate="inhibits", _kg_edge=kg_edge_b)

        from autoreview.knowledge_graph.nli import _build_claims

        claims, _ = _build_claims(g)
        return g, claims

    def test_beta_binomial_math_is_correct(self):
        """Verify alpha, beta, confidence_mean, controversy_score calculations."""
        from autoreview.knowledge_graph.nli import _update_graph_posteriors

        g, claims = self._make_graph_with_evidence()
        claim_keys = sorted(claims.keys())
        key_a, key_b = claim_keys[0], claim_keys[1]

        # Force a contradiction between claim_a and claim_b with p_contra = 0.8
        nli_results = {
            (key_a, key_b): {"p_contra": 0.8, "p_entail": 0.1, "p_neutral": 0.1, "method": "nli"}
        }

        updated = _update_graph_posteriors(g, nli_results, claims, contradiction_threshold=0.3)
        assert updated == 2

        for u, v, k, data in g.edges(keys=True, data=True):
            if "confidence_mean" not in data:
                continue
            alpha = data["_nli_alpha"]
            beta_val = data["_nli_beta"]
            cross_beta = data["_nli_cross_beta"]

            # base_alpha = 1.0 + sum(evidence weights for THIS edge)
            # base_beta  = 1.0
            # new_beta   = 1.0 + cross_beta
            assert alpha >= 1.0
            assert beta_val == pytest.approx(1.0 + cross_beta)

            expected_mean = alpha / (alpha + beta_val)
            assert data["confidence_mean"] == pytest.approx(expected_mean)

            expected_controversy = min(alpha, beta_val) / max(alpha, beta_val)
            assert data["controversy_score"] == pytest.approx(expected_controversy)

    def test_base_alpha_includes_evidence_weights(self):
        """Edge with direct_experimental evidence should have alpha = 1.0 + 1.0 = 2.0."""
        from autoreview.knowledge_graph.nli import _update_graph_posteriors

        g, claims = self._make_graph_with_evidence()
        claim_keys = sorted(claims.keys())
        key_a, key_b = claim_keys[0], claim_keys[1]

        nli_results = {
            (key_a, key_b): {
                "p_contra": 1.0,
                "p_entail": 0.0,
                "p_neutral": 0.0,
                "method": "predicate_opposition",
            }
        }
        _update_graph_posteriors(g, nli_results, claims, contradiction_threshold=0.3)

        # Find the edge whose claim is key_a and check its alpha
        claim_a_data = claims[key_a]
        u, v, k = claim_a_data["subj_id"], claim_a_data["obj_id"], claim_a_data["key"]
        edge_data = g[u][v][k]

        kg_edge = edge_data.get("_kg_edge")
        if kg_edge and any(
            str(ev.evidence_strength) == "direct_experimental" for ev in kg_edge.evidence_links
        ):
            # This edge had direct_experimental — alpha should reflect weight 1.0
            assert edge_data.get("_nli_alpha", 0) == pytest.approx(2.0)

    def test_below_threshold_pairs_do_not_update_graph(self):
        """Pairs below contradiction_threshold must not mutate graph edges."""
        from autoreview.knowledge_graph.nli import _update_graph_posteriors

        g, claims = self._make_graph_with_evidence()
        claim_keys = sorted(claims.keys())
        key_a, key_b = claim_keys[0], claim_keys[1]

        nli_results = {
            (key_a, key_b): {"p_contra": 0.1, "p_entail": 0.8, "p_neutral": 0.1, "method": "nli"}
        }
        updated = _update_graph_posteriors(g, nli_results, claims, contradiction_threshold=0.3)
        assert updated == 0

        for _, _, data in g.edges(data=True):
            assert "confidence_mean" not in data

    def test_no_nli_results_returns_zero_updates(self):
        from autoreview.knowledge_graph.nli import _update_graph_posteriors

        g, claims = self._make_graph_with_evidence()
        updated = _update_graph_posteriors(g, {}, claims, contradiction_threshold=0.3)
        assert updated == 0

    def test_cross_beta_accumulates_from_opposing_evidence(self):
        """Each evidence link from claim B adds p_contra * weight to claim A's cross_beta."""
        from autoreview.knowledge_graph.nli import _update_graph_posteriors

        g: nx.MultiDiGraph = nx.MultiDiGraph()
        g.add_node("A", canonical_name="GeneA")
        g.add_node("B", canonical_name="ProteinB")

        # Claim A: two direct_experimental evidence links (weights 1.0 each)
        kg_edge_a = _make_kg_edge(
            "A", "B", "activates", ["direct_experimental", "direct_experimental"]
        )
        g.add_edge("A", "B", predicate="activates", _kg_edge=kg_edge_a)
        # Claim B: one observational evidence link (weight 0.5)
        kg_edge_b = _make_kg_edge("A", "B", "inhibits", ["observational"])
        g.add_edge("A", "B", predicate="inhibits", _kg_edge=kg_edge_b)

        from autoreview.knowledge_graph.nli import _build_claims

        claims, _ = _build_claims(g)
        claim_keys = sorted(claims.keys())
        key_a, key_b = claim_keys[0], claim_keys[1]

        p_contra = 0.9
        nli_results = {
            (key_a, key_b): {
                "p_contra": p_contra,
                "p_entail": 0.0,
                "p_neutral": 0.1,
                "method": "nli",
            }
        }
        _update_graph_posteriors(g, nli_results, claims, contradiction_threshold=0.3)

        # claim_a cross_beta: B's ev3 contributes p_contra * 0.5
        claim_a_meta = claims[key_a]
        u_a, v_a, k_a = claim_a_meta["subj_id"], claim_a_meta["obj_id"], claim_a_meta["key"]
        data_a = g[u_a][v_a][k_a]
        if "confidence_mean" in data_a:
            expected_cross_beta_a = p_contra * 0.5  # one ev from B (observational)
            assert data_a["_nli_cross_beta"] == pytest.approx(expected_cross_beta_a, rel=1e-4)

        # claim_b cross_beta: A's ev1 + ev2 contribute p_contra * (1.0 + 1.0)
        claim_b_meta = claims[key_b]
        u_b, v_b, k_b = claim_b_meta["subj_id"], claim_b_meta["obj_id"], claim_b_meta["key"]
        data_b = g[u_b][v_b][k_b]
        if "confidence_mean" in data_b:
            expected_cross_beta_b = p_contra * (1.0 + 1.0)  # two evs from A
            assert data_b["_nli_cross_beta"] == pytest.approx(expected_cross_beta_b, rel=1e-4)


class TestContradictionTaxonomy:
    """Test the v2 contradiction type classification."""

    def test_nli_pair_result_has_contradiction_type(self):
        from autoreview.knowledge_graph.nli import NLIPairResult

        result = NLIPairResult(
            claim_a_id="a",
            claim_b_id="b",
            p_contradiction=0.9,
            p_entailment=0.05,
            p_neutral=0.05,
            method="nli",
            shared_entities=["BMP4"],
            contradiction_type="within_context",
        )
        assert result.contradiction_type == "within_context"

    def test_nli_pair_result_contradiction_type_default_none(self):
        from autoreview.knowledge_graph.nli import NLIPairResult

        result = NLIPairResult(
            claim_a_id="a",
            claim_b_id="b",
            p_contradiction=0.9,
            p_entailment=0.05,
            p_neutral=0.05,
            method="nli",
            shared_entities=["BMP4"],
        )
        assert result.contradiction_type is None

    def test_classify_contradiction_type_within_context(self):
        from autoreview.knowledge_graph.nli import _classify_contradiction_type

        claim_a = {
            "subj_id": "e1",
            "obj_id": "e2",
            "predicate": "induces",
            "edge_data": {"condition_signature": "abc123"},
        }
        claim_b = {
            "subj_id": "e1",
            "obj_id": "e2",
            "predicate": "induces",
            "edge_data": {"condition_signature": "abc123"},
        }
        assert _classify_contradiction_type(claim_a, claim_b) == "within_context"

    def test_classify_contradiction_type_cross_context(self):
        from autoreview.knowledge_graph.nli import _classify_contradiction_type

        claim_a = {
            "subj_id": "e1",
            "obj_id": "e2",
            "predicate": "induces",
            "edge_data": {"condition_signature": "abc123"},
        }
        claim_b = {
            "subj_id": "e1",
            "obj_id": "e2",
            "predicate": "induces",
            "edge_data": {"condition_signature": "def456"},
        }
        assert _classify_contradiction_type(claim_a, claim_b) == "cross_context"

    def test_classify_contradiction_type_structural(self):
        from autoreview.knowledge_graph.nli import _classify_contradiction_type

        claim_a = {
            "subj_id": "e1",
            "obj_id": "e2",
            "predicate": "induces",
            "edge_data": {"condition_signature": "abc123"},
        }
        claim_b = {
            "subj_id": "e1",
            "obj_id": "e2",
            "predicate": "inhibits",
            "edge_data": {"condition_signature": "abc123"},
        }
        assert _classify_contradiction_type(claim_a, claim_b) == "structural"

    def test_classify_contradiction_type_nli_semantic(self):
        from autoreview.knowledge_graph.nli import _classify_contradiction_type

        claim_a = {
            "subj_id": "e1",
            "obj_id": "e2",
            "predicate": "induces",
            "edge_data": {"condition_signature": "abc123"},
        }
        claim_b = {
            "subj_id": "e3",
            "obj_id": "e4",
            "predicate": "regulates",
            "edge_data": {"condition_signature": "abc123"},
        }
        assert _classify_contradiction_type(claim_a, claim_b) == "nli_semantic"

    def test_classify_missing_condition_signature(self):
        """v1 graphs without condition_signature default to nli_semantic."""
        from autoreview.knowledge_graph.nli import _classify_contradiction_type

        claim_a = {
            "subj_id": "e1",
            "obj_id": "e2",
            "predicate": "induces",
            "edge_data": {},
        }
        claim_b = {
            "subj_id": "e1",
            "obj_id": "e2",
            "predicate": "induces",
            "edge_data": {},
        }
        assert _classify_contradiction_type(claim_a, claim_b) == "nli_semantic"
