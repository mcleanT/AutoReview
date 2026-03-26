"""Tests for graph analysis: communities, contradictions, gaps."""

from __future__ import annotations

import networkx as nx


def _make_test_graph() -> nx.MultiDiGraph:
    """Build a small test graph with known properties."""
    graph = nx.MultiDiGraph()
    # Cluster 1: Wnt-mesoderm
    graph.add_node("wnt", canonical_name="Wnt signaling", entity_type="pathway", paper_count=5)
    graph.add_node(
        "meso",
        canonical_name="mesoderm formation",
        entity_type="biological_process",
        paper_count=4,
    )
    graph.add_node("bra", canonical_name="Brachyury", entity_type="gene", paper_count=3)
    graph.add_edge(
        "wnt",
        "meso",
        key="e1",
        predicate="is_required_for",
        confidence_mean=0.8,
        evidence_count=5,
        controversy_score=0.2,
        evidence_diversity=3,
        independent_source_count=3,
    )
    graph.add_edge(
        "wnt",
        "bra",
        key="e2",
        predicate="induces",
        confidence_mean=0.7,
        evidence_count=3,
        controversy_score=0.3,
        evidence_diversity=2,
        independent_source_count=2,
    )
    # Cluster 2: BMP-dorsal
    graph.add_node("bmp", canonical_name="BMP signaling", entity_type="pathway", paper_count=3)
    graph.add_node(
        "dorsal",
        canonical_name="dorsal-ventral axis",
        entity_type="biological_process",
        paper_count=2,
    )
    graph.add_edge(
        "bmp",
        "dorsal",
        key="e3",
        predicate="induces",
        confidence_mean=0.6,
        evidence_count=2,
        controversy_score=0.7,
        evidence_diversity=2,
        independent_source_count=2,
    )
    # Cross-cluster link
    graph.add_edge(
        "wnt",
        "bmp",
        key="e4",
        predicate="inhibits",
        confidence_mean=0.5,
        evidence_count=1,
        controversy_score=0.9,
        evidence_diversity=1,
        independent_source_count=1,
    )
    return graph


class TestCommunityDetection:
    def test_finds_communities(self):
        from autoreview.knowledge_graph.analysis import detect_communities

        graph = _make_test_graph()
        communities = detect_communities(graph)
        assert len(communities) >= 1
        all_nodes = set()
        for comm in communities:
            all_nodes.update(comm)
        assert all_nodes == set(graph.nodes)

    def test_returns_list_of_sets(self):
        from autoreview.knowledge_graph.analysis import detect_communities

        graph = _make_test_graph()
        communities = detect_communities(graph)
        assert isinstance(communities, list)
        for c in communities:
            assert isinstance(c, set)

    def test_no_overlap_between_communities(self):
        from autoreview.knowledge_graph.analysis import detect_communities

        graph = _make_test_graph()
        communities = detect_communities(graph)
        seen: set[str] = set()
        for c in communities:
            assert c.isdisjoint(seen), "Communities must not overlap"
            seen.update(c)

    def test_single_node_graph(self):
        from autoreview.knowledge_graph.analysis import detect_communities

        solo_graph = nx.MultiDiGraph()
        solo_graph.add_node("solo")
        communities = detect_communities(solo_graph)
        assert len(communities) == 1
        assert communities[0] == {"solo"}


class TestHubEntities:
    def test_hub_ranking(self):
        from autoreview.knowledge_graph.analysis import find_hub_entities

        graph = _make_test_graph()
        hubs = find_hub_entities(graph, top_n=3)
        # Wnt has the most edges (3 outgoing)
        assert hubs[0][0] == "wnt"

    def test_returns_at_most_top_n(self):
        from autoreview.knowledge_graph.analysis import find_hub_entities

        graph = _make_test_graph()
        hubs = find_hub_entities(graph, top_n=2)
        assert len(hubs) <= 2

    def test_scores_are_descending(self):
        from autoreview.knowledge_graph.analysis import find_hub_entities

        graph = _make_test_graph()
        hubs = find_hub_entities(graph, top_n=5)
        scores = [score for _, score in hubs]
        assert scores == sorted(scores, reverse=True)

    def test_tuple_structure(self):
        from autoreview.knowledge_graph.analysis import find_hub_entities

        graph = _make_test_graph()
        hubs = find_hub_entities(graph, top_n=3)
        for node_id, score in hubs:
            assert isinstance(node_id, str)
            assert 0.0 <= score <= 1.0


class TestContradictionDetection:
    def test_finds_high_controversy_edges(self):
        from autoreview.knowledge_graph.analysis import find_contradictions

        graph = _make_test_graph()
        contradictions = find_contradictions(graph, threshold=0.5)
        # Edges e3 (0.7) and e4 (0.9) are above threshold
        assert len(contradictions) >= 2
        edge_ids = [c["edge_key"] for c in contradictions]
        assert "e4" in edge_ids

    def test_excludes_low_controversy_edges(self):
        from autoreview.knowledge_graph.analysis import find_contradictions

        graph = _make_test_graph()
        contradictions = find_contradictions(graph, threshold=0.5)
        edge_ids = [c["edge_key"] for c in contradictions]
        # e1 (0.2) and e2 (0.3) are below threshold
        assert "e1" not in edge_ids
        assert "e2" not in edge_ids

    def test_dict_structure(self):
        from autoreview.knowledge_graph.analysis import find_contradictions

        graph = _make_test_graph()
        contradictions = find_contradictions(graph, threshold=0.5)
        assert len(contradictions) > 0
        required_keys = {
            "edge_key",
            "subject",
            "object",
            "predicate",
            "controversy_score",
            "confidence_mean",
        }
        for c in contradictions:
            assert required_keys.issubset(c.keys())

    def test_high_threshold_returns_empty(self):
        from autoreview.knowledge_graph.analysis import find_contradictions

        graph = _make_test_graph()
        contradictions = find_contradictions(graph, threshold=0.99)
        assert contradictions == []

    def test_zero_threshold_returns_all_edges(self):
        from autoreview.knowledge_graph.analysis import find_contradictions

        graph = _make_test_graph()
        contradictions = find_contradictions(graph, threshold=0.0)
        # All edges have controversy_score > 0.0
        assert len(contradictions) == graph.number_of_edges()


class TestGapAnalysis:
    def test_low_evidence_entities(self):
        from autoreview.knowledge_graph.analysis import find_low_evidence_entities

        graph = _make_test_graph()
        gaps = find_low_evidence_entities(graph, min_degree=2, max_evidence=3)
        assert isinstance(gaps, list)
        # Check structure of results
        if gaps:
            assert "node_id" in gaps[0]
            assert "degree" in gaps[0]

    def test_low_evidence_dict_has_required_keys(self):
        from autoreview.knowledge_graph.analysis import find_low_evidence_entities

        graph = _make_test_graph()
        gaps = find_low_evidence_entities(graph, min_degree=1, max_evidence=100)
        required_keys = {"node_id", "canonical_name", "degree", "total_evidence"}
        for gap in gaps:
            assert required_keys.issubset(gap.keys())

    def test_min_degree_filter(self):
        from autoreview.knowledge_graph.analysis import find_low_evidence_entities

        graph = _make_test_graph()
        # With min_degree=10, no node qualifies
        gaps = find_low_evidence_entities(graph, min_degree=10, max_evidence=100)
        assert gaps == []

    def test_temporal_gaps(self):
        from autoreview.knowledge_graph.analysis import find_temporal_gaps

        graph = _make_test_graph()
        for u, v, k in graph.edges(keys=True):
            graph.edges[u, v, k]["publication_date"] = "2018-01-01"
        gaps = find_temporal_gaps(graph, cutoff_year=2020)
        assert len(gaps) >= 1

    def test_temporal_gaps_all_recent_returns_empty(self):
        from autoreview.knowledge_graph.analysis import find_temporal_gaps

        graph = _make_test_graph()
        for u, v, k in graph.edges(keys=True):
            graph.edges[u, v, k]["publication_date"] = "2024-06-01"
        gaps = find_temporal_gaps(graph, cutoff_year=2020)
        assert gaps == []

    def test_temporal_gaps_missing_date_skipped(self):
        from autoreview.knowledge_graph.analysis import find_temporal_gaps

        graph = _make_test_graph()
        # No publication_date set on any edge
        gaps = find_temporal_gaps(graph, cutoff_year=2020)
        assert gaps == []

    def test_temporal_gaps_dict_structure(self):
        from autoreview.knowledge_graph.analysis import find_temporal_gaps

        graph = _make_test_graph()
        for u, v, k in graph.edges(keys=True):
            graph.edges[u, v, k]["publication_date"] = "2015-03-10"
        gaps = find_temporal_gaps(graph, cutoff_year=2020)
        required_keys = {"edge_key", "subject", "object", "predicate", "publication_date"}
        for gap in gaps:
            assert required_keys.issubset(gap.keys())


class TestExtractSubgraph:
    def test_subgraph_contains_only_requested_nodes(self):
        from autoreview.knowledge_graph.analysis import extract_subgraph

        graph = _make_test_graph()
        sub = extract_subgraph(graph, {"wnt", "meso"})
        assert set(sub.nodes()) == {"wnt", "meso"}

    def test_subgraph_preserves_edges(self):
        from autoreview.knowledge_graph.analysis import extract_subgraph

        graph = _make_test_graph()
        sub = extract_subgraph(graph, {"wnt", "meso", "bra"})
        # e1 (wnt->meso) and e2 (wnt->bra) should be present
        assert sub.number_of_edges() == 2

    def test_subgraph_accepts_set_or_list(self):
        from autoreview.knowledge_graph.analysis import extract_subgraph

        graph = _make_test_graph()
        sub_list = extract_subgraph(graph, ["wnt", "meso"])
        sub_set = extract_subgraph(graph, {"wnt", "meso"})
        assert set(sub_list.nodes()) == set(sub_set.nodes())

    def test_subgraph_unknown_nodes_ignored(self):
        from autoreview.knowledge_graph.analysis import extract_subgraph

        graph = _make_test_graph()
        sub = extract_subgraph(graph, {"wnt", "nonexistent"})
        assert "nonexistent" not in sub.nodes()
        assert "wnt" in sub.nodes()
