"""Tests for graph analysis: communities, contradictions, gaps."""

from __future__ import annotations

import networkx as nx

from autoreview.knowledge_graph.cluster import (
    build_topic_clusters,
    detect_finding_contradictions,
    form_findings,
)


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


# ---------------------------------------------------------------------------
# Helpers for contradiction centrality tests
# ---------------------------------------------------------------------------


def _make_contradiction_edge(
    graph: nx.MultiDiGraph,
    u: str,
    v: str,
    key: str,
    predicate: str,
    direction: str | None = None,
    organism: str | None = "mouse",
    model_system: str | None = "in_vivo",
    in_vitro: bool | None = False,
    conditions: dict | None = None,
) -> None:
    """Add a minimally-valid edge for contradiction detection tests."""
    graph.add_edge(
        u,
        v,
        key=key,
        edge_id=key,
        predicate=predicate,
        direction=direction,
        organism=organism,
        model_system=model_system,
        in_vitro=in_vitro,
        conditions=conditions or {},
    )


class TestContradictionCentrality:
    def test_contradiction_centrality_basic(self):
        """Node involved in contradictions should have positive score."""
        from autoreview.knowledge_graph.analysis import score_contradiction_centrality

        graph = nx.MultiDiGraph()
        graph.add_node("A", canonical_name="Entity A")
        graph.add_node("B", canonical_name="Entity B")
        # A->B with "induces" and "inhibits" — a predicate opposition
        _make_contradiction_edge(graph, "A", "B", "e1", "induces")
        _make_contradiction_edge(graph, "A", "B", "e2", "inhibits")

        results = score_contradiction_centrality(graph)

        assert len(results) > 0
        node_ids = {r["node_id"] for r in results}
        assert "A" in node_ids
        assert "B" in node_ids
        for r in results:
            assert r["raw_score"] > 0.0

    def test_contradiction_centrality_empty(self):
        """Graph with no contradictions should return empty list."""
        from autoreview.knowledge_graph.analysis import score_contradiction_centrality

        graph = nx.MultiDiGraph()
        graph.add_node("A", canonical_name="Entity A")
        graph.add_node("B", canonical_name="Entity B")
        # Only one edge — no pair to contradict
        _make_contradiction_edge(graph, "A", "B", "e1", "induces")

        results = score_contradiction_centrality(graph)
        assert results == []

    def test_contradiction_centrality_hub_detection(self):
        """Node involved in multiple contradictions should score higher than one with one."""
        from autoreview.knowledge_graph.analysis import score_contradiction_centrality

        # Node B is involved in two contradiction pairs: A->B and B->C
        graph = nx.MultiDiGraph()
        graph.add_node("A", canonical_name="Entity A")
        graph.add_node("B", canonical_name="Entity B")
        graph.add_node("C", canonical_name="Entity C")
        graph.add_node("D", canonical_name="Entity D")

        # Contradiction pair on A->B (B and A both involved)
        _make_contradiction_edge(graph, "A", "B", "e1", "induces")
        _make_contradiction_edge(graph, "A", "B", "e2", "inhibits")

        # Contradiction pair on B->C (B and C both involved)
        _make_contradiction_edge(graph, "B", "C", "e3", "induces")
        _make_contradiction_edge(graph, "B", "C", "e4", "inhibits")

        # Contradiction pair on A->D (A and D involved, but not B)
        _make_contradiction_edge(graph, "A", "D", "e5", "induces")
        _make_contradiction_edge(graph, "A", "D", "e6", "inhibits")

        results = score_contradiction_centrality(graph)
        by_node = {r["node_id"]: r for r in results}

        # B is in 2 contradiction pairs; A is also in 2 pairs (A->B and A->D),
        # C and D are each in 1 pair
        assert by_node["B"]["n_contradictions"] == 2
        assert by_node["C"]["n_contradictions"] == 1
        assert by_node["D"]["n_contradictions"] == 1
        assert by_node["B"]["raw_score"] > by_node["C"]["raw_score"]

    def test_contradiction_centrality_normalization(self):
        """High-degree node with same contradiction count should have lower normalized score."""
        from autoreview.knowledge_graph.analysis import score_contradiction_centrality

        # Node HUB has many edges (high degree) with just one contradiction pair
        # Node LOW has fewer edges with the same one contradiction pair
        graph = nx.MultiDiGraph()
        for name in ["HUB", "LOW", "X", "Y", "Z", "W", "Q"]:
            graph.add_node(name, canonical_name=name)

        # One contradiction pair for HUB (HUB->X)
        _make_contradiction_edge(graph, "HUB", "X", "h1", "induces")
        _make_contradiction_edge(graph, "HUB", "X", "h2", "inhibits")

        # Extra non-contradicting edges to inflate HUB's degree
        _make_contradiction_edge(graph, "HUB", "Y", "h3", "induces")
        _make_contradiction_edge(graph, "HUB", "Z", "h4", "induces")
        _make_contradiction_edge(graph, "HUB", "W", "h5", "induces")
        _make_contradiction_edge(graph, "HUB", "Q", "h6", "induces")

        # One contradiction pair for LOW (LOW->X) — same raw contribution
        _make_contradiction_edge(graph, "LOW", "X", "l1", "induces")
        _make_contradiction_edge(graph, "LOW", "X", "l2", "inhibits")

        results = score_contradiction_centrality(graph)
        by_node = {r["node_id"]: r for r in results}

        # Both should have the same n_contradictions (1 pair each)
        assert by_node["HUB"]["n_contradictions"] == 1
        assert by_node["LOW"]["n_contradictions"] == 1

        # HUB has higher degree so its normalized_score should be lower
        assert by_node["HUB"]["degree"] > by_node["LOW"]["degree"]
        assert by_node["HUB"]["normalized_score"] < by_node["LOW"]["normalized_score"]

    def test_contradiction_centrality_writes_node_attrs(self):
        """Should write contradiction_centrality and n_contradictions onto graph nodes."""
        from autoreview.knowledge_graph.analysis import score_contradiction_centrality

        graph = nx.MultiDiGraph()
        graph.add_node("A", canonical_name="Entity A")
        graph.add_node("B", canonical_name="Entity B")
        graph.add_node("C", canonical_name="Entity C")

        # Contradiction on A->B; C has no contradictions
        _make_contradiction_edge(graph, "A", "B", "e1", "induces")
        _make_contradiction_edge(graph, "A", "B", "e2", "inhibits")

        score_contradiction_centrality(graph)

        # Involved nodes get non-zero attributes
        assert graph.nodes["A"]["contradiction_centrality"] > 0.0
        assert graph.nodes["A"]["n_contradictions"] == 1
        assert graph.nodes["B"]["contradiction_centrality"] > 0.0
        assert graph.nodes["B"]["n_contradictions"] == 1

        # Non-involved node gets zero
        assert graph.nodes["C"]["contradiction_centrality"] == 0.0
        assert graph.nodes["C"]["n_contradictions"] == 0

    def test_contradiction_centrality_dict_structure(self):
        """Each result dict should have all required keys with correct types."""
        from autoreview.knowledge_graph.analysis import score_contradiction_centrality

        graph = nx.MultiDiGraph()
        graph.add_node("A", canonical_name="Entity A")
        graph.add_node("B", canonical_name="Entity B")
        _make_contradiction_edge(graph, "A", "B", "e1", "induces")
        _make_contradiction_edge(graph, "A", "B", "e2", "inhibits")

        results = score_contradiction_centrality(graph)

        required_keys = {
            "node_id",
            "canonical_name",
            "raw_score",
            "normalized_score",
            "n_contradictions",
            "degree",
        }
        for r in results:
            assert required_keys.issubset(r.keys())
            assert isinstance(r["node_id"], str)
            assert isinstance(r["raw_score"], float)
            assert isinstance(r["normalized_score"], float)
            assert isinstance(r["n_contradictions"], int)
            assert isinstance(r["degree"], int)

    def test_contradiction_centrality_sorted_descending(self):
        """Results should be sorted by raw_score descending."""
        from autoreview.knowledge_graph.analysis import score_contradiction_centrality

        graph = nx.MultiDiGraph()
        for name in ["A", "B", "C", "D"]:
            graph.add_node(name, canonical_name=name)

        # A->B contradiction (A and B each get 1 pair)
        _make_contradiction_edge(graph, "A", "B", "e1", "induces")
        _make_contradiction_edge(graph, "A", "B", "e2", "inhibits")
        # B->C contradiction (B gets a 2nd pair, C gets 1)
        _make_contradiction_edge(graph, "B", "C", "e3", "induces")
        _make_contradiction_edge(graph, "B", "C", "e4", "inhibits")

        results = score_contradiction_centrality(graph)
        scores = [r["raw_score"] for r in results]
        assert scores == sorted(scores, reverse=True)


def _make_finding_test_graph() -> nx.MultiDiGraph:
    """Graph with topic clusters for finding-level analysis testing."""
    graph = nx.MultiDiGraph()
    graph.add_node("A", canonical_name="Entity A")
    graph.add_node("B", canonical_name="Entity B")
    graph.add_node("C", canonical_name="Entity C")

    # Cluster 1: A→B activating (3 edges, 2 directions → contradiction)
    graph.add_edge(
        "A",
        "B",
        edge_id="ab1",
        predicate="induces",
        direction="positive",
        confidence_mean=0.8,
        organism="Mus musculus",
        in_vitro=True,
        model_system="mESC",
        conditions={},
        section_source="primary_empirical",
        natural_language="A induces B",
        _kg_edge=None,
    )
    graph.add_edge(
        "A",
        "B",
        edge_id="ab2",
        predicate="is_sufficient_for",
        direction="positive",
        confidence_mean=0.7,
        organism="Mus musculus",
        in_vitro=True,
        model_system="mESC",
        conditions={},
        section_source="primary_empirical",
        natural_language="A is sufficient for B",
        _kg_edge=None,
    )
    graph.add_edge(
        "A",
        "B",
        edge_id="ab3",
        predicate="induces",
        direction="negative",
        confidence_mean=0.4,
        organism="Mus musculus",
        in_vitro=True,
        model_system="mESC",
        conditions={},
        section_source="primary_empirical",
        natural_language="A does not induce B",
        _kg_edge=None,
    )

    # Cluster 2: B→C activating (2 edges, same direction → no contradiction)
    graph.add_edge(
        "B",
        "C",
        edge_id="bc1",
        predicate="induces",
        direction="positive",
        confidence_mean=0.9,
        organism="Mus musculus",
        in_vitro=True,
        model_system="mESC",
        conditions={},
        section_source="primary_empirical",
        natural_language="B induces C",
        _kg_edge=None,
    )
    graph.add_edge(
        "B",
        "C",
        edge_id="bc2",
        predicate="is_sufficient_for",
        direction="positive",
        confidence_mean=0.85,
        organism="Mus musculus",
        in_vitro=True,
        model_system="mESC",
        conditions={},
        section_source="primary_empirical",
        natural_language="B is sufficient for C",
        _kg_edge=None,
    )
    return graph


class TestSummarizeTopicClusters:
    def test_returns_list_of_dicts(self):
        from autoreview.knowledge_graph.analysis import summarize_topic_clusters

        graph = _make_finding_test_graph()
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        summary = summarize_topic_clusters(clusters, findings)
        assert isinstance(summary, list)
        assert len(summary) >= 1

    def test_dict_structure(self):
        from autoreview.knowledge_graph.analysis import summarize_topic_clusters

        graph = _make_finding_test_graph()
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        summary = summarize_topic_clusters(clusters, findings)
        required_keys = {
            "cluster_id",
            "subject_id",
            "object_id",
            "predicate_class",
            "n_edges",
            "n_findings",
            "member_predicates",
        }
        for s in summary:
            assert required_keys.issubset(s.keys())

    def test_edge_counts_correct(self):
        from autoreview.knowledge_graph.analysis import summarize_topic_clusters

        graph = _make_finding_test_graph()
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        summary = summarize_topic_clusters(clusters, findings)
        ab_cluster = [s for s in summary if s["subject_id"] == "A" and s["object_id"] == "B"]
        assert len(ab_cluster) == 1
        assert ab_cluster[0]["n_edges"] == 3


class TestScoreFindingContradictionCentrality:
    def test_returns_list_of_dicts(self):
        from autoreview.knowledge_graph.analysis import score_finding_contradiction_centrality

        graph = _make_finding_test_graph()
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        contradictions = detect_finding_contradictions(findings, clusters, graph=graph)
        results = score_finding_contradiction_centrality(graph, contradictions)
        assert isinstance(results, list)

    def test_dict_structure(self):
        from autoreview.knowledge_graph.analysis import score_finding_contradiction_centrality

        graph = _make_finding_test_graph()
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        contradictions = detect_finding_contradictions(findings, clusters, graph=graph)
        results = score_finding_contradiction_centrality(graph, contradictions)
        if results:
            required_keys = {
                "node_id",
                "canonical_name",
                "raw_score",
                "n_finding_contradictions",
            }
            for r in results:
                assert required_keys.issubset(r.keys())

    def test_contradicted_nodes_have_positive_scores(self):
        from autoreview.knowledge_graph.analysis import score_finding_contradiction_centrality

        graph = _make_finding_test_graph()
        clusters = build_topic_clusters(graph)
        findings = form_findings(clusters, graph)
        contradictions = detect_finding_contradictions(findings, clusters, graph=graph)
        results = score_finding_contradiction_centrality(graph, contradictions)
        if results:
            for r in results:
                assert r["raw_score"] > 0.0

    def test_empty_contradictions_returns_empty(self):
        from autoreview.knowledge_graph.analysis import score_finding_contradiction_centrality

        graph = _make_finding_test_graph()
        results = score_finding_contradiction_centrality(graph, [])
        assert results == []
