"""NetworkX graph construction and serialization for the AutoReview knowledge graph."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import networkx as nx
import structlog

from autoreview.knowledge_graph.models import KGEdge, KGEntity

log = structlog.get_logger(__name__)

# Attribute types that are safe for GraphML serialization
_GRAPHML_SAFE_TYPES = (str, int, float, bool)


def build_nx_graph(
    entities: dict[str, KGEntity],
    edges: list[KGEdge],
) -> nx.MultiDiGraph:
    """Build a NetworkX MultiDiGraph from deduplicated entities and assertion edges.

    Args:
        entities: Mapping of entity_id to KGEntity (deduplicated nodes).
        edges: List of KGEdge objects (merged assertion edges).

    Returns:
        A MultiDiGraph where nodes are entities and edges are typed assertions.
        Edge keys are edge_id strings to allow parallel edges with stable lookup.
    """
    G: nx.MultiDiGraph = nx.MultiDiGraph()

    # Add nodes
    for entity_id, entity in entities.items():
        G.add_node(
            entity_id,
            canonical_name=entity.canonical_name,
            entity_type=str(entity.entity_type),
            ontology_id=entity.ontology_id or "",
            ontology_source=entity.ontology_source or "",
            paper_count=entity.paper_count,
            # GraphML-compatible: comma-separated string
            aliases=",".join(entity.aliases),
            source_paper_ids=",".join(entity.source_paper_ids),
        )

    # Add edges
    for edge in edges:
        G.add_edge(
            edge.subject_id,
            edge.object_id,
            key=edge.edge_id,
            # Flat, GraphML-safe attributes
            predicate=edge.predicate,
            direction=edge.direction or "",
            assertion_type=str(edge.assertion_type),
            confidence_mean=edge.confidence.mean,
            evidence_count=len(edge.evidence_links),
            publication_date=edge.publication_date or "",
            source_assertions=",".join(edge.source_assertions),
            # Rich object for programmatic access (not GraphML-safe)
            _kg_edge=edge,
        )

    log.info(
        "kg.graph_built",
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
    )
    return G


def _make_graphml_copy(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Return a copy of G with only GraphML-serializable attributes."""
    H: nx.MultiDiGraph = nx.MultiDiGraph()

    for node, attrs in G.nodes(data=True):
        H.add_node(node, **{k: v for k, v in attrs.items() if isinstance(v, _GRAPHML_SAFE_TYPES)})

    for u, v, key, attrs in G.edges(data=True, keys=True):
        safe_attrs: dict[str, Any] = {
            k: v for k, v in attrs.items() if isinstance(v, _GRAPHML_SAFE_TYPES)
        }
        H.add_edge(u, v, key=key, **safe_attrs)

    return H


def save_graph(G: nx.MultiDiGraph, path: Path) -> None:
    """Serialize the graph to pickle and GraphML.

    Args:
        G: The MultiDiGraph to serialize.
        path: Base path without extension. Creates ``{path}.pkl`` and ``{path}.graphml``.
    """
    path = Path(path)
    pkl_path = path.with_suffix(".pkl")
    graphml_path = path.with_suffix(".graphml")

    # Pickle preserves all attrs including _kg_edge
    with open(pkl_path, "wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # GraphML needs a clean copy without non-serializable attrs
    H = _make_graphml_copy(G)
    nx.write_graphml(H, graphml_path)

    log.info(
        "kg.graph_saved",
        pkl=str(pkl_path),
        graphml=str(graphml_path),
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
    )


def load_graph(path: Path) -> nx.MultiDiGraph:
    """Load a graph from a pickle file.

    Args:
        path: Path to the ``.pkl`` file produced by :func:`save_graph`.

    Returns:
        The deserialized MultiDiGraph.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        G: nx.MultiDiGraph = pickle.load(fh)  # noqa: S301

    log.info(
        "kg.graph_loaded",
        path=str(path),
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
    )
    return G
