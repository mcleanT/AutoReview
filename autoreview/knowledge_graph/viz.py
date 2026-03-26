"""Visualization and export utilities for the knowledge graph."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from collections.abc import Collection
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

_ENTITY_COLORS: dict[str, str] = {
    "gene": "#0072B2",
    "protein": "#D55E00",
    "pathway": "#009E73",
    "biological_process": "#CC79A7",
    "cell_type": "#F0E442",
    "chemical": "#56B4E9",
    "organism": "#E69F00",
    "other": "#999999",
}

_FONT_FAMILY = "Arial"
_DPI = 300

_SERIALIZABLE = (str, int, float, bool)


def export_graphml(graph: nx.MultiDiGraph, path: Path | str) -> None:
    """Export the graph to GraphML format, stripping non-serializable attributes.

    Args:
        graph: The knowledge graph.
        path: Destination file path (will be created or overwritten).
    """
    path = Path(path)
    clean = nx.MultiDiGraph()

    for node, data in graph.nodes(data=True):
        clean_data = {k: v for k, v in data.items() if isinstance(v, _SERIALIZABLE)}
        clean.add_node(node, **clean_data)

    for u, v, key, data in graph.edges(keys=True, data=True):
        clean_data = {k: v for k, v in data.items() if isinstance(v, _SERIALIZABLE)}
        clean.add_edge(u, v, key=key, **clean_data)

    nx.write_graphml(clean, path)


def plot_subgraph(
    graph: nx.MultiDiGraph,
    output_path: Path | str,
    node_ids: Collection[str] | None = None,
    seed: int = 42,
) -> None:
    """Render a network plot of the graph (or a subgraph).

    Nodes are colored by entity_type using a colorblind-safe palette.
    Edge width is proportional to evidence_count.

    Args:
        graph: The knowledge graph.
        output_path: Destination file path for the PNG/PDF figure.
        node_ids: Optional subset of node IDs to render; uses full graph if None.
        seed: Random seed for spring layout reproducibility.
    """
    output_path = Path(output_path)

    view = graph.subgraph(list(node_ids)) if node_ids is not None else graph

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    pos = nx.spring_layout(view, seed=seed)

    node_colors = [
        _ENTITY_COLORS.get(data.get("entity_type", "other"), _ENTITY_COLORS["other"])
        for _, data in view.nodes(data=True)
    ]

    labels = {n: data.get("canonical_name", n) for n, data in view.nodes(data=True)}

    edge_widths = [
        max(1.0, float(data.get("evidence_count", 1))) for _, _, data in view.edges(data=True)
    ]

    nx.draw_networkx_nodes(view, pos, node_color=node_colors, node_size=500, ax=ax)
    nx.draw_networkx_edges(
        view, pos, width=edge_widths, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=15
    )
    nx.draw_networkx_labels(view, pos, labels=labels, font_family=_FONT_FAMILY, font_size=9, ax=ax)

    ax.set_title("Knowledge Graph", fontsize=14, fontweight="bold", fontfamily=_FONT_FAMILY)
    ax.axis("off")

    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_distribution(
    graph: nx.MultiDiGraph,
    output_path: Path | str,
    bins: int = 20,
) -> None:
    """Plot a histogram of edge confidence scores.

    Args:
        graph: The knowledge graph.
        output_path: Destination file path for the figure.
        bins: Number of histogram bins.
    """
    output_path = Path(output_path)

    confidences = [
        float(data["confidence_mean"])
        for _, _, data in graph.edges(data=True)
        if "confidence_mean" in data
    ]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    ax.hist(confidences, bins=bins, color="#0072B2", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Confidence Score", fontsize=12, fontweight="bold", fontfamily=_FONT_FAMILY)
    ax.set_ylabel("Number of Edges", fontsize=12, fontweight="bold", fontfamily=_FONT_FAMILY)
    ax.set_title(
        "Edge Confidence Distribution", fontsize=14, fontweight="bold", fontfamily=_FONT_FAMILY
    )
    ax.tick_params(labelsize=10)

    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_controversy_map(
    graph: nx.MultiDiGraph,
    output_path: Path | str,
    threshold: float = 0.5,
    seed: int = 42,
) -> None:
    """Render a network plot highlighting controversial edges.

    Edges with controversy_score above threshold are drawn in red; others in gray.
    Edge width is proportional to controversy_score.

    Args:
        graph: The knowledge graph.
        output_path: Destination file path for the figure.
        threshold: Controversy score cutoff for red highlighting.
        seed: Random seed for spring layout reproducibility.
    """
    output_path = Path(output_path)

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    pos = nx.spring_layout(graph, seed=seed)

    node_colors = [
        _ENTITY_COLORS.get(data.get("entity_type", "other"), _ENTITY_COLORS["other"])
        for _, data in graph.nodes(data=True)
    ]

    labels = {n: data.get("canonical_name", n) for n, data in graph.nodes(data=True)}

    edge_colors = []
    edge_widths = []
    for _, _, data in graph.edges(data=True):
        score = float(data.get("controversy_score", 0.0))
        edge_colors.append("#D55E00" if score > threshold else "#999999")
        edge_widths.append(max(1.0, score * 4))

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=500, ax=ax)
    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color=edge_colors,
        width=edge_widths,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
    )
    nx.draw_networkx_labels(graph, pos, labels=labels, font_family=_FONT_FAMILY, font_size=9, ax=ax)

    ax.set_title(
        f"Controversy Map (threshold={threshold})",
        fontsize=14,
        fontweight="bold",
        fontfamily=_FONT_FAMILY,
    )
    ax.axis("off")

    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
