"""Publication-quality figure generation for review papers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import structlog

from autoreview.models.visuals import FigureMetadata, VisualInsertionAnchor

logger = structlog.get_logger()

# Colorblind-safe palette (Wong)
CB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]

# Attempt SchematicEngine import
HAS_SCHEMATIC_ENGINE = False
SchematicEngine = None
WONG = None
TINTS = None

try:
    _spec = importlib.util.spec_from_file_location(
        "schematic_engine",
        Path.home() / ".claude/skills/programmatic-schematics/schematic_engine.py",
    )
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        SchematicEngine = _mod.SchematicEngine
        WONG = _mod.WONG
        TINTS = _mod.TINTS
        HAS_SCHEMATIC_ENGINE = True
except (ImportError, OSError, FileNotFoundError, AttributeError):
    logger.warning("schematic_engine.unavailable", fallback="matplotlib")


def _ensure_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def generate_pipeline_schematic(output_dir: Path) -> Path:
    """Generate RAG pipeline architecture diagram (Fig 1)."""
    fig_dir = _ensure_dir(output_dir)
    out_path = fig_dir / "fig1_rag_pipeline.png"

    if HAS_SCHEMATIC_ENGINE and SchematicEngine:
        assert WONG is not None and TINTS is not None
        engine = SchematicEngine(width=14, height=5, dpi=300, bg="white")
        # Pipeline boxes
        stages_se = [
            ("Query", 0.03),
            ("Retrieval\n(Dense/Sparse/Hybrid)", 0.18),
            ("Re-ranking", 0.38),
            ("Context\nIntegration", 0.53),
            ("LLM\nGenerator", 0.68),
            ("Output +\nCitations", 0.83),
        ]
        refs = []
        for label, x in stages_se:
            ref = engine.box(
                x,
                0.45,
                0.13,
                0.30,
                label,
                bold=True,
                facecolor=TINTS["blue"],
                edgecolor=WONG["blue"],
            )
            refs.append(ref)
        for a, b in zip(refs, refs[1:], strict=False):
            engine.arrow(a.right_center, b.left_center, color=WONG["blue"], lw=2)
        # Tier labels
        tiers = [
            ("Naive RAG", 0.08, WONG["green"]),
            ("Advanced RAG", 0.38, WONG["orange"]),
            ("Agentic RAG", 0.68, WONG["pink"]),
        ]
        for label, x, color in tiers:
            engine.text(x + 0.08, 0.18, label, fontsize=10, color=color, ha="center")
        engine.save(str(out_path.with_suffix("")), formats=("png",))
    else:
        # Matplotlib fallback
        fig, ax = plt.subplots(1, 1, figsize=(14, 4), dpi=300)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        stages = ["Query", "Retrieval", "Re-ranking", "Context\nIntegration", "LLM", "Output"]
        for i, label in enumerate(stages):
            x = 0.05 + i * 0.15
            rect = mpatches.FancyBboxPatch(
                (x, 0.3),
                0.12,
                0.35,
                boxstyle="round,pad=0.02",
                facecolor=CB_PALETTE[0] + "30",
                edgecolor=CB_PALETTE[0],
                lw=2,
            )
            ax.add_patch(rect)
            ax.text(x + 0.06, 0.48, label, ha="center", va="center", fontsize=9, fontfamily="Arial")
            if i < len(stages) - 1:
                ax.annotate(
                    "",
                    xy=(x + 0.14, 0.48),
                    xytext=(x + 0.12, 0.48),
                    arrowprops=dict(arrowstyle="->", color=CB_PALETTE[0], lw=1.5),
                )
        fig.savefig(out_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    return out_path


def generate_temporal_chart(year_counts: dict[int, int], output_dir: Path) -> Path:
    """Generate temporal distribution bar chart (Fig 2)."""
    fig_dir = _ensure_dir(output_dir)
    out_path = fig_dir / "fig2_temporal.png"

    years = sorted(year_counts.keys())
    counts = [year_counts[y] for y in years]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bars = ax.bar([str(y) for y in years], counts, color=CB_PALETTE[0], edgecolor="white", lw=0.5)
    for bar, count in zip(bars, counts, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
            fontfamily="Arial",
        )

    # Annotate inflection
    if 2023 in year_counts:
        idx_2023 = years.index(2023)
        ax.annotate(
            "ChatGPT-era\ninflection",
            xy=(str(2023), year_counts[2023]),  # type: ignore[arg-type]
            xytext=(str(years[max(0, idx_2023 - 1)]), max(counts) * 0.6),  # type: ignore[arg-type]
            arrowprops=dict(arrowstyle="->", color=CB_PALETTE[1], lw=1.5),
            fontsize=9,
            color=CB_PALETTE[1],
            fontfamily="Arial",
        )

    ax.set_xlabel("Publication Year", fontsize=12, fontfamily="Arial", fontweight="bold")
    ax.set_ylabel("Number of Papers", fontsize=12, fontfamily="Arial", fontweight="bold")
    ax.set_title(
        "Temporal Distribution of Reviewed Papers",
        fontsize=14,
        fontfamily="Arial",
        fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def generate_taxonomy_tree(output_dir: Path) -> Path:
    """Generate RAG architecture taxonomy (Fig 3)."""
    fig_dir = _ensure_dir(output_dir)
    out_path = fig_dir / "fig3_taxonomy.png"

    if HAS_SCHEMATIC_ENGINE and SchematicEngine:
        assert WONG is not None and TINTS is not None
        engine = SchematicEngine(width=16, height=9, dpi=300, bg="white")
        root = engine.box(
            0.40,
            0.85,
            0.20,
            h=None,
            label="RAG Architectures",
            bold=True,
            facecolor=TINTS["blue"],
            edgecolor=WONG["blue"],
            name="root",
        )
        branches = [
            (
                "Naive RAG",
                0.02,
                WONG["green"],
                TINTS["green"],
                ["Single retrieval → Generate", "BM25/DPR baseline"],
            ),
            (
                "Advanced RAG",
                0.27,
                WONG["blue"],
                TINTS["blue"],
                ["Query expansion (HyDE)", "Re-ranking (ColBERT)", "Iterative retrieval"],
            ),
            (
                "Modular RAG",
                0.52,
                WONG["orange"],
                TINTS["yellow"],
                ["Interchangeable components", "Self-RAG reflection", "CRAG correction"],
            ),
            (
                "Agentic RAG",
                0.77,
                WONG["pink"],
                TINTS["pink"],
                ["Tool-use agents", "Multi-step planning", "RL-trained (R3-RAG)"],
            ),
        ]
        for label, x, edge_c, face_c, children in branches:
            branch = engine.box(
                x,
                0.58,
                0.20,
                h=None,
                label=label,
                bold=True,
                facecolor=face_c,
                edgecolor=edge_c,
                name=label,
            )
            engine.arrow(root.bottom_center, branch.top_center, color=edge_c, lw=1.5)
            for i, child in enumerate(children):
                child_ref = engine.box(
                    x,
                    0.38 - i * 0.12,
                    0.20,
                    h=None,
                    label=child,
                    facecolor="white",
                    edgecolor=edge_c,
                )
                engine.arrow(branch.bottom_center, child_ref.top_center, color=edge_c, lw=1)
        engine.save(str(out_path.with_suffix("")), formats=("png",))
    else:
        # Simple matplotlib fallback
        fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
        ax.axis("off")
        ax.text(
            0.5,
            0.92,
            "RAG Architectures",
            ha="center",
            fontsize=16,
            fontweight="bold",
            fontfamily="Arial",
        )
        branch_labels = ["Naive RAG", "Advanced RAG", "Modular RAG", "Agentic RAG"]
        for i, label in enumerate(branch_labels):
            x = 0.12 + i * 0.22
            ax.text(
                x,
                0.65,
                label,
                ha="center",
                fontsize=12,
                fontweight="bold",
                fontfamily="Arial",
                color=CB_PALETTE[i],
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=CB_PALETTE[i] + "20",
                    edgecolor=CB_PALETTE[i],
                ),
            )
        fig.savefig(out_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return out_path


def generate_evidence_chart(themes: list[dict[str, Any]], output_dir: Path) -> Path:
    """Generate evidence strength stacked bar chart (Fig 4)."""
    fig_dir = _ensure_dir(output_dir)
    out_path = fig_dir / "fig4_evidence.png"

    strength_order = ["strong", "moderate", "weak", "preliminary"]
    colors = {
        "strong": CB_PALETTE[0],
        "moderate": CB_PALETTE[5],
        "weak": CB_PALETTE[6],
        "preliminary": CB_PALETTE[3],
    }

    names = [t["name"][:30] for t in themes]
    data: dict[str, list[int]] = {s: [] for s in strength_order}
    for t in themes:
        dist = t.get("evidence_strength_distribution", {})
        for s in strength_order:
            data[s].append(dist.get(s, 0))

    fig, ax = plt.subplots(figsize=(10, max(5, len(themes) * 0.6)), dpi=300)
    y_pos = np.arange(len(names))
    left = np.zeros(len(names))
    for strength in strength_order:
        values = np.array(data[strength])
        ax.barh(
            y_pos,
            values,
            left=left,
            color=colors[strength],
            label=strength.capitalize(),
            height=0.6,
        )
        left += values

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9, fontfamily="Arial")
    ax.set_xlabel("Number of Findings", fontsize=12, fontfamily="Arial", fontweight="bold")
    ax.set_title(
        "Evidence Strength Distribution Across Research Themes",
        fontsize=14,
        fontfamily="Arial",
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def generate_all_figures(kb: Any) -> dict[str, FigureMetadata]:
    """Generate all 4 figures and return metadata dict."""
    output_dir = Path(kb.output_dir) / "figures"

    # Fig 1: Pipeline schematic (static)
    generate_pipeline_schematic(output_dir)

    # Fig 2: Temporal distribution
    year_counts: dict[int, int] = {}
    for sp in kb.screened_papers:
        y = sp.paper.year if hasattr(sp, "paper") else getattr(sp, "year", None)
        if y:
            year_counts[y] = year_counts.get(y, 0) + 1
    if year_counts:
        generate_temporal_chart(year_counts, output_dir)

    # Fig 3: Taxonomy tree (static)
    generate_taxonomy_tree(output_dir)

    # Fig 4: Evidence strength
    themes_data = []
    if kb.evidence_map and hasattr(kb.evidence_map, "themes"):
        for t in kb.evidence_map.themes:
            td = {
                "name": t.name if hasattr(t, "name") else str(t),
                "evidence_strength_distribution": (
                    t.evidence_strength_distribution
                    if hasattr(t, "evidence_strength_distribution")
                    else {}
                ),
            }
            themes_data.append(td)
    if themes_data:
        generate_evidence_chart(themes_data, output_dir)

    figures: dict[str, FigureMetadata] = {}

    figures["fig1_rag_pipeline"] = FigureMetadata(
        key="fig1_rag_pipeline",
        path="figures/fig1_rag_pipeline.png",
        caption=(
            "Figure 1. Canonical RAG pipeline architecture showing the progression"
            " from naive single-retrieval to agentic multi-tool systems."
        ),
        anchor=VisualInsertionAnchor(section_id="sec_1", position="after"),
    )

    if year_counts:
        figures["fig2_temporal"] = FigureMetadata(
            key="fig2_temporal",
            path="figures/fig2_temporal.png",
            caption=(
                "Figure 2. Temporal distribution of reviewed papers, showing the inflection"
                " point driven by ChatGPT-era deployment interest."
            ),
            anchor=VisualInsertionAnchor(section_id="sec_2", position="after"),
            data_driven=True,
        )
    else:
        logger.warning("figure.skipped", figure="fig2_temporal", reason="year_counts is empty")

    figures["fig3_taxonomy"] = FigureMetadata(
        key="fig3_taxonomy",
        path="figures/fig3_taxonomy.png",
        caption=(
            "Figure 3. Taxonomy of RAG architectural variants from naive single-pass"
            " to autonomous agentic systems."
        ),
        anchor=VisualInsertionAnchor(section_id="sec_6", position="before"),
    )

    if themes_data:
        figures["fig4_evidence"] = FigureMetadata(
            key="fig4_evidence",
            path="figures/fig4_evidence.png",
            caption=(
                "Figure 4. Evidence strength distribution across thematic clusters"
                " identified in the reviewed corpus."
            ),
            anchor=VisualInsertionAnchor(section_id="sec_13", position="before"),
            data_driven=True,
        )
    else:
        logger.warning("figure.skipped", figure="fig4_evidence", reason="themes_data is empty")

    return figures
