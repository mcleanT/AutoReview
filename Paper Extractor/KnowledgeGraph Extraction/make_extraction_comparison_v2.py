"""
Publication-quality comparison figure for three extraction models tested on the Mycelium framework.
Saves to: outputs/extraction_comparison_v2.png
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/Users/mst36/Desktop/Projects/Science/Autonomous Science/src")
from mycelium.extraction_schema import ExtractionResult

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

BASE = "/Users/mst36/Desktop/Projects/Science/Autonomous Science/outputs"

results = {}
for name, path in [
    ("Opus (v1)", f"{BASE}/extraction_test_opus.json"),
    ("Sonnet (v2)", f"{BASE}/extraction_test_sonnet_v2.json"),
    ("Haiku (v2)", f"{BASE}/extraction_test_haiku.json"),
]:
    with open(path) as f:
        results[name] = ExtractionResult.model_validate(json.load(f))

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

OPUS_COLOR = "#0072B2"
SONNET_COLOR = "#D55E00"
HAIKU_COLOR = "#009E73"
MODEL_COLORS = [OPUS_COLOR, SONNET_COLOR, HAIKU_COLOR]
MODEL_NAMES = list(results.keys())

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.35,
        "grid.color": "#cccccc",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
fig.suptitle(
    "Mycelium Extraction Framework: Three-Model Comparison (v2 Schema)\n"
    "Test Paper: Lobos Patorniti et al. (2023) — Rai14/Invariant chain",
    fontsize=13,
    fontweight="bold",
)


def panel_label(ax, letter, x=-0.13, y=1.04):
    ax.text(
        x, y, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left"
    )


# ---------------------------------------------------------------------------
# Panel A — Coverage Overview
# ---------------------------------------------------------------------------
ax = axes[0, 0]
panel_label(ax, "A")

coverage_data = {
    "Opus (v1)": {
        "Total assertions": 37,
        "Evidence units": 23,
        "Citation contexts": 0,
        "Novel findings": 22,
    },
    "Sonnet (v2)": {
        "Total assertions": 17,
        "Evidence units": 19,
        "Citation contexts": 12,
        "Novel findings": 13,
    },
    "Haiku (v2)": {
        "Total assertions": 11,
        "Evidence units": 16,
        "Citation contexts": 4,
        "Novel findings": 11,
    },
}

bar_labels = ["Total assertions", "Evidence units", "Citation contexts", "Novel findings"]
n_groups = len(MODEL_NAMES)
n_bars = len(bar_labels)
width = 0.18
group_gap = 0.05
x = np.arange(n_groups) * (n_bars * width + group_gap + 0.1)

bar_colors = ["#333333", "#888888", "#aaaaaa", "#cccccc"]
for bi, bl in enumerate(bar_labels):
    vals = [coverage_data[m][bl] for m in MODEL_NAMES]
    offsets = x + bi * width
    bars = ax.bar(offsets, vals, width=width, color=bar_colors[bi], label=bl, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(v),
            ha="center",
            va="bottom",
            fontsize=8,
        )

group_centers = x + (n_bars - 1) * width / 2
ax.set_xticks(group_centers)
ax.set_xticklabels([m.replace(" ", "\n") for m in MODEL_NAMES])
ax.set_ylabel("Count")
ax.set_title("Coverage Overview")
ax.legend(loc="upper right", framealpha=0.8)
ax.set_ylim(0, 45)
# Add note about citation contexts
ax.annotate(
    "*Opus v1: 0 citation contexts\n(15 background assertions\ncaptured as such in v2)",
    xy=(group_centers[0], 1),
    xytext=(group_centers[0] - 0.3, 36),
    fontsize=7,
    color="#555555",
    style="italic",
    arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8),
)

# ---------------------------------------------------------------------------
# Panel B — Assertion Type Distribution (stacked horizontal bars)
# ---------------------------------------------------------------------------
ax = axes[0, 1]
panel_label(ax, "B")

type_data = {
    "Opus (v1)": {
        "mechanistic_causal": 24,
        "existence": 12,
        "conditional": 1,
        "correlational": 0,
        "methodological": 0,
    },
    "Sonnet (v2)": {
        "mechanistic_causal": 13,
        "existence": 2,
        "conditional": 0,
        "correlational": 1,
        "methodological": 1,
    },
    "Haiku (v2)": {
        "mechanistic_causal": 9,
        "existence": 1,
        "conditional": 0,
        "correlational": 1,
        "methodological": 0,
    },
}

type_keys = ["mechanistic_causal", "existence", "correlational", "conditional", "methodological"]
type_colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442"]
type_labels = [
    "Mechanistic / causal",
    "Existence",
    "Correlational",
    "Conditional",
    "Methodological",
]

y_pos = np.arange(len(MODEL_NAMES))
lefts = np.zeros(len(MODEL_NAMES))
for ti, tk in enumerate(type_keys):
    vals = np.array([type_data[m].get(tk, 0) for m in MODEL_NAMES])
    bars = ax.barh(
        y_pos, vals, left=lefts, color=type_colors[ti], label=type_labels[ti], height=0.55, zorder=3
    )
    # Label non-zero segments
    for bar, v, l in zip(bars, vals, lefts):
        if v > 0:
            ax.text(
                l + v / 2,
                bar.get_y() + bar.get_height() / 2,
                str(v),
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if ti < 2 else "black",
                fontweight="bold",
            )
    lefts += vals

ax.set_yticks(y_pos)
ax.set_yticklabels(MODEL_NAMES)
ax.set_xlabel("Number of assertions")
ax.set_title("Assertion Type Distribution")
ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
ax.set_xlim(0, 42)

# ---------------------------------------------------------------------------
# Panel C — Scope & Field Completeness
# ---------------------------------------------------------------------------
ax = axes[0, 2]
panel_label(ax, "C")

scope_fields = ["Species", "Cell type", "in_vitro", "Verbatim hedge"]
scope_data = {
    "Opus (v1)": [59, 70, 59, 70],
    "Sonnet (v2)": [100, 100, 100, 94],
    "Haiku (v2)": [100, 100, 100, 100],
}

x_sc = np.arange(len(scope_fields))
n_models = len(MODEL_NAMES)
width_sc = 0.22

for mi, (mname, mcolor) in enumerate(zip(MODEL_NAMES, MODEL_COLORS)):
    offsets = x_sc + (mi - (n_models - 1) / 2) * width_sc
    bars = ax.bar(
        offsets, scope_data[mname], width=width_sc, color=mcolor, label=mname, alpha=0.88, zorder=3
    )
    for bar, v in zip(bars, scope_data[mname]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{v}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

ax.axhline(100, color="#444444", linestyle="--", linewidth=1.2, zorder=4, label="100% target")
ax.set_xticks(x_sc)
ax.set_xticklabels(scope_fields, rotation=10, ha="right")
ax.set_ylabel("Completeness (%)")
ax.set_ylim(0, 118)
ax.set_title("Scope & Field Completeness")
ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

# ---------------------------------------------------------------------------
# Panel D — Citation Context Analysis (Sonnet and Haiku only)
# ---------------------------------------------------------------------------
ax = axes[1, 0]
panel_label(ax, "D")

cite_data = {
    "Sonnet (v2)": {"contextualizes": 9, "extends": 2, "supports": 1},
    "Haiku (v2)": {"contextualizes": 4, "extends": 0, "supports": 0},
}
rel_types = ["contextualizes", "extends", "supports"]
rel_colors = ["#D55E00", "#009E73", "#CC79A7"]

models_cite = ["Sonnet (v2)", "Haiku (v2)"]
x_cite = np.arange(len(models_cite))
width_cite = 0.25
lefts_cite = np.zeros(len(models_cite))

for ri, rtype in enumerate(rel_types):
    vals = np.array([cite_data[m].get(rtype, 0) for m in models_cite])
    bars = ax.bar(
        x_cite + ri * width_cite - width_cite,
        vals,
        width=width_cite,
        color=rel_colors[ri],
        label=rtype.capitalize(),
        zorder=3,
    )
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(v),
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

ax.set_xticks([0, 1])
ax.set_xticklabels(models_cite)
ax.set_ylabel("Count")
ax.set_title("Citation Context Analysis\n(Sonnet & Haiku only)")
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.85)
ax.set_ylim(0, 13)

# Annotation note
ax.text(
    0.5,
    0.16,
    '"Citation contexts replace background\nassertions — fewer but higher quality\nedge signals"',
    transform=ax.transAxes,
    fontsize=7.5,
    color="#444444",
    style="italic",
    ha="center",
    va="bottom",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="#cccccc", alpha=0.9),
)

# Add note about Opus
ax.text(
    -0.25,
    -0.14,
    "Opus v1: 0 citation contexts (old schema)",
    transform=ax.transAxes,
    fontsize=7.5,
    color="#666666",
    style="italic",
)

# ---------------------------------------------------------------------------
# Panel E — Entity Quality (dual bars: count + avg name length)
# ---------------------------------------------------------------------------
ax = axes[1, 1]
panel_label(ax, "E")

entity_counts = {"Opus (v1)": 42, "Sonnet (v2)": 19, "Haiku (v2)": 13}
entity_len = {"Opus (v1)": 15.1, "Sonnet (v2)": 14.4, "Haiku (v2)": 11.5}

ax2 = ax.twinx()
ax2.spines["right"].set_visible(True)
ax2.spines["top"].set_visible(False)

x_ent = np.arange(len(MODEL_NAMES))
width_ent = 0.32

b1 = ax.bar(
    x_ent - width_ent / 2,
    [entity_counts[m] for m in MODEL_NAMES],
    width=width_ent,
    color=MODEL_COLORS,
    alpha=0.85,
    zorder=3,
    label="Unique entity forms",
)

b2 = ax2.bar(
    x_ent + width_ent / 2,
    [entity_len[m] for m in MODEL_NAMES],
    width=width_ent,
    color=MODEL_COLORS,
    alpha=0.45,
    zorder=3,
    hatch="//",
    label="Avg name length (chars)",
)

# Value labels
for bar, v in zip(b1, [entity_counts[m] for m in MODEL_NAMES]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.4,
        str(v),
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

for bar, v in zip(b2, [entity_len[m] for m in MODEL_NAMES]):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f"{v:.1f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

ax.set_xticks(x_ent)
ax.set_xticklabels([m.replace(" ", "\n") for m in MODEL_NAMES])
ax.set_ylabel("Unique entity surface forms", color="#333333")
ax2.set_ylabel("Avg entity name length (chars)", color="#666666", fontsize=10, fontweight="bold")
ax.set_title("Entity Quality")
ax.set_ylim(0, 55)
ax2.set_ylim(0, 22)

patch1 = mpatches.Patch(color="#555555", alpha=0.85, label="Unique entity forms")
patch2 = mpatches.Patch(color="#555555", alpha=0.45, hatch="//", label="Avg name length (chars)")
ax.legend(handles=[patch1, patch2], loc="upper right", fontsize=8.5, framealpha=0.85)

# Story annotation
ax.text(
    0.5,
    0.95,
    "Haiku: cleanest entities; Opus: most verbose",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=8,
    color="#444444",
    style="italic",
)

# ---------------------------------------------------------------------------
# Panel F — Cost-Quality Summary Table
# ---------------------------------------------------------------------------
ax = axes[1, 2]
panel_label(ax, "F")
ax.axis("off")
ax.set_title("Cost-Quality Summary", pad=12)

col_labels = ["Model", "Assertions", "Evidence", "Citations", "Scope %", "Entity Len", "Verdict"]
cell_data = [
    ["Opus (v1)", "37", "23", "0", "59%", "15.1", "Most comprehensive"],
    ["Sonnet (v2)", "17", "19", "12", "100%", "14.4", "Best balance"],
    ["Haiku (v2)", "11", "16", "4", "100%", "11.5", "Cleanest extraction"],
]

col_widths = [0.16, 0.12, 0.10, 0.10, 0.10, 0.12, 0.22]
n_cols = len(col_labels)
n_rows = len(cell_data)

# Header
header_y = 0.88
row_height = 0.14
header_bg = "#333333"
row_bgs = [OPUS_COLOR, SONNET_COLOR, HAIKU_COLOR]

# Draw column headers
x_start = 0.01
xs = [x_start + sum(col_widths[:ci]) for ci in range(n_cols)]

ax.add_patch(
    plt.Rectangle(
        (x_start - 0.01, header_y - 0.04),
        sum(col_widths) + 0.02,
        row_height,
        transform=ax.transAxes,
        facecolor=header_bg,
        zorder=2,
        clip_on=False,
    )
)

for ci, (clab, cw, cx) in enumerate(zip(col_labels, col_widths, xs)):
    ax.text(
        cx + cw / 2,
        header_y,
        clab,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="white",
        zorder=3,
    )

# Data rows
for ri, (row, bg) in enumerate(zip(cell_data, row_bgs)):
    ry = header_y - row_height * (ri + 1) - 0.01 * ri
    ax.add_patch(
        plt.Rectangle(
            (x_start - 0.01, ry - 0.04),
            sum(col_widths) + 0.02,
            row_height,
            transform=ax.transAxes,
            facecolor=bg,
            alpha=0.15,
            zorder=1,
            clip_on=False,
        )
    )
    for ci, (val, cw, cx) in enumerate(zip(row, col_widths, xs)):
        weight = "bold" if ci == 0 or ci == n_cols - 1 else "normal"
        color = bg if ci == 0 else ("#333333" if ci < n_cols - 1 else bg)
        ax.text(
            cx + cw / 2,
            ry,
            val,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight=weight,
            color=color,
            zorder=3,
        )

# Divider line — use a Line2D in axes coords instead of axhline (which disallows transform kwarg)
ax.add_line(
    plt.matplotlib.lines.Line2D(
        [0, 1], [0, 0], transform=ax.transAxes, color="#cccccc", linewidth=0.5
    )
)

# Caption below table
cap_y = header_y - row_height * (n_rows + 1) - 0.08
ax.text(
    0.5,
    cap_y,
    "v2 schema introduces citation contexts and scope discipline.\n"
    "Sonnet and Haiku achieve 100% scope completeness.\n"
    "Trade-off: fewer assertions but higher precision.",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=8,
    color="#444444",
    style="italic",
    linespacing=1.5,
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = (
    "/Users/mst36/Desktop/Projects/Science/Autonomous Science/outputs/extraction_comparison_v2.png"
)
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
plt.close(fig)
