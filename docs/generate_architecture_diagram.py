#!/usr/bin/env python3
"""Generate publication-quality architecture diagram for AutoReview pipeline."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLORS = {
    "input_output": "#E8E8E8",
    "search": "#CADCF0",
    "llm_struct": "#C8E6C9",
    "llm_gen": "#FFE0B2",
    "critique": "#F8C8C8",
    "analysis": "#E1D5F0",
}

BORDER = {
    "input_output": "#888888",
    "search": "#4A7AB5",
    "llm_struct": "#388E3C",
    "llm_gen": "#E65100",
    "critique": "#C62828",
    "analysis": "#6A1B9A",
}

TEXT_COLOR = "#1A1A1A"
ARROW_COLOR = "#555555"
LOOP_ARROW_COLOR = "#B71C1C"
FEEDBACK_COLOR = "#4A148C"


# ---------------------------------------------------------------------------
# Figure setup
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 20), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 20)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

# Title
ax.text(
    8,
    19.55,
    "AutoReview \u2014 Pipeline Architecture",
    ha="center",
    va="center",
    fontsize=22,
    fontweight="bold",
    color="#1A1A1A",
    fontfamily="sans-serif",
)
ax.text(
    8,
    19.2,
    "Autonomous Scientific Review Paper Generation",
    ha="center",
    va="center",
    fontsize=12,
    color="#555555",
    fontfamily="sans-serif",
    fontstyle="italic",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def draw_box(
    ax, cx, cy, w, h, label, color_key, sublabel=None, fontsize=10, sublabel_fontsize=8, bold=True
):
    """Draw a rounded-corner box centered at (cx, cy)."""
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.08",
        facecolor=COLORS[color_key],
        edgecolor=BORDER[color_key],
        linewidth=1.6,
        zorder=3,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    y_text = cy + 0.07 if sublabel else cy
    ax.text(
        cx,
        y_text,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=TEXT_COLOR,
        fontfamily="sans-serif",
        zorder=4,
    )
    if sublabel:
        ax.text(
            cx,
            cy - 0.22,
            sublabel,
            ha="center",
            va="center",
            fontsize=sublabel_fontsize,
            color="#555555",
            fontfamily="sans-serif",
            zorder=4,
        )


def draw_arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    color=ARROW_COLOR,
    style="-|>",
    lw=1.5,
    connection_style="arc3,rad=0",
    ls="-",
    zorder=2,
):
    """Draw an arrow from (x1,y1) to (x2,y2)."""
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=14,
        color=color,
        linewidth=lw,
        connectionstyle=connection_style,
        linestyle=ls,
        zorder=zorder,
    )
    ax.add_patch(arrow)


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
CX = 8.0  # main column centre
BW = 5.0  # standard box width
BH = 0.62  # standard box height
GAP = 0.28  # vertical gap between boxes

# We track a y_cursor that moves downward.
y = 18.5


# =========================================================================
# STAGE 0 — Topic Input
# =========================================================================
topic_y = y
draw_box(ax, CX, topic_y, 3.2, 0.55, "Topic / Research Question", "input_output")
y -= 0.55 / 2 + GAP + BH / 2

# =========================================================================
# STAGE 1 — Query Expansion
# =========================================================================
qe_y = y
draw_box(
    ax,
    CX,
    qe_y,
    BW,
    BH,
    "1. Query Expansion  [LLM-G]",
    "llm_gen",
    sublabel="Boolean queries  +  semantic queries  +  scope document",
)
draw_arrow(ax, CX, topic_y - 0.275, CX, qe_y + BH / 2 + 0.04)
y -= BH / 2 + GAP + (BH + 0.12) / 2

# =========================================================================
# STAGE 2 — Multi-Source Search
# =========================================================================
ms_y = y
ms_h = BH + 0.12
draw_box(
    ax,
    CX,
    ms_y,
    BW + 1.2,
    ms_h,
    "2. Multi-Source Search",
    "search",
    sublabel="PubMed  |  Semantic Scholar  |  OpenAlex  |  Perplexity Sonar",
)
draw_arrow(ax, CX, qe_y - BH / 2 - 0.02, CX, ms_y + ms_h / 2 + 0.04)
y -= ms_h / 2 + GAP + 0.50 / 2

# =========================================================================
# Aggregator / Dedup
# =========================================================================
agg_y = y
agg_h = 0.50
draw_box(
    ax,
    CX,
    agg_y,
    4.2,
    agg_h,
    "Aggregator / Deduplicate by DOI",
    "search",
    sublabel="Target: 200\u2013500 raw candidates",
    fontsize=9,
)
draw_arrow(ax, CX, ms_y - ms_h / 2 - 0.02, CX, agg_y + agg_h / 2 + 0.04)
y -= agg_h / 2 + GAP + BH / 2

# =========================================================================
# STAGE 3 — Screening
# =========================================================================
scr_y = y
draw_box(
    ax,
    CX,
    scr_y,
    BW,
    BH,
    "3. Screening  [LLM-S]",
    "llm_struct",
    sublabel="Batched relevance scoring, threshold \u2265 3 \u2192 ScreenedPapers",
)
draw_arrow(ax, CX, agg_y - agg_h / 2 - 0.02, CX, scr_y + BH / 2 + 0.04)
y -= BH / 2 + GAP + BH / 2 + 0.05

# =========================================================================
# STAGE 4 — Parallel Extraction
# =========================================================================
ext_y = y
draw_box(
    ax,
    CX,
    ext_y,
    BW,
    BH,
    "4. Parallel Extraction  [LLM-S]",
    "llm_struct",
    sublabel="Per-paper, Semaphore(10) \u2192 PaperExtractions",
)
draw_arrow(ax, CX, scr_y - BH / 2 - 0.02, CX, ext_y + BH / 2 + 0.04)
y -= BH / 2 + GAP + (BH + 0.15) / 2 + 0.05

# =========================================================================
# STAGE 5 — Analysis Engine
# =========================================================================
ana_y = y
ana_h = BH + 0.15
draw_box(
    ax,
    CX,
    ana_y,
    BW + 1.0,
    ana_h,
    "5. Analysis Engine  [LLM-S]",
    "analysis",
    sublabel="Thematic Clustering + Contradiction Detection + Gap Detection \u2192 EvidenceMap",
    sublabel_fontsize=7.5,
)
draw_arrow(ax, CX, ext_y - BH / 2 - 0.02, CX, ana_y + ana_h / 2 + 0.04)

# ---- Gap-Aware Search (side branch) ----
gap_x = CX + 5.0
gap_y = ana_y
gap_h = BH + 0.10
draw_box(
    ax,
    gap_x,
    gap_y,
    3.8,
    gap_h,
    "6. Gap-Aware Search",
    "search",
    sublabel="Conditional, at most once",
    fontsize=9,
    sublabel_fontsize=7.5,
)
# Arrow: analysis -> gap search
draw_arrow(
    ax,
    CX + (BW + 1.0) / 2 + 0.04,
    ana_y,
    gap_x - 3.8 / 2 - 0.04,
    gap_y,
    color=FEEDBACK_COLOR,
    ls="--",
    lw=1.3,
)
ax.text(
    (CX + (BW + 1.0) / 2 + gap_x - 3.8 / 2) / 2,
    ana_y + 0.22,
    "if gaps found",
    ha="center",
    va="bottom",
    fontsize=7,
    color=FEEDBACK_COLOR,
    fontstyle="italic",
    fontfamily="sans-serif",
)
# Feedback: gap search -> back up to extraction level, then left into extraction box
draw_arrow(ax, gap_x, gap_y + gap_h / 2 + 0.04, gap_x, ext_y, color=FEEDBACK_COLOR, ls="--", lw=1.3)
draw_arrow(ax, gap_x, ext_y, CX + BW / 2 + 0.04, ext_y, color=FEEDBACK_COLOR, ls="--", lw=1.3)
ax.text(
    gap_x + 0.15,
    (gap_y + gap_h / 2 + ext_y) / 2,
    "new papers \u2192\nextract & merge",
    ha="left",
    va="center",
    fontsize=7,
    color=FEEDBACK_COLOR,
    fontfamily="sans-serif",
    fontstyle="italic",
)

y -= ana_h / 2 + GAP + BH / 2 + 0.10

# =========================================================================
# STAGE 7 — Outline Generation + Critique
# =========================================================================
out_y = y
out_gen_x = CX - 1.5
out_crit_x = CX + 2.2
out_gen_w = 3.4
out_crit_w = 3.0

draw_box(
    ax,
    out_gen_x,
    out_y,
    out_gen_w,
    BH,
    "7. Outline Gen  [LLM-G]",
    "llm_gen",
    sublabel="Hierarchical outline",
    fontsize=9,
    sublabel_fontsize=7.5,
)
draw_box(
    ax,
    out_crit_x,
    out_y,
    out_crit_w,
    BH,
    "Outline Critique  [LLM-S]",
    "critique",
    sublabel="Up to 2 revision cycles",
    fontsize=9,
    sublabel_fontsize=7.5,
)

# Arrow from analysis down to outline gen (slight diagonal)
draw_arrow(ax, CX, ana_y - ana_h / 2 - 0.02, out_gen_x, out_y + BH / 2 + 0.04)
# Arrow: outline gen -> critique
draw_arrow(ax, out_gen_x + out_gen_w / 2 + 0.04, out_y, out_crit_x - out_crit_w / 2 - 0.04, out_y)
# Loop arrow: critique back to gen
draw_arrow(
    ax,
    out_crit_x - 0.3,
    out_y + BH / 2 + 0.04,
    out_gen_x + 0.3,
    out_y + BH / 2 + 0.04,
    color=LOOP_ARROW_COLOR,
    ls="--",
    lw=1.2,
    connection_style="arc3,rad=-0.4",
)
ax.text(
    CX + 0.35,
    out_y + BH / 2 + 0.40,
    "revise (\u22642\u00d7)",
    ha="center",
    va="center",
    fontsize=7,
    color=LOOP_ARROW_COLOR,
    fontstyle="italic",
    fontfamily="sans-serif",
)

y -= BH / 2 + GAP + BH / 2 + 0.12

# =========================================================================
# STAGE 8 — Section Writing + Critique
# =========================================================================
sec_y = y
sec_gen_x = CX - 1.5
sec_crit_x = CX + 2.2
sec_gen_w = 3.4
sec_crit_w = 3.0

draw_box(
    ax,
    sec_gen_x,
    sec_y,
    sec_gen_w,
    BH,
    "8. Section Writing  [LLM-G]",
    "llm_gen",
    sublabel="Sequential, cross-section context",
    fontsize=9,
    sublabel_fontsize=7.5,
)
draw_box(
    ax,
    sec_crit_x,
    sec_y,
    sec_crit_w,
    BH,
    "Section Critique  [LLM-S]",
    "critique",
    sublabel="Up to 2 revision cycles",
    fontsize=9,
    sublabel_fontsize=7.5,
)

# Arrow from outline gen down to section writing
draw_arrow(ax, out_gen_x, out_y - BH / 2 - 0.02, sec_gen_x, sec_y + BH / 2 + 0.04)
# Arrow: section writing -> critique
draw_arrow(ax, sec_gen_x + sec_gen_w / 2 + 0.04, sec_y, sec_crit_x - sec_crit_w / 2 - 0.04, sec_y)
# Loop arrow
draw_arrow(
    ax,
    sec_crit_x - 0.3,
    sec_y + BH / 2 + 0.04,
    sec_gen_x + 0.3,
    sec_y + BH / 2 + 0.04,
    color=LOOP_ARROW_COLOR,
    ls="--",
    lw=1.2,
    connection_style="arc3,rad=-0.4",
)
ax.text(
    CX + 0.35,
    sec_y + BH / 2 + 0.40,
    "revise (\u22642\u00d7)",
    ha="center",
    va="center",
    fontsize=7,
    color=LOOP_ARROW_COLOR,
    fontstyle="italic",
    fontfamily="sans-serif",
)

y -= BH / 2 + GAP + 0.50 / 2 + 0.08

# =========================================================================
# STAGE 9 — Draft Assembly
# =========================================================================
asm_y = y
asm_h = 0.50
draw_box(
    ax,
    CX,
    asm_y,
    BW,
    asm_h,
    "9. Draft Assembly",
    "llm_gen",
    sublabel="Assemble sections into full draft",
    fontsize=10,
    sublabel_fontsize=8,
)
draw_arrow(ax, sec_gen_x, sec_y - BH / 2 - 0.02, CX, asm_y + asm_h / 2 + 0.04)

y -= asm_h / 2 + GAP + BH / 2 + 0.08

# =========================================================================
# Holistic Critique loop
# =========================================================================
hol_y = y
hol_gen_x = CX - 1.5
hol_crit_x = CX + 2.2
hol_gen_w = 3.4
hol_crit_w = 3.0

draw_box(
    ax,
    hol_gen_x,
    hol_y,
    hol_gen_w,
    BH,
    "Cross-Section Revision  [LLM-G]",
    "llm_gen",
    sublabel="Narrative arc, balance, transitions",
    fontsize=9,
    sublabel_fontsize=7.5,
)
draw_box(
    ax,
    hol_crit_x,
    hol_y,
    hol_crit_w,
    BH,
    "Holistic Critique  [LLM-S]",
    "critique",
    sublabel="Up to 3 revision cycles",
    fontsize=9,
    sublabel_fontsize=7.5,
)

draw_arrow(ax, CX, asm_y - asm_h / 2 - 0.02, hol_gen_x, hol_y + BH / 2 + 0.04)
draw_arrow(ax, hol_gen_x + hol_gen_w / 2 + 0.04, hol_y, hol_crit_x - hol_crit_w / 2 - 0.04, hol_y)
draw_arrow(
    ax,
    hol_crit_x - 0.3,
    hol_y + BH / 2 + 0.04,
    hol_gen_x + 0.3,
    hol_y + BH / 2 + 0.04,
    color=LOOP_ARROW_COLOR,
    ls="--",
    lw=1.2,
    connection_style="arc3,rad=-0.4",
)
ax.text(
    CX + 0.35,
    hol_y + BH / 2 + 0.40,
    "revise (\u22643\u00d7)",
    ha="center",
    va="center",
    fontsize=7,
    color=LOOP_ARROW_COLOR,
    fontstyle="italic",
    fontfamily="sans-serif",
)

y -= BH / 2 + GAP + BH / 2 + 0.08

# =========================================================================
# STAGE 10 — Final Polish
# =========================================================================
pol_y = y
draw_box(
    ax,
    CX,
    pol_y,
    BW,
    BH,
    "10. Final Polish  [LLM-G]",
    "llm_gen",
    sublabel="Language polishing, terminology consistency",
)
draw_arrow(ax, hol_gen_x, hol_y - BH / 2 - 0.02, CX, pol_y + BH / 2 + 0.04)

y -= BH / 2 + GAP + 0.55 / 2 + 0.05

# =========================================================================
# Format Output
# =========================================================================
fmt_y = y
fmt_h = 0.55
draw_box(
    ax,
    CX,
    fmt_y,
    BW + 0.5,
    fmt_h,
    "Format Output",
    "input_output",
    sublabel="Markdown  |  LaTeX  |  DOCX  (Jinja2 + Pandoc)",
)
draw_arrow(ax, CX, pol_y - BH / 2 - 0.02, CX, fmt_y + fmt_h / 2 + 0.04)


# =========================================================================
# LEGEND — colour swatches
# =========================================================================
# Place legend just below the last box
leg_y = fmt_y - fmt_h / 2 - 0.65
leg_x = 0.4
leg_gap = 0.42

legend_items = [
    ("Input / Output", "input_output"),
    ("Search Stages", "search"),
    ("LLM \u2014 Structured  [LLM-S]", "llm_struct"),
    ("LLM \u2014 Free-form  [LLM-G]", "llm_gen"),
    ("Critique Loops", "critique"),
    ("Analysis", "analysis"),
]

legend_border = FancyBboxPatch(
    (leg_x - 0.2, leg_y - len(legend_items) * leg_gap - 0.1),
    3.1,
    len(legend_items) * leg_gap + 0.55,
    boxstyle="round,pad=0.12",
    facecolor="white",
    edgecolor="#BBBBBB",
    linewidth=1.0,
    zorder=2,
)
ax.add_patch(legend_border)
ax.text(
    leg_x + 1.25,
    leg_y + 0.15,
    "Legend",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
    color=TEXT_COLOR,
    fontfamily="sans-serif",
    zorder=3,
)

for i, (lbl, ckey) in enumerate(legend_items):
    ly = leg_y - i * leg_gap - 0.15
    swatch = FancyBboxPatch(
        (leg_x, ly - 0.16),
        0.45,
        0.32,
        boxstyle="round,pad=0.04",
        facecolor=COLORS[ckey],
        edgecolor=BORDER[ckey],
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(swatch)
    ax.text(
        leg_x + 0.60,
        ly,
        lbl,
        ha="left",
        va="center",
        fontsize=8.5,
        color=TEXT_COLOR,
        fontfamily="sans-serif",
        zorder=3,
    )


# =========================================================================
# LEGEND — arrow types
# =========================================================================
ann_x = 12.5
ann_y = leg_y

ann_border = FancyBboxPatch(
    (ann_x - 0.3, ann_y - 1.6),
    3.6,
    2.05,
    boxstyle="round,pad=0.12",
    facecolor="white",
    edgecolor="#BBBBBB",
    linewidth=1.0,
    zorder=2,
)
ax.add_patch(ann_border)
ax.text(
    ann_x + 1.5,
    ann_y + 0.15,
    "Arrow Types",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
    color=TEXT_COLOR,
    fontfamily="sans-serif",
    zorder=3,
)

draw_arrow(ax, ann_x, ann_y - 0.25, ann_x + 0.8, ann_y - 0.25, color=ARROW_COLOR, lw=1.5)
ax.text(
    ann_x + 1.0,
    ann_y - 0.25,
    "Data flow",
    ha="left",
    va="center",
    fontsize=8.5,
    color=TEXT_COLOR,
    fontfamily="sans-serif",
    zorder=3,
)

draw_arrow(
    ax, ann_x, ann_y - 0.67, ann_x + 0.8, ann_y - 0.67, color=LOOP_ARROW_COLOR, lw=1.2, ls="--"
)
ax.text(
    ann_x + 1.0,
    ann_y - 0.67,
    "Critique / revision loop",
    ha="left",
    va="center",
    fontsize=8.5,
    color=TEXT_COLOR,
    fontfamily="sans-serif",
    zorder=3,
)

draw_arrow(
    ax, ann_x, ann_y - 1.09, ann_x + 0.8, ann_y - 1.09, color=FEEDBACK_COLOR, lw=1.2, ls="--"
)
ax.text(
    ann_x + 1.0,
    ann_y - 1.09,
    "Conditional feedback",
    ha="left",
    va="center",
    fontsize=8.5,
    color=TEXT_COLOR,
    fontfamily="sans-serif",
    zorder=3,
)


# =========================================================================
# KnowledgeBase sidebar annotation
# =========================================================================
kb_x = 0.8
kb_y = 10.0

kb_box = FancyBboxPatch(
    (kb_x - 0.1, kb_y - 1.7),
    2.9,
    3.4,
    boxstyle="round,pad=0.12",
    facecolor="#FFF9E6",
    edgecolor="#B8860B",
    linewidth=1.2,
    linestyle="--",
    zorder=2,
)
ax.add_patch(kb_box)
ax.text(
    kb_x + 1.35,
    kb_y + 1.4,
    "KnowledgeBase",
    ha="center",
    va="center",
    fontsize=9,
    fontweight="bold",
    color="#6D4C00",
    fontfamily="sans-serif",
    zorder=3,
)

kb_lines = [
    "Serialized to JSON",
    "after every DAG node",
    "",
    "\u2022 Search queries",
    "\u2022 CandidatePapers",
    "\u2022 ScreenedPapers",
    "\u2022 PaperExtractions",
    "\u2022 EvidenceMap",
    "\u2022 Outline + Drafts",
    "\u2022 Critique history",
]
for j, line in enumerate(kb_lines):
    ax.text(
        kb_x + 0.05,
        kb_y + 1.0 - j * 0.28,
        line,
        ha="left",
        va="center",
        fontsize=7,
        color="#6D4C00",
        fontfamily="sans-serif",
        zorder=3,
    )

# Dotted connector from KB box to pipeline spine
draw_arrow(
    ax, kb_x + 2.8, kb_y, CX - BW / 2 - 0.25, kb_y, color="#B8860B", ls=":", lw=0.9, style="-"
)


# =========================================================================
# Save
# =========================================================================
out = Path(__file__).parent / "architecture.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.3)
plt.close(fig)
print(f"Saved: {out}")
