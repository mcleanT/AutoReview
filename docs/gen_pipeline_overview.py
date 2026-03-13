"""Generate AutoReview pipeline overview PNG using matplotlib."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Colour palette ────────────────────────────────────────────────────────────
SEARCH_COLOR = "#4A90D9"  # blue
EXTRACT_COLOR = "#7B68EE"  # medium-slate
ANALYSIS_COLOR = "#50C878"  # emerald
WRITING_COLOR = "#FF8C42"  # orange
CRITIQUE_COLOR = "#E05C5C"  # red
OUTPUT_COLOR = "#6BAB8C"  # sage green
CHECK_COLOR = "#F5A623"  # amber  (comprehensiveness checks)
ENRICH_COLOR = "#9B59B6"  # purple (enrichment / narrative planning)

BG = "#F8F9FA"
ARROW = "#555555"
LOOP_ARROW = "#999999"

TEXT_MAIN = "#1A1A2E"
TEXT_LIGHT = "#FFFFFF"

# ── Node definitions ──────────────────────────────────────────────────────────
# (label, x_center, y_center, width, height, color)
# Layout: y decreases top→bottom

W, H = 3.4, 0.52  # default node size
MAIN_X = 4.5  # x of the main spine

nodes = [
    # ── RETRIEVAL PHASE ──
    # label                              x       y      w     h     color
    ("Query Expansion", MAIN_X, 13.0, W, H, SEARCH_COLOR),
    (
        "Multi-Source Search\n(PubMed · S2 · OpenAlex · Perplexity)",
        MAIN_X,
        12.1,
        W,
        H,
        SEARCH_COLOR,
    ),
    ("Screen & Deduplicate", MAIN_X, 11.2, W, H, SEARCH_COLOR),
    # ── EXTRACTION PHASE ──
    ("Parallel Extraction", MAIN_X, 10.3, W, H, EXTRACT_COLOR),
    # ── ANALYSIS PHASE ──
    ("Thematic Clustering\n+ Contradiction Detection", MAIN_X, 9.35, W, H, ANALYSIS_COLOR),
    ("Gap-Aware\nSupplementary Search", MAIN_X, 8.25, W, H, SEARCH_COLOR),
    # ── PLANNING PHASE ──
    ("Outline Generation", MAIN_X, 7.15, W, H, WRITING_COLOR),
    ("Narrative Planning", MAIN_X, 6.25, W, H, ENRICH_COLOR),
    ("Contextual Enrichment", MAIN_X, 5.35, W, H, ENRICH_COLOR),
    # ── WRITING PHASE ──
    ("Section Writing", MAIN_X, 4.25, W, H, WRITING_COLOR),
    ("Passage Search\n& Section Revision", MAIN_X, 3.15, W, H, SEARCH_COLOR),
    # ── ASSEMBLY & OUTPUT ──
    ("Assemble Draft", MAIN_X, 2.05, W, H, WRITING_COLOR),
    ("Final Polish", MAIN_X, 0.95, W, H, OUTPUT_COLOR),
    ("Format Output\n(MD · LaTeX · DOCX)", MAIN_X, -0.05, W, H, OUTPUT_COLOR),
    # ── Critique / revision nodes (right side) ──
    ("Outline Critique\n& Revision", 8.35, 7.15, 2.6, H, CRITIQUE_COLOR),
    ("Per-Section Critique\n& Revision", 8.35, 4.25, 2.6, H, CRITIQUE_COLOR),
    ("Holistic Critique\n& Revision", 8.35, 2.05, 2.6, H, CRITIQUE_COLOR),
    # ── Comprehensiveness check nodes (left side) ──
    ("Query Coverage\nCheck", 0.8, 13.0, 2.4, H, CHECK_COLOR),
    ("Coverage Anomaly\nCheck", 0.8, 11.2, 2.4, H, CHECK_COLOR),
    ("Borderline\nRescreening", 0.8, 10.5, 2.4, H, CHECK_COLOR),
    ("Post-Gap\nRevalidation", 0.8, 8.25, 2.4, H, CHECK_COLOR),
    ("Benchmark\nValidation", 0.8, 7.55, 2.4, H, CHECK_COLOR),
]

node_dict = {n[0].split("\n")[0]: n for n in nodes}


def node_bbox(n):
    """Return (x_left, x_right, y_bottom, y_top) of node."""
    _, x, y, w, h, _ = n
    return x - w / 2, x + w / 2, y - h / 2, y + h / 2


# ── Draw ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 18))
ax.set_xlim(-0.5, 11.5)
ax.set_ylim(-1.6, 14.0)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# Title
ax.text(
    5.5,
    13.75,
    "AutoReview Pipeline",
    ha="center",
    va="center",
    fontsize=17,
    fontweight="bold",
    color=TEXT_MAIN,
    fontfamily="DejaVu Sans",
)


def draw_node(ax, n):
    label, x, y, w, h, color = n
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.05",
        linewidth=1.2,
        edgecolor="#333333",
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=7.8,
        color=TEXT_LIGHT,
        fontweight="bold",
        zorder=4,
        multialignment="center",
    )


def arrow(ax, x1, y1, x2, y2, color=ARROW, lw=1.6, style="->", ls="-"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style, color=color, lw=lw, linestyle=ls, connectionstyle="arc3,rad=0.0"
        ),
        zorder=2,
    )


def curved_arrow(ax, x1, y1, x2, y2, color=ARROW, lw=1.4, rad=0.25):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, connectionstyle=f"arc3,rad={rad}"),
        zorder=2,
    )


# Draw all nodes
for n in nodes:
    draw_node(ax, n)

# ── Phase labels (left margin) ───────────────────────────────────────────────
phases = [
    ("RETRIEVAL", 12.1, 11.2, SEARCH_COLOR),
    ("EXTRACTION", 10.3, 10.3, EXTRACT_COLOR),
    ("ANALYSIS", 9.35, 8.25, ANALYSIS_COLOR),
    ("PLANNING", 7.15, 5.35, ENRICH_COLOR),
    ("WRITING", 4.25, 3.15, WRITING_COLOR),
    ("ASSEMBLY\n& OUTPUT", 2.05, -0.05, OUTPUT_COLOR),
]

for label, y_top, y_bot, color in phases:
    mid_y = (y_top + y_bot) / 2
    ax.text(
        -0.35,
        mid_y,
        label,
        ha="center",
        va="center",
        fontsize=6.5,
        fontweight="bold",
        color=color,
        rotation=90,
        fontstyle="italic",
        alpha=0.7,
    )

# ── Main spine arrows ─────────────────────────────────────────────────────────
spine = [
    "Query Expansion",
    "Multi-Source Search",
    "Screen & Deduplicate",
    "Parallel Extraction",
    "Thematic Clustering",
    "Gap-Aware",
    "Outline Generation",
    "Narrative Planning",
    "Contextual Enrichment",
    "Section Writing",
    "Passage Search",
    "Assemble Draft",
    "Final Polish",
    "Format Output",
]

spine_nodes = []
for label in spine:
    match = next(n for n in nodes if n[0].startswith(label))
    spine_nodes.append(match)

for i in range(len(spine_nodes) - 1):
    n1 = spine_nodes[i]
    n2 = spine_nodes[i + 1]
    _, x1, y1, _, h1, _ = n1
    _, x2, y2, _, h2, _ = n2
    arrow(ax, x1, y1 - h1 / 2, x2, y2 + h2 / 2)

# ── Critique loop arrows ──────────────────────────────────────────────────────
# Outline Generation ↔ Outline Critique
n_out = next(n for n in nodes if n[0] == "Outline Generation")
n_oc = next(n for n in nodes if n[0].startswith("Outline Critique"))
arrow(ax, n_out[1] + n_out[3] / 2, n_out[2], n_oc[1] - n_oc[3] / 2, n_oc[2], color=CRITIQUE_COLOR)
curved_arrow(
    ax,
    n_oc[1] - n_oc[3] / 2,
    n_oc[2] - 0.12,
    n_out[1] + n_out[3] / 2,
    n_out[2] - 0.12,
    color=LOOP_ARROW,
    rad=-0.25,
)

# Section Writing ↔ Per-Section Critique
n_sw = next(n for n in nodes if n[0] == "Section Writing")
n_sc = next(n for n in nodes if n[0].startswith("Per-Section Critique"))
arrow(ax, n_sw[1] + n_sw[3] / 2, n_sw[2], n_sc[1] - n_sc[3] / 2, n_sc[2], color=CRITIQUE_COLOR)
curved_arrow(
    ax,
    n_sc[1] - n_sc[3] / 2,
    n_sc[2] - 0.12,
    n_sw[1] + n_sw[3] / 2,
    n_sw[2] - 0.12,
    color=LOOP_ARROW,
    rad=-0.25,
)

# Assemble Draft ↔ Holistic Critique
n_ad = next(n for n in nodes if n[0] == "Assemble Draft")
n_hc = next(n for n in nodes if n[0].startswith("Holistic Critique"))
arrow(ax, n_ad[1] + n_ad[3] / 2, n_ad[2], n_hc[1] - n_hc[3] / 2, n_hc[2], color=CRITIQUE_COLOR)
curved_arrow(
    ax,
    n_hc[1] - n_hc[3] / 2,
    n_hc[2] - 0.12,
    n_ad[1] + n_ad[3] / 2,
    n_ad[2] - 0.12,
    color=LOOP_ARROW,
    rad=-0.25,
)

# ── Comprehensiveness check arrows (left side) ────────────────────────────────
# Query Coverage Check  ← Query Expansion
n_qe = next(n for n in nodes if n[0] == "Query Expansion")
n_qcc = next(n for n in nodes if n[0].startswith("Query Coverage"))
arrow(
    ax,
    n_qe[1] - n_qe[3] / 2,
    n_qe[2],
    n_qcc[1] + n_qcc[3] / 2,
    n_qcc[2],
    color=CHECK_COLOR,
    lw=1.2,
    ls="--",
)

# Coverage Anomaly ← Screen & Deduplicate
n_sd = next(n for n in nodes if n[0] == "Screen & Deduplicate")
n_cac = next(n for n in nodes if n[0].startswith("Coverage Anomaly"))
arrow(
    ax,
    n_sd[1] - n_sd[3] / 2,
    n_sd[2],
    n_cac[1] + n_cac[3] / 2,
    n_cac[2],
    color=CHECK_COLOR,
    lw=1.2,
    ls="--",
)

# Borderline Rescreening ← Screen & Deduplicate (slightly lower)
n_br = next(n for n in nodes if n[0].startswith("Borderline"))
ax.annotate(
    "",
    xy=(n_br[1] + n_br[3] / 2, n_br[2]),
    xytext=(n_sd[1] - n_sd[3] / 2, n_sd[2] - 0.18),
    arrowprops=dict(
        arrowstyle="->", color=CHECK_COLOR, lw=1.2, linestyle="--", connectionstyle="arc3,rad=-0.2"
    ),
    zorder=2,
)

# Post-Gap Revalidation ← Gap-Aware Search
n_gs = next(n for n in nodes if n[0].startswith("Gap-Aware"))
n_pgr = next(n for n in nodes if n[0].startswith("Post-Gap"))
arrow(
    ax,
    n_gs[1] - n_gs[3] / 2,
    n_gs[2],
    n_pgr[1] + n_pgr[3] / 2,
    n_pgr[2],
    color=CHECK_COLOR,
    lw=1.2,
    ls="--",
)

# Benchmark Validation ← Gap-Aware Search (slightly lower)
n_bv = next(n for n in nodes if n[0].startswith("Benchmark"))
ax.annotate(
    "",
    xy=(n_bv[1] + n_bv[3] / 2, n_bv[2]),
    xytext=(n_gs[1] - n_gs[3] / 2, n_gs[2] - 0.18),
    arrowprops=dict(
        arrowstyle="->", color=CHECK_COLOR, lw=1.2, linestyle="--", connectionstyle="arc3,rad=0.25"
    ),
    zorder=2,
)

# ── "conditional" label on Gap-Aware Search node ─────────────────────────────
n_gs_node = next(n for n in nodes if n[0].startswith("Gap-Aware"))
ax.text(
    n_gs_node[1],
    n_gs_node[2] - n_gs_node[4] / 2 - 0.13,
    "conditional",
    ha="center",
    va="top",
    fontsize=6.5,
    color=SEARCH_COLOR,
    style="italic",
)

# ── Enrichment annotation ────────────────────────────────────────────────────
n_ce = next(n for n in nodes if n[0] == "Contextual Enrichment")
ax.text(
    n_ce[1] + n_ce[3] / 2 + 0.15,
    n_ce[2],
    "adjacent material\n(background, cross-field,\nclinical implications)",
    ha="left",
    va="center",
    fontsize=6.0,
    color=ENRICH_COLOR,
    style="italic",
)

# ── Passage search annotation ────────────────────────────────────────────────
n_ps = next(n for n in nodes if n[0].startswith("Passage Search"))
ax.text(
    n_ps[1] + n_ps[3] / 2 + 0.15,
    n_ps[2],
    "mine undercited claims,\nretrieve + revise sections",
    ha="left",
    va="center",
    fontsize=6.0,
    color=SEARCH_COLOR,
    style="italic",
)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    (SEARCH_COLOR, "Search / Retrieval"),
    (EXTRACT_COLOR, "Extraction"),
    (ANALYSIS_COLOR, "Analysis"),
    (WRITING_COLOR, "Writing / Outline"),
    (CRITIQUE_COLOR, "Critique & Revision"),
    (OUTPUT_COLOR, "Output"),
    (ENRICH_COLOR, "Narrative / Enrichment"),
    (CHECK_COLOR, "Comprehensiveness Checks"),
]

lx, ly = 0.0, -0.75
for i, (color, label) in enumerate(legend_items):
    col = i % 4
    row = i // 4
    bx = lx + col * 2.85
    by = ly - row * 0.42
    box = FancyBboxPatch(
        (bx, by - 0.14),
        0.30,
        0.28,
        boxstyle="round,pad=0.03",
        facecolor=color,
        edgecolor="#333",
        lw=0.8,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(bx + 0.42, by, label, va="center", fontsize=7.2, color=TEXT_MAIN)

# Arrow style legend
ax.annotate(
    "", xy=(9.6, -0.75), xytext=(9.0, -0.75), arrowprops=dict(arrowstyle="->", color=ARROW, lw=1.6)
)
ax.text(9.72, -0.75, "pipeline flow", va="center", fontsize=7.2, color=TEXT_MAIN)

ax.annotate(
    "",
    xy=(9.6, -1.17),
    xytext=(9.0, -1.17),
    arrowprops=dict(arrowstyle="->", color=CHECK_COLOR, lw=1.2, linestyle="--"),
)
ax.text(9.72, -1.17, "check output", va="center", fontsize=7.2, color=TEXT_MAIN)

ax.annotate(
    "",
    xy=(9.6, -1.59),
    xytext=(9.0, -1.59),
    arrowprops=dict(arrowstyle="->", color=LOOP_ARROW, lw=1.4, linestyle="-"),
)
ax.text(9.72, -1.59, "revision loop", va="center", fontsize=7.2, color=TEXT_MAIN)

plt.tight_layout(pad=0.3)
out = "docs/pipeline_overview.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")
