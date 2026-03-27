"""Generate publication-quality 6-panel comparison figure: Opus vs Sonnet extraction results."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mycelium.extraction_schema import ExtractionResult

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/Users/mst36/Desktop/Projects/Science/Autonomous Science/outputs")
OPUS_PATH = BASE / "extraction_test_opus.json"
SONNET_PATH = BASE / "extraction_test_sonnet.json"
OUT_PATH = BASE / "extraction_comparison.png"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
opus_raw = OPUS_PATH.read_text()
sonnet_raw = SONNET_PATH.read_text()

opus: ExtractionResult = ExtractionResult.model_validate_json(opus_raw)
sonnet: ExtractionResult = ExtractionResult.model_validate_json(sonnet_raw)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
OPUS_COLOR = "#0072B2"  # Blue
SONNET_COLOR = "#D55E00"  # Orange
ACCENT = "#009E73"  # Green

CB_PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
    "#000000",
]

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
fig.suptitle(
    "Mycelium Extraction Framework: Opus vs Sonnet Comparison\n"
    "Test Paper: Lobos Patorniti et al. (2023) — Rai14/Invariant chain",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)

# Shared legend patches
opus_patch = mpatches.Patch(color=OPUS_COLOR, label="Opus")
sonnet_patch = mpatches.Patch(color=SONNET_COLOR, label="Sonnet")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def add_panel_label(ax, label):
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )


def light_grid(ax, axis="y"):
    ax.set_axisbelow(True)
    if axis == "y":
        ax.yaxis.grid(True, color="#dddddd", linewidth=0.7, linestyle="--")
    else:
        ax.xaxis.grid(True, color="#dddddd", linewidth=0.7, linestyle="--")


def bar_labels(ax, bars, fmt="{:.0f}", offset_frac=0.01):
    """Add value labels on top of bars."""
    max_h = max(b.get_height() for b in bars if b.get_height() > 0) if bars else 1
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + max_h * offset_frac,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=7.5,
            )


# ---------------------------------------------------------------------------
# Panel A — Coverage Overview
# ---------------------------------------------------------------------------
ax = axes[0, 0]

o_assertions = len(opus.assertion_drafts)
s_assertions = len(sonnet.assertion_drafts)
o_evidence = len(opus.evidence_units)
s_evidence = len(sonnet.evidence_units)
o_primary = sum(1 for a in opus.assertion_drafts if a.epistemic_status.is_primary)
s_primary = sum(1 for a in sonnet.assertion_drafts if a.epistemic_status.is_primary)
o_bg = o_assertions - o_primary
s_bg = s_assertions - s_primary

categories = [
    "Total\nassertions",
    "Total\nevidence units",
    "Novel findings\n(is_primary=T)",
    "Background\n(is_primary=F)",
]
opus_vals = [o_assertions, o_evidence, o_primary, o_bg]
sonnet_vals = [s_assertions, s_evidence, s_primary, s_bg]

x = np.arange(len(categories))
w = 0.35
bars_o = ax.bar(x - w / 2, opus_vals, w, color=OPUS_COLOR, label="Opus", zorder=3)
bars_s = ax.bar(x + w / 2, sonnet_vals, w, color=SONNET_COLOR, label="Sonnet", zorder=3)
bar_labels(ax, bars_o)
bar_labels(ax, bars_s)

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=8.5)
ax.set_ylabel("Count")
ax.set_title("A  Coverage Overview")
add_panel_label(ax, "A")
light_grid(ax)
ax.legend(handles=[opus_patch, sonnet_patch], loc="upper right", frameon=False)

# Annotation: which model extracted more
diff = o_assertions - s_assertions
sign = "more" if diff > 0 else "fewer"
ax.annotate(
    f"Opus: {abs(diff)} {sign}\nassertions than Sonnet",
    xy=(0, max(o_assertions, s_assertions)),
    xytext=(0.05, 0.92),
    textcoords="axes fraction",
    fontsize=7.5,
    color="#444444",
    ha="left",
)

# ---------------------------------------------------------------------------
# Panel B — Assertion Type Distribution (stacked horizontal bar)
# ---------------------------------------------------------------------------
ax = axes[0, 1]

ALL_TYPES = [
    "mechanistic_causal",
    "correlational",
    "existence",
    "comparative",
    "conditional",
    "absence",
    "methodological",
]
TYPE_LABELS = [
    "Mechanistic\ncausal",
    "Correlational",
    "Existence",
    "Comparative",
    "Conditional",
    "Absence",
    "Methodological",
]


def count_types(result):
    c = Counter(a.assertion_type.value for a in result.assertion_drafts)
    return [c.get(t, 0) for t in ALL_TYPES]


opus_type_counts = np.array(count_types(opus))
sonnet_type_counts = np.array(count_types(sonnet))
# Normalise to % of total assertions
o_total = opus_type_counts.sum()
s_total = sonnet_type_counts.sum()
opus_pct = opus_type_counts / o_total * 100
sonnet_pct = sonnet_type_counts / s_total * 100

y_pos = np.array([1, 0])  # Opus=1, Sonnet=0
colors_types = CB_PALETTE[: len(ALL_TYPES)]

left_o = 0
left_s = 0
for i, (lbl, co, cs, col) in enumerate(zip(TYPE_LABELS, opus_pct, sonnet_pct, colors_types)):
    ax.barh(1, co, left=left_o, color=col, label=lbl if i == 0 else "", height=0.5)
    ax.barh(0, cs, left=left_s, color=col, height=0.5)
    left_o += co
    left_s += cs

ax.set_yticks([0, 1])
ax.set_yticklabels(["Sonnet", "Opus"], fontsize=10)
ax.set_xlabel("% of assertions")
ax.set_title("B  Assertion Type Distribution")
add_panel_label(ax, "B")
light_grid(ax, axis="x")

# Legend for types
type_patches = [
    mpatches.Patch(color=col, label=lbl.replace("\n", " "))
    for lbl, col in zip(TYPE_LABELS, colors_types)
]
ax.legend(
    handles=type_patches, loc="lower right", fontsize=7, frameon=True, framealpha=0.85, ncol=1
)

# Annotation: mechanistic causal
o_mech_pct = opus_pct[0]
s_mech_pct = sonnet_pct[0]
ax.annotate(
    f"Opus: {o_mech_pct:.0f}% mechanistic\nvs Sonnet: {s_mech_pct:.0f}%",
    xy=(0.01, 1.05),
    xytext=(0.05, 0.92),
    textcoords="axes fraction",
    fontsize=7.5,
    color="#444444",
)

# ---------------------------------------------------------------------------
# Panel C — Scope Completeness
# ---------------------------------------------------------------------------
ax = axes[0, 2]


def scope_completeness(result):
    assertions = result.assertion_drafts
    n = len(assertions)
    if n == 0:
        return [0, 0, 0, 0]
    has_species = sum(1 for a in assertions if len(a.scope.species) > 0) / n * 100
    has_cell_type = sum(1 for a in assertions if len(a.scope.cell_type) > 0) / n * 100
    has_in_vitro = sum(1 for a in assertions if a.scope.in_vitro is not None) / n * 100
    has_hedge = (
        sum(
            1
            for a in assertions
            if a.hedging.verbatim_hedge is not None and a.hedging.verbatim_hedge.strip() != ""
        )
        / n
        * 100
    )
    return [has_species, has_cell_type, has_in_vitro, has_hedge]


field_labels = ["Species", "Cell type", "in_vitro\nflag", "Verbatim\nhedge"]
opus_scope = scope_completeness(opus)
sonnet_scope = scope_completeness(sonnet)

x = np.arange(len(field_labels))
w = 0.35
bars_o = ax.bar(x - w / 2, opus_scope, w, color=OPUS_COLOR, zorder=3)
bars_s = ax.bar(x + w / 2, sonnet_scope, w, color=SONNET_COLOR, zorder=3)
bar_labels(ax, bars_o, fmt="{:.0f}%")
bar_labels(ax, bars_s, fmt="{:.0f}%")

ax.set_xticks(x)
ax.set_xticklabels(field_labels, fontsize=8.5)
ax.set_ylabel("% assertions with field")
ax.set_ylim(0, 115)
ax.set_title("C  Scope Completeness")
add_panel_label(ax, "C")
light_grid(ax)
ax.legend(handles=[opus_patch, sonnet_patch], loc="upper right", frameon=False)

# Annotation
best_field_idx = np.argmax(np.array(sonnet_scope) - np.array(opus_scope))
best_field = field_labels[best_field_idx].replace("\n", " ")
diff_field = sonnet_scope[best_field_idx] - opus_scope[best_field_idx]
ax.annotate(
    f"Sonnet leads on\n'{best_field}'\nby {diff_field:.0f} pp",
    xy=(0.6, 0.85),
    xytext=(0.6, 0.85),
    textcoords="axes fraction",
    fontsize=7.5,
    color=SONNET_COLOR,
    ha="center",
)

# ---------------------------------------------------------------------------
# Panel D — Epistemic Function Distribution
# ---------------------------------------------------------------------------
ax = axes[1, 0]

ALL_FUNCS = [
    "novel_finding",
    "background",
    "interpretation",
    "hypothesis",
    "limitation",
    "replication",
    "methodological_note",
]
FUNC_LABELS = [
    "Novel finding",
    "Background",
    "Interpretation",
    "Hypothesis",
    "Limitation",
    "Replication",
    "Method. note",
]


def count_funcs(result):
    c = Counter(a.epistemic_status.function.value for a in result.assertion_drafts)
    return [c.get(f, 0) for f in ALL_FUNCS]


opus_func = np.array(count_funcs(opus))
sonnet_func = np.array(count_funcs(sonnet))
o_tot = opus_func.sum()
s_tot = sonnet_func.sum()
opus_func_pct = opus_func / o_tot * 100 if o_tot else opus_func
sonnet_func_pct = sonnet_func / s_tot * 100 if s_tot else sonnet_func

colors_funcs = CB_PALETTE[: len(ALL_FUNCS)]
left_o = 0
left_s = 0
for i, (lbl, co, cs, col) in enumerate(
    zip(FUNC_LABELS, opus_func_pct, sonnet_func_pct, colors_funcs)
):
    ax.barh(1, co, left=left_o, color=col, height=0.5)
    ax.barh(0, cs, left=left_s, color=col, height=0.5)
    left_o += co
    left_s += cs

ax.set_yticks([0, 1])
ax.set_yticklabels(["Sonnet", "Opus"], fontsize=10)
ax.set_xlabel("% of assertions")
ax.set_title("D  Epistemic Function Distribution")
add_panel_label(ax, "D")
light_grid(ax, axis="x")

func_patches = [mpatches.Patch(color=col, label=lbl) for lbl, col in zip(FUNC_LABELS, colors_funcs)]
ax.legend(
    handles=func_patches, loc="lower right", fontsize=7, frameon=True, framealpha=0.85, ncol=1
)

# Annotation
o_novel = opus_func_pct[0]
s_novel = sonnet_func_pct[0]
ax.annotate(
    f"Novel finding:\nOpus {o_novel:.0f}% vs Sonnet {s_novel:.0f}%",
    xy=(0.05, 0.88),
    xytext=(0.05, 0.88),
    textcoords="axes fraction",
    fontsize=7.5,
    color="#444444",
)

# ---------------------------------------------------------------------------
# Panel E — Evidence Unit Assay Coverage
# ---------------------------------------------------------------------------
ax = axes[1, 1]


def get_approach_categories(result):
    """Collect approach_category counts across evidence units."""
    cats = []
    for eu in result.evidence_units:
        cat = eu.methodological_tags.approach_category
        # Also include first assay type if available
        assays = eu.methodological_tags.assay_types
        if assays:
            cats.extend(assays)
        else:
            cats.append(cat)
    return cats


opus_cats = Counter(get_approach_categories(opus))
sonnet_cats = Counter(get_approach_categories(sonnet))

# Union of top categories
all_cats = sorted(
    set(list(opus_cats.keys()) + list(sonnet_cats.keys())),
    key=lambda k: -(opus_cats.get(k, 0) + sonnet_cats.get(k, 0)),
)[:12]


# Shorten long labels
def shorten(s):
    replacements = {
        "co-immunoprecipitation": "co-IP",
        "co_immunoprecipitation": "co-IP",
        "immunofluorescence": "immunofluor.",
        "yeast_two_hybrid": "Y2H",
        "western_blot": "Western blot",
        "biochemical_assay": "biochemical",
        "genetic_loss_of_function": "LOF",
        "imaging": "imaging",
        "pharmacological_inhibition": "pharm. inhib.",
        "macropinocytosis_assay": "macropinocytosis",
        "migration_assay": "migration",
        "flow_cytometry": "flow cytometry",
        "proximity_ligation_assay": "PLA",
        "confocal_microscopy": "confocal",
        "fluorescence_microscopy": "fluorescence",
    }
    return replacements.get(s, s.replace("_", " "))


cat_labels = [shorten(c) for c in all_cats]
opus_ev_vals = [opus_cats.get(c, 0) for c in all_cats]
sonnet_ev_vals = [sonnet_cats.get(c, 0) for c in all_cats]

y = np.arange(len(all_cats))
h = 0.35
ax.barh(y + h / 2, opus_ev_vals, h, color=OPUS_COLOR, label="Opus", zorder=3)
ax.barh(y - h / 2, sonnet_ev_vals, h, color=SONNET_COLOR, label="Sonnet", zorder=3)

ax.set_yticks(y)
ax.set_yticklabels(cat_labels, fontsize=8)
ax.set_xlabel("Count")
ax.set_title("E  Evidence Unit Assay Coverage")
add_panel_label(ax, "E")
light_grid(ax, axis="x")
ax.legend(handles=[opus_patch, sonnet_patch], loc="lower right", frameon=False)

# Annotation
max_assay = all_cats[0] if all_cats else ""
ax.annotate(
    f"Top assay: {shorten(max_assay)}",
    xy=(0.55, 0.97),
    xytext=(0.55, 0.97),
    textcoords="axes fraction",
    fontsize=7.5,
    color="#444444",
    ha="center",
)

# ---------------------------------------------------------------------------
# Panel F — Entity Extraction Quality
# ---------------------------------------------------------------------------
ax = axes[1, 2]


def entity_stats(result):
    all_entities = []
    for a in result.assertion_drafts:
        all_entities.append(a.subject_entity)
        if a.object_entity is not None:
            all_entities.append(a.object_entity)

    surface_forms = [e.surface_form for e in all_entities]
    unique_forms = set(surface_forms)
    avg_len = np.mean([len(s) for s in surface_forms]) if surface_forms else 0
    with_ontology = sum(
        1 for e in all_entities if e.ontology_id is not None and e.ontology_id.strip() != ""
    )
    return len(unique_forms), avg_len, with_ontology


o_uniq, o_avg_len, o_onto = entity_stats(opus)
s_uniq, s_avg_len, s_onto = entity_stats(sonnet)

metrics = ["Unique surface\nforms", "Avg name\nlength (chars)", "With\nontology ID"]
opus_entity_vals = [o_uniq, o_avg_len, o_onto]
sonnet_entity_vals = [s_uniq, s_avg_len, s_onto]

x = np.arange(len(metrics))
w = 0.35
bars_o = ax.bar(x - w / 2, opus_entity_vals, w, color=OPUS_COLOR, zorder=3)
bars_s = ax.bar(x + w / 2, sonnet_entity_vals, w, color=SONNET_COLOR, zorder=3)
bar_labels(ax, bars_o, fmt="{:.1f}")
bar_labels(ax, bars_s, fmt="{:.1f}")

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=8.5)
ax.set_ylabel("Value")
ax.set_title("F  Entity Extraction Quality")
add_panel_label(ax, "F")
light_grid(ax)
ax.legend(handles=[opus_patch, sonnet_patch], loc="upper right", frameon=False)

# Annotation
len_diff = o_avg_len - s_avg_len
sign = "longer" if len_diff > 0 else "shorter"
ax.annotate(
    f"Opus names {abs(len_diff):.1f} chars {sign}\n(more verbose compound phrases)",
    xy=(1, 0.85),
    xytext=(0.5, 0.88),
    textcoords="axes fraction",
    fontsize=7.5,
    color="#444444",
    ha="center",
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")

# ---------------------------------------------------------------------------
# Summary stats printout
# ---------------------------------------------------------------------------
print("\n=== Summary Statistics ===")
print(f"{'Metric':<40} {'Opus':>8} {'Sonnet':>8}")
print("-" * 58)
print(f"{'Assertions':<40} {o_assertions:>8} {s_assertions:>8}")
print(f"{'Evidence units':<40} {o_evidence:>8} {s_evidence:>8}")
print(f"{'Novel findings (is_primary=T)':<40} {o_primary:>8} {s_primary:>8}")
print(f"{'Background (is_primary=F)':<40} {o_bg:>8} {s_bg:>8}")
print(f"{'Scope: species filled (%)':<40} {opus_scope[0]:>7.1f}% {sonnet_scope[0]:>7.1f}%")
print(f"{'Scope: cell_type filled (%)':<40} {opus_scope[1]:>7.1f}% {sonnet_scope[1]:>7.1f}%")
print(f"{'Scope: in_vitro filled (%)':<40} {opus_scope[2]:>7.1f}% {sonnet_scope[2]:>7.1f}%")
print(f"{'Hedging: verbatim hedge (%)':<40} {opus_scope[3]:>7.1f}% {sonnet_scope[3]:>7.1f}%")
print(f"{'Entities: unique surface forms':<40} {o_uniq:>8} {s_uniq:>8}")
print(f"{'Entities: avg name length (chars)':<40} {o_avg_len:>8.1f} {s_avg_len:>8.1f}")
print(f"{'Entities: with ontology ID':<40} {o_onto:>8} {s_onto:>8}")
