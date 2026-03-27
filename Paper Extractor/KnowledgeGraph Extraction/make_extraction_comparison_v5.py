"""
Publication-quality 6-panel comparison figure for v5 Haiku extraction vs previous iterations.
Handles both v5 (claims format) and v4/v2/v1 (assertion_drafts format).

Output: extraction_comparison_v5.png (300 DPI, colorblind-safe palette)
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE = Path(
    "/Users/mst36/Desktop/Projects/Science/AutoReview/Paper Extractor/KnowledgeGraph Extraction"
)

FILE_MAP = {
    "Opus v1": BASE / "extraction_test_opus.json",
    "Sonnet v2": BASE / "extraction_test_sonnet_v2.json",
    "Haiku v4": BASE / "extraction_test_haiku.json",
    "Haiku v5": BASE / "extraction_test_haiku_v5.json",
}

# ---------------------------------------------------------------------------
# Helpers — unified accessors for both formats
# ---------------------------------------------------------------------------


def load_and_parse(path: Path) -> dict:
    with open(path) as f:
        raw = json.load(f)

    # Detect format
    if "claims" in raw:
        # v5 format
        claims = raw["claims"]
        evidence = raw.get("evidence", [])
        citation_contexts = raw.get("citation_contexts", [])
        assertion_types = Counter(c.get("claim_type", "unknown") for c in claims)
        evidence_strengths = Counter(c.get("evidence_strength", "unknown") for c in claims)
        return {
            "format": "v5",
            "raw": raw,
            "claims": claims,
            "evidence": evidence,
            "citation_contexts": citation_contexts,
            "n_claims": len(claims),
            "n_evidence": len(evidence),
            "n_citations": len(citation_contexts),
            "assertion_types": assertion_types,
            "evidence_strengths": evidence_strengths,
        }
    else:
        # v4/v2/v1 format
        drafts = raw.get("assertion_drafts", [])
        evidence_units = raw.get("evidence_units", [])
        citation_contexts = raw.get("citation_contexts", [])
        assertion_types = Counter(a.get("assertion_type", "unknown") for a in drafts)
        evidence_strengths = Counter(e.get("evidence_strength", "unknown") for e in evidence_units)
        return {
            "format": "legacy",
            "raw": raw,
            "claims": drafts,
            "evidence": evidence_units,
            "citation_contexts": citation_contexts,
            "n_claims": len(drafts),
            "n_evidence": len(evidence_units),
            "n_citations": len(citation_contexts),
            "assertion_types": assertion_types,
            "evidence_strengths": evidence_strengths,
        }


def pct(n: int, total: int) -> float:
    return 100.0 * n / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

data: dict[str, dict] = {}
for label, path in FILE_MAP.items():
    data[label] = load_and_parse(path)

MODEL_NAMES = list(FILE_MAP.keys())

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]  # Opus, Sonnet, Haiku v4, Haiku v5
MODEL_COLORS = dict(zip(MODEL_NAMES, PALETTE))

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

fig, axes = plt.subplots(2, 3, figsize=(14, 10), constrained_layout=True)
fig.suptitle(
    "KG Extraction Framework: Four-Version Comparison (v1→v5)\n"
    "Same paper: Rai14/Invariant chain (Lobos Patorniti 2023)",
    fontsize=13,
    fontweight="bold",
)


def panel_label(ax, letter, x=-0.13, y=1.04):
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


# ---------------------------------------------------------------------------
# Panel A — Coverage Overview (grouped bar chart)
# ---------------------------------------------------------------------------
ax = axes[0, 0]
panel_label(ax, "A")

bar_labels = ["Claims /\nAssertions", "Evidence\nUnits", "Citation\nContexts"]
n_groups = len(MODEL_NAMES)
n_bars = len(bar_labels)
width = 0.18
group_gap = 0.15
x = np.arange(n_groups) * (n_bars * width + group_gap + 0.2)

bar_colors = ["#333333", "#888888", "#cccccc"]
for bi, bl in enumerate(bar_labels):
    if bi == 0:
        vals = [data[m]["n_claims"] for m in MODEL_NAMES]
    elif bi == 1:
        vals = [data[m]["n_evidence"] for m in MODEL_NAMES]
    else:
        vals = [data[m]["n_citations"] for m in MODEL_NAMES]
    offsets = x + bi * width
    bars = ax.bar(offsets, vals, width=width, color=bar_colors[bi], label=bl, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            str(v),
            ha="center",
            va="bottom",
            fontsize=8,
        )

group_centers = x + (n_bars - 1) * width / 2
ax.set_xticks(group_centers)
ax.set_xticklabels(MODEL_NAMES, fontsize=9)
ax.set_ylabel("Count")
ax.set_title("Coverage Overview")
ax.legend(loc="upper right", framealpha=0.85, fontsize=8)
ax.set_ylim(0, max(data[m]["n_claims"] for m in MODEL_NAMES) * 1.3)


# ---------------------------------------------------------------------------
# Panel B — Claim/Assertion Type Distribution (stacked horizontal bars)
# ---------------------------------------------------------------------------
ax = axes[0, 1]
panel_label(ax, "B")

# Unified type key set
ALL_TYPES = [
    "mechanistic_causal",
    "existence",
    "correlational",
    "conditional",
    "comparative",
    "methodological",
]
type_colors_map = {
    "mechanistic_causal": "#0072B2",
    "existence": "#D55E00",
    "correlational": "#009E73",
    "conditional": "#CC79A7",
    "comparative": "#F0E442",
    "methodological": "#56B4E9",
}
type_labels_map = {
    "mechanistic_causal": "Mechanistic / causal",
    "existence": "Existence",
    "correlational": "Correlational",
    "conditional": "Conditional",
    "comparative": "Comparative",
    "methodological": "Methodological",
}

y_pos = np.arange(len(MODEL_NAMES))
lefts = np.zeros(len(MODEL_NAMES))

for tk in ALL_TYPES:
    vals = np.array([data[m]["assertion_types"].get(tk, 0) for m in MODEL_NAMES])
    if vals.sum() == 0:
        continue
    bars = ax.barh(
        y_pos,
        vals,
        left=lefts,
        color=type_colors_map[tk],
        label=type_labels_map[tk],
        height=0.55,
        zorder=3,
    )
    for bar, v, left in zip(bars, vals, lefts):
        if v > 0:
            ax.text(
                left + v / 2,
                bar.get_y() + bar.get_height() / 2,
                str(int(v)),
                ha="center",
                va="center",
                fontsize=8.5,
                color="white",
                fontweight="bold",
            )
    lefts += vals

ax.set_yticks(y_pos)
ax.set_yticklabels(MODEL_NAMES, fontsize=9)
ax.set_xlabel("Number of claims / assertions")
ax.set_title("Claim Type Distribution")
ax.legend(loc="lower right", fontsize=7.5, framealpha=0.85)
ax.set_xlim(0, max(data[m]["n_claims"] for m in MODEL_NAMES) * 1.15)

# ---------------------------------------------------------------------------
# Panel C — V5 Field Completeness (bar chart, v5 only)
# ---------------------------------------------------------------------------
ax = axes[0, 2]
panel_label(ax, "C")

v5_claims = data["Haiku v5"]["claims"]
n_v5 = len(v5_claims)

fields = [
    "model_system",
    "organism",
    "certainty",
    "section_source",
    "quantitative_context\n(has values)",
]


def v5_completeness(claims):
    n = len(claims)
    return [
        pct(sum(1 for c in claims if c.get("model_system")), n),
        pct(sum(1 for c in claims if c.get("organism")), n),
        pct(sum(1 for c in claims if c.get("certainty")), n),
        pct(sum(1 for c in claims if c.get("section_source")), n),
        pct(
            sum(
                1
                for c in claims
                if c.get("quantitative_context")
                and any(v is not None for v in c["quantitative_context"].values())
            ),
            n,
        ),
    ]


completeness_vals = v5_completeness(v5_claims)
x_c = np.arange(len(fields))
bars = ax.bar(x_c, completeness_vals, color=MODEL_COLORS["Haiku v5"], alpha=0.88, zorder=3)
for bar, v in zip(bars, completeness_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f"{v:.0f}%",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )

ax.axhline(100, color="#444444", linestyle="--", linewidth=1.2, zorder=4, label="100% target")
ax.set_xticks(x_c)
ax.set_xticklabels(fields, rotation=15, ha="right", fontsize=8.5)
ax.set_ylabel("Completeness (%)")
ax.set_ylim(0, 118)
ax.set_title(f"V5 Field Completeness (n={n_v5} claims)")
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.85)

# ---------------------------------------------------------------------------
# Panel D — Section Source Distribution (v5 only, horizontal bar)
# ---------------------------------------------------------------------------
ax = axes[1, 0]
panel_label(ax, "D")

section_counts = Counter(c.get("section_source", "unknown") for c in v5_claims)
sections = sorted(section_counts.keys())
counts = [section_counts[s] for s in sections]
y_d = np.arange(len(sections))
section_colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"][: len(sections)]

bars_d = ax.barh(y_d, counts, color=section_colors, alpha=0.88, height=0.55, zorder=3)
for bar, v in zip(bars_d, counts):
    pct_val = pct(v, n_v5)
    ax.text(
        bar.get_width() + 0.15,
        bar.get_y() + bar.get_height() / 2,
        f"{v} ({pct_val:.0f}%)",
        va="center",
        ha="left",
        fontsize=8.5,
    )

ax.set_yticks(y_d)
ax.set_yticklabels(sections, fontsize=9)
ax.set_xlabel("Number of claims")
ax.set_title(f"Section Source Distribution\n(V5 Haiku, n={n_v5})")
ax.set_xlim(0, max(counts) * 1.45)

# ---------------------------------------------------------------------------
# Panel E — Evidence Strength Comparison (grouped bars, all 4)
# ---------------------------------------------------------------------------
ax = axes[1, 1]
panel_label(ax, "E")

ALL_STRENGTHS = [
    "direct_experimental",
    "observational_uncontrolled",
    "computational",
    "indirect",
]
strength_labels = {
    "direct_experimental": "Direct exp.",
    "observational_uncontrolled": "Observational",
    "computational": "Computational",
    "indirect": "Indirect",
}

# Collect which strengths are actually present
present_strengths = [
    s
    for s in ALL_STRENGTHS
    if any(data[m]["evidence_strengths"].get(s, 0) > 0 for m in MODEL_NAMES)
]

n_str = len(present_strengths)
width_e = 0.18
x_e = np.arange(len(present_strengths))

for mi, mname in enumerate(MODEL_NAMES):
    vals = [data[mname]["evidence_strengths"].get(s, 0) for s in present_strengths]
    offsets = x_e + (mi - (len(MODEL_NAMES) - 1) / 2) * width_e
    bars_e = ax.bar(
        offsets,
        vals,
        width=width_e,
        color=MODEL_COLORS[mname],
        alpha=0.88,
        label=mname,
        zorder=3,
    )
    for bar, v in zip(bars_e, vals):
        if v > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                str(v),
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

ax.set_xticks(x_e)
ax.set_xticklabels([strength_labels.get(s, s) for s in present_strengths], fontsize=9)
ax.set_ylabel("Count")
ax.set_title("Evidence Strength Comparison")
ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

# ---------------------------------------------------------------------------
# Panel F — Summary Table
# ---------------------------------------------------------------------------
ax = axes[1, 2]
panel_label(ax, "F")
ax.axis("off")
ax.set_title("Summary Comparison Table", pad=12)


# Build table data
def verdict(mname):
    verdicts = {
        "Opus v1": "Most comprehensive",
        "Sonnet v2": "Best balance",
        "Haiku v4": "Clean, efficient",
        "Haiku v5": "Richest metadata",
    }
    return verdicts.get(mname, "—")


col_labels = ["Model", "Claims", "Evidence", "Citations", "Schema", "Verdict"]
col_widths = [0.19, 0.10, 0.11, 0.11, 0.13, 0.30]

cell_data = []
for mname in MODEL_NAMES:
    d = data[mname]
    schema = "v5" if d["format"] == "v5" else ("v1" if mname == "Opus v1" else "v2/v4")
    cell_data.append(
        [
            mname,
            str(d["n_claims"]),
            str(d["n_evidence"]),
            str(d["n_citations"]),
            schema,
            verdict(mname),
        ]
    )

n_cols = len(col_labels)
n_rows = len(cell_data)
header_y = 0.88
row_height = 0.155
header_bg = "#333333"
row_bgs = PALETTE

x_start = 0.01
xs = [x_start + sum(col_widths[:ci]) for ci in range(n_cols)]

# Header background
ax.add_patch(
    plt.Rectangle(
        (x_start - 0.01, header_y - 0.045),
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
        fontsize=8.5,
        fontweight="bold",
        color="white",
        zorder=3,
    )

# Data rows
for ri, (row, bg) in enumerate(zip(cell_data, row_bgs)):
    ry = header_y - row_height * (ri + 1) - 0.01 * ri
    ax.add_patch(
        plt.Rectangle(
            (x_start - 0.01, ry - 0.045),
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
            fontsize=8.0,
            fontweight=weight,
            color=color,
            zorder=3,
        )

# Caption
cap_y = header_y - row_height * (n_rows + 1) - 0.06
ax.text(
    0.5,
    cap_y,
    "v5 introduces: certainty, model_system, organism, section_source,\n"
    "quantitative_context, evidence_links per claim.\n"
    "All four extractions on same paper: Rai14/Invariant chain (Lobos Patorniti 2023).",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=7.5,
    color="#444444",
    style="italic",
    linespacing=1.5,
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = BASE / "extraction_comparison_v5.png"
fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Text Summary
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("  KG EXTRACTION COMPARISON SUMMARY — v1 through v5")
print("=" * 70)
print()

for mname in MODEL_NAMES:
    d = data[mname]
    fmt = d["format"]
    print(f"  {mname.upper()} ({fmt.upper()} FORMAT) [Rai14/Invariant chain, Lobos Patorniti 2023]")
    print(f"    Claims / Assertions : {d['n_claims']}")
    print(f"    Evidence units      : {d['n_evidence']}")
    print(f"    Citation contexts   : {d['n_citations']}")
    print("    Claim type breakdown:")
    for k, v in sorted(d["assertion_types"].items(), key=lambda x: -x[1]):
        print(f"      {k:30s}: {v}")
    print("    Evidence strengths:")
    for k, v in sorted(d["evidence_strengths"].items(), key=lambda x: -x[1]):
        print(f"      {k:30s}: {v}")
    print()

print("-" * 70)
print("  V5-SPECIFIC FIELD COMPLETENESS")
print("-" * 70)
v5c = data["Haiku v5"]["claims"]
n5 = len(v5c)
fields_display = [
    ("model_system", lambda c: bool(c.get("model_system"))),
    ("organism", lambda c: bool(c.get("organism"))),
    ("certainty", lambda c: bool(c.get("certainty"))),
    ("section_source", lambda c: bool(c.get("section_source"))),
    (
        "quantitative_context",
        lambda c: (
            bool(c.get("quantitative_context"))
            and any(v is not None for v in c["quantitative_context"].values())
        ),
    ),
    ("conditions", lambda c: bool(c.get("conditions"))),
    ("evidence_links", lambda c: bool(c.get("evidence_links"))),
]
for fname, fn in fields_display:
    n_filled = sum(1 for c in v5c if fn(c))
    print(f"    {fname:30s}: {n_filled}/{n5} ({pct(n_filled, n5):.0f}%)")

print()
print("-" * 70)
print("  V5 SECTION SOURCE BREAKDOWN")
print("-" * 70)
for src, cnt in sorted(Counter(c.get("section_source") for c in v5c).items(), key=lambda x: -x[1]):
    print(f"    {str(src):30s}: {cnt} claims ({pct(cnt, n5):.0f}%)")

print()
print("-" * 70)
print("  V5 CERTAINTY BREAKDOWN")
print("-" * 70)
for cert, cnt in sorted(Counter(c.get("certainty") for c in v5c).items(), key=lambda x: -x[1]):
    print(f"    {str(cert):25s}: {cnt} claims ({pct(cnt, n5):.0f}%)")

print()
print("-" * 70)
print("  KEY OBSERVATIONS")
print("-" * 70)
print("  1. All four extractions on the same paper (Rai14/Invariant chain, Lobos Patorniti 2023).")
print("     Opus v1 produces the most assertions (37) but lacks citation contexts")
print("     and structured metadata fields.")
print("  2. Sonnet v2 / Haiku v4 introduced citation contexts and scope discipline.")
print("  3. Haiku v5 (42 claims) adds per-claim: certainty, model_system, organism,")
print("     section_source, quantitative_context, and evidence_links — the richest")
print("     metadata schema to date.")
print(
    f"  4. V5 field completeness: model_system={pct(sum(1 for c in v5c if c.get('model_system')), n5):.0f}%,"
)
print(f"     organism={pct(sum(1 for c in v5c if c.get('organism')), n5):.0f}%,")
print(f"     certainty={pct(sum(1 for c in v5c if c.get('certainty')), n5):.0f}%,")
print(
    f"     quantitative_context={pct(sum(1 for c in v5c if c.get('quantitative_context') and any(v is not None for v in c['quantitative_context'].values())), n5):.0f}%."
)
print("  5. V5 moves evidence_strength to the claim level (not evidence unit level),")
print("     enabling per-claim confidence tracking.")
print("=" * 70)
