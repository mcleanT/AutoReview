"""Analysis 10: Depth level comparison.

Compares AutoReview output quality, information content, and cost
across low/medium/deep depth levels.

Usage:
    python -m paper.analysis.depth_comparison --results-dir paper/results/depth_comparison --output-dir paper/output/depth_comparison
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import structlog
from scipy import stats as sp_stats

from autoreview.evaluation.models import Claim, InformationMetrics

logger = structlog.get_logger()

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

DEPTHS = ["low", "medium", "deep"]
DEPTH_PAIRS = [("low", "medium"), ("medium", "deep")]


def load_depth_runs(results_dir: Path) -> pd.DataFrame:
    """Load evaluation results for all depth runs into a DataFrame.

    Expects results_dir/depth_runs.json with run configs,
    and each run's evaluation_path pointing to an evaluation JSON.
    """
    config_path = results_dir / "depth_runs.json"
    with open(config_path) as f:
        config = json.load(f)

    rows: list[dict[str, Any]] = []
    for run in config["runs"]:
        eval_path = Path(run["evaluation_path"])
        if not eval_path.exists():
            logger.warning("depth_comparison.missing_eval", path=str(eval_path))
            continue

        with open(eval_path) as f:
            eval_data = json.load(f)

        row = {
            "topic": run["topic"],
            "domain": run["domain"],
            "depth": run["depth"],
            "tier": run.get("tier"),
            "generated_path": run["generated_path"],
            "reference_path": run["reference_path"],
            "overall_score": eval_data["overall_score"],
            "citation_recall": eval_data["citation_score"]["recall"],
            "citation_precision": eval_data["citation_score"]["precision"],
            "citation_f1": eval_data["citation_score"]["f1"],
            "synthesis_score": eval_data["synthesis_score"]["generated_score"],
            "topic_coverage": eval_data["topic_coverage"]["generated_coverage"],
            "writing_quality": eval_data["writing_quality"]["generated_score"],
        }
        if eval_data.get("structural_metrics"):
            sm = eval_data["structural_metrics"]
            row["word_count"] = sm["word_count"]
            row["section_count"] = sm["section_count"]
            row["citation_count"] = sm["citation_count"]
            row["citations_per_1000_words"] = sm["citations_per_1000_words"]
            row["flesch_kincaid_grade"] = sm["flesch_kincaid_grade"]

        rows.append(row)

    return pd.DataFrame(rows)


def split_sections_from_markdown(text: str) -> list[dict[str, str]]:
    """Split markdown into sections by headings. Returns [{"id": ..., "text": ...}]."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [{"id": "full_text", "text": text.strip()}]

    sections: list[dict[str, str]] = []
    for i, match in enumerate(matches):
        title = match.group(2).strip()
        section_id = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append({"id": section_id, "text": body})

    return sections if sections else [{"id": "full_text", "text": text.strip()}]


def compute_information_metrics(
    claims_by_depth: dict[str, list[Claim]],
    novel_claims: dict[str, list[Claim]],
    concepts_by_depth: dict[str, list[str]],
    word_counts: dict[str, int],
    citation_counts: dict[str, int],
    reference_claim_coverage: dict[str, float] | None = None,
) -> InformationMetrics:
    """Compute all derived information metrics across depths."""
    claims_per_depth = {d: len(claims_by_depth[d]) for d in DEPTHS if d in claims_by_depth}

    new_claims_per_increment: dict[str, int] = {}
    new_claims_by_category: dict[str, dict[str, int]] = {}
    claim_novelty_rate: dict[str, float] = {}

    for d_from, d_to in DEPTH_PAIRS:
        key = f"{d_from}_to_{d_to}"
        if key in novel_claims:
            new = novel_claims[key]
            new_claims_per_increment[key] = len(new)
            new_claims_by_category[key] = dict(Counter(c.category for c in new))
            total_at_target = claims_per_depth.get(d_to, 0)
            claim_novelty_rate[key] = len(new) / total_at_target if total_at_target > 0 else 0.0
        else:
            new_claims_per_increment[key] = 0
            new_claims_by_category[key] = {}
            claim_novelty_rate[key] = 0.0

    concepts_per_depth = {d: len(concepts_by_depth[d]) for d in DEPTHS if d in concepts_by_depth}
    concept_growth: dict[str, int] = {}
    concept_overlap: dict[str, float] = {}
    for d_from, d_to in DEPTH_PAIRS:
        key = f"{d_from}_to_{d_to}"
        set_from = set(concepts_by_depth.get(d_from, []))
        set_to = set(concepts_by_depth.get(d_to, []))
        concept_growth[key] = len(set_to - set_from)
        min_size = min(len(set_from), len(set_to))
        concept_overlap[key] = len(set_from & set_to) / min_size if min_size > 0 else 0.0

    claims_per_1k = {
        d: (claims_per_depth[d] / word_counts[d] * 1000) if word_counts.get(d, 0) > 0 else 0.0
        for d in DEPTHS
        if d in claims_per_depth
    }
    concepts_per_1k = {
        d: (concepts_per_depth[d] / word_counts[d] * 1000) if word_counts.get(d, 0) > 0 else 0.0
        for d in DEPTHS
        if d in concepts_per_depth
    }
    citations_per_claim = {
        d: (citation_counts[d] / claims_per_depth[d]) if claims_per_depth.get(d, 0) > 0 else 0.0
        for d in DEPTHS
        if d in claims_per_depth
    }

    return InformationMetrics(
        claims_per_depth=claims_per_depth,
        new_claims_per_increment=new_claims_per_increment,
        new_claims_by_category=new_claims_by_category,
        claim_novelty_rate=claim_novelty_rate,
        concepts_per_depth=concepts_per_depth,
        concept_growth=concept_growth,
        claims_per_1k_words=claims_per_1k,
        concepts_per_1k_words=concepts_per_1k,
        citations_per_claim=citations_per_claim,
        concept_overlap=concept_overlap,
        reference_claim_coverage=reference_claim_coverage,
    )


def compute_pairwise_tests(
    df: pd.DataFrame,
    metric_columns: list[str],
    min_pairs: int = 5,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Wilcoxon signed-rank tests for low->medium and medium->deep.

    Returns nested dict: metric -> pair_key -> {statistic, p_value, n_pairs, p_adjusted}.
    FDR correction (Benjamini-Hochberg) applied across all tests.
    """
    results: dict[str, dict[str, dict[str, Any]]] = {}
    all_p_values: list[tuple[str, str, float]] = []

    for metric in metric_columns:
        results[metric] = {}
        for d_from, d_to in DEPTH_PAIRS:
            key = f"{d_from}_to_{d_to}"
            from_scores = df[df["depth"] == d_from].set_index("topic")[metric]
            to_scores = df[df["depth"] == d_to].set_index("topic")[metric]
            common = from_scores.index.intersection(to_scores.index)

            n_pairs = len(common)
            if n_pairs < min_pairs:
                results[metric][key] = {
                    "statistic": float("nan"),
                    "p_value": float("nan"),
                    "n_pairs": n_pairs,
                    "p_adjusted": float("nan"),
                    "mean_diff": float("nan"),
                }
                continue

            a = from_scores.loc[common].values
            b = to_scores.loc[common].values
            diff = b - a

            if np.all(diff == 0):
                stat, p_val = 0.0, 1.0
            else:
                w_result = sp_stats.wilcoxon(diff, alternative="two-sided")
                stat = float(w_result[0])  # type: ignore[arg-type]  # statistic
                p_val = float(w_result[1])  # type: ignore[arg-type]  # pvalue

            results[metric][key] = {
                "statistic": stat,
                "p_value": p_val,
                "n_pairs": n_pairs,
                "mean_diff": float(np.mean(diff)),
            }
            all_p_values.append((metric, key, p_val))

    # FDR correction (Benjamini-Hochberg) with cumulative minimum pass
    if all_p_values:
        sorted_pvals = sorted(all_p_values, key=lambda x: x[2])
        m = len(sorted_pvals)
        raw_adj = [min(p * m / (rank + 1), 1.0) for rank, (_, _, p) in enumerate(sorted_pvals)]
        # Cumulative minimum from the right (enforces monotonicity)
        for i in range(len(raw_adj) - 2, -1, -1):
            raw_adj[i] = min(raw_adj[i], raw_adj[i + 1])
        for i, (metric, key, _) in enumerate(sorted_pvals):
            results[metric][key]["p_adjusted"] = raw_adj[i]

    return results


@dataclass
class DepthAnalysisOutput:
    """Container for all analysis results."""

    summary_stats: dict[str, dict[str, dict[str, float]]]
    pairwise_tests: dict[str, dict[str, dict[str, Any]]]
    information_metrics_per_topic: dict[str, InformationMetrics]
    cost_data: dict[str, dict[str, float]] | None = None


def write_analysis_json(output: DepthAnalysisOutput, path: Path) -> None:
    """Write all computed metrics and test results to JSON."""
    data: dict[str, Any] = {
        "summary_stats": output.summary_stats,
        "pairwise_tests": output.pairwise_tests,
        "information_metrics_per_topic": {
            topic: metrics.model_dump()
            for topic, metrics in output.information_metrics_per_topic.items()
        },
    }
    if output.cost_data:
        data["cost_data"] = output.cost_data

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("depth_comparison.json_written", path=str(path))


def generate_report(output: DepthAnalysisOutput, path: Path) -> None:
    """Generate markdown summary report."""
    lines: list[str] = []
    lines.append("# Analysis 10: Depth Level Comparison\n")
    lines.append("## Summary Statistics\n")

    lines.append("| Depth | Metric | Mean | Std | N |")
    lines.append("|-------|--------|------|-----|---|")
    for depth in DEPTHS:
        if depth in output.summary_stats:
            for metric, stats in output.summary_stats[depth].items():
                lines.append(
                    f"| {depth.capitalize()} | {metric} | "
                    f"{stats.get('mean', 0):.3f} | {stats.get('std', 0):.3f} | "
                    f"{stats.get('n', 0)} |"
                )
    lines.append("")

    lines.append("## Pairwise Statistical Tests (Wilcoxon Signed-Rank)\n")
    lines.append("| Metric | Comparison | Mean Diff | p-value | p-adjusted | N pairs |")
    lines.append("|--------|------------|-----------|---------|------------|---------|")
    for metric, pairs in output.pairwise_tests.items():
        for pair_key, result in pairs.items():
            lines.append(
                f"| {metric} | {pair_key} | "
                f"{result.get('mean_diff', 0):.3f} | "
                f"{result.get('p_value', float('nan')):.4f} | "
                f"{result.get('p_adjusted', float('nan')):.4f} | "
                f"{result.get('n_pairs', 0)} |"
            )
    lines.append("")

    lines.append("## Information Metrics\n")
    if output.information_metrics_per_topic:
        lines.append("### Claims per Depth (averaged across topics)\n")
        avg_claims: dict[str, float] = {}
        for depth in DEPTHS:
            values = [
                m.claims_per_depth.get(depth, 0)
                for m in output.information_metrics_per_topic.values()
            ]
            avg_claims[depth] = float(np.mean(values)) if values else 0.0
        for depth in DEPTHS:
            lines.append(f"- **{depth.capitalize()}**: {avg_claims[depth]:.1f} claims")
        lines.append("")

        lines.append("### Novelty per Increment (averaged across topics)\n")
        for d_from, d_to in DEPTH_PAIRS:
            key = f"{d_from}_to_{d_to}"
            values = [
                m.new_claims_per_increment.get(key, 0)
                for m in output.information_metrics_per_topic.values()
            ]
            avg = float(np.mean(values)) if values else 0.0
            lines.append(f"- **{d_from} -> {d_to}**: {avg:.1f} new claims")
        lines.append("")

        lines.append("### Information Density (averaged across topics)\n")
        for depth in DEPTHS:
            values = [
                m.claims_per_1k_words.get(depth, 0)
                for m in output.information_metrics_per_topic.values()
            ]
            avg = float(np.mean(values)) if values else 0.0
            lines.append(f"- **{depth.capitalize()}**: {avg:.2f} claims / 1K words")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    logger.info("depth_comparison.report_written", path=str(path))


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt  # noqa: E402

# Colorblind-safe palette
CB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]
DEPTH_COLORS = {"low": CB_PALETTE[0], "medium": CB_PALETTE[5], "deep": CB_PALETTE[1]}
CATEGORY_ORDER = ["empirical", "methodological", "contextual", "synthesis", "limitation"]
CATEGORY_COLORS = {cat: CB_PALETTE[i] for i, cat in enumerate(CATEGORY_ORDER)}

_FONT_CONFIG = {
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
}


def _apply_style() -> None:
    plt.rcParams.update(_FONT_CONFIG)


def plot_quality_by_depth(df: pd.DataFrame, out: Path) -> None:
    """Line chart: quality score x depth level, one line per metric."""
    _apply_style()
    metrics = {
        "synthesis_score": "Synthesis",
        "topic_coverage": "Topic Coverage",
        "writing_quality": "Writing Quality",
        "citation_f1": "Citation F1",
    }
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for i, (col, label) in enumerate(metrics.items()):
        means = df.groupby("depth")[col].mean().reindex(DEPTHS)
        sems = df.groupby("depth")[col].sem().reindex(DEPTHS)
        ax.errorbar(
            DEPTHS,
            means,
            yerr=sems,
            marker="o",
            label=label,
            color=CB_PALETTE[i],
            linewidth=2,
            capsize=4,
        )

    ax.set_xlabel("Depth Level")
    ax.set_ylabel("Score")
    ax.set_title("Quality Metrics by Depth Level")
    ax.legend(loc="best")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_claims(
    claims_by_depth_and_category: dict[str, dict[str, int]],
    out: Path,
) -> None:
    """Stacked bar: claim counts by category at each depth."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    x = np.arange(len(DEPTHS))
    width = 0.6
    bottom = np.zeros(len(DEPTHS))

    for cat in CATEGORY_ORDER:
        values = [claims_by_depth_and_category.get(d, {}).get(cat, 0) for d in DEPTHS]
        ax.bar(x, values, width, bottom=bottom, label=cat.capitalize(), color=CATEGORY_COLORS[cat])
        bottom += np.array(values, dtype=float)

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in DEPTHS])
    ax.set_xlabel("Depth Level")
    ax.set_ylabel("Number of Claims")
    ax.set_title("Cumulative Claims by Category and Depth")
    ax.legend(loc="upper left")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_information_density(density_df: pd.DataFrame, out: Path) -> None:
    """Scatter: word count (x) vs claims/1000 words (y), color=domain, shape=depth."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    domains = density_df["domain"].unique()
    domain_colors = {d: CB_PALETTE[i % len(CB_PALETTE)] for i, d in enumerate(domains)}
    depth_markers = {"low": "o", "medium": "s", "deep": "D"}

    for _, row in density_df.iterrows():
        ax.scatter(
            row["word_count"],
            row["claims_per_1k_words"],
            c=domain_colors[row["domain"]],
            marker=depth_markers.get(str(row["depth"]), "o"),
            s=80,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )

    for domain, color in domain_colors.items():
        ax.scatter([], [], c=color, marker="o", label=domain.capitalize(), s=80)
    for depth, marker in depth_markers.items():
        ax.scatter([], [], c="gray", marker=marker, label=f"{depth.capitalize()} depth", s=80)

    ax.set_xlabel("Word Count")
    ax.set_ylabel("Claims per 1,000 Words")
    ax.set_title("Information Density vs. Review Length")
    ax.legend(loc="best", ncol=2)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cost_efficiency(cost_df: pd.DataFrame, out: Path) -> None:
    """Dual-axis chart: quality score and cost by depth."""
    _apply_style()
    fig, ax1 = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax2 = ax1.twinx()

    x = np.arange(len(cost_df))
    width = 0.35

    ax1.bar(
        x - width / 2,
        cost_df["mean_score"],
        width,
        label="Quality Score",
        color=CB_PALETTE[0],
        alpha=0.8,
    )
    ax2.bar(
        x + width / 2,
        cost_df["mean_cost"],
        width,
        label="Cost ($)",
        color=CB_PALETTE[1],
        alpha=0.8,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels([d.capitalize() for d in cost_df["depth"]])
    ax1.set_xlabel("Depth Level")
    ax1.set_ylabel("Quality Score", color=CB_PALETTE[0])
    ax2.set_ylabel("Cost ($)", color=CB_PALETTE[1])
    ax1.set_title("Quality-Cost Tradeoff by Depth")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_domain_depth_heatmap(
    df: pd.DataFrame,
    metric: str,
    out: Path,
) -> None:
    """Heatmap: domain (rows) x depth (cols), cell = mean metric value."""
    _apply_style()
    pivot = df.pivot_table(values=metric, index="domain", columns="depth", aggfunc="mean")
    pivot = pivot.reindex(columns=DEPTHS)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")

    ax.set_xticks(range(len(DEPTHS)))
    ax.set_xticklabels([d.capitalize() for d in DEPTHS])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([d.capitalize() for d in pivot.index])
    ax.set_xlabel("Depth Level")
    ax.set_ylabel("Domain")
    ax.set_title(f"{metric.replace('_', ' ').title()} by Domain and Depth")

    for i in range(len(pivot.index)):
        for j in range(len(DEPTHS)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white" if val < pivot.values.mean() else "black",
                    fontsize=10,
                )

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


async def run_extraction_for_topic(
    topic: str,
    depth_reviews: dict[str, str],
    extractor: Any,
    embedder: Any,
    word_counts: dict[str, int],
    citation_counts: dict[str, int],
    reference_text: str | None = None,
) -> InformationMetrics:
    """Run claim/concept extraction for a single topic across all depths."""
    from autoreview.evaluation.claim_dedup import deduplicate_claims

    claims_by_depth: dict[str, list[Claim]] = {}
    concepts_by_depth: dict[str, list[str]] = {}

    for depth in DEPTHS:
        if depth not in depth_reviews:
            continue
        claims_by_depth[depth] = await extractor.extract_claims(depth_reviews[depth])
        concepts_by_depth[depth] = await extractor.extract_concepts(depth_reviews[depth])
        logger.info(
            "depth_comparison.extracted",
            topic=topic,
            depth=depth,
            n_claims=len(claims_by_depth[depth]),
            n_concepts=len(concepts_by_depth[depth]),
        )

    novel_claims: dict[str, list[Claim]] = {}
    if "low" in claims_by_depth and "medium" in claims_by_depth:
        novel, _ = await deduplicate_claims(
            claims_by_depth["medium"], claims_by_depth["low"], embedder
        )
        novel_claims["low_to_medium"] = novel

    if "medium" in claims_by_depth and "deep" in claims_by_depth:
        baseline = claims_by_depth.get("low", []) + claims_by_depth.get("medium", [])
        novel, _ = await deduplicate_claims(claims_by_depth["deep"], baseline, embedder)
        novel_claims["medium_to_deep"] = novel

    ref_coverage: dict[str, float] | None = None
    if reference_text:
        ref_claims = await extractor.extract_claims(reference_text)
        if ref_claims:
            ref_coverage = {}
            for depth in DEPTHS:
                if depth in claims_by_depth:
                    _, shared = await deduplicate_claims(
                        ref_claims, claims_by_depth[depth], embedder
                    )
                    ref_coverage[depth] = len(shared) / len(ref_claims)

    return compute_information_metrics(
        claims_by_depth=claims_by_depth,
        novel_claims=novel_claims,
        concepts_by_depth=concepts_by_depth,
        word_counts=word_counts,
        citation_counts=citation_counts,
        reference_claim_coverage=ref_coverage,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analysis 10: Depth level comparison")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing depth_runs.json and evaluation results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/output/depth_comparison"),
        help="Output directory for figures, JSON, and report",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip claim/concept extraction (use cached results from depth_claims.json)",
    )
    return parser.parse_args(argv)


async def main(results_dir: Path, output_dir: Path, skip_extraction: bool = False) -> None:
    """Full depth comparison analysis pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("depth_comparison.loading_runs", results_dir=str(results_dir))
    df = load_depth_runs(results_dir)
    logger.info("depth_comparison.loaded", n_runs=len(df))

    metric_cols = [
        "overall_score",
        "citation_recall",
        "citation_f1",
        "synthesis_score",
        "topic_coverage",
        "writing_quality",
    ]
    summary_stats: dict[str, dict[str, dict[str, float]]] = {}
    for depth in DEPTHS:
        depth_df = df[df["depth"] == depth]
        summary_stats[depth] = {}
        for col in metric_cols:
            if col in depth_df.columns:
                summary_stats[depth][col] = {
                    "mean": float(depth_df[col].mean()),
                    "std": float(depth_df[col].std()),
                    "n": int(len(depth_df)),
                }

    pairwise = compute_pairwise_tests(df, metric_cols)

    info_metrics: dict[str, InformationMetrics] = {}
    claims_cache = results_dir / "depth_claims.json"

    if skip_extraction and claims_cache.exists():
        logger.info("depth_comparison.loading_cached_claims")
        with open(claims_cache) as f:
            cached = json.load(f)
        for topic_key, metrics_data in cached.items():
            info_metrics[topic_key] = InformationMetrics.model_validate(metrics_data)
    elif not skip_extraction:
        logger.warning(
            "depth_comparison.extraction_requires_llm",
            msg="Live extraction requires LLM client and embedder. Use --skip-extraction with cached results.",
        )
    else:
        logger.info("depth_comparison.no_extraction_no_cache")

    analysis_output = DepthAnalysisOutput(
        summary_stats=summary_stats,
        pairwise_tests=pairwise,
        information_metrics_per_topic=info_metrics,
    )

    write_analysis_json(analysis_output, output_dir / "depth_analysis.json")
    generate_report(analysis_output, output_dir / "report.md")

    plot_quality_by_depth(df, output_dir / "quality_by_depth.pdf")
    plot_domain_depth_heatmap(df, "overall_score", output_dir / "domain_depth_heatmap.pdf")

    if info_metrics and not claims_cache.exists():
        cache_data = {t: m.model_dump() for t, m in info_metrics.items()}
        with open(claims_cache, "w") as f:
            json.dump(cache_data, f, indent=2, default=str)
        logger.info("depth_comparison.claims_cached", path=str(claims_cache))

    logger.info("depth_comparison.complete", output_dir=str(output_dir))


if __name__ == "__main__":
    import asyncio

    _args = parse_args()
    asyncio.run(main(_args.results_dir, _args.output_dir, _args.skip_extraction))
