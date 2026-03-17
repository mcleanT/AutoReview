"""Analysis 1: Main System Comparison.

Compares AutoReview vs ARISE baseline on overall quality metrics
across all benchmark topics.

Usage:
    python -m paper.analysis.main_comparison --results-dir paper/results --output-dir paper/output/main_comparison
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import structlog
from scipy import stats as sp_stats

from paper.analysis.common import CB_PALETTE, apply_style, fdr_correct, save_analysis_json

logger = structlog.get_logger()

METRICS = [
    "overall_score",
    "synthesis_score",
    "topic_coverage",
    "writing_quality",
    "citation_f1",
]

METRIC_LABELS = {
    "overall_score": "Overall Score",
    "synthesis_score": "Synthesis",
    "topic_coverage": "Topic Coverage",
    "writing_quality": "Writing Quality",
    "citation_f1": "Citation F1",
}


def compute_system_summary(df: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    """Group by system and compute mean/std per metric.

    Returns: system -> metric -> {mean, std, n}
    """
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for system, group in df.groupby("system"):
        summary[str(system)] = {}
        for metric in METRICS:
            if metric not in group.columns:
                continue
            vals = group[metric].dropna()
            summary[str(system)][metric] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "n": int(len(vals)),
            }
    return summary


def compute_statistical_tests(df: pd.DataFrame) -> dict[str, Any]:
    """Friedman test across systems + post-hoc Wilcoxon with FDR correction.

    Returns dict with per-metric Friedman result and pairwise Wilcoxon tests.
    """
    results: dict[str, Any] = {}
    systems = sorted(df["system"].unique())

    for metric in METRICS:
        if metric not in df.columns:
            continue

        metric_result: dict[str, Any] = {"systems": systems}

        # Friedman test (requires >= 3 systems; fall back to Wilcoxon for 2)
        groups = [df[df["system"] == s][metric].dropna().values for s in systems]
        # Use only common topics for paired tests
        topic_col = "topic_id" if "topic_id" in df.columns else "topic"
        pivot = df.pivot_table(values=metric, index=topic_col, columns="system", aggfunc="mean")
        pivot_clean = pivot.dropna()

        if len(systems) >= 3 and len(pivot_clean) >= 3:
            try:
                stat, p = sp_stats.friedmanchisquare(
                    *[pivot_clean[s].values for s in systems if s in pivot_clean.columns]
                )
                metric_result["friedman"] = {"statistic": float(stat), "p_value": float(p)}
            except Exception:
                metric_result["friedman"] = {"statistic": float("nan"), "p_value": float("nan")}
        elif len(systems) == 2 and len(pivot_clean) >= 3:
            s1, s2 = systems[0], systems[1]
            if s1 in pivot_clean.columns and s2 in pivot_clean.columns:
                diff = pivot_clean[s1].values - pivot_clean[s2].values
                if np.all(diff == 0):
                    stat, p = 0.0, 1.0
                else:
                    w_result = sp_stats.wilcoxon(diff, alternative="two-sided")
                    stat, p = float(w_result.statistic), float(w_result.pvalue)
                metric_result["friedman"] = {"statistic": float(stat), "p_value": float(p)}
            else:
                metric_result["friedman"] = {"statistic": float("nan"), "p_value": float("nan")}
        else:
            metric_result["friedman"] = {"statistic": float("nan"), "p_value": float("nan")}

        # Post-hoc pairwise Wilcoxon (all pairs)
        pairwise: dict[str, dict[str, float]] = {}
        p_values_raw: list[float] = []
        pair_keys: list[str] = []

        for i, s1 in enumerate(systems):
            for s2 in systems[i + 1 :]:
                pair_key = f"{s1}_vs_{s2}"
                if (
                    s1 in pivot_clean.columns
                    and s2 in pivot_clean.columns
                    and len(pivot_clean) >= 3
                ):
                    diff = pivot_clean[s1].values - pivot_clean[s2].values
                    if np.all(diff == 0):
                        stat, p_val = 0.0, 1.0
                    else:
                        w_result = sp_stats.wilcoxon(diff, alternative="two-sided")
                        stat, p_val = float(w_result.statistic), float(w_result.pvalue)
                    mean_diff = float(np.mean(diff))
                else:
                    stat, p_val, mean_diff = float("nan"), float("nan"), float("nan")

                pairwise[pair_key] = {
                    "statistic": stat,
                    "p_value": p_val,
                    "mean_diff": mean_diff,
                    "n_pairs": int(len(pivot_clean)),
                }
                p_values_raw.append(p_val)
                pair_keys.append(pair_key)

        # FDR correction over all pairwise p-values for this metric
        valid_p = [p for p in p_values_raw if not np.isnan(p)]
        if valid_p:
            adj = fdr_correct(p_values_raw)
            for pair_key, p_adj in zip(pair_keys, adj):
                pairwise[pair_key]["p_adjusted"] = p_adj
        else:
            for pair_key in pair_keys:
                pairwise[pair_key]["p_adjusted"] = float("nan")

        metric_result["pairwise"] = pairwise
        results[metric] = metric_result

    return results


def plot_system_comparison_bar(df: pd.DataFrame, out: Path) -> None:
    """Grouped bar chart: one bar per system per metric."""
    apply_style()
    systems = sorted(df["system"].unique())
    n_systems = len(systems)
    n_metrics = len(METRICS)

    x = np.arange(n_metrics)
    width = 0.8 / n_systems

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    for i, system in enumerate(systems):
        sdf = df[df["system"] == system]
        means = [sdf[m].mean() if m in sdf.columns else 0.0 for m in METRICS]
        sems = [sdf[m].sem() if m in sdf.columns else 0.0 for m in METRICS]
        offsets = x + (i - n_systems / 2 + 0.5) * width
        ax.bar(
            offsets,
            means,
            width,
            yerr=sems,
            label=system.capitalize(),
            color=CB_PALETTE[i % len(CB_PALETTE)],
            alpha=0.85,
            capsize=4,
            error_kw={"elinewidth": 1.5},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in METRICS], rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("System Comparison: AutoReview vs Baseline")
    ax.legend(loc="upper right")
    ax.set_ylim(0, None)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("main_comparison.bar_chart_saved", path=str(out))


def plot_radar_chart(summary: dict[str, dict[str, dict[str, float]]], out: Path) -> None:
    """Radar/spider chart of dimension scores per system."""
    apply_style()
    systems = list(summary.keys())
    # Use metrics that are present in all systems
    available_metrics = [m for m in METRICS if all(m in summary[s] for s in systems)]
    if not available_metrics:
        logger.warning("main_comparison.radar_no_metrics")
        return

    n = len(available_metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True}, constrained_layout=True)

    for i, system in enumerate(systems):
        values = [summary[system][m]["mean"] for m in available_metrics]
        # Normalize: synthesis_score and writing_quality are 1-5, others 0-1
        norm_values: list[float] = []
        for m, v in zip(available_metrics, values):
            if m in ("synthesis_score", "writing_quality"):
                norm_values.append(v / 5.0)
            else:
                norm_values.append(float(v))
        norm_values += norm_values[:1]

        ax.plot(
            angles,
            norm_values,
            "o-",
            linewidth=2,
            label=system.capitalize(),
            color=CB_PALETTE[i % len(CB_PALETTE)],
        )
        ax.fill(angles, norm_values, alpha=0.15, color=CB_PALETTE[i % len(CB_PALETTE)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in available_metrics], size=10)
    ax.set_ylim(0, 1)
    ax.set_title("System Comparison — Radar Chart", size=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("main_comparison.radar_chart_saved", path=str(out))


def generate_report(
    summary: dict[str, dict[str, dict[str, float]]],
    tests: dict[str, Any],
    out: Path,
) -> None:
    """Generate markdown summary report."""
    lines: list[str] = ["# Analysis 1: Main System Comparison\n"]

    lines.append("## Summary Statistics\n")
    lines.append("| System | Metric | Mean | Std | N |")
    lines.append("|--------|--------|------|-----|---|")
    for system, metrics in summary.items():
        for metric, stats in metrics.items():
            lines.append(
                f"| {system} | {METRIC_LABELS.get(metric, metric)} | "
                f"{stats.get('mean', 0):.3f} | {stats.get('std', 0):.3f} | "
                f"{stats.get('n', 0)} |"
            )
    lines.append("")

    lines.append("## Statistical Tests\n")
    lines.append("### Friedman Test\n")
    lines.append("| Metric | Statistic | p-value |")
    lines.append("|--------|-----------|---------|")
    for metric, result in tests.items():
        fr = result.get("friedman", {})
        lines.append(
            f"| {METRIC_LABELS.get(metric, metric)} | "
            f"{fr.get('statistic', float('nan')):.3f} | "
            f"{fr.get('p_value', float('nan')):.4f} |"
        )
    lines.append("")

    lines.append("### Post-hoc Pairwise Wilcoxon (FDR-corrected)\n")
    lines.append("| Metric | Pair | Mean Diff | p-value | p-adjusted | N |")
    lines.append("|--------|------|-----------|---------|------------|---|")
    for metric, result in tests.items():
        for pair_key, pw in result.get("pairwise", {}).items():
            lines.append(
                f"| {METRIC_LABELS.get(metric, metric)} | {pair_key} | "
                f"{pw.get('mean_diff', float('nan')):.3f} | "
                f"{pw.get('p_value', float('nan')):.4f} | "
                f"{pw.get('p_adjusted', float('nan')):.4f} | "
                f"{pw.get('n_pairs', 0)} |"
            )
    lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    logger.info("main_comparison.report_written", path=str(out))


async def main(results_dir: Path, output_dir: Path, df: pd.DataFrame) -> None:
    """Orchestrate Analysis 1: main system comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("main_comparison.start", n_rows=len(df))

    summary = compute_system_summary(df)
    tests = compute_statistical_tests(df)

    save_analysis_json(
        {"summary": summary, "tests": tests},
        output_dir / "analysis.json",
    )

    plot_system_comparison_bar(df, output_dir / "system_comparison_bar.pdf")
    plot_radar_chart(summary, output_dir / "radar_chart.pdf")
    generate_report(summary, tests, output_dir / "report.md")

    logger.info("main_comparison.complete", output_dir=str(output_dir))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analysis 1: Main System Comparison")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/output/main_comparison"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    from paper.analysis.common import load_all_evaluations
    from paper.models import TopicsConfig

    _args = parse_args()
    import yaml

    with open("paper/topics.yaml") as f:
        _topics = TopicsConfig.model_validate(yaml.safe_load(f))
    _df = load_all_evaluations(_args.results_dir, _topics)
    asyncio.run(main(_args.results_dir, _args.output_dir, _df))
