"""Analysis 6: Citation quality.

Computes citation recall/precision/F1 and hallucination rate broken down by
domain, tier, or model. Generates grouped bar charts and a hallucination
histogram.

Usage:
    python -m paper.analysis.citation_analysis --results-dir paper/results --output-dir paper/output/citation
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import structlog

from paper.analysis.common import CB_PALETTE, apply_style, save_analysis_json

logger = structlog.get_logger()


def compute_citation_breakdown(
    df: pd.DataFrame,
    group_by: str = "domain",
) -> dict[str, dict[str, float]]:
    """Compute mean recall/precision/F1/hallucination_rate grouped by a column.

    Args:
        df: DataFrame with citation_recall, citation_precision, citation_f1,
            hallucination_rate columns plus grouping column.
        group_by: Column to group by — "domain", "tier", or "model".

    Returns:
        Dict mapping group value to dict of mean metrics:
        {group: {mean_recall, mean_precision, mean_f1, mean_hallucination_rate}}.
    """
    metrics = {
        "mean_recall": "citation_recall",
        "mean_precision": "citation_precision",
        "mean_f1": "citation_f1",
        "mean_hallucination_rate": "hallucination_rate",
    }

    result: dict[str, dict[str, float]] = {}
    for group_val, group_df in df.groupby(group_by):
        group_stats: dict[str, float] = {}
        for stat_name, col in metrics.items():
            if col in group_df.columns:
                group_stats[stat_name] = float(group_df[col].mean())
            else:
                group_stats[stat_name] = float("nan")
        result[str(group_val)] = group_stats

    return result


def compute_hallucination_rate(df: pd.DataFrame) -> float:
    """Compute overall hallucination rate across all rows.

    Args:
        df: DataFrame with a hallucination_rate column.

    Returns:
        Mean hallucination rate in [0, 1].
    """
    if "hallucination_rate" not in df.columns or df.empty:
        return 0.0
    return float(df["hallucination_rate"].mean())


def plot_citation_bar(
    breakdown: dict[str, dict[str, float]],
    out: Path,
) -> None:
    """Grouped bar chart: groups on x-axis, bars = recall/precision/F1.

    Args:
        breakdown: Output of compute_citation_breakdown.
        out: Output path (PDF/PNG).
    """
    apply_style()

    groups = sorted(breakdown.keys())
    metric_keys = ["mean_recall", "mean_precision", "mean_f1"]
    metric_labels = ["Recall", "Precision", "F1"]

    x = np.arange(len(groups))
    n_metrics = len(metric_keys)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.5), 5), constrained_layout=True)

    for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        values = [breakdown[g].get(key, float("nan")) for g in groups]
        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=label, color=CB_PALETTE[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("_", " ").title() for g in groups], rotation=15, ha="right")
    ax.set_xlabel("Group")
    ax.set_ylabel("Score")
    ax.set_title("Citation Quality by Group")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("citation_analysis.bar_saved", path=str(out))


def plot_hallucination_histogram(df: pd.DataFrame, out: Path) -> None:
    """Histogram of per-run hallucination rates.

    Args:
        df: DataFrame with hallucination_rate column.
        out: Output path (PDF/PNG).
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    rates = df["hallucination_rate"].dropna()
    ax.hist(rates, bins=20, color=CB_PALETTE[1], alpha=0.85, edgecolor="black", linewidth=0.5)

    mean_rate = float(rates.mean()) if len(rates) > 0 else 0.0
    ax.axvline(
        mean_rate, color=CB_PALETTE[0], linewidth=2, linestyle="--", label=f"Mean = {mean_rate:.3f}"
    )

    ax.set_xlabel("Hallucination Rate")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Hallucination Rates")
    ax.legend(loc="upper right")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("citation_analysis.histogram_saved", path=str(out))


async def main(results_dir: Path, output_dir: Path, df: pd.DataFrame) -> None:
    """Run citation quality analysis.

    Args:
        results_dir: Directory with benchmark results (unused if df is provided).
        output_dir: Output directory for figures and JSON.
        df: Pre-loaded unified DataFrame from load_all_evaluations().
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("citation_analysis.start", n_rows=len(df))

    # Compute breakdowns
    breakdown_domain = compute_citation_breakdown(df, group_by="domain")
    breakdown_tier = compute_citation_breakdown(df, group_by="tier")
    breakdown_model = compute_citation_breakdown(df, group_by="model")

    overall_hallucination = compute_hallucination_rate(df)
    logger.info("citation_analysis.overall_hallucination_rate", rate=overall_hallucination)

    # Save JSON
    results: dict[str, Any] = {
        "overall_hallucination_rate": overall_hallucination,
        "breakdown_by_domain": breakdown_domain,
        "breakdown_by_tier": breakdown_tier,
        "breakdown_by_model": breakdown_model,
    }
    save_analysis_json(results, output_dir / "citation_analysis.json")

    # Plots
    plot_citation_bar(breakdown_domain, output_dir / "citation_bar_domain.pdf")
    plot_citation_bar(breakdown_tier, output_dir / "citation_bar_tier.pdf")
    plot_hallucination_histogram(df, output_dir / "hallucination_histogram.pdf")

    logger.info("citation_analysis.complete", output_dir=str(output_dir))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analysis 6: Citation quality")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/output/citation"),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    from paper.analysis.common import load_all_evaluations
    from paper.models import TopicsConfig

    _args = parse_args()
    _topics_path = _args.results_dir / "topics.yaml"
    _topics = TopicsConfig.from_yaml(_topics_path)
    _df = load_all_evaluations(_args.results_dir, _topics)
    main(_args.results_dir, _args.output_dir, _df)
