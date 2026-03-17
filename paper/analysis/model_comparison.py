"""Analysis 7: Model tier comparison.

Compares AutoReview output quality across model tiers at medium depth.
Runs Friedman test + post-hoc Wilcoxon with FDR correction.

Usage:
    python -m paper.analysis.model_comparison --results-dir paper/results --output-dir paper/output/model_comparison
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
from scipy import stats as sp_stats

from paper.analysis.common import CB_PALETTE, apply_style, fdr_correct, save_analysis_json

logger = structlog.get_logger()

METRIC_COLUMNS = [
    "overall_score",
    "citation_f1",
    "synthesis_score",
    "topic_coverage",
    "writing_quality",
]

METRIC_LABELS = {
    "overall_score": "Overall",
    "citation_f1": "Citation F1",
    "synthesis_score": "Synthesis",
    "topic_coverage": "Topic Coverage",
    "writing_quality": "Writing Quality",
}


def compute_model_summary(
    df: pd.DataFrame,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute mean and std per metric for each model (medium depth, AutoReview only).

    Args:
        df: Unified benchmark DataFrame. Will be filtered to medium depth and
            autoreview system internally.

    Returns:
        Nested dict: {model: {metric: {mean, std, n}}}.
    """
    filtered = df[(df["depth"] == "medium") & (df["system"] == "autoreview")]

    result: dict[str, dict[str, dict[str, float]]] = {}
    for model, model_df in filtered.groupby("model"):
        result[str(model)] = {}
        for col in METRIC_COLUMNS:
            if col not in model_df.columns:
                continue
            values = model_df[col].dropna()
            result[str(model)][col] = {
                "mean": float(values.mean()) if len(values) > 0 else float("nan"),
                "std": float(values.std()) if len(values) > 1 else 0.0,
                "n": int(len(values)),
            }

    return result


def compute_model_tests(df: pd.DataFrame) -> dict[str, Any]:
    """Friedman test + post-hoc Wilcoxon pairwise tests with FDR correction.

    Filters to medium depth, AutoReview system. Rows must have the same
    topic_id to be paired in Wilcoxon tests.

    Args:
        df: Unified benchmark DataFrame.

    Returns:
        Dict with keys:
            - "friedman": {metric: {statistic, p_value, n_groups, n_topics}}
            - "wilcoxon": {metric: {model_a|model_b: {statistic, p_value, p_adjusted,
                           mean_diff, n_pairs}}}
    """
    filtered = df[(df["depth"] == "medium") & (df["system"] == "autoreview")]
    models = sorted(filtered["model"].unique())

    friedman_results: dict[str, dict[str, Any]] = {}
    wilcoxon_results: dict[str, dict[str, dict[str, Any]]] = {}

    # Collect all p-values for FDR correction across all Wilcoxon tests
    all_pvals: list[tuple[str, str, float]] = []  # (metric, pair_key, p)

    for metric in METRIC_COLUMNS:
        if metric not in filtered.columns:
            continue

        # Friedman test: need same topics across all models
        pivot = filtered.pivot_table(
            values=metric, index="topic_id", columns="model", aggfunc="mean"
        )
        pivot = pivot.dropna()
        n_topics = len(pivot)
        n_groups = len(models)

        if n_topics >= 2 and n_groups >= 3:
            groups = [pivot[m].values for m in pivot.columns if m in pivot.columns]
            try:
                friedman_stat, friedman_p = sp_stats.friedmanchisquare(*groups)
                friedman_results[metric] = {
                    "statistic": float(friedman_stat),
                    "p_value": float(friedman_p),
                    "n_groups": n_groups,
                    "n_topics": n_topics,
                }
            except Exception:
                friedman_results[metric] = {
                    "statistic": float("nan"),
                    "p_value": float("nan"),
                    "n_groups": n_groups,
                    "n_topics": n_topics,
                }
        else:
            friedman_results[metric] = {
                "statistic": float("nan"),
                "p_value": float("nan"),
                "n_groups": n_groups,
                "n_topics": n_topics,
            }

        # Pairwise Wilcoxon tests
        wilcoxon_results[metric] = {}
        for i, model_a in enumerate(models):
            for model_b in models[i + 1 :]:
                pair_key = f"{model_a}|{model_b}"
                a_scores = filtered[filtered["model"] == model_a].set_index("topic_id")[metric]
                b_scores = filtered[filtered["model"] == model_b].set_index("topic_id")[metric]
                common_topics = a_scores.index.intersection(b_scores.index)
                n_pairs = len(common_topics)

                if n_pairs < 2:
                    wilcoxon_results[metric][pair_key] = {
                        "statistic": float("nan"),
                        "p_value": float("nan"),
                        "p_adjusted": float("nan"),
                        "mean_diff": float("nan"),
                        "n_pairs": n_pairs,
                    }
                    continue

                a_vals = a_scores.loc[common_topics].values
                b_vals = b_scores.loc[common_topics].values
                diff = b_vals - a_vals

                if np.all(diff == 0):
                    stat, p_val = 0.0, 1.0
                else:
                    w_result = sp_stats.wilcoxon(diff, alternative="two-sided")
                    stat = float(w_result[0])
                    p_val = float(w_result[1])

                wilcoxon_results[metric][pair_key] = {
                    "statistic": stat,
                    "p_value": p_val,
                    "p_adjusted": float("nan"),  # filled below
                    "mean_diff": float(np.mean(diff)),
                    "n_pairs": n_pairs,
                }
                all_pvals.append((metric, pair_key, p_val))

    # FDR correction across all pairwise tests
    if all_pvals:
        raw_pvals = [p for _, _, p in all_pvals]
        adjusted = fdr_correct(raw_pvals)
        for (metric, pair_key, _), p_adj in zip(all_pvals, adjusted):
            wilcoxon_results[metric][pair_key]["p_adjusted"] = p_adj

    return {
        "friedman": friedman_results,
        "wilcoxon": wilcoxon_results,
    }


def plot_model_line(
    summary: dict[str, dict[str, dict[str, float]]],
    out: Path,
) -> None:
    """Line chart: model (x-axis) x metric (one line per metric).

    Args:
        summary: Output of compute_model_summary.
        out: Output path (PDF/PNG).
    """
    apply_style()

    models = sorted(summary.keys())
    metrics_to_plot = [m for m in METRIC_COLUMNS if m in METRIC_LABELS]

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.5), 5), constrained_layout=True)

    x = np.arange(len(models))
    for i, metric in enumerate(metrics_to_plot):
        means = []
        stds = []
        for model in models:
            stats = summary.get(model, {}).get(metric, {})
            means.append(stats.get("mean", float("nan")))
            stds.append(stats.get("std", 0.0))

        ax.errorbar(
            x,
            means,
            yerr=stds,
            marker="o",
            label=METRIC_LABELS.get(metric, metric),
            color=CB_PALETTE[i % len(CB_PALETTE)],
            linewidth=2,
            capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("-", "\n") for m in models], rotation=0, ha="center")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Quality Metrics by Model Tier")
    ax.legend(loc="best", ncol=2)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("model_comparison.line_saved", path=str(out))


def plot_cost_frontier(df: pd.DataFrame, out: Path) -> None:
    """Scatter: cost_usd (x) vs overall_score (y), colored by model.

    Filters to medium depth, AutoReview system.

    Args:
        df: Unified benchmark DataFrame.
        out: Output path (PDF/PNG).
    """
    apply_style()

    filtered = df[
        (df["depth"] == "medium") & (df["system"] == "autoreview") & df["cost_usd"].notna()
    ]

    models = sorted(filtered["model"].unique())
    model_colors = {m: CB_PALETTE[i % len(CB_PALETTE)] for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for model in models:
        model_df = filtered[filtered["model"] == model]
        ax.scatter(
            model_df["cost_usd"],
            model_df["overall_score"],
            c=model_colors[model],
            label=model,
            s=80,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.5,
        )

    ax.set_xlabel("Cost (USD)")
    ax.set_ylabel("Overall Score")
    ax.set_title("Cost–Quality Frontier by Model")
    ax.legend(loc="best")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("model_comparison.cost_frontier_saved", path=str(out))


async def main(results_dir: Path, output_dir: Path, df: pd.DataFrame) -> None:
    """Run model tier comparison analysis.

    Args:
        results_dir: Directory with benchmark results (unused if df provided).
        output_dir: Output directory for figures and JSON.
        df: Pre-loaded unified DataFrame from load_all_evaluations().
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("model_comparison.start", n_rows=len(df))

    summary = compute_model_summary(df)
    tests = compute_model_tests(df)

    results: dict[str, Any] = {
        "model_summary": summary,
        "statistical_tests": tests,
    }
    save_analysis_json(results, output_dir / "model_comparison.json")

    plot_model_line(summary, output_dir / "model_comparison_line.pdf")
    plot_cost_frontier(df, output_dir / "model_cost_frontier.pdf")

    logger.info("model_comparison.complete", output_dir=str(output_dir))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analysis 7: Model tier comparison")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/output/model_comparison"),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    import asyncio

    from paper.analysis.common import load_all_evaluations
    from paper.models import load_topics

    _args = parse_args()
    _topics = load_topics(Path("paper/topics.yaml"))
    _df = load_all_evaluations(_args.results_dir, _topics)
    asyncio.run(main(_args.results_dir, _args.output_dir, _df))
