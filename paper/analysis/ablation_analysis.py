"""Analysis 4: Component Ablation.

Compares AutoReview output quality when individual components are ablated
(disabled) versus the full end-to-end pipeline.

Usage:
    python -m paper.analysis.ablation_analysis --results-dir paper/results --output-dir paper/output/ablation
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

from paper.analysis.common import apply_style, fdr_correct, save_analysis_json

logger = structlog.get_logger()

METRICS = [
    "overall_score",
    "citation_f1",
    "synthesis_score",
    "topic_coverage",
    "writing_quality",
]

# Conditions that represent ablated variants (not full pipeline, not retrieval_controlled)
_ABLATION_EXCLUDE = {"end_to_end", "retrieval_controlled"}


def compute_ablation_deltas(
    df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Compute mean delta from end_to_end baseline per ablation condition per metric.

    For each ablation condition, finds matching baseline rows (same topic, model,
    depth with condition="end_to_end") and computes mean(ablation - baseline).

    Args:
        df: DataFrame with columns: topic_id, model, depth, condition, + metric columns.

    Returns:
        Nested dict: condition -> metric -> mean_delta (float).
    """
    ablation_conditions = [c for c in df["condition"].unique() if c not in _ABLATION_EXCLUDE]

    baseline_df = df[df["condition"] == "end_to_end"].copy()
    results: dict[str, dict[str, float]] = {}

    for condition in ablation_conditions:
        cond_df = df[df["condition"] == condition].copy()
        metric_deltas: dict[str, float] = {}

        for metric in METRICS:
            if metric not in df.columns:
                continue

            deltas: list[float] = []
            for _, abl_row in cond_df.iterrows():
                # Find matching baseline row (same topic, model, depth)
                mask = (
                    (baseline_df["topic_id"] == abl_row["topic_id"])
                    & (baseline_df["model"] == abl_row["model"])
                    & (baseline_df["depth"] == abl_row["depth"])
                )
                matched = baseline_df[mask]
                if matched.empty:
                    continue
                baseline_val = float(matched.iloc[0][metric])
                ablation_val = float(abl_row[metric])
                deltas.append(ablation_val - baseline_val)

            metric_deltas[metric] = float(np.mean(deltas)) if deltas else float("nan")

        results[condition] = metric_deltas
        logger.info(
            "ablation_analysis.condition_computed",
            condition=condition,
            n_pairs=len(cond_df),
        )

    return results


def compute_ablation_pairwise_tests(
    df: pd.DataFrame,
    min_pairs: int = 3,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Wilcoxon signed-rank tests: ablation condition vs end_to_end baseline.

    Args:
        df: DataFrame with condition, topic_id, model, depth, metric columns.
        min_pairs: Minimum matched pairs to run a test.

    Returns:
        Nested dict: condition -> metric -> {statistic, p_value, n_pairs, mean_diff, p_adjusted}.
    """
    ablation_conditions = [c for c in df["condition"].unique() if c not in _ABLATION_EXCLUDE]
    baseline_df = df[df["condition"] == "end_to_end"].copy()

    all_results: dict[str, dict[str, dict[str, Any]]] = {}
    all_p_values: list[tuple[str, str, float]] = []

    for condition in ablation_conditions:
        cond_df = df[df["condition"] == condition].copy()
        all_results[condition] = {}

        for metric in METRICS:
            if metric not in df.columns:
                continue

            abl_vals: list[float] = []
            base_vals: list[float] = []

            for _, abl_row in cond_df.iterrows():
                mask = (
                    (baseline_df["topic_id"] == abl_row["topic_id"])
                    & (baseline_df["model"] == abl_row["model"])
                    & (baseline_df["depth"] == abl_row["depth"])
                )
                matched = baseline_df[mask]
                if matched.empty:
                    continue
                base_vals.append(float(matched.iloc[0][metric]))
                abl_vals.append(float(abl_row[metric]))

            n_pairs = len(abl_vals)
            if n_pairs < min_pairs:
                all_results[condition][metric] = {
                    "statistic": float("nan"),
                    "p_value": float("nan"),
                    "n_pairs": n_pairs,
                    "mean_diff": float("nan"),
                    "p_adjusted": float("nan"),
                }
                continue

            diff = np.array(abl_vals) - np.array(base_vals)
            if np.all(diff == 0):
                stat, p_val = 0.0, 1.0
            else:
                w_result = sp_stats.wilcoxon(diff, alternative="two-sided")
                stat = float(w_result[0])  # type: ignore[arg-type]
                p_val = float(w_result[1])  # type: ignore[arg-type]

            all_results[condition][metric] = {
                "statistic": stat,
                "p_value": p_val,
                "n_pairs": n_pairs,
                "mean_diff": float(np.mean(diff)),
                "p_adjusted": float("nan"),  # filled in below
            }
            all_p_values.append((condition, metric, p_val))

    # FDR correction across all tests
    if all_p_values:
        raw_pvals = [p for _, _, p in all_p_values]
        adj_pvals = fdr_correct(raw_pvals)
        for i, (condition, metric, _) in enumerate(all_p_values):
            all_results[condition][metric]["p_adjusted"] = adj_pvals[i]

    return all_results


def plot_ablation_heatmap(
    deltas: dict[str, dict[str, float]],
    out: Path,
) -> None:
    """Heatmap of ablation deltas: conditions (rows) x metrics (cols).

    Cells show mean delta from end_to_end baseline. Blue = improvement,
    red = degradation.

    Args:
        deltas: Output of compute_ablation_deltas().
        out: Output path for the PDF figure.
    """
    apply_style()

    conditions = sorted(deltas.keys())
    metrics = [m for m in METRICS if any(m in deltas[c] for c in conditions)]

    if not conditions or not metrics:
        logger.warning("ablation_analysis.empty_deltas_skipping_heatmap")
        return

    data = np.array(
        [[deltas[c].get(m, float("nan")) for m in metrics] for c in conditions],
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(max(6, len(metrics) * 1.4), max(3, len(conditions) * 0.8 + 1)),
        constrained_layout=True,
    )

    # Diverging colormap centered at 0
    vmax = np.nanmax(np.abs(data)) if not np.all(np.isnan(data)) else 1.0
    vmax = max(vmax, 0.01)
    im = ax.imshow(data, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels([c.replace("_", " ") for c in conditions])
    ax.set_xlabel("Metric")
    ax.set_ylabel("Ablation Condition")
    ax.set_title("Ablation Deltas from End-to-End Baseline")

    for i, condition in enumerate(conditions):
        for j, metric in enumerate(metrics):
            val = data[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:+.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

    fig.colorbar(im, ax=ax, shrink=0.8, label="Delta from baseline")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("ablation_analysis.heatmap_saved", path=str(out))


async def main(results_dir: Path, output_dir: Path, df: pd.DataFrame) -> None:
    """Full ablation analysis pipeline.

    Args:
        results_dir: Directory containing run results (unused if df is provided).
        output_dir: Directory for output figures and JSON.
        df: Pre-loaded unified evaluations DataFrame from load_all_evaluations().
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("ablation_analysis.start", n_rows=len(df))

    deltas = compute_ablation_deltas(df)
    tests = compute_ablation_pairwise_tests(df)

    # Save JSON
    save_analysis_json(
        {"deltas": deltas, "pairwise_tests": tests},
        output_dir / "ablation_analysis.json",
    )

    # Plot heatmap
    plot_ablation_heatmap(deltas, output_dir / "ablation_heatmap.pdf")

    logger.info("ablation_analysis.complete", output_dir=str(output_dir))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analysis 4: Component Ablation")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/output/ablation"),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    import asyncio

    from paper.analysis.common import load_all_evaluations
    from paper.models import TopicsConfig

    _args = parse_args()
    # In standalone mode, load topics config and evaluations
    _topics = TopicsConfig.load(_args.results_dir.parent / "paper/topics.yaml")
    _df = load_all_evaluations(_args.results_dir, _topics)
    asyncio.run(main(_args.results_dir, _args.output_dir, _df))
