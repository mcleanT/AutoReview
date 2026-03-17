"""Analysis 5: Retrieval vs Synthesis Decomposition.

Compares end_to_end vs retrieval_controlled conditions on Tier A topics
to isolate the contribution of retrieval quality vs synthesis quality.

Usage:
    python -m paper.analysis.retrieval_decomposition --results-dir paper/results --output-dir paper/output/retrieval
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

CONDITIONS = ["end_to_end", "retrieval_controlled"]
METRICS = [
    "overall_score",
    "citation_f1",
    "synthesis_score",
    "topic_coverage",
    "writing_quality",
]


def compute_condition_comparison(
    df: pd.DataFrame,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute mean/std per condition per metric for Tier A topics.

    Filters to Tier A rows with end_to_end or retrieval_controlled conditions.

    Args:
        df: DataFrame with columns: tier, condition, + metric columns.

    Returns:
        Nested dict: condition -> metric -> {mean, std, n}.
    """
    tier_a = df[df["tier"] == "A"].copy()
    filtered = tier_a[tier_a["condition"].isin(CONDITIONS)].copy()

    results: dict[str, dict[str, dict[str, float]]] = {}

    for condition in CONDITIONS:
        cond_df = filtered[filtered["condition"] == condition]
        results[condition] = {}

        for metric in METRICS:
            if metric not in cond_df.columns or cond_df.empty:
                results[condition][metric] = {"mean": float("nan"), "std": float("nan"), "n": 0}
                continue

            vals = cond_df[metric].dropna()
            results[condition][metric] = {
                "mean": float(vals.mean()) if len(vals) > 0 else float("nan"),
                "std": float(vals.std()) if len(vals) > 1 else 0.0,
                "n": int(len(vals)),
            }

        logger.info(
            "retrieval_decomposition.condition_computed",
            condition=condition,
            n_rows=len(cond_df),
        )

    return results


def compute_condition_pairwise_tests(
    df: pd.DataFrame,
    min_pairs: int = 3,
) -> dict[str, dict[str, Any]]:
    """Wilcoxon signed-rank tests: end_to_end vs retrieval_controlled on Tier A topics.

    Matched by topic_id + model + depth.

    Args:
        df: DataFrame with tier, condition, topic_id, model, depth, metric columns.
        min_pairs: Minimum matched pairs to run a test.

    Returns:
        Dict: metric -> {statistic, p_value, n_pairs, mean_diff, p_adjusted}.
    """
    tier_a = df[df["tier"] == "A"].copy()
    e2e = tier_a[tier_a["condition"] == "end_to_end"].copy()
    ctrl = tier_a[tier_a["condition"] == "retrieval_controlled"].copy()

    all_results: dict[str, dict[str, Any]] = {}
    all_p_values: list[tuple[str, float]] = []

    for metric in METRICS:
        if metric not in df.columns:
            continue

        e2e_vals: list[float] = []
        ctrl_vals: list[float] = []

        for _, ctrl_row in ctrl.iterrows():
            mask = (
                (e2e["topic_id"] == ctrl_row["topic_id"])
                & (e2e["model"] == ctrl_row["model"])
                & (e2e["depth"] == ctrl_row["depth"])
            )
            matched = e2e[mask]
            if matched.empty:
                continue
            e2e_vals.append(float(matched.iloc[0][metric]))
            ctrl_vals.append(float(ctrl_row[metric]))

        n_pairs = len(e2e_vals)
        if n_pairs < min_pairs:
            all_results[metric] = {
                "statistic": float("nan"),
                "p_value": float("nan"),
                "n_pairs": n_pairs,
                "mean_diff": float("nan"),
                "p_adjusted": float("nan"),
            }
            continue

        diff = np.array(ctrl_vals) - np.array(e2e_vals)
        if np.all(diff == 0):
            stat, p_val = 0.0, 1.0
        else:
            w_result = sp_stats.wilcoxon(diff, alternative="two-sided")
            stat = float(w_result[0])  # type: ignore[arg-type]
            p_val = float(w_result[1])  # type: ignore[arg-type]

        all_results[metric] = {
            "statistic": stat,
            "p_value": p_val,
            "n_pairs": n_pairs,
            "mean_diff": float(np.mean(diff)),
            "p_adjusted": float("nan"),  # filled in below
        }
        all_p_values.append((metric, p_val))

    # FDR correction
    if all_p_values:
        raw_pvals = [p for _, p in all_p_values]
        adj_pvals = fdr_correct(raw_pvals)
        for i, (metric, _) in enumerate(all_p_values):
            all_results[metric]["p_adjusted"] = adj_pvals[i]

    return all_results


def plot_condition_bar(
    comp: dict[str, dict[str, dict[str, float]]],
    out: Path,
) -> None:
    """Grouped bar chart: end_to_end vs retrieval_controlled across metrics.

    Args:
        comp: Output of compute_condition_comparison().
        out: Output path for the PDF figure.
    """
    apply_style()

    metrics = [m for m in METRICS if any(m in comp.get(c, {}) for c in CONDITIONS)]
    present_conditions = [c for c in CONDITIONS if c in comp]

    if not metrics or not present_conditions:
        logger.warning("retrieval_decomposition.empty_comparison_skipping_bar")
        return

    x = np.arange(len(metrics))
    n_conds = len(present_conditions)
    width = 0.35
    offsets = np.linspace(-(n_conds - 1) * width / 2, (n_conds - 1) * width / 2, n_conds)

    condition_labels = {
        "end_to_end": "End-to-End",
        "retrieval_controlled": "Retrieval Controlled",
    }
    condition_colors = {
        "end_to_end": CB_PALETTE[0],
        "retrieval_controlled": CB_PALETTE[1],
    }

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    for i, condition in enumerate(present_conditions):
        means = [comp[condition].get(m, {}).get("mean", 0.0) for m in metrics]
        stds = [comp[condition].get(m, {}).get("std", 0.0) for m in metrics]
        ax.bar(
            x + offsets[i],
            means,
            width,
            yerr=stds,
            label=condition_labels.get(condition, condition),
            color=condition_colors.get(condition, CB_PALETTE[i]),
            alpha=0.85,
            capsize=4,
            error_kw={"elinewidth": 1.5},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Retrieval vs Synthesis Decomposition\n(Tier A Topics)")
    ax.legend(loc="best")
    ax.set_ylim(bottom=0)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("retrieval_decomposition.bar_saved", path=str(out))


async def main(results_dir: Path, output_dir: Path, df: pd.DataFrame) -> None:
    """Full retrieval decomposition analysis pipeline.

    Args:
        results_dir: Directory containing run results (unused if df is provided).
        output_dir: Directory for output figures and JSON.
        df: Pre-loaded unified evaluations DataFrame from load_all_evaluations().
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("retrieval_decomposition.start", n_rows=len(df))

    comp = compute_condition_comparison(df)
    tests = compute_condition_pairwise_tests(df)

    save_analysis_json(
        {"condition_comparison": comp, "pairwise_tests": tests},
        output_dir / "retrieval_decomposition.json",
    )

    plot_condition_bar(comp, output_dir / "retrieval_decomposition_bar.pdf")

    logger.info("retrieval_decomposition.complete", output_dir=str(output_dir))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analysis 5: Retrieval vs Synthesis Decomposition")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/output/retrieval"),
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
