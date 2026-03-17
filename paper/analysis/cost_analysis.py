"""Analysis 8: Cost-Quality Tradeoff.

Computes cost efficiency metrics and plots cost vs. quality scatter.

Usage:
    python -m paper.analysis.cost_analysis --results-dir paper/results --output-dir paper/output/cost_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import structlog

from paper.analysis.common import CB_PALETTE, apply_style, save_analysis_json

logger = structlog.get_logger()


def compute_cost_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Add cost efficiency columns to the DataFrame.

    Adds:
        cost_per_quality_point: cost_usd / overall_score (0.0 if overall_score == 0)
        cost_per_citation_f1:   cost_usd / citation_f1  (0.0 if citation_f1 == 0)

    Args:
        df: DataFrame with columns cost_usd, overall_score, citation_f1.

    Returns:
        Copy of df with two new columns appended.
    """
    out = df.copy()

    cost = pd.to_numeric(out["cost_usd"], errors="coerce").fillna(0.0)

    overall = pd.to_numeric(out["overall_score"], errors="coerce").fillna(0.0)
    out["cost_per_quality_point"] = np.where(overall > 0, cost / overall, 0.0)

    cit_f1 = pd.to_numeric(out["citation_f1"], errors="coerce").fillna(0.0)
    out["cost_per_citation_f1"] = np.where(cit_f1 > 0, cost / cit_f1, 0.0)

    return out


def plot_cost_quality_scatter(df: pd.DataFrame, out: Path) -> None:
    """Scatter plot: cost_usd (x) vs overall_score (y), colored by model.

    Args:
        df: DataFrame with columns cost_usd, overall_score, model.
        out: Output file path (PDF/PNG).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    models = df["model"].unique() if "model" in df.columns else ["unknown"]
    model_colors = {m: CB_PALETTE[i % len(CB_PALETTE)] for i, m in enumerate(models)}

    for model in models:
        subset = df[df["model"] == model] if "model" in df.columns else df
        ax.scatter(
            pd.to_numeric(subset["cost_usd"], errors="coerce"),
            pd.to_numeric(subset["overall_score"], errors="coerce"),
            c=model_colors[model],
            label=str(model),
            s=80,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.5,
        )

    ax.set_xlabel("Cost (USD)")
    ax.set_ylabel("Overall Score")
    ax.set_title("Cost vs. Quality Tradeoff")
    ax.legend(loc="best", title="Model")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("cost_analysis.scatter_saved", path=str(out))


async def main(results_dir: Path, output_dir: Path, df: pd.DataFrame) -> None:
    """Run Analysis 8: cost-quality tradeoff.

    Args:
        results_dir: Directory containing pipeline results (unused when df is provided).
        output_dir: Output directory for figures and JSON.
        df: Pre-loaded evaluations DataFrame from load_all_evaluations().
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    eff_df = compute_cost_efficiency(df)

    # Save summary JSON
    summary: dict = {
        "n_runs": len(eff_df),
        "mean_cost_usd": float(pd.to_numeric(eff_df["cost_usd"], errors="coerce").mean()),
        "mean_cost_per_quality_point": float(eff_df["cost_per_quality_point"].mean()),
        "mean_cost_per_citation_f1": float(eff_df["cost_per_citation_f1"].mean()),
    }
    save_analysis_json(summary, output_dir / "cost_analysis.json")

    # Plot scatter
    plot_cost_quality_scatter(eff_df, output_dir / "cost_quality_scatter.pdf")

    logger.info("cost_analysis.complete", output_dir=str(output_dir))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analysis 8: Cost-quality tradeoff")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/output/cost_analysis")
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    import asyncio

    _args = parse_args()
    # When run standalone, load evaluations via common utility
    from paper.analysis.common import load_all_evaluations
    from paper.models import TopicsConfig

    _topics = TopicsConfig.load(_args.results_dir / "../topics.yaml")
    _df = load_all_evaluations(_args.results_dir, _topics)
    asyncio.run(main(_args.results_dir, _args.output_dir, _df))
