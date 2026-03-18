"""Analysis 3: Cross-Rubric Agreement.

Examines agreement between the AutoReview evaluation rubric (overall_score)
and the ARISE rubric (arise_total) via Spearman/Pearson correlation
and Bland-Altman agreement analysis.

Usage:
    python -m paper.analysis.rubric_agreement \
        --results-dir paper/results --output-dir paper/output/rubric_agreement
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

from paper.analysis.common import CB_PALETTE, apply_style, save_analysis_json

logger = structlog.get_logger()


def _get_paired(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract paired (overall_score, arise_total_norm) arrays with no NaNs.

    arise_total is on 0–100 scale; normalised to 0–1 for comparison.
    """
    subset = df[["overall_score", "arise_total"]].dropna()
    x = subset["overall_score"].values.astype(float)
    y = subset["arise_total"].values.astype(float) / 100.0
    return x, y


def compute_rubric_correlation(df: pd.DataFrame) -> dict[str, Any]:
    """Compute Spearman and Pearson correlations between overall_score and arise_total.

    Returns dict with spearman_rho, spearman_p, pearson_r, pearson_p, n.
    """
    x, y = _get_paired(df)
    n = len(x)

    if n < 3:
        return {
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "n": n,
        }

    spearman_result = sp_stats.spearmanr(x, y)
    pearson_result = sp_stats.pearsonr(x, y)

    return {
        "spearman_rho": float(spearman_result.statistic),
        "spearman_p": float(spearman_result.pvalue),
        "pearson_r": float(pearson_result.statistic),
        "pearson_p": float(pearson_result.pvalue),
        "n": n,
    }


def plot_rubric_scatter(df: pd.DataFrame, out: Path) -> None:
    """Scatter plot: overall_score (x) vs arise_total normalised (y).

    One point per evaluation. Color by system if available.
    """
    apply_style()
    x, y = _get_paired(df)

    # Compute correlation for annotation
    corr = compute_rubric_correlation(df)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    # Color by system if column present
    if "system" in df.columns:
        subset = df[["overall_score", "arise_total", "system"]].dropna(
            subset=["overall_score", "arise_total"]
        )
        systems = sorted(subset["system"].unique())
        for i, system in enumerate(systems):
            mask = subset["system"] == system
            sx = subset.loc[mask, "overall_score"].values.astype(float)
            sy = subset.loc[mask, "arise_total"].values.astype(float) / 100.0
            ax.scatter(
                sx,
                sy,
                color=CB_PALETTE[i % len(CB_PALETTE)],
                alpha=0.7,
                edgecolors="black",
                linewidths=0.5,
                s=60,
                label=system.capitalize(),
            )
        ax.legend(loc="upper left")
    else:
        ax.scatter(x, y, color=CB_PALETTE[0], alpha=0.7, edgecolors="black", linewidths=0.5, s=60)

    # Regression line
    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m * x_line + b, "--", color=CB_PALETTE[1], linewidth=1.5, label="OLS fit")

    # Diagonal reference line (perfect agreement)
    lims = [min(x.min(), y.min()) - 0.05, max(x.max(), y.max()) + 0.05]
    ax.plot(lims, lims, ":", color="gray", linewidth=1, alpha=0.7, label="Identity")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    rho = corr.get("spearman_rho", float("nan"))
    n = corr.get("n", 0)
    ax.annotate(
        f"Spearman ρ = {rho:.3f}\nn = {n}",
        xy=(0.05, 0.93),
        xycoords="axes fraction",
        fontsize=10,
        va="top",
    )

    ax.set_xlabel("AutoReview Overall Score")
    ax.set_ylabel("ARISE Total Score (normalised 0–1)")
    ax.set_title("Rubric Agreement: AutoReview vs ARISE")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("rubric_agreement.scatter_saved", path=str(out))


def plot_bland_altman(df: pd.DataFrame, out: Path) -> None:
    """Bland-Altman agreement plot.

    x-axis: mean of both scores; y-axis: difference (overall_score - arise_total_norm).
    Shows mean difference ± 1.96 SD limits of agreement.
    """
    apply_style()
    x, y = _get_paired(df)

    means = (x + y) / 2.0
    diffs = x - y
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    ax.scatter(
        means,
        diffs,
        color=CB_PALETTE[0],
        alpha=0.7,
        edgecolors="black",
        linewidths=0.5,
        s=60,
    )

    ax.axhline(mean_diff, color=CB_PALETTE[1], linewidth=2, label=f"Mean diff: {mean_diff:.3f}")
    ax.axhline(
        loa_upper,
        color=CB_PALETTE[1],
        linewidth=1.5,
        linestyle="--",
        label=f"+1.96 SD: {loa_upper:.3f}",
    )
    ax.axhline(
        loa_lower,
        color=CB_PALETTE[1],
        linewidth=1.5,
        linestyle="--",
        label=f"−1.96 SD: {loa_lower:.3f}",
    )
    ax.axhline(0, color="gray", linewidth=1, linestyle=":", alpha=0.7)

    ax.set_xlabel("Mean of AutoReview and ARISE Scores")
    ax.set_ylabel("Difference (AutoReview − ARISE)")
    ax.set_title("Bland-Altman Plot: Rubric Agreement")
    ax.legend(loc="upper right", fontsize=9)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("rubric_agreement.bland_altman_saved", path=str(out))


def generate_report(corr: dict[str, Any], out: Path) -> None:
    """Generate markdown summary report."""
    lines: list[str] = ["# Analysis 3: Cross-Rubric Agreement\n"]

    lines.append("## Correlation: AutoReview Overall Score vs ARISE Total\n")
    lines.append("| Measure | Value | p-value | N |")
    lines.append("|---------|-------|---------|---|")
    lines.append(
        f"| Spearman ρ | {corr.get('spearman_rho', float('nan')):.4f} | "
        f"{corr.get('spearman_p', float('nan')):.4f} | {corr.get('n', 0)} |"
    )
    lines.append(
        f"| Pearson r | {corr.get('pearson_r', float('nan')):.4f} | "
        f"{corr.get('pearson_p', float('nan')):.4f} | {corr.get('n', 0)} |"
    )
    lines.append("")
    lines.append("## Interpretation\n")

    rho = corr.get("spearman_rho", float("nan"))
    if not np.isnan(rho):
        if abs(rho) >= 0.8:
            strength = "very strong"
        elif abs(rho) >= 0.6:
            strength = "strong"
        elif abs(rho) >= 0.4:
            strength = "moderate"
        else:
            strength = "weak"
        direction = "positive" if rho >= 0 else "negative"
        lines.append(
            f"The two rubrics show a **{strength} {direction} correlation** "
            f"(Spearman ρ = {rho:.3f}), suggesting they "
            + ("measure similar constructs." if abs(rho) >= 0.6 else "diverge on some criteria.")
        )
    lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    logger.info("rubric_agreement.report_written", path=str(out))


async def main(results_dir: Path, output_dir: Path, df: pd.DataFrame) -> None:
    """Orchestrate Analysis 3: cross-rubric agreement."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("rubric_agreement.start", n_rows=len(df))

    corr = compute_rubric_correlation(df)

    save_analysis_json({"correlation": corr}, output_dir / "analysis.json")

    plot_rubric_scatter(df, output_dir / "rubric_scatter.pdf")
    plot_bland_altman(df, output_dir / "bland_altman.pdf")
    generate_report(corr, output_dir / "report.md")

    logger.info("rubric_agreement.complete", output_dir=str(output_dir))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analysis 3: Cross-Rubric Agreement")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/output/rubric_agreement"))
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
