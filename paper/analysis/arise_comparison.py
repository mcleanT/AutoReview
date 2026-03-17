"""Analysis 11: ARISE Head-to-Head Comparison.

Runs AutoReview on ARISE's exact 10 benchmark topics and compares
results against ARISE's published scores using their rubric.

Usage:
    python -m paper.analysis.arise_comparison --results-dir paper/results --output-dir paper/output/arise_comparison
"""

from __future__ import annotations

import argparse
import asyncio
import json
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

# ARISE published results (Tables 2, 3 from Wang et al. 2025)
ARISE_PUBLISHED: dict[str, Any] = {
    "overall": {
        "ARISE": 92.48,
        "SurveyForge": 87.58,
        "Human_Baseline": 85.94,
        "AutoSurvey": 82.75,
        "SurveyX": 81.91,
    },
    "by_category": {
        "Analysis": {
            "ARISE": 4.56,
            "AutoSurvey": 3.94,
            "Human": 3.74,
            "SurveyForge": 4.45,
            "SurveyX": 3.64,
        },
        "Literature": {
            "ARISE": 4.95,
            "AutoSurvey": 4.53,
            "Human": 4.70,
            "SurveyForge": 4.79,
            "SurveyX": 4.29,
        },
        "Organization": {
            "ARISE": 4.82,
            "AutoSurvey": 4.26,
            "Human": 4.52,
            "SurveyForge": 4.55,
            "SurveyX": 4.41,
        },
        "Originality": {
            "ARISE": 4.44,
            "AutoSurvey": 3.88,
            "Human": 3.97,
            "SurveyForge": 4.10,
            "SurveyX": 3.68,
        },
        "Presentation": {
            "ARISE": 4.84,
            "AutoSurvey": 3.57,
            "Human": 4.35,
            "SurveyForge": 3.87,
            "SurveyX": 4.22,
        },
        "References": {
            "ARISE": 4.98,
            "AutoSurvey": 4.81,
            "Human": 4.94,
            "SurveyForge": 4.85,
            "SurveyX": 4.45,
        },
        "Scope": {
            "ARISE": 4.30,
            "AutoSurvey": 4.10,
            "Human": 4.14,
            "SurveyForge": 4.23,
            "SurveyX": 4.00,
        },
    },
    "per_topic_final": {
        "arise_llm_evaluation": 91.17,
        "arise_rag": 90.90,
        "arise_time_series": 90.70,
        "arise_disease_detection": 89.67,
        "arise_multimodal_llm": 87.10,
        "arise_genai_mobile": 86.60,
        "arise_genai_manufacturing": 86.07,
        "arise_fractal_squares": 84.27,
    },
}

CATEGORIES = [
    "Analysis",
    "Literature",
    "Organization",
    "Originality",
    "Presentation",
    "References",
    "Scope",
]


def load_arise_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Filter df to ARISE benchmark topics (tier == 'ARISE')."""
    return df[df["tier"] == "ARISE"].copy()


def compute_arise_comparison(arise_df: pd.DataFrame) -> dict[str, Any]:
    """Compute AutoReview's mean ARISE rubric score and compare against published results.

    Returns dict with autoreview_mean, autoreview_std, autoreview_n,
    arise_published overall scores, and per-system deltas.
    """
    valid = arise_df["arise_total"].dropna()
    autoreview_mean = float(valid.mean()) if len(valid) > 0 else 0.0
    autoreview_std = float(valid.std()) if len(valid) > 1 else 0.0
    autoreview_n = int(len(valid))

    published_overall = ARISE_PUBLISHED["overall"]
    autoreview_vs_published = {
        system: autoreview_mean - score for system, score in published_overall.items()
    }

    return {
        "autoreview_mean": autoreview_mean,
        "autoreview_std": autoreview_std,
        "autoreview_n": autoreview_n,
        "arise_published": published_overall,
        "autoreview_vs_published": autoreview_vs_published,
    }


def load_category_scores(arise_df: pd.DataFrame, results_dir: Path) -> dict[str, list[float]]:
    """Load per-category ARISE rubric scores from evaluation JSONs.

    Looks for evaluation JSONs at results_dir/runs/{topic_id}/evaluation.json.
    Returns dict: category_name -> list[float] of scores across topics.
    """
    category_scores: dict[str, list[float]] = {}

    for _, row in arise_df.iterrows():
        topic_id = row.get("topic_id", "")
        eval_path = results_dir / "runs" / str(topic_id) / "evaluation.json"

        if not eval_path.exists():
            logger.debug("arise_comparison.eval_not_found", topic=topic_id, path=str(eval_path))
            continue

        try:
            eval_data = json.loads(eval_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("arise_comparison.eval_load_error", topic=topic_id, error=str(exc))
            continue

        arise_result = eval_data.get("arise_result")
        if not arise_result:
            continue

        for cat_entry in arise_result.get("category_scores", []):
            cat_name = cat_entry.get("category", "")
            cat_score = cat_entry.get("category_score")
            if cat_name and cat_score is not None:
                category_scores.setdefault(cat_name, []).append(float(cat_score))

    return category_scores


def compute_category_means(category_scores: dict[str, list[float]]) -> dict[str, float]:
    """Compute mean score per category."""
    return {cat: float(np.mean(scores)) for cat, scores in category_scores.items() if scores}


def plot_system_bar(comparison: dict[str, Any], out: Path) -> None:
    """Bar chart comparing AutoReview vs all published systems on ARISE overall score."""
    apply_style()

    systems = ["AutoReview", "ARISE", "SurveyForge", "Human Baseline", "AutoSurvey", "SurveyX"]
    scores = [
        comparison["autoreview_mean"],
        ARISE_PUBLISHED["overall"]["ARISE"],
        ARISE_PUBLISHED["overall"]["SurveyForge"],
        ARISE_PUBLISHED["overall"]["Human_Baseline"],
        ARISE_PUBLISHED["overall"]["AutoSurvey"],
        ARISE_PUBLISHED["overall"]["SurveyX"],
    ]

    x = np.arange(len(systems))
    colors = [CB_PALETTE[i % len(CB_PALETTE)] for i in range(len(systems))]

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars = ax.bar(x, scores, color=colors, alpha=0.85, width=0.6)

    # Horizontal dashed line at AutoReview score
    autoreview_score = comparison["autoreview_mean"]
    ax.axhline(
        y=autoreview_score,
        color=CB_PALETTE[0],
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"AutoReview ({autoreview_score:.1f})",
    )

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{score:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=20, ha="right")
    ax.set_ylabel("ARISE Overall Score (0–100)")
    ax.set_title("System Comparison: ARISE Benchmark")
    ax.set_ylim(0, max(scores) * 1.15)
    ax.legend(loc="upper right")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("arise_comparison.bar_saved", path=str(out))


def plot_category_radar(category_means: dict[str, float], out: Path) -> None:
    """Radar chart: AutoReview vs ARISE per rubric category (scores normalized to 0-1)."""
    apply_style()

    out.parent.mkdir(parents=True, exist_ok=True)

    if not category_means:
        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
        ax.text(
            0.5,
            0.5,
            "No per-category data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title("Category Radar: AutoReview vs ARISE")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("arise_comparison.radar_placeholder_saved", path=str(out))
        return

    n = len(CATEGORIES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    # AutoReview per-category (normalize 1-5 -> 0-1)
    ar_values = [category_means.get(cat, 0.0) / 5.0 for cat in CATEGORIES]
    ar_values += ar_values[:1]

    # ARISE published per-category (normalize 1-5 -> 0-1)
    arise_values = [ARISE_PUBLISHED["by_category"][cat]["ARISE"] / 5.0 for cat in CATEGORIES]
    arise_values += arise_values[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True}, constrained_layout=True)

    ax.plot(angles, ar_values, "o-", linewidth=2, label="AutoReview", color=CB_PALETTE[0])
    ax.fill(angles, ar_values, alpha=0.15, color=CB_PALETTE[0])

    ax.plot(angles, arise_values, "o-", linewidth=2, label="ARISE", color=CB_PALETTE[1])
    ax.fill(angles, arise_values, alpha=0.15, color=CB_PALETTE[1])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CATEGORIES, size=10)
    ax.set_ylim(0, 1)
    ax.set_title("Category Radar: AutoReview vs ARISE", size=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("arise_comparison.radar_saved", path=str(out))


def generate_report(
    comparison: dict[str, Any],
    category_means: dict[str, float],
    path: Path,
) -> None:
    """Generate markdown summary report."""
    lines: list[str] = ["# Analysis 11: ARISE Head-to-Head Comparison\n"]

    ar_mean = comparison["autoreview_mean"]

    lines.append("## Overall Score Comparison\n")
    lines.append("| System | Score | Delta vs AutoReview |")
    lines.append("|--------|-------|---------------------|")
    lines.append(f"| AutoReview | {ar_mean:.2f} | — |")
    for system, score in comparison["arise_published"].items():
        delta = ar_mean - score
        delta_str = f"{delta:+.2f}"
        lines.append(f"| {system} | {score:.2f} | {delta_str} |")
    lines.append("")

    lines.append("## Per-Category Comparison\n")
    if category_means:
        lines.append("| Category | AutoReview | ARISE Published | Delta |")
        lines.append("|----------|------------|-----------------|-------|")
        for cat in CATEGORIES:
            ar_cat = category_means.get(cat)
            arise_cat = ARISE_PUBLISHED["by_category"].get(cat, {}).get("ARISE")
            if ar_cat is not None and arise_cat is not None:
                delta = ar_cat - arise_cat
                lines.append(f"| {cat} | {ar_cat:.2f} | {arise_cat:.2f} | {delta:+.2f} |")
    else:
        lines.append("No per-category data available.\n")
    lines.append("")

    lines.append("## Key Findings\n")
    beats = [s for s, d in comparison.get("autoreview_vs_published", {}).items() if d > 0]
    trails = [s for s, d in comparison.get("autoreview_vs_published", {}).items() if d <= 0]
    if beats:
        lines.append(f"- AutoReview outperforms: {', '.join(beats)}")
    if trails:
        lines.append(f"- AutoReview trails: {', '.join(trails)}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    logger.info("arise_comparison.report_written", path=str(path))


async def main(results_dir: Path, output_dir: Path, df: pd.DataFrame) -> None:
    """Orchestrate Analysis 11: ARISE head-to-head comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("arise_comparison.start", n_rows=len(df))

    arise_df = load_arise_runs(df)
    logger.info("arise_comparison.arise_rows", n=len(arise_df))

    comparison = compute_arise_comparison(arise_df)
    category_scores = load_category_scores(arise_df, results_dir)
    category_means = compute_category_means(category_scores)

    save_analysis_json(
        {"comparison": comparison, "category_means": category_means},
        output_dir / "arise_comparison.json",
    )

    plot_system_bar(comparison, output_dir / "system_comparison_bar.pdf")
    plot_category_radar(category_means, output_dir / "category_radar.pdf")
    generate_report(comparison, category_means, output_dir / "report.md")

    logger.info("arise_comparison.complete", output_dir=str(output_dir))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analysis 11: ARISE Head-to-Head Comparison")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/output/arise_comparison"),
    )
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
