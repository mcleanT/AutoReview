"""Analysis 2: Cross-Domain Variation.

Examines how AutoReview performance varies across scientific domains
(cs_ai, biomedical, climate_science, social_science).

Usage:
    python -m paper.analysis.domain_analysis --results-dir paper/results --output-dir paper/output/domain_analysis
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


def compute_domain_summary(df: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    """Group by domain and compute mean/std per metric.

    Returns: domain -> metric -> {mean, std, n}
    """
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for domain, group in df.groupby("domain"):
        summary[str(domain)] = {}
        for metric in METRICS:
            if metric not in group.columns:
                continue
            vals = group[metric].dropna()
            summary[str(domain)][metric] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "n": int(len(vals)),
            }
    return summary


def compute_domain_tests(df: pd.DataFrame) -> dict[str, Any]:
    """Kruskal-Wallis test per metric across domains, with FDR correction.

    Returns: metric -> {statistic, p_value, p_adjusted, domains, n_per_domain}
    """
    results: dict[str, Any] = {}
    domains = sorted(df["domain"].unique())
    p_values_raw: list[float] = []
    metric_order: list[str] = []

    for metric in METRICS:
        if metric not in df.columns:
            continue
        groups = [df[df["domain"] == d][metric].dropna().values for d in domains]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            results[metric] = {
                "statistic": float("nan"),
                "p_value": float("nan"),
                "domains": domains,
                "n_per_domain": {},
            }
            p_values_raw.append(float("nan"))
        else:
            try:
                stat, p = sp_stats.kruskal(*groups)
                results[metric] = {
                    "statistic": float(stat),
                    "p_value": float(p),
                    "domains": domains,
                    "n_per_domain": {
                        d: int(len(df[df["domain"] == d][metric].dropna())) for d in domains
                    },
                }
                p_values_raw.append(float(p))
            except Exception:
                results[metric] = {
                    "statistic": float("nan"),
                    "p_value": float("nan"),
                    "domains": domains,
                    "n_per_domain": {},
                }
                p_values_raw.append(float("nan"))

        metric_order.append(metric)

    # FDR correction across all metrics
    valid_p = [p for p in p_values_raw if not np.isnan(p)]
    if valid_p:
        adjusted = fdr_correct(p_values_raw)
        for metric, p_adj in zip(metric_order, adjusted):
            if metric in results:
                results[metric]["p_adjusted"] = p_adj
    else:
        for metric in metric_order:
            if metric in results:
                results[metric]["p_adjusted"] = float("nan")

    return results


def plot_domain_heatmap(df: pd.DataFrame, out: Path) -> None:
    """Heatmap: domain (rows) x metric (cols), cell = mean score."""
    apply_style()
    domains = sorted(df["domain"].unique())
    available_metrics = [m for m in METRICS if m in df.columns]

    data = np.full((len(domains), len(available_metrics)), np.nan)
    for i, domain in enumerate(domains):
        ddf = df[df["domain"] == domain]
        for j, metric in enumerate(available_metrics):
            if metric in ddf.columns:
                val = ddf[metric].mean()
                # Normalize 1-5 scales
                if metric in ("synthesis_score", "writing_quality"):
                    val = val / 5.0
                data[i, j] = val

    fig, ax = plt.subplots(
        figsize=(max(6, len(available_metrics) * 1.2), max(4, len(domains) * 0.8)),
        constrained_layout=True,
    )
    im = ax.imshow(data, cmap="viridis", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(available_metrics)))
    ax.set_xticklabels(
        [METRIC_LABELS.get(m, m) for m in available_metrics],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels([d.replace("_", " ").title() for d in domains])
    ax.set_xlabel("Metric")
    ax.set_ylabel("Domain")
    ax.set_title("Performance by Domain and Metric")

    for i in range(len(domains)):
        for j in range(len(available_metrics)):
            val = data[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Score (normalized 0–1)")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("domain_analysis.heatmap_saved", path=str(out))


def plot_domain_bar(df: pd.DataFrame, out: Path) -> None:
    """Grouped bar chart: overall_score per domain, split by system."""
    apply_style()
    domains = sorted(df["domain"].unique())
    systems = sorted(df["system"].unique())
    n_domains = len(domains)
    n_systems = len(systems)
    x = np.arange(n_domains)
    width = 0.8 / n_systems

    fig, ax = plt.subplots(figsize=(max(8, n_domains * 1.5), 5), constrained_layout=True)

    for i, system in enumerate(systems):
        sdf = df[df["system"] == system]
        means = []
        sems = []
        for domain in domains:
            vals = sdf[sdf["domain"] == domain]["overall_score"].dropna()
            means.append(float(vals.mean()) if len(vals) > 0 else 0.0)
            sems.append(float(vals.sem()) if len(vals) > 1 else 0.0)

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
    ax.set_xticklabels([d.replace("_", " ").title() for d in domains], rotation=20, ha="right")
    ax.set_ylabel("Overall Score")
    ax.set_title("Overall Score by Domain and System")
    ax.legend(loc="upper right")
    ax.set_ylim(0, None)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("domain_analysis.bar_chart_saved", path=str(out))


def generate_report(
    summary: dict[str, dict[str, dict[str, float]]],
    tests: dict[str, Any],
    out: Path,
) -> None:
    """Generate markdown summary report."""
    lines: list[str] = ["# Analysis 2: Cross-Domain Variation\n"]

    lines.append("## Summary Statistics\n")
    lines.append("| Domain | Metric | Mean | Std | N |")
    lines.append("|--------|--------|------|-----|---|")
    for domain, metrics in summary.items():
        for metric, stats in metrics.items():
            lines.append(
                f"| {domain.replace('_', ' ').title()} | "
                f"{METRIC_LABELS.get(metric, metric)} | "
                f"{stats.get('mean', 0):.3f} | {stats.get('std', 0):.3f} | "
                f"{stats.get('n', 0)} |"
            )
    lines.append("")

    lines.append("## Kruskal-Wallis Tests (Domain Effect)\n")
    lines.append("| Metric | Statistic | p-value | p-adjusted |")
    lines.append("|--------|-----------|---------|------------|")
    for metric, result in tests.items():
        lines.append(
            f"| {METRIC_LABELS.get(metric, metric)} | "
            f"{result.get('statistic', float('nan')):.3f} | "
            f"{result.get('p_value', float('nan')):.4f} | "
            f"{result.get('p_adjusted', float('nan')):.4f} |"
        )
    lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    logger.info("domain_analysis.report_written", path=str(out))


async def main(results_dir: Path, output_dir: Path, df: pd.DataFrame) -> None:
    """Orchestrate Analysis 2: cross-domain variation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("domain_analysis.start", n_rows=len(df))

    summary = compute_domain_summary(df)
    tests = compute_domain_tests(df)

    save_analysis_json(
        {"summary": summary, "tests": tests},
        output_dir / "analysis.json",
    )

    plot_domain_heatmap(df, output_dir / "domain_heatmap.pdf")
    plot_domain_bar(df, output_dir / "domain_bar.pdf")
    generate_report(summary, tests, output_dir / "report.md")

    logger.info("domain_analysis.complete", output_dir=str(output_dir))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analysis 2: Cross-Domain Variation")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/output/domain_analysis"))
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
