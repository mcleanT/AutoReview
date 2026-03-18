# paper/analysis/common.py
"""Shared utilities for benchmark analysis scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import structlog

if TYPE_CHECKING:
    from paper.models import TopicsConfig

logger = structlog.get_logger()

# Colorblind-safe palette (Wong 2011, matches depth_comparison.py)
CB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]

FONT_CONFIG = {
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


def apply_style() -> None:
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update(FONT_CONFIG)


def fdr_correct(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction.

    Returns adjusted p-values in the same order as input.
    """
    if not p_values:
        return []

    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    # Compute raw adjusted values
    raw_adj = [0.0] * m
    for rank, (_orig_idx, p) in enumerate(indexed):
        raw_adj[rank] = min(p * m / (rank + 1), 1.0)

    # Cumulative minimum from right (enforce monotonicity)
    for i in range(len(raw_adj) - 2, -1, -1):
        raw_adj[i] = min(raw_adj[i], raw_adj[i + 1])

    # Map back to original order
    result = [0.0] * m
    for rank, (orig_idx, _) in enumerate(indexed):
        result[orig_idx] = raw_adj[rank]

    return result


def save_analysis_json(data: dict[str, Any], path: Path) -> None:
    """Save analysis results as JSON with directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info("analysis.json_saved", path=str(path))


def generate_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Generate a markdown table string."""
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def load_all_evaluations(results_dir: Path, topics: TopicsConfig) -> pd.DataFrame:
    """Load all evaluation results into a unified DataFrame.

    Reads run_registry.json for AutoReview runs, scans arise/ for ARISE outputs.
    """
    from paper.models import RunRegistry, parse_run_key

    topic_lookup = {t.id: t for t in topics.topics}
    registry = RunRegistry.load(results_dir / "run_registry.json")
    rows: list[dict[str, Any]] = []

    # AutoReview runs from registry
    for key, entry in registry.runs.items():
        if entry.status != "completed" or not entry.evaluation_path:
            continue

        eval_path = Path(entry.evaluation_path)
        if not eval_path.exists():
            logger.warning("common.missing_eval", key=key, path=str(eval_path))
            continue

        eval_data = json.loads(eval_path.read_text())
        topic_id, model, depth, condition = parse_run_key(key)
        topic_info = topic_lookup.get(topic_id)

        row = _eval_to_row(eval_data)
        row.update(
            {
                "topic_id": topic_id,
                "domain": topic_info.domain if topic_info else "unknown",
                "tier": topic_info.tier if topic_info else "unknown",
                "system": "autoreview",
                "model": model,
                "depth": depth,
                "condition": condition,
                "cost_usd": entry.cost_usd,
                "tokens_input": entry.tokens_input,
                "tokens_output": entry.tokens_output,
            }
        )
        rows.append(row)

    # ARISE outputs
    arise_dir = results_dir / "arise"
    if arise_dir.exists():
        for topic_dir in arise_dir.iterdir():
            if not topic_dir.is_dir():
                continue
            eval_path = topic_dir / "evaluation.json"
            if not eval_path.exists():
                continue

            eval_data = json.loads(eval_path.read_text())
            topic_id = topic_dir.name
            topic_info = topic_lookup.get(topic_id)

            row = _eval_to_row(eval_data)
            row.update(
                {
                    "topic_id": topic_id,
                    "domain": topic_info.domain if topic_info else "unknown",
                    "tier": topic_info.tier if topic_info else "unknown",
                    "system": "arise",
                    "model": "arise",
                    "depth": "medium",
                    "condition": "end_to_end",
                    "cost_usd": None,
                    "tokens_input": None,
                    "tokens_output": None,
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def _eval_to_row(eval_data: dict[str, Any]) -> dict[str, Any]:
    """Extract flat row from an evaluation JSON dict."""
    row: dict[str, Any] = {
        "overall_score": eval_data.get("overall_score", 0.0),
        "citation_recall": eval_data.get("citation_score", {}).get("recall", 0.0),
        "citation_precision": eval_data.get("citation_score", {}).get("precision", 0.0),
        "citation_f1": eval_data.get("citation_score", {}).get("f1", 0.0),
        "synthesis_score": eval_data.get("synthesis_score", {}).get("generated_score", 0.0),
        "topic_coverage": eval_data.get("topic_coverage", {}).get("generated_coverage", 0.0),
        "writing_quality": eval_data.get("writing_quality", {}).get("generated_score", 0.0),
    }

    # ARISE rubric
    arise = eval_data.get("arise_result")
    row["arise_total"] = arise.get("total_score") if arise else None

    # Hallucination rate (for Analysis 6)
    cs = eval_data.get("citation_score", {})
    hallucinated = len(cs.get("hallucinated_titles", []))
    generated = cs.get("generated_count", 0)
    row["hallucination_rate"] = hallucinated / generated if generated > 0 else 0.0

    # Structural metrics
    sm = eval_data.get("structural_metrics")
    if sm:
        row["word_count"] = sm.get("word_count", 0)
        row["section_count"] = sm.get("section_count", 0)
        row["citation_count"] = sm.get("citation_count", 0)
        row["citations_per_1000_words"] = sm.get("citations_per_1000_words", 0.0)
        row["flesch_kincaid_grade"] = sm.get("flesch_kincaid_grade", 0.0)
    else:
        row.update(
            {
                "word_count": 0,
                "section_count": 0,
                "citation_count": 0,
                "citations_per_1000_words": 0.0,
                "flesch_kincaid_grade": 0.0,
            }
        )

    return row
