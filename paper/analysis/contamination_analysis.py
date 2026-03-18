"""Analysis 9: Contamination Analysis.

Measures n-gram overlap between generated reviews and reference papers
to detect potential training data contamination.

Usage:
    python -m paper.analysis.contamination_analysis \
        --results-dir paper/results --output-dir paper/output/contamination
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import structlog

from paper.analysis.common import CB_PALETTE, apply_style, save_analysis_json

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Core n-gram logic
# ---------------------------------------------------------------------------


def compute_ngram_overlap(text_a: str, text_b: str, n: int) -> float:
    """Compute Jaccard index of n-gram sets between two texts.

    Tokenizes by splitting on whitespace and lowercasing. Generates n-grams
    as tuples of consecutive tokens.

    Args:
        text_a: First text string.
        text_b: Second text string.
        n: N-gram size (e.g. 3 for trigrams).

    Returns:
        |intersection| / |union| as a float in [0.0, 1.0].
        Returns 0.0 if both texts produce empty n-gram sets.
    """

    def _ngrams(text: str) -> set[tuple[str, ...]]:
        tokens = text.lower().split()
        if len(tokens) < n:
            return set()
        return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

    set_a = _ngrams(text_a)
    set_b = _ngrams(text_b)

    union = set_a | set_b
    if not union:
        return 0.0

    return len(set_a & set_b) / len(union)


# ---------------------------------------------------------------------------
# Overlap computation over a results directory
# ---------------------------------------------------------------------------


def compute_all_overlaps(
    results_dir: Path,
    df: pd.DataFrame,
    ns: list[int] | None = None,
) -> pd.DataFrame:
    """Compute n-gram overlaps between generated reviews and reference PDFs.

    Looks up review paths from the run registry and reference PDF paths from
    topics.yaml (via the topic_id column in df). Deduplicates by topic to
    avoid computing overlaps for every model/depth variant of the same topic.

    Args:
        results_dir: Results directory containing run_registry.json.
        df: DataFrame with at minimum columns: topic_id, tier.
        ns: List of n values to compute. Defaults to [1, 2, 3, 4, 5].

    Returns:
        DataFrame with columns: topic_id, tier, n, overlap.
    """
    from paper.models import RunRegistry, load_topics, parse_run_key

    if ns is None:
        ns = [1, 2, 3, 4, 5]

    # Load registry for review paths
    registry = RunRegistry.load(results_dir / "run_registry.json")

    # Load topics for reference PDF paths
    topics_path = results_dir.parent / "topics.yaml"
    if not topics_path.exists():
        topics_path = Path("paper/topics.yaml")
    topic_lookup: dict[str, Any] = {}
    if topics_path.exists():
        topics_config = load_topics(topics_path)
        topic_lookup = {t.id: t for t in topics_config.topics}

    # Build a map: topic_id -> (review_path, reference_pdf_path, tier)
    # Use the first completed Sonnet/medium/end_to_end run per topic
    topic_paths: dict[str, tuple[Path, Path, str]] = {}
    for key, entry in registry.runs.items():
        if entry.status != "completed" or not entry.review_path:
            continue
        topic_id, _model, _depth, condition = parse_run_key(key)
        if topic_id in topic_paths or condition != "end_to_end":
            continue

        topic_info = topic_lookup.get(topic_id)
        if not topic_info:
            continue

        review_file = Path(entry.review_path)
        ref_file = Path(topic_info.reference.pdf_path)

        if review_file.exists() and ref_file.exists():
            tier = topic_info.tier
            topic_paths[topic_id] = (review_file, ref_file, tier)

    rows: list[dict[str, Any]] = []
    for topic_id, (review_file, ref_file, tier) in topic_paths.items():
        generated_text = review_file.read_text(encoding="utf-8", errors="replace")

        try:
            from autoreview.evaluation.pdf_extractor import extract_text_from_pdf

            reference_text = extract_text_from_pdf(ref_file)
        except Exception as exc:
            logger.warning("contamination.pdf_extract_failed", topic_id=topic_id, error=str(exc))
            continue

        for n_val in ns:
            overlap = compute_ngram_overlap(generated_text, reference_text, n=n_val)
            rows.append({"topic_id": topic_id, "tier": tier, "n": n_val, "overlap": overlap})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------


def plot_overlap_histogram(overlaps_df: pd.DataFrame, out: Path) -> None:
    """Histogram of overlap scores across all topics and n values.

    Args:
        overlaps_df: DataFrame with column 'overlap'.
        out: Output file path.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    ax.hist(
        overlaps_df["overlap"],
        bins=30,
        color=CB_PALETTE[0],
        edgecolor="white",
        alpha=0.85,
    )
    ax.set_xlabel("N-gram Overlap (Jaccard Index)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of N-gram Overlap Scores")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("contamination.histogram_saved", path=str(out))


def plot_tier_comparison(overlaps_df: pd.DataFrame, out: Path) -> None:
    """Side-by-side box plots of overlap scores for Tier A vs Tier B.

    Args:
        overlaps_df: DataFrame with columns 'overlap' and 'tier'.
        out: Output file path.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    tiers = sorted(overlaps_df["tier"].dropna().unique())
    data = [overlaps_df.loc[overlaps_df["tier"] == t, "overlap"].values for t in tiers]
    colors = [CB_PALETTE[i % len(CB_PALETTE)] for i in range(len(tiers))]

    bp = ax.boxplot(data, patch_artist=True, tick_labels=[str(t).capitalize() for t in tiers])
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xlabel("Tier")
    ax.set_ylabel("N-gram Overlap (Jaccard Index)")
    ax.set_title("Overlap by Reference Tier")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("contamination.tier_comparison_saved", path=str(out))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def main(results_dir: Path, output_dir: Path, df: pd.DataFrame) -> None:
    """Run Analysis 9: contamination analysis.

    Reads text files referenced by df. Missing files are skipped with warnings.

    Args:
        results_dir: Base directory for resolving relative paths in df.
        output_dir: Output directory for figures and JSON.
        df: Pre-loaded evaluations DataFrame (columns: topic_id, tier, review_path, reference_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    overlaps_df = compute_all_overlaps(results_dir, df)

    if overlaps_df.empty:
        logger.warning(
            "contamination.no_overlaps_computed",
            msg="No text files found; skipping figures and JSON output.",
        )
        return

    # Summary JSON
    summary: dict = {
        "n_pairs": int(len(overlaps_df) // 5),  # 5 n values per pair
        "mean_overlap": float(overlaps_df["overlap"].mean()),
        "max_overlap": float(overlaps_df["overlap"].max()),
        "by_tier": {
            str(tier): float(grp["overlap"].mean()) for tier, grp in overlaps_df.groupby("tier")
        },
    }
    save_analysis_json(summary, output_dir / "contamination_analysis.json")

    plot_overlap_histogram(overlaps_df, output_dir / "overlap_histogram.pdf")
    plot_tier_comparison(overlaps_df, output_dir / "tier_comparison.pdf")

    logger.info("contamination.complete", output_dir=str(output_dir))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analysis 9: Contamination analysis")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/output/contamination"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    import asyncio

    from paper.analysis.common import load_all_evaluations
    from paper.models import load_topics

    _args = parse_args()
    _topics = load_topics(Path("paper/topics.yaml"))
    _df = load_all_evaluations(_args.results_dir, _topics)
    asyncio.run(main(_args.results_dir, _args.output_dir, _df))
