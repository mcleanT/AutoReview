from __future__ import annotations

import json
from pathlib import Path

import structlog

from autoreview.evaluation.models import EvaluationResult

logger = structlog.get_logger()


def generate_markdown_report(result: EvaluationResult) -> str:
    cs = result.citation_score
    ss = result.synthesis_score
    tc = result.topic_coverage
    wq = result.writing_quality

    lines = [
        "# AutoReview Evaluation Report",
        f"\n**Generated:** {result.timestamp}",
        f"**Generated review:** `{result.generated_path}`",
        f"**Reference review:** `{result.reference_path}`",
        "",
        "---",
        "",
        "## Summary Scores",
        "",
        "| Dimension | Generated | Reference | Delta |",
        "|---|---|---|---|",
        f"| Citation Recall | {cs.recall:.2f} | — | — |",
        (
            f"| Synthesis Depth | {ss.generated_score:.1f}/5 "
            f"| {ss.reference_score:.1f}/5 | {ss.delta:+.2f} |"
        ),
        (
            f"| Topical Coverage | {tc.generated_coverage:.2f} "
            f"| {tc.reference_coverage:.2f} "
            f"| {tc.generated_coverage - tc.reference_coverage:+.2f} |"
        ),
        (
            f"| Writing Quality | {wq.generated_score:.1f}/5 "
            f"| {wq.reference_score:.1f}/5 | {wq.delta:+.2f} |"
        ),
        f"| **Overall** | **{result.overall_score:.2f}** | — | — |",
        "",
        "---",
        "",
        "## Citation Coverage",
        "",
        (
            f"- **Recall:** {cs.recall:.1%} "
            f"({cs.matched_count}/{cs.reference_count} reference papers matched)"
        ),
        f"- **Generated bibliography size:** {cs.generated_count} papers",
    ]
    if cs.missed_titles:
        lines += ["", "**Notable missed papers:**"]
        for t in cs.missed_titles[:10]:
            lines.append(f"- {t}")
        if len(cs.missed_titles) > 10:
            lines.append(f"- ... and {len(cs.missed_titles) - 10} more")

    lines += [
        "",
        "---",
        "",
        "## Synthesis Depth",
        "",
        "| Dimension | Generated | Reference |",
        "|---|---|---|",
    ]
    for dim, val in ss.dimension_scores.items():
        lines.append(f"| {dim.replace('_', ' ').title()} | {val:.1f} | — |")
    lines += [
        "",
        f"**Generated observations:** {ss.generated_observations}",
        f"**Reference observations:** {ss.reference_observations}",
    ]

    lines += [
        "",
        "---",
        "",
        "## Topical Coverage",
        "",
        f"- **Coverage:** {tc.generated_coverage:.1%} of reference sub-topics covered",
        f"- **Topics in both:** {', '.join(tc.topics_in_both) if tc.topics_in_both else 'none'}",
    ]
    if tc.topics_only_in_reference:
        lines.append(
            f"- **Topics only in reference (gaps):** {', '.join(tc.topics_only_in_reference)}"
        )
    if tc.topics_only_in_generated:
        lines.append(
            f"- **Topics only in generated (extras):** {', '.join(tc.topics_only_in_generated)}"
        )

    lines += [
        "",
        "---",
        "",
        "## Writing Quality",
        "",
        "| Dimension | Generated | Reference |",
        "|---|---|---|",
    ]
    for dim, val in wq.dimension_scores.items():
        lines.append(f"| {dim.replace('_', ' ').title()} | {val:.1f} | — |")

    return "\n".join(lines)


def save_report(result: EvaluationResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = result.timestamp.replace(":", "").replace("T", "_")[:15]

    json_path = output_dir / f"evaluation_{ts}.json"
    json_path.write_text(json.dumps(result.model_dump(), indent=2))

    md_path = output_dir / f"evaluation_{ts}_report.md"
    md_path.write_text(generate_markdown_report(result))

    logger.info("report_generator.saved", json=str(json_path), md=str(md_path))
    return json_path, md_path
