import json
import tempfile
from pathlib import Path
from autoreview.evaluation.models import (
    EvaluationResult, CitationScore, SynthesisScore, TopicCoverageScore, WritingQualityScore,
)
from autoreview.evaluation.report_generator import generate_markdown_report, save_report


def _make_result() -> EvaluationResult:
    return EvaluationResult(
        timestamp="2026-02-24T00:00:00",
        generated_path="output/review.md",
        reference_path="reference.pdf",
        citation_score=CitationScore(recall=0.62, matched_count=22, reference_count=35, generated_count=36, matched_titles=["Smith 2020", "Jones 2021"], missed_titles=["Lee 2019"]),
        synthesis_score=SynthesisScore(generated_score=3.8, reference_score=4.2, delta=-0.4, dimension_scores={"cross_paper": 3.5, "gap_identification": 4.0}, generated_observations="Decent synthesis.", reference_observations="Strong synthesis."),
        topic_coverage=TopicCoverageScore(generated_coverage=0.70, reference_coverage=1.0, topics_in_both=["SASP", "telomere"], topics_only_in_reference=["immune senescence", "lysosomal pathway"], topics_only_in_generated=["gut microbiome"]),
        writing_quality=WritingQualityScore(generated_score=3.5, reference_score=4.0, delta=-0.5, dimension_scores={"clarity": 3.5, "flow": 3.5}),
        overall_score=0.65,
    )


def test_markdown_report_has_summary_table():
    md = generate_markdown_report(_make_result())
    assert "Citation Recall" in md
    assert "0.62" in md
    assert "Synthesis Depth" in md
    assert "3.8" in md
    assert "Topical Coverage" in md
    assert "Writing Quality" in md
    assert "Overall" in md


def test_markdown_report_has_missed_topics():
    md = generate_markdown_report(_make_result())
    assert "immune senescence" in md
    assert "lysosomal pathway" in md


def test_save_report_creates_json_and_markdown():
    result = _make_result()
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path, md_path = save_report(result, Path(tmpdir))
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["overall_score"] == 0.65
        assert md_path.exists()
        md = md_path.read_text()
        assert "Citation Recall" in md
