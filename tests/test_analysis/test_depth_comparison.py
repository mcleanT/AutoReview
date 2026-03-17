from __future__ import annotations

import json

import matplotlib
import pandas as pd
import pytest

from autoreview.evaluation.models import Claim, InformationMetrics

matplotlib.use("Agg")

from paper.analysis.depth_comparison import (
    DepthAnalysisOutput,
    compute_information_metrics,
    compute_pairwise_tests,
    generate_report,
    load_depth_runs,
    parse_args,
    plot_cost_efficiency,
    plot_cumulative_claims,
    plot_domain_depth_heatmap,
    plot_information_density,
    plot_quality_by_depth,
    split_sections_from_markdown,
    write_analysis_json,
)


@pytest.fixture
def mock_results_dir(tmp_path):
    """Create a mock results directory with depth run data."""
    config = {
        "runs": [
            {
                "topic": "gut_microbiome",
                "domain": "biomedical",
                "depth": depth,
                "generated_path": str(tmp_path / depth / "review.md"),
                "reference_path": str(tmp_path / "ref.pdf"),
                "evaluation_path": str(tmp_path / depth / "eval.json"),
                "tier": "B",
            }
            for depth in ["low", "medium", "deep"]
        ]
    }
    (tmp_path / "depth_runs.json").write_text(json.dumps(config))

    for depth in ["low", "medium", "deep"]:
        d = tmp_path / depth
        d.mkdir(exist_ok=True)
        score = {"low": 0.65, "medium": 0.78, "deep": 0.84}[depth]
        eval_data = {
            "overall_score": score,
            "citation_score": {
                "recall": 0.5,
                "precision": 0.6,
                "f1": 0.55,
                "matched_count": 10,
                "reference_count": 20,
                "generated_count": 15,
                "matched_titles": [],
                "missed_titles": [],
                "hallucinated_titles": [],
            },
            "synthesis_score": {
                "generated_score": score * 5,
                "reference_score": 4.0,
                "delta": 0.0,
                "dimension_scores": {},
                "generated_observations": "",
                "reference_observations": "",
            },
            "topic_coverage": {
                "generated_coverage": score,
                "reference_coverage": 1.0,
                "topics_in_both": [],
                "topics_only_in_reference": [],
                "topics_only_in_generated": [],
            },
            "writing_quality": {
                "generated_score": score * 5,
                "reference_score": 4.0,
                "delta": 0.0,
                "dimension_scores": {},
            },
            "timestamp": "2026-03-17",
            "generated_path": str(d / "review.md"),
            "reference_path": str(tmp_path / "ref.pdf"),
            "structural_metrics": {
                "word_count": {"low": 4000, "medium": 8000, "deep": 25000}[depth],
                "section_count": 5,
                "citation_count": 30,
                "citations_per_1000_words": 7.5,
                "avg_section_length_words": 800,
                "section_balance": 0.2,
                "flesch_kincaid_grade": 14.0,
            },
        }
        (d / "eval.json").write_text(json.dumps(eval_data))
        (d / "review.md").write_text(
            f"## Introduction\nSome intro text.\n\n## Results\nSome results for {depth}.\n"
        )

    return tmp_path


def test_load_depth_runs(mock_results_dir):
    df = load_depth_runs(mock_results_dir)
    assert len(df) == 3
    assert set(df["depth"]) == {"low", "medium", "deep"}
    assert "overall_score" in df.columns
    assert "word_count" in df.columns


def test_split_sections_from_markdown():
    md = "## Intro\nParagraph 1.\n\n## Methods\nParagraph 2.\n\n## Results\nParagraph 3.\n"
    sections = split_sections_from_markdown(md)
    assert len(sections) == 3
    assert sections[0]["id"] == "intro"
    assert "Paragraph 1" in sections[0]["text"]


def test_compute_information_metrics():
    claims_by_depth = {
        "low": [
            Claim(text="A", category="empirical"),
            Claim(text="B", category="methodological"),
        ],
        "medium": [
            Claim(text="A", category="empirical"),
            Claim(text="B", category="methodological"),
            Claim(text="C", category="synthesis"),
        ],
        "deep": [
            Claim(text="A", category="empirical"),
            Claim(text="B", category="methodological"),
            Claim(text="C", category="synthesis"),
            Claim(text="D", category="empirical"),
            Claim(text="E", category="limitation"),
        ],
    }
    novel_claims = {
        "low_to_medium": [Claim(text="C", category="synthesis")],
        "medium_to_deep": [
            Claim(text="D", category="empirical"),
            Claim(text="E", category="limitation"),
        ],
    }
    concepts_by_depth = {
        "low": ["gene therapy", "crispr"],
        "medium": ["gene therapy", "crispr", "off-target effects"],
        "deep": ["gene therapy", "crispr", "off-target effects", "delivery vector", "aav"],
    }
    word_counts = {"low": 4000, "medium": 8000, "deep": 25000}
    citation_counts = {"low": 20, "medium": 45, "deep": 100}

    metrics = compute_information_metrics(
        claims_by_depth=claims_by_depth,
        novel_claims=novel_claims,
        concepts_by_depth=concepts_by_depth,
        word_counts=word_counts,
        citation_counts=citation_counts,
    )
    assert metrics.claims_per_depth == {"low": 2, "medium": 3, "deep": 5}
    assert metrics.new_claims_per_increment["low_to_medium"] == 1
    assert metrics.new_claims_per_increment["medium_to_deep"] == 2
    assert metrics.claim_novelty_rate["low_to_medium"] == pytest.approx(1 / 3)
    assert metrics.concepts_per_depth == {"low": 2, "medium": 3, "deep": 5}
    assert metrics.concept_growth["low_to_medium"] == 1
    assert metrics.concept_growth["medium_to_deep"] == 2
    assert metrics.claims_per_1k_words["low"] == pytest.approx(0.5)
    assert metrics.citations_per_claim["low"] == pytest.approx(10.0)
    assert metrics.concept_overlap is not None
    assert metrics.concept_overlap["low_to_medium"] == pytest.approx(1.0)  # low is subset of medium


def test_compute_pairwise_tests_requires_min_samples():
    """Pairwise tests need at least 5 paired observations."""
    df = pd.DataFrame(
        {
            "topic": ["t1", "t1"],
            "depth": ["low", "medium"],
            "overall_score": [0.5, 0.6],
        }
    )
    results = compute_pairwise_tests(df, metric_columns=["overall_score"])
    assert results["overall_score"]["low_to_medium"]["n_pairs"] < 5


@pytest.fixture
def sample_analysis_output():
    return DepthAnalysisOutput(
        summary_stats={
            "low": {"overall_score": {"mean": 0.65, "std": 0.05, "n": 20}},
            "medium": {"overall_score": {"mean": 0.78, "std": 0.04, "n": 20}},
            "deep": {"overall_score": {"mean": 0.84, "std": 0.03, "n": 20}},
        },
        pairwise_tests={
            "overall_score": {
                "low_to_medium": {
                    "statistic": 15.0,
                    "p_value": 0.001,
                    "n_pairs": 20,
                    "p_adjusted": 0.002,
                    "mean_diff": 0.13,
                },
                "medium_to_deep": {
                    "statistic": 20.0,
                    "p_value": 0.01,
                    "n_pairs": 20,
                    "p_adjusted": 0.015,
                    "mean_diff": 0.06,
                },
            },
        },
        information_metrics_per_topic={
            "gut_microbiome": InformationMetrics(
                claims_per_depth={"low": 45, "medium": 102, "deep": 230},
                new_claims_per_increment={"low_to_medium": 57, "medium_to_deep": 128},
                new_claims_by_category={
                    "low_to_medium": {"empirical": 20},
                    "medium_to_deep": {"empirical": 50},
                },
                claim_novelty_rate={"low_to_medium": 0.56, "medium_to_deep": 0.56},
                concepts_per_depth={"low": 30, "medium": 65, "deep": 120},
                concept_growth={"low_to_medium": 35, "medium_to_deep": 55},
                claims_per_1k_words={"low": 11.0, "medium": 12.0, "deep": 9.0},
                concepts_per_1k_words={"low": 7.0, "medium": 8.0, "deep": 5.0},
                citations_per_claim={"low": 1.2, "medium": 1.4, "deep": 1.1},
            ),
        },
    )


def test_write_analysis_json(sample_analysis_output, tmp_path):
    write_analysis_json(sample_analysis_output, tmp_path / "depth_analysis.json")
    assert (tmp_path / "depth_analysis.json").exists()
    with open(tmp_path / "depth_analysis.json") as f:
        data = json.load(f)
    assert "summary_stats" in data
    assert "pairwise_tests" in data
    assert "information_metrics_per_topic" in data


def test_generate_report(sample_analysis_output, tmp_path):
    generate_report(sample_analysis_output, tmp_path / "report.md")
    assert (tmp_path / "report.md").exists()
    content = (tmp_path / "report.md").read_text()
    assert "Depth Level Comparison" in content
    assert "low" in content.lower()
    assert "medium" in content.lower()
    assert "deep" in content.lower()


# ---------------------------------------------------------------------------
# Figure function tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    """DataFrame with 2 topics x 3 depths for figure testing."""
    rows = []
    for topic, domain in [("t1", "biomedical"), ("t2", "cs")]:
        for depth, score in [("low", 0.6), ("medium", 0.75), ("deep", 0.82)]:
            rows.append(
                {
                    "topic": topic,
                    "domain": domain,
                    "depth": depth,
                    "overall_score": score + (0.02 if topic == "t2" else 0),
                    "synthesis_score": score * 5,
                    "topic_coverage": score,
                    "writing_quality": score * 4.5,
                    "citation_f1": score * 0.8,
                    "word_count": {"low": 4000, "medium": 8000, "deep": 25000}[depth],
                    "citation_count": {"low": 20, "medium": 45, "deep": 100}[depth],
                }
            )
    return pd.DataFrame(rows)


def test_plot_quality_by_depth(sample_df, tmp_path):
    plot_quality_by_depth(sample_df, tmp_path / "quality.pdf")
    assert (tmp_path / "quality.pdf").exists()


def test_plot_cumulative_claims(tmp_path):
    claims_data = {
        "low": {
            "empirical": 10,
            "methodological": 5,
            "contextual": 8,
            "synthesis": 3,
            "limitation": 2,
        },
        "medium": {
            "empirical": 25,
            "methodological": 12,
            "contextual": 15,
            "synthesis": 8,
            "limitation": 5,
        },
        "deep": {
            "empirical": 55,
            "methodological": 28,
            "contextual": 30,
            "synthesis": 18,
            "limitation": 12,
        },
    }
    plot_cumulative_claims(claims_data, tmp_path / "claims.pdf")
    assert (tmp_path / "claims.pdf").exists()


def test_plot_information_density(sample_df, tmp_path):
    density_data = []
    for _, row in sample_df.iterrows():
        density_data.append(
            {
                "topic": row["topic"],
                "domain": row["domain"],
                "depth": row["depth"],
                "word_count": row["word_count"],
                "claims_per_1k_words": 12.0 if row["depth"] != "deep" else 8.5,
            }
        )
    density_df = pd.DataFrame(density_data)
    plot_information_density(density_df, tmp_path / "density.pdf")
    assert (tmp_path / "density.pdf").exists()


def test_plot_cost_efficiency(tmp_path):
    cost_data = pd.DataFrame(
        {
            "depth": ["low", "medium", "deep"],
            "mean_score": [0.6, 0.75, 0.82],
            "mean_cost": [1.50, 3.20, 8.75],
            "claims_per_dollar": [30, 32, 26],
        }
    )
    plot_cost_efficiency(cost_data, tmp_path / "cost.pdf")
    assert (tmp_path / "cost.pdf").exists()


def test_plot_domain_depth_heatmap(sample_df, tmp_path):
    plot_domain_depth_heatmap(sample_df, "overall_score", tmp_path / "heatmap.pdf")
    assert (tmp_path / "heatmap.pdf").exists()


# ---------------------------------------------------------------------------
# CLI parse_args tests
# ---------------------------------------------------------------------------


def test_parse_args_defaults():
    args = parse_args(["--results-dir", "/tmp/results"])
    assert str(args.results_dir) == "/tmp/results"
    assert str(args.output_dir) == "paper/output/depth_comparison"


def test_parse_args_custom_output():
    args = parse_args(["--results-dir", "/tmp/results", "--output-dir", "/tmp/out"])
    assert str(args.output_dir) == "/tmp/out"


def test_parse_args_skip_extraction():
    args = parse_args(["--results-dir", "/tmp/r", "--skip-extraction"])
    assert args.skip_extraction is True
