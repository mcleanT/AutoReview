from __future__ import annotations


def test_analyze_structural_profile_returns_expected_keys():
    from autoreview.analysis.reference_analyzer import ReferenceAnalyzer

    # Create a minimal mock PDF text
    analyzer = ReferenceAnalyzer()
    profile = analyzer.analyze_text(
        "Abstract\nSome words here.\n\n1. Introduction\nMore text.\n\n"
        "References\n[1] Author A. Title. 2020.\n[2] Author B. Title. 2021.\n"
    )
    assert "word_count" in profile
    assert "citation_count" in profile
    assert "citation_density" in profile
    assert "section_count" in profile
    assert "citation_style" in profile


def test_analyze_text_counts_references():
    from autoreview.analysis.reference_analyzer import ReferenceAnalyzer

    analyzer = ReferenceAnalyzer()
    text = "Some body text [1] and [2].\n\nReferences\n[1] Paper A.\n[2] Paper B.\n[3] Paper C.\n"
    profile = analyzer.analyze_text(text)
    assert profile["citation_count"] == 3


def test_analyze_text_detects_numbered_style():
    from autoreview.analysis.reference_analyzer import ReferenceAnalyzer

    analyzer = ReferenceAnalyzer()
    text = "Body [1] text [2].\n\nReferences\n[1] A.\n[2] B.\n"
    profile = analyzer.analyze_text(text)
    assert profile["citation_style"] == "numbered"


def test_analyze_text_detects_author_year_style():
    from autoreview.analysis.reference_analyzer import ReferenceAnalyzer

    analyzer = ReferenceAnalyzer()
    text = "Body (Smith et al., 2020) text (Jones, 2021).\n\nReferences\nSmith 2020.\nJones 2021.\n"
    profile = analyzer.analyze_text(text)
    assert profile["citation_style"] == "author-year"


def test_analyze_text_empty_references_returns_none_count():
    from autoreview.analysis.reference_analyzer import ReferenceAnalyzer

    analyzer = ReferenceAnalyzer()
    text = "Body text without references section."
    profile = analyzer.analyze_text(text)
    assert profile["citation_count"] is None
