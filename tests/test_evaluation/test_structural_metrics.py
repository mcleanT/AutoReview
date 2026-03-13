from __future__ import annotations

from autoreview.evaluation.structural_metrics import compute_structural_metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MULTI_SECTION_MD = """\
# Introduction

This is the introduction section with several words to count here.
It has two sentences and some more filler text to pad the word count nicely.

## Methods

We used advanced computational methods.
The methods section is shorter.

## Results

Results are presented here. We found significant effects [1] [2] [3].
(Smith et al., 2023) also reported similar findings.

## Conclusion

In conclusion, the study demonstrates important insights.
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic_metrics() -> None:
    result = compute_structural_metrics(_MULTI_SECTION_MD)

    # Word count should be positive
    assert result.word_count > 0

    # Four headings: #, ##, ##, ##
    assert result.section_count == 4

    # Citations: [1] [2] [3] (three unique bracket cites) + (Smith et al., 2023) = 4
    assert result.citation_count == 4

    # Derived fields are non-negative
    assert result.citations_per_1000_words >= 0.0
    assert result.avg_section_length_words > 0.0
    assert result.section_balance >= 0.0
    assert isinstance(result.flesch_kincaid_grade, float)


def test_empty_text() -> None:
    result = compute_structural_metrics("")

    assert result.word_count == 0
    assert result.section_count == 0
    assert result.citation_count == 0
    assert result.citations_per_1000_words == 0.0
    assert result.avg_section_length_words == 0.0
    assert result.section_balance == 0.0
    assert result.flesch_kincaid_grade == 0.0


def test_whitespace_only_text() -> None:
    result = compute_structural_metrics("   \n\t  \n  ")

    assert result.word_count == 0
    assert result.section_count == 0


def test_single_section_balance_is_zero() -> None:
    md = "## Only Section\n\nThis is the only section with some words in it.\n"
    result = compute_structural_metrics(md)

    assert result.section_count == 1
    assert result.section_balance == 0.0


def test_citation_counting_bracket_only() -> None:
    md = "Some text [1] more text [2] repeated [1] again [3].\n"
    result = compute_structural_metrics(md)

    # Unique bracket cites: [1], [2], [3]
    assert result.citation_count == 3


def test_citation_counting_author_only() -> None:
    md = (
        "As shown by (Smith et al., 2020) and (Jones et al. 2021) and again (Smith et al., 2020).\n"
    )
    result = compute_structural_metrics(md)

    # Unique author cites: (Smith et al., 2020), (Jones et al. 2021)
    assert result.citation_count == 2


def test_citation_counting_mixed() -> None:
    md = "See [1] and [2] and (Smith et al., 2019).\n"
    result = compute_structural_metrics(md)

    # [1], [2] = 2 bracket; (Smith et al., 2019) = 1 author → total 3
    assert result.citation_count == 3


def test_citations_per_1000_words_known_ratio() -> None:
    # Construct text with exactly 1000 words and 5 unique citations.
    body = ("word " * 996).strip()
    md = f"## Section\n\n{body} [1] [2] [3] [4] [5]\n"
    result = compute_structural_metrics(md)

    assert result.citation_count == 5
    # citations_per_1000_words should be close to 5.0
    assert (
        abs(result.citations_per_1000_words - result.citation_count / (result.word_count / 1000))
        < 0.01
    )


def test_flesch_kincaid_simple_vs_complex() -> None:
    simple_md = "## Title\n\nThe cat sat on the mat. The dog ran fast.\n"
    complex_md = (
        "## Title\n\n"
        "The utilization of sophisticated computational methodologies necessitates "
        "comprehensive understanding of multidimensional analytical frameworks, "
        "particularly within the context of high-dimensional genomic investigations "
        "encompassing transcriptomic and proteomic data integration.\n"
    )

    simple_result = compute_structural_metrics(simple_md)
    complex_result = compute_structural_metrics(complex_md)

    # Complex text should have a higher (harder) Flesch-Kincaid grade
    assert complex_result.flesch_kincaid_grade > simple_result.flesch_kincaid_grade


def test_no_headings_treated_as_one_section() -> None:
    md = "Just some plain text without any headings at all. Lots of words here.\n"
    result = compute_structural_metrics(md)

    assert result.section_count == 1
    assert result.section_balance == 0.0


def test_markdown_stripped_before_word_count() -> None:
    # Bold markers and link syntax should not inflate word count
    md = "## Section\n\n**Bold text** and [link](http://example.com).\n"
    result = compute_structural_metrics(md)

    # Plain text would be: "Section Bold text and link."
    # (heading text "Section" remains after stripping the ## marker)
    assert result.word_count == 5
