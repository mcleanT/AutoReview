from autoreview.evaluation.pdf_extractor import (
    extract_bibliography_lines,
    normalize_title_for_matching,
)


def test_extract_bibliography_lines_bracket_format():
    text = (
        "Introduction text.\n\n"
        "References\n"
        "[1] Smith J. Senescence in aging. Nature, 2020.\n"
        "[2] Jones K. SASP mechanisms. Cell, 2021.\n"
    )
    lines = extract_bibliography_lines(text)
    assert len(lines) == 2
    assert "Smith J" in lines[0]


def test_extract_bibliography_lines_numbered_dot_format():
    text = "References\n1. Smith J. Title. Journal, 2020.\n2. Jones K. Another. Cell, 2021."
    lines = extract_bibliography_lines(text)
    assert len(lines) == 2


def test_extract_bibliography_lines_no_section():
    lines = extract_bibliography_lines("No references here.")
    assert lines == []


def test_normalize_title_for_matching_case_insensitive():
    t1 = normalize_title_for_matching("The Role of p16INK4a in Cellular Senescence")
    t2 = normalize_title_for_matching("role p16ink4a cellular senescence")
    assert t1 == t2


def test_normalize_title_strips_stopwords():
    t = normalize_title_for_matching("The mechanisms of a cellular aging")
    tokens = t.split()
    assert "the" not in tokens
    assert "a" not in tokens
    assert "of" not in tokens
