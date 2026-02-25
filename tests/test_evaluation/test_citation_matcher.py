from autoreview.evaluation.citation_matcher import (
    parse_bibliography_from_markdown,
    match_citations,
)


def test_parse_bibliography_from_markdown():
    md = (
        "Some text.\n\n"
        "## References\n"
        "[1] Smith J. Senescence in aging. Nature, 2020.\n"
        "[2] Jones K. SASP mechanisms. Cell, 2021.\n"
    )
    refs = parse_bibliography_from_markdown(md)
    assert len(refs) == 2
    assert any("Smith" in r for r in refs)


def test_match_citations_exact():
    gen = ["Smith J. Senescence in aging. Nature 2020.", "Jones K. SASP. Cell 2021."]
    ref = ["Smith J. Senescence in aging. Nature 2020.", "Jones K. SASP. Cell 2021.", "Lee M. Paper. PNAS 2019."]
    score = match_citations(gen, ref)
    assert score.matched_count == 2
    assert score.reference_count == 3
    assert abs(score.recall - 2 / 3) < 0.01


def test_match_citations_fuzzy():
    gen = ["Smith J et al. Senescence in aging and disease. Nature Reviews 2020."]
    ref = ["Smith J, Brown A. Senescence in aging and disease. Nat Rev Aging, 2020."]
    score = match_citations(gen, ref)
    assert score.matched_count == 1


def test_match_citations_no_overlap():
    gen = ["Author A. Paper One. J1 2020."]
    ref = ["Author B. Paper Two. J2 2021.", "Author C. Paper Three. J3 2022."]
    score = match_citations(gen, ref)
    assert score.matched_count == 0
    assert score.recall == 0.0
