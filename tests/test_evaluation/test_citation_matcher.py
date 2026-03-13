from autoreview.evaluation.citation_matcher import (
    match_citations,
    parse_bibliography_from_markdown,
)

# ---------------------------------------------------------------------------
# Existing tests (kept intact)
# ---------------------------------------------------------------------------


def test_parse_bibliography_from_markdown() -> None:
    md = (
        "Some text.\n\n"
        "## References\n"
        "[1] Smith J. Senescence in aging. Nature, 2020.\n"
        "[2] Jones K. SASP mechanisms. Cell, 2021.\n"
    )
    refs = parse_bibliography_from_markdown(md)
    assert len(refs) == 2
    assert any("Smith" in r for r in refs)


def test_match_citations_exact() -> None:
    gen = ["Smith J. Senescence in aging. Nature 2020.", "Jones K. SASP. Cell 2021."]
    ref = [
        "Smith J. Senescence in aging. Nature 2020.",
        "Jones K. SASP. Cell 2021.",
        "Lee M. Paper. PNAS 2019.",
    ]
    score = match_citations(gen, ref)
    assert score.matched_count == 2
    assert score.reference_count == 3
    assert abs(score.recall - 2 / 3) < 0.01


def test_match_citations_fuzzy() -> None:
    gen = ["Smith J et al. Senescence in aging and disease. Nature Reviews 2020."]
    ref = ["Smith J, Brown A. Senescence in aging and disease. Nat Rev Aging, 2020."]
    score = match_citations(gen, ref)
    assert score.matched_count == 1


def test_match_citations_no_overlap() -> None:
    gen = ["Author A. Paper One. J1 2020."]
    ref = ["Author B. Paper Two. J2 2021.", "Author C. Paper Three. J3 2022."]
    score = match_citations(gen, ref)
    assert score.matched_count == 0
    assert score.recall == 0.0


# ---------------------------------------------------------------------------
# New tests
# ---------------------------------------------------------------------------


def test_doi_matching() -> None:
    """Refs with matching DOIs are matched even when titles differ substantially."""
    doi = "10.1038/s41586-020-00001-x"
    gen = [f"Smith J. Completely Different Title Text. Nature 2020. doi:{doi}"]
    ref = [f"Brown A. Entirely Other Heading Words Here. Cell 2019. doi:{doi}"]
    score = match_citations(gen, ref)
    # DOI match → forward pass should match the single reference ref
    assert score.matched_count == 1
    assert score.recall == 1.0
    # Reverse pass: generated ref also matches → precision = 1.0
    assert score.precision == 1.0
    assert score.f1 == 1.0


def test_rapidfuzz_matching() -> None:
    """Refs with minor typos / journal abbreviations are matched via fuzzy."""
    gen = ["Jones K. Cellular senescense and the SASP pathway in inflamation. Cell 2021."]
    ref = ["Jones K. Cellular senescence and the SASP pathway in inflammation. Cell 2021."]
    score = match_citations(gen, ref)
    # The titles are very similar; token_sort_ratio should be ≥ 80
    assert score.matched_count == 1
    assert score.recall == 1.0


def test_precision_calculation() -> None:
    """Precision = fraction of generated refs that match any reference ref."""
    gen = [
        "Smith J. Senescence in aging. Nature 2020.",  # matches
        "Fake Author. Made up paper. Journal X 2099.",  # no match
    ]
    ref = [
        "Smith J. Senescence in aging. Nature 2020.",
        "Lee M. SASP review. Cell 2021.",
    ]
    score = match_citations(gen, ref)
    # 1 out of 2 generated refs matches → precision = 0.5
    assert abs(score.precision - 0.5) < 0.01
    assert score.generated_count == 2


def test_f1_calculation() -> None:
    """F1 is harmonic mean of precision and recall."""
    gen = [
        "Smith J. Senescence in aging. Nature 2020.",  # matches ref[0]
        "Fake Author. Made up paper. Journal X 2099.",  # hallucinated
    ]
    ref = [
        "Smith J. Senescence in aging. Nature 2020.",  # matched
        "Lee M. SASP review. Cell 2021.",  # missed
    ]
    score = match_citations(gen, ref)
    expected_precision = 1 / 2
    expected_recall = 1 / 2
    expected_f1 = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)
    assert abs(score.precision - expected_precision) < 0.01
    assert abs(score.recall - expected_recall) < 0.01
    assert abs(score.f1 - expected_f1) < 0.01


def test_hallucinated_titles() -> None:
    """Generated refs with no match in the reference set are flagged as hallucinated."""
    gen = [
        "Smith J. Senescence in aging. Nature 2020.",  # real
        "Fictional A. Nonexistent study on dragons. Fantasy J 2077.",  # hallucinated
        "Ghost B. Imaginary results. Made Up Letters 2050.",  # hallucinated
    ]
    ref = [
        "Smith J. Senescence in aging. Nature 2020.",
    ]
    score = match_citations(gen, ref)
    assert len(score.hallucinated_titles) == 2
    assert any("dragon" in t.lower() for t in score.hallucinated_titles)
    assert any("imaginary" in t.lower() for t in score.hallucinated_titles)


def test_bidirectional_matching() -> None:
    """Mixed scenario: overlapping, hallucinated, and missed refs all handled."""
    gen = [
        "Smith J. Senescence in aging. Nature 2020.",  # matches ref[0]
        "Jones K. SASP mechanisms. Cell 2021.",  # matches ref[1]
        "Hallucination X. Never published anywhere. Journal Z 2099.",  # hallucinated
    ]
    ref = [
        "Smith J. Senescence in aging. Nature 2020.",  # matched
        "Jones K. SASP mechanisms. Cell 2021.",  # matched
        "Lee M. Missing paper. PNAS 2019.",  # missed
    ]
    score = match_citations(gen, ref)

    # Forward pass: 2 of 3 reference refs matched
    assert score.matched_count == 2
    assert abs(score.recall - 2 / 3) < 0.01
    assert len(score.missed_titles) == 1
    assert "Lee" in score.missed_titles[0]

    # Reverse pass: 2 of 3 generated refs matched → precision = 2/3
    assert abs(score.precision - 2 / 3) < 0.01
    assert len(score.hallucinated_titles) == 1
    assert "Hallucination" in score.hallucinated_titles[0]

    # F1 should be 2/3 (since precision == recall == 2/3)
    assert abs(score.f1 - 2 / 3) < 0.01
