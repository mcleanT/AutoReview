"""Tests for factual accuracy scoring functions."""

from autoreview.extraction.scoring_factual import (
    _extract_facts,
    _extract_numbers,
    _fact_overlap,
)


class TestExtractNumbers:
    def test_percentages(self):
        assert _extract_numbers("achieves 67.6% accuracy") == {67.6}

    def test_multiple_numbers(self):
        nums = _extract_numbers("from 61.9% to 92.6%, comparable to 92.9%")
        assert nums == {61.9, 92.6, 92.9}

    def test_integers(self):
        assert _extract_numbers("540B parameters and 3173 questions") == {540.0, 3173.0}

    def test_no_numbers(self):
        assert _extract_numbers("no quantitative data here") == set()

    def test_empty_string(self):
        assert _extract_numbers("") == set()

    def test_none_input(self):
        assert _extract_numbers(None) == set()


class TestExtractFacts:
    def test_proper_nouns(self):
        facts = _extract_facts("Flan-PaLM achieves 67.6% on MedQA")
        assert "MedQA" in facts
        assert "67.6" in facts

    def test_filters_common_words(self):
        facts = _extract_facts("The model However performs well")
        assert "The" not in facts
        assert "However" not in facts

    def test_empty(self):
        assert _extract_facts("") == set()


class TestFactOverlap:
    def test_full_overlap(self):
        assert _fact_overlap({"A", "B"}, {"A", "B", "C"}) == 1.0

    def test_partial_overlap(self):
        assert _fact_overlap({"A", "B"}, {"A", "C"}) == 0.5

    def test_no_overlap(self):
        assert _fact_overlap({"A", "B"}, {"C", "D"}) == 0.0

    def test_empty_gold(self):
        assert _fact_overlap(set(), {"A"}) == 1.0

    def test_empty_pred(self):
        assert _fact_overlap({"A"}, set()) == 0.0
