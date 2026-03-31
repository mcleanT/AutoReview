"""Tests for factual accuracy scoring functions."""

from autoreview.extraction.models import EvidenceStrength, Finding, PaperExtraction, StudyDesign
from autoreview.extraction.scoring_factual import (
    _detect_limitation_topics,
    _extract_facts,
    _extract_numbers,
    _fact_overlap,
    _factual_key_findings_score,
    _factual_limitations_score,
    _factual_methods_score,
    _factual_quant_score,
    score_extraction_pair_factual,
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


# --- New tests for Task 2 ---


class TestDetectLimitationTopics:
    def test_sample_size_detected(self):
        topics = _detect_limitation_topics("The study has a small sample size.")
        assert "sample_size" in topics

    def test_generalizability_detected(self):
        topics = _detect_limitation_topics("Results may not generaliz to other populations.")
        assert "generalizability" in topics

    def test_multiple_topics(self):
        text = "small sample and results may not generaliz beyond this dataset"
        topics = _detect_limitation_topics(text)
        assert "sample_size" in topics
        assert "generalizability" in topics

    def test_no_topics(self):
        # Neutral statement with no limitation keywords
        topics = _detect_limitation_topics("The model performs well on all tasks.")
        assert len(topics) == 0

    def test_empty_text(self):
        assert _detect_limitation_topics("") == set()

    def test_none_text(self):
        assert _detect_limitation_topics(None) == set()

    def test_case_insensitive(self):
        topics = _detect_limitation_topics("SMALL SAMPLE SIZE in this study")
        assert "sample_size" in topics

    def test_hallucination_topic(self):
        topics = _detect_limitation_topics("The system may produce hallucination errors.")
        assert "hallucination" in topics


class TestFactualKeyFindingsScore:
    def test_exact_match(self):
        # Same claims → perfect score
        claims = ["GPT-4 achieves 90% on MedQA"]
        score = _factual_key_findings_score(claims, claims)
        assert score == 1.0

    def test_different_wording_same_facts(self):
        # Same facts (GPT-4, 90, MedQA) in different wording
        pred = ["GPT-4 scores 90% accuracy on MedQA benchmark"]
        gold = ["GPT-4 achieves 90% on MedQA"]
        score = _factual_key_findings_score(pred, gold)
        assert score > 0.5

    def test_missing_facts(self):
        # Pred mentions different model and number
        pred = ["BERT achieves 75% on SQuAD"]
        gold = ["GPT-4 achieves 90% on MedQA"]
        score = _factual_key_findings_score(pred, gold)
        assert score < 1.0

    def test_empty_gold(self):
        score = _factual_key_findings_score(["some claim"], [])
        assert score == 1.0

    def test_empty_pred(self):
        score = _factual_key_findings_score([], ["some claim"])
        assert score == 0.0

    def test_multiple_gold_claims(self):
        pred = ["GPT-4 achieves 90% on MedQA", "BERT scores 75% on SQuAD"]
        gold = ["GPT-4 achieves 90% on MedQA", "BERT scores 75% on SQuAD"]
        score = _factual_key_findings_score(pred, gold)
        assert score == 1.0

    def test_partial_gold_coverage(self):
        # pred covers first gold claim but not second
        pred = ["GPT-4 achieves 90% on MedQA"]
        gold = ["GPT-4 achieves 90% on MedQA", "BERT scores 75% on SQuAD"]
        score = _factual_key_findings_score(pred, gold)
        # First claim matches (score ~1.0), second does not (~0.0) → ~0.5
        assert 0.3 < score < 0.8


class TestFactualQuantScore:
    def test_matching_numbers(self):
        score = _factual_quant_score("90% accuracy", "achieves 90%")
        assert score == 1.0

    def test_no_match(self):
        score = _factual_quant_score("75% accuracy", "achieves 90%")
        assert score == 0.0

    def test_both_empty(self):
        score = _factual_quant_score(None, None)
        assert score == 1.0

    def test_gold_empty(self):
        # Gold has no numbers → score 1.0
        score = _factual_quant_score("90% accuracy", "no numbers here")
        assert score == 1.0

    def test_pred_empty(self):
        # Gold has numbers but pred is empty → score 0.0
        score = _factual_quant_score(None, "achieves 90%")
        assert score == 0.0

    def test_partial_match(self):
        # Gold has one number (90), pred has {90, 75} → 1/1 = 1.0
        score = _factual_quant_score("achieves 90% and 75%", "90% on MedQA")
        assert score == 1.0  # Gold has {90}, pred has {90, 75} → 1/1 = 1.0

    def test_multiple_gold_numbers_partial(self):
        # Gold has two numbers (90, 85), pred only has {85} → 1/2 = 0.5
        score = _factual_quant_score("accuracy: 85%", "achieved 90% and 85%")
        assert score == 0.5  # Gold has {90, 85}, pred has {85} → 1/2 = 0.5


class TestFactualMethodsScore:
    def test_keyword_coverage(self):
        pred = "fine-tuned BERT on SQuAD with Adam optimizer"
        gold = "fine-tuned BERT on SQuAD dataset"
        score = _factual_methods_score(pred, gold)
        assert score > 0.5

    def test_no_overlap(self):
        pred = "trained ResNet on ImageNet"
        gold = "fine-tuned BERT on SQuAD"
        score = _factual_methods_score(pred, gold)
        # BERT and SQuAD are in gold, not in pred (different proper nouns)
        assert score < 1.0

    def test_empty_gold(self):
        score = _factual_methods_score("some methods", None)
        assert score == 1.0

    def test_empty_pred(self):
        score = _factual_methods_score(None, "BERT fine-tuned on SQuAD")
        assert score == 0.0


class TestFactualLimitationsScore:
    def test_topic_match(self):
        pred = "The model has a small sample and may not generaliz to other domains."
        gold = "small sample size and limited generalizability"
        score = _factual_limitations_score(pred, gold)
        assert score == 1.0

    def test_no_match(self):
        pred = "The model performs well in most scenarios."
        gold = "small sample size"
        score = _factual_limitations_score(pred, gold)
        assert score == 0.0

    def test_empty_gold(self):
        score = _factual_limitations_score("something", None)
        assert score == 1.0

    def test_empty_pred(self):
        score = _factual_limitations_score(None, "small sample size")
        assert score == 0.0

    def test_partial_match(self):
        # Gold has two topics, pred only covers one
        pred = "small sample size is a concern"
        gold = "small sample size and hallucination risk"
        score = _factual_limitations_score(pred, gold)
        assert 0.3 < score < 0.8


# --- Helper to build PaperExtraction objects for integration tests ---


def _make_finding(
    claim: str,
    quant: str | None = None,
    strength: EvidenceStrength = EvidenceStrength.MODERATE,
    paper_id: str = "paper_1",
) -> Finding:
    return Finding(
        claim=claim,
        evidence_strength=strength,
        quantitative_result=quant,
        paper_id=paper_id,
    )


def _make_extraction(
    paper_id: str,
    findings: list[Finding] | None = None,
    methods: str = "standard methods",
    limitations: str = "some limitations",
    study_design: StudyDesign | None = StudyDesign.COMPUTATIONAL,
    quality_score: float | None = 0.8,
    sample_size: int | None = 100,
) -> PaperExtraction:
    return PaperExtraction(
        paper_id=paper_id,
        key_findings=findings or [],
        methods_summary=methods,
        limitations=limitations,
        study_design=study_design,
        quality_score=quality_score,
        sample_size=sample_size,
    )


class TestScoreExtractionPairFactual:
    def test_identical_extractions(self):
        finding = _make_finding("GPT-4 achieves 90% on MedQA", quant="90%")
        pred = _make_extraction(
            "pred",
            findings=[finding],
            methods="fine-tuned BERT on SQuAD",
            limitations="small sample size",
        )
        gold = _make_extraction(
            "gold",
            findings=[_make_finding("GPT-4 achieves 90% on MedQA", quant="90%", paper_id="gold")],
            methods="fine-tuned BERT on SQuAD",
            limitations="small sample size",
        )
        sim_scores = {
            "evidence_strength": 1.0,
            "study_design": 1.0,
            "quality_score": 1.0,
            "sample_size": 1.0,
        }
        scores = score_extraction_pair_factual(pred, gold, sim_scores)

        assert scores["key_findings_factual"] == 1.0
        assert scores["quantitative_result_factual"] == 1.0
        assert scores["methods_summary_factual"] == 1.0
        assert scores["limitations_factual"] == 1.0

    def test_pass_through_fields(self):
        pred = _make_extraction("pred")
        gold = _make_extraction("gold")
        sim_scores = {
            "evidence_strength": 0.9,
            "study_design": 0.8,
            "quality_score": 0.7,
            "sample_size": 0.6,
        }
        scores = score_extraction_pair_factual(pred, gold, sim_scores)

        assert scores["evidence_strength"] == 0.9
        assert scores["study_design"] == 0.8
        assert scores["quality_score"] == 0.7
        assert scores["sample_size"] == 0.6

    def test_missing_similarity_fields_not_included(self):
        pred = _make_extraction("pred")
        gold = _make_extraction("gold")
        scores = score_extraction_pair_factual(pred, gold, {})

        assert "evidence_strength" not in scores
        assert "study_design" not in scores

    def test_empty_findings(self):
        pred = _make_extraction("pred", findings=[])
        gold = _make_extraction("gold", findings=[])
        scores = score_extraction_pair_factual(pred, gold, {})

        assert scores["key_findings_factual"] == 1.0
        assert scores["quantitative_result_factual"] == 1.0

    def test_wrong_methods(self):
        pred = _make_extraction("pred", methods="trained ResNet on ImageNet")
        gold = _make_extraction("gold", methods="fine-tuned BERT on SQuAD")
        scores = score_extraction_pair_factual(pred, gold, {})

        # BERT and SQuAD appear in gold but not pred
        assert scores["methods_summary_factual"] < 1.0

    def test_wrong_limitations(self):
        pred = _make_extraction("pred", limitations="The model performs well.")
        gold = _make_extraction("gold", limitations="small sample size")
        scores = score_extraction_pair_factual(pred, gold, {})

        assert scores["limitations_factual"] == 0.0

    def test_returns_all_four_factual_keys(self):
        pred = _make_extraction("pred")
        gold = _make_extraction("gold")
        scores = score_extraction_pair_factual(pred, gold, {})

        for key in (
            "key_findings_factual",
            "quantitative_result_factual",
            "methods_summary_factual",
            "limitations_factual",
        ):
            assert key in scores, f"Missing key: {key}"


import pytest  # noqa: E402

from autoreview.extraction.scoring import compute_dual_composite  # noqa: E402


class TestDualComposite:
    def test_equal_weight(self):
        sim = {
            "key_findings": 0.8,
            "evidence_strength": 0.9,
            "quantitative_result": 0.3,
            "methods_summary": 0.7,
            "limitations": 0.6,
            "study_design": 0.9,
            "quality_score": 0.9,
            "sample_size": 0.5,
        }
        fact = {
            "key_findings": 0.9,
            "evidence_strength": 0.9,
            "quantitative_result": 0.7,
            "methods_summary": 0.8,
            "limitations": 0.7,
            "study_design": 0.9,
            "quality_score": 0.9,
            "sample_size": 0.5,
        }
        result = compute_dual_composite(sim, fact, alpha=0.5)
        assert "similarity" in result
        assert "factual" in result
        assert "combined" in result
        assert result["combined"] == pytest.approx(
            0.5 * result["similarity"] + 0.5 * result["factual"], abs=1e-6
        )

    def test_alpha_zero_is_factual_only(self):
        sim = {
            f: 0.5
            for f in [
                "key_findings",
                "evidence_strength",
                "quantitative_result",
                "methods_summary",
                "limitations",
                "study_design",
                "quality_score",
                "sample_size",
            ]
        }
        fact = {f: 1.0 for f in sim}
        result = compute_dual_composite(sim, fact, alpha=0.0)
        assert result["combined"] == pytest.approx(result["factual"], abs=1e-6)

    def test_alpha_one_is_similarity_only(self):
        sim = {
            f: 0.8
            for f in [
                "key_findings",
                "evidence_strength",
                "quantitative_result",
                "methods_summary",
                "limitations",
                "study_design",
                "quality_score",
                "sample_size",
            ]
        }
        fact = {f: 0.2 for f in sim}
        result = compute_dual_composite(sim, fact, alpha=1.0)
        assert result["combined"] == pytest.approx(result["similarity"], abs=1e-6)
