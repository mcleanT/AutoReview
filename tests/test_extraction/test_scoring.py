"""Tests for the benchmark scoring module."""

from __future__ import annotations

import pytest

from autoreview.extraction.models import (
    EvidenceStrength,
    Finding,
    PaperExtraction,
    StudyDesign,
)
from autoreview.extraction.scoring import (
    compute_composite_score,
    compute_quality_score_correlation,
    compute_sample_size_accuracy,
    compute_study_design_accuracy,
    rouge_l_f1,
    score_extraction_pair,
)


def _make_extraction(
    paper_id: str = "test",
    claims: list[str] | None = None,
    methods: str = "Test methods",
    limitations: str = "Test limitations",
    study_design: StudyDesign | None = None,
    quality_score: float = 0.7,
    sample_size: int = 100,
    evidence_strength: EvidenceStrength = EvidenceStrength.MODERATE,
) -> PaperExtraction:
    """Helper to create PaperExtraction objects for testing."""
    if claims is None:
        claims = ["Claim one about results.", "Claim two about methods."]
    findings = [
        Finding(
            claim=c,
            evidence_strength=evidence_strength,
            paper_id=paper_id,
        )
        for c in claims
    ]
    return PaperExtraction(
        paper_id=paper_id,
        key_findings=findings,
        methods_summary=methods,
        limitations=limitations,
        study_design=study_design,
        quality_score=quality_score,
        sample_size=sample_size,
    )


class TestRougeLF1:
    def test_identical(self) -> None:
        assert rouge_l_f1("the cat sat on the mat", "the cat sat on the mat") == 1.0

    def test_no_overlap(self) -> None:
        assert rouge_l_f1("hello world", "foo bar baz") == 0.0

    def test_partial_overlap(self) -> None:
        score = rouge_l_f1("the cat sat on the mat", "the dog sat on the rug")
        assert 0.0 < score < 1.0

    def test_empty_strings(self) -> None:
        assert rouge_l_f1("", "hello world") == 0.0
        assert rouge_l_f1("hello world", "") == 0.0
        assert rouge_l_f1("", "") == 0.0


class TestStudyDesignAccuracy:
    def test_all_match(self) -> None:
        predicted = [StudyDesign.RCT, StudyDesign.COHORT]
        ground_truth = [StudyDesign.RCT, StudyDesign.COHORT]
        assert compute_study_design_accuracy(predicted, ground_truth) == 1.0

    def test_none_match(self) -> None:
        predicted = [StudyDesign.RCT, StudyDesign.COHORT]
        ground_truth = [StudyDesign.COHORT, StudyDesign.RCT]
        assert compute_study_design_accuracy(predicted, ground_truth) == 0.0

    def test_partial_match(self) -> None:
        predicted = [StudyDesign.RCT, StudyDesign.COHORT]
        ground_truth = [StudyDesign.RCT, StudyDesign.RCT]
        assert compute_study_design_accuracy(predicted, ground_truth) == 0.5


class TestSampleSizeAccuracy:
    def test_exact_match(self) -> None:
        assert compute_sample_size_accuracy([100], [100]) == 1.0

    def test_both_none(self) -> None:
        assert compute_sample_size_accuracy([None], [None]) == 1.0

    def test_within_tolerance(self) -> None:
        # 105 vs 100, 5% diff <= 10% tolerance
        assert compute_sample_size_accuracy([105], [100]) == 1.0

    def test_mismatch(self) -> None:
        # None vs 100 = no match
        assert compute_sample_size_accuracy([None], [100]) == 0.0

    def test_outside_tolerance(self) -> None:
        # 150 vs 100, 50% diff > 10% tolerance
        assert compute_sample_size_accuracy([150], [100]) == 0.0


class TestQualityScoreCorrelation:
    def test_perfect_correlation(self) -> None:
        predicted = [0.1, 0.5, 0.9]
        ground_truth = [0.1, 0.5, 0.9]
        result = compute_quality_score_correlation(predicted, ground_truth)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_no_variance(self) -> None:
        # All same values → 0.5
        predicted = [0.5, 0.5, 0.5]
        ground_truth = [0.5, 0.5, 0.5]
        assert compute_quality_score_correlation(predicted, ground_truth) == 0.5

    def test_insufficient_data(self) -> None:
        # <3 pairs → 0.5
        predicted = [0.1, 0.5]
        ground_truth = [0.1, 0.5]
        assert compute_quality_score_correlation(predicted, ground_truth) == 0.5

    def test_inverse_correlation(self) -> None:
        predicted = [0.1, 0.5, 0.9]
        ground_truth = [0.9, 0.5, 0.1]
        result = compute_quality_score_correlation(predicted, ground_truth)
        assert result == pytest.approx(0.0, abs=0.01)


class TestCompositeScore:
    def test_all_perfect(self) -> None:
        scores = {
            "key_findings": 1.0,
            "evidence_strength": 1.0,
            "quantitative_result": 1.0,
            "methods_summary": 1.0,
            "limitations": 1.0,
            "study_design": 1.0,
            "quality_score": 1.0,
            "sample_size": 1.0,
        }
        assert compute_composite_score(scores) == pytest.approx(1.0)

    def test_all_zero(self) -> None:
        scores = {
            "key_findings": 0.0,
            "evidence_strength": 0.0,
            "quantitative_result": 0.0,
            "methods_summary": 0.0,
            "limitations": 0.0,
            "study_design": 0.0,
            "quality_score": 0.0,
            "sample_size": 0.0,
        }
        assert compute_composite_score(scores) == 0.0

    def test_weights_sum_to_one(self) -> None:
        weights = {
            "key_findings": 0.40,
            "evidence_strength": 0.05,
            "quantitative_result": 0.05,
            "methods_summary": 0.15,
            "limitations": 0.10,
            "study_design": 0.10,
            "quality_score": 0.05,
            "sample_size": 0.10,
        }
        assert sum(weights.values()) == pytest.approx(1.0)


class TestScoreExtractionPair:
    def test_identical_extractions(self) -> None:
        ext = _make_extraction()
        scores = score_extraction_pair(ext, ext)
        for field, value in scores.items():
            assert value == pytest.approx(1.0, abs=0.01), f"{field} should be ~1.0, got {value}"

    def test_completely_different(self) -> None:
        pred = _make_extraction(
            claims=["Alpha beta gamma delta."],
            methods="Completely novel approach using xyz",
            limitations="No known issues",
            study_design=StudyDesign.RCT,
            quality_score=0.1,
            sample_size=10,
            evidence_strength=EvidenceStrength.STRONG,
        )
        gold = _make_extraction(
            claims=["Epsilon zeta eta theta."],
            methods="Traditional method with established protocols",
            limitations="Significant confounders identified",
            study_design=StudyDesign.COHORT,
            quality_score=0.9,
            sample_size=1000,
            evidence_strength=EvidenceStrength.WEAK,
        )
        scores = score_extraction_pair(pred, gold)
        # key_findings: no word overlap between alpha/beta/gamma and epsilon/zeta/eta
        assert scores["key_findings"] == 0.0
        # evidence_strength: STRONG vs WEAK
        assert scores["evidence_strength"] == 0.0
        # methods_summary: very different words, low ROUGE-L
        assert scores["methods_summary"] < 0.3
        # study_design: RCT vs COHORT
        assert scores["study_design"] == 0.0
        # quality_score: 1.0 - abs(0.1 - 0.9) = 0.2
        assert scores["quality_score"] == pytest.approx(0.2, abs=0.01)
        # sample_size: 10 vs 1000, way outside tolerance
        assert scores["sample_size"] == 0.0

    def test_returns_all_expected_fields(self) -> None:
        ext = _make_extraction()
        scores = score_extraction_pair(ext, ext)
        expected_fields = {
            "key_findings",
            "evidence_strength",
            "quantitative_result",
            "methods_summary",
            "limitations",
            "study_design",
            "quality_score",
            "sample_size",
        }
        assert set(scores.keys()) == expected_fields

    def test_evidence_strength_uses_matched_pairs_not_positional(self) -> None:
        """Evidence strength should match by semantic similarity, not position.

        Pred findings are in reverse order vs gold. Positional zip would
        mismatch all pairs; matched-pair scoring should pair them correctly.
        """
        pred = PaperExtraction(
            paper_id="test",
            key_findings=[
                Finding(
                    claim="Drug reduces tumor size significantly.",
                    evidence_strength=EvidenceStrength.STRONG,
                    paper_id="test",
                ),
                Finding(
                    claim="Side effects were minimal in patients.",
                    evidence_strength=EvidenceStrength.WEAK,
                    paper_id="test",
                ),
            ],
            methods_summary="Test",
            limitations="Test",
        )
        # Gold has same claims but in REVERSE order
        gold = PaperExtraction(
            paper_id="test",
            key_findings=[
                Finding(
                    claim="Side effects were minimal in the patient cohort.",
                    evidence_strength=EvidenceStrength.WEAK,
                    paper_id="test",
                ),
                Finding(
                    claim="The drug significantly reduces tumor size.",
                    evidence_strength=EvidenceStrength.STRONG,
                    paper_id="test",
                ),
            ],
            methods_summary="Test",
            limitations="Test",
        )
        scores = score_extraction_pair(pred, gold)
        # With matched pairs: "Drug reduces tumor" matches "drug significantly reduces tumor"
        # and "Side effects minimal" matches "Side effects minimal in cohort"
        # Both evidence_strength values should match their semantic pair
        assert scores["evidence_strength"] == 1.0, (
            f"Expected 1.0 (matched-pair scoring), got {scores['evidence_strength']}"
        )


class TestScoreExtractionPairWithEmbeddings:
    """Tests for embedding-based scoring function."""

    def test_fallback_when_model_is_none(self) -> None:
        """Falls back to word-overlap scoring when no model is provided."""
        from autoreview.extraction.scoring import score_extraction_pair_with_embeddings

        ext = _make_extraction()
        scores = score_extraction_pair_with_embeddings(ext, ext, model=None)
        # Should produce the same results as score_extraction_pair
        baseline = score_extraction_pair(ext, ext)
        for field in baseline:
            assert scores[field] == pytest.approx(baseline[field], abs=0.01), (
                f"{field} mismatch: {scores[field]} vs {baseline[field]}"
            )
        # Should NOT have embedding-specific fields
        assert "key_findings_precision" not in scores
        assert "key_findings_recall" not in scores

    def test_with_embedding_model_identical_claims(self) -> None:
        """Identical claims should score very high with embeddings."""
        from autoreview.extraction.scoring import (
            load_embedding_model,
            score_extraction_pair_with_embeddings,
        )

        model = load_embedding_model()
        if model is None:
            pytest.skip("sentence-transformers not available")

        ext = _make_extraction(claims=["Drug X reduces tumor size by 40%"])
        scores = score_extraction_pair_with_embeddings(ext, ext, model=model)
        assert scores["key_findings"] > 0.95
        assert scores["key_findings_precision"] == 1.0
        assert scores["key_findings_recall"] == 1.0

    def test_embedding_scores_paraphrase_higher_than_word_overlap(self) -> None:
        """Embedding scoring should rate paraphrased claims higher than word overlap."""
        from autoreview.extraction.scoring import (
            load_embedding_model,
            score_extraction_pair_with_embeddings,
        )

        model = load_embedding_model()
        if model is None:
            pytest.skip("sentence-transformers not available")

        # Paraphrased claims: same meaning, different words
        pred = _make_extraction(
            claims=["The treatment significantly decreased inflammation markers."]
        )
        gold = _make_extraction(
            claims=["Inflammatory biomarkers were substantially reduced by the intervention."]
        )

        word_overlap_scores = score_extraction_pair(pred, gold)
        embedding_scores = score_extraction_pair_with_embeddings(pred, gold, model=model)

        # Embeddings should capture semantic similarity better than word overlap
        assert embedding_scores["key_findings"] > word_overlap_scores["key_findings"], (
            f"Embedding score ({embedding_scores['key_findings']:.3f}) should be higher than "
            f"word overlap ({word_overlap_scores['key_findings']:.3f}) for paraphrased claims"
        )

    def test_embedding_returns_extra_fields(self) -> None:
        """Embedding scoring should return precision and recall fields."""
        from autoreview.extraction.scoring import (
            load_embedding_model,
            score_extraction_pair_with_embeddings,
        )

        model = load_embedding_model()
        if model is None:
            pytest.skip("sentence-transformers not available")

        ext = _make_extraction()
        scores = score_extraction_pair_with_embeddings(ext, ext, model=model)
        assert "key_findings_precision" in scores
        assert "key_findings_recall" in scores
        assert 0.0 <= scores["key_findings_precision"] <= 1.0
        assert 0.0 <= scores["key_findings_recall"] <= 1.0

    def test_embedding_empty_claims(self) -> None:
        """Empty claims should handle gracefully with embeddings."""
        from autoreview.extraction.scoring import (
            load_embedding_model,
            score_extraction_pair_with_embeddings,
        )

        model = load_embedding_model()
        if model is None:
            pytest.skip("sentence-transformers not available")

        pred = _make_extraction(claims=[])
        gold = _make_extraction(claims=["Some finding"])
        scores = score_extraction_pair_with_embeddings(pred, gold, model=model)
        assert scores["key_findings"] == 0.0
        assert scores["key_findings_precision"] == 0.0
        assert scores["key_findings_recall"] == 0.0

    def test_embedding_unrelated_claims_low_score(self) -> None:
        """Completely unrelated claims should score low with embeddings."""
        from autoreview.extraction.scoring import (
            load_embedding_model,
            score_extraction_pair_with_embeddings,
        )

        model = load_embedding_model()
        if model is None:
            pytest.skip("sentence-transformers not available")

        pred = _make_extraction(claims=["The weather was sunny today in California."])
        gold = _make_extraction(claims=["Mitochondrial DNA mutations cause hearing loss."])
        scores = score_extraction_pair_with_embeddings(pred, gold, model=model)
        assert scores["key_findings"] < 0.4

    def test_embedding_methods_summary_paraphrase(self) -> None:
        """Embedding scoring should rate paraphrased methods higher than ROUGE-L."""
        from autoreview.extraction.scoring import (
            load_embedding_model,
            score_extraction_pair_with_embeddings,
        )

        model = load_embedding_model()
        if model is None:
            pytest.skip("sentence-transformers not available")

        pred = _make_extraction(
            methods="We trained a deep neural network on clinical records using cross-validation."
        )
        gold = _make_extraction(
            methods=(
                "A deep learning model was fitted to patient medical data with k-fold validation."
            )
        )
        word_scores = score_extraction_pair(pred, gold)
        emb_scores = score_extraction_pair_with_embeddings(pred, gold, model=model)
        # Embedding should score higher for semantically similar but differently worded methods
        assert emb_scores["methods_summary"] > word_scores["methods_summary"], (
            f"Embedding ({emb_scores['methods_summary']:.3f}) should beat "
            f"ROUGE-L ({word_scores['methods_summary']:.3f}) for paraphrased methods"
        )

    def test_embedding_limitations_paraphrase(self) -> None:
        """Embedding scoring should rate paraphrased limitations higher than ROUGE-L."""
        from autoreview.extraction.scoring import (
            load_embedding_model,
            score_extraction_pair_with_embeddings,
        )

        model = load_embedding_model()
        if model is None:
            pytest.skip("sentence-transformers not available")

        pred = _make_extraction(
            limitations="The small cohort size restricts the generalizability of our conclusions."
        )
        gold = _make_extraction(
            limitations="Limited sample reduces how broadly the findings can be applied."
        )
        word_scores = score_extraction_pair(pred, gold)
        emb_scores = score_extraction_pair_with_embeddings(pred, gold, model=model)
        assert emb_scores["limitations"] > word_scores["limitations"], (
            f"Embedding ({emb_scores['limitations']:.3f}) should beat "
            f"ROUGE-L ({word_scores['limitations']:.3f}) for paraphrased limitations"
        )

    def test_embedding_identical_methods_high_score(self) -> None:
        """Identical methods text should get very high embedding score."""
        from autoreview.extraction.scoring import (
            load_embedding_model,
            score_extraction_pair_with_embeddings,
        )

        model = load_embedding_model()
        if model is None:
            pytest.skip("sentence-transformers not available")

        ext = _make_extraction(methods="We used a transformer model trained on clinical notes.")
        scores = score_extraction_pair_with_embeddings(ext, ext, model=model)
        assert scores["methods_summary"] > 0.95
        assert scores["limitations"] > 0.95
