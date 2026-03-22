"""Comprehensive tests for the programmatic extractor."""

from __future__ import annotations

import pytest

from autoreview.config.models import ExtractionConfig
from autoreview.extraction.models import (
    EvidenceStrength,
    PaperExtraction,
    StudyDesign,
)
from autoreview.extraction.programmatic import (
    ProgrammaticExtractor,
    _find_section,
    _truncate_at_sentence_boundary,
    classify_study_design,
    compute_quality_score,
    determine_evidence_strength,
    extract_key_findings,
    extract_limitations,
    extract_methods_summary,
    extract_quantitative_results,
    extract_sample_size,
    score_sentence,
    split_sentences,
    word_overlap_similarity,
)
from autoreview.extraction.truncation import ParsedSection
from autoreview.models.paper import CandidatePaper, ScreenedPaper

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_paper(
    title: str = "Evaluating LLMs",
    abstract: str = (
        "Large language models have shown remarkable performance. "
        "We evaluate GPT-4 on medical benchmarks. "
        "Our results show GPT-4 achieves 86.7% accuracy on MedQA. "
        "The model significantly outperforms prior approaches. "
        "However, challenges remain in hallucination."
    ),
    full_text: str | None = None,
    citation_count: int = 50,
) -> CandidatePaper:
    return CandidatePaper(
        title=title,
        authors=["A"],
        year=2024,
        journal="Nature",
        doi="10.1234/test",
        abstract=abstract,
        source_database="semantic_scholar",
        full_text=full_text,
        citation_count=citation_count,
    )


def _make_screened(paper: CandidatePaper | None = None) -> ScreenedPaper:
    paper = paper or _make_paper()
    return ScreenedPaper(
        paper=paper,
        relevance_score=4,
        rationale="Relevant",
        include=True,
    )


# ---------------------------------------------------------------------------
# TestSplitSentences
# ---------------------------------------------------------------------------


class TestSplitSentences:
    def test_basic_splitting(self):
        text = "This is sentence one. This is sentence two. And three."
        result = split_sentences(text)
        assert len(result) == 3
        assert result[0] == "This is sentence one."
        assert result[1] == "This is sentence two."

    def test_empty_string(self):
        assert split_sentences("") == []
        assert split_sentences("   ") == []

    def test_single_sentence(self):
        result = split_sentences("Just one sentence here.")
        assert len(result) == 1
        assert result[0] == "Just one sentence here."

    def test_double_newline_paragraph(self):
        text = "First paragraph sentence.\n\nSecond paragraph sentence."
        result = split_sentences(text)
        assert len(result) == 2

    def test_single_newline_normalized(self):
        text = "This continues\non the next line. Then a new sentence."
        result = split_sentences(text)
        assert len(result) == 2
        assert "continues on" in result[0]


# ---------------------------------------------------------------------------
# TestWordOverlapSimilarity
# ---------------------------------------------------------------------------


class TestWordOverlapSimilarity:
    def test_identical(self):
        assert word_overlap_similarity("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert word_overlap_similarity("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        sim = word_overlap_similarity("the cat sat", "the dog sat")
        assert 0.0 < sim < 1.0
        # Jaccard: {the, sat} / {the, cat, sat, dog} = 2/4 = 0.5
        assert sim == pytest.approx(0.5)

    def test_empty_string(self):
        assert word_overlap_similarity("", "hello") == 0.0
        assert word_overlap_similarity("hello", "") == 0.0
        assert word_overlap_similarity("", "") == 0.0


# ---------------------------------------------------------------------------
# TestScoreSentence
# ---------------------------------------------------------------------------


class TestScoreSentence:
    def test_result_sentence_scores_high(self):
        sent = "Our results show the model achieves 95% accuracy."
        score = score_sentence(sent, "results", 0.5)
        # Position: results=0.30, keywords: "results show"+"achieves"=0.10,
        # quant: percentage=0.10, novelty: 0.0 -> ~0.50
        assert score >= 0.4

    def test_methods_sentence_scores_low(self):
        sent = "We used a standard preprocessing pipeline."
        score = score_sentence(sent, "methods", 0.5)
        # Position: methods=0.05, few keywords, no quant, no novelty
        assert score < 0.2

    def test_abstract_conclusion_scores_high(self):
        sent = "We propose a novel framework that significantly outperforms baselines."
        score = score_sentence(sent, "abstract", 0.95)
        # Position: abstract_last=0.40, keywords: "we propose"+"significant"+"outperforms"=0.15,
        # novelty: "novel"=0.10 -> ~0.65
        assert score >= 0.5

    def test_quantitative_bonus(self):
        sent_quant = "The accuracy improved to 92.3% (p < 0.001)."
        sent_plain = "The accuracy improved notably."
        score_q = score_sentence(sent_quant, "results", 0.5)
        score_p = score_sentence(sent_plain, "results", 0.5)
        assert score_q > score_p


# ---------------------------------------------------------------------------
# TestExtractQuantitativeResults
# ---------------------------------------------------------------------------


class TestExtractQuantitativeResults:
    def test_percentage(self):
        result = extract_quantitative_results("Achieved 95.2% accuracy.")
        assert result is not None
        assert "95.2%" in result

    def test_p_value(self):
        result = extract_quantitative_results("The difference was significant (p < 0.001).")
        assert result is not None
        assert "p" in result.lower()

    def test_no_numbers(self):
        result = extract_quantitative_results("The model performed well overall.")
        assert result is None

    def test_sample_size_n(self):
        result = extract_quantitative_results("We tested on N=500 samples.")
        assert result is not None
        assert "N=500" in result

    def test_emdash_range(self):
        result = extract_quantitative_results("Performance gap: 8–18 percentage points.")
        assert result is not None
        assert "8–18" in result

    def test_decimal_comparison(self):
        result = extract_quantitative_results("Scores were 35.9 vs 4.2 under noise.")
        assert result is not None
        assert "vs" in result

    def test_dropped_from_to(self):
        result = extract_quantitative_results(
            "F1 dropped from 35.9% to 4.2% under noise injection."
        )
        assert result is not None

    def test_at_k_pattern(self):
        result = extract_quantitative_results("GPT-4 base: 100% at k=1, declining to 25% at k=10.")
        assert result is not None

    def test_score_equals(self):
        result = extract_quantitative_results("BLEU score of 0.42 on the test set.")
        assert result is not None

    def test_percentage_points(self):
        result = extract_quantitative_results(
            "The average gap was approximately 24 percentage points on XNLI."
        )
        assert result is not None


# ---------------------------------------------------------------------------
# TestDetermineEvidenceStrength
# ---------------------------------------------------------------------------


class TestDetermineEvidenceStrength:
    def test_full_text_with_quant_and_specific_claim_is_strong(self):
        # Requires: full_text + quant + specific claim + high quant density
        strength = determine_evidence_strength(
            "Our results show the model achieves state-of-the-art performance.",
            has_full_text=True,
            has_quantitative=True,
            paper_quant_density=0.5,
        )
        assert strength == EvidenceStrength.STRONG

    def test_full_text_with_quant_default_is_moderate(self):
        # Full text + quant but no specific claim → MODERATE (76% in LLM corpus)
        strength = determine_evidence_strength(
            "Good results were observed.",
            has_full_text=True,
            has_quantitative=True,
            paper_quant_density=0.5,
        )
        assert strength == EvidenceStrength.MODERATE

    def test_abstract_only_is_always_preliminary(self):
        # LLM labels 100% of abstract-only findings as preliminary
        strength = determine_evidence_strength(
            "Some results observed.", has_full_text=False, has_quantitative=False
        )
        assert strength == EvidenceStrength.PRELIMINARY

    def test_abstract_with_quant_still_preliminary(self):
        # Even with quant data, abstract-only → PRELIMINARY
        strength = determine_evidence_strength(
            "Got 95% accuracy.", has_full_text=False, has_quantitative=True
        )
        assert strength == EvidenceStrength.PRELIMINARY

    def test_full_text_no_quant_is_moderate(self):
        strength = determine_evidence_strength(
            "Results look good.", has_full_text=True, has_quantitative=False
        )
        assert strength == EvidenceStrength.MODERATE

    def test_hedging_downgrades_moderate_to_weak(self):
        # "may" combined with "suggest" triggers mild hedging downgrade
        strength = determine_evidence_strength(
            "This may suggest improvement in outcomes.",
            has_full_text=True,
            has_quantitative=True,
            paper_quant_density=0.5,
        )
        # Base MODERATE (no specific claim), hedged → WEAK
        assert strength == EvidenceStrength.WEAK

    def test_strong_hedging_downgrades(self):
        # "preliminary" is a strong hedging signal; specific claim → STRONG, then hedged → MODERATE
        strength = determine_evidence_strength(
            "Our results show preliminary but significant improvements.",
            has_full_text=True,
            has_quantitative=True,
            paper_quant_density=0.5,
        )
        assert strength == EvidenceStrength.MODERATE

    def test_may_alone_no_downgrade(self):
        # "may" without uncertainty words is standard academic phrasing
        strength = determine_evidence_strength(
            "This approach may be applied to other domains.",
            has_full_text=True,
            has_quantitative=True,
            paper_quant_density=0.5,
        )
        # No specific claim → MODERATE, no hedging → stays MODERATE
        assert strength == EvidenceStrength.MODERATE

    def test_hedging_preliminary_stays(self):
        strength = determine_evidence_strength(
            "This preliminary study may suggest something.",
            has_full_text=False,
            has_quantitative=False,
        )
        # Abstract-only → always PRELIMINARY
        assert strength == EvidenceStrength.PRELIMINARY

    def test_generic_claim_downgrades(self):
        strength = determine_evidence_strength(
            "This has emerged as a promising research direction.",
            has_full_text=True,
            has_quantitative=False,
        )
        # MODERATE -> WEAK due to generic claim pattern
        assert strength == EvidenceStrength.WEAK

    def test_study_design_meta_analysis_strong(self):
        # Meta-analysis with quant + specific claim + high density → STRONG
        strength = determine_evidence_strength(
            "Our results demonstrate a clear pooled effect.",
            has_full_text=True,
            has_quantitative=True,
            study_design=StudyDesign.META_ANALYSIS,
            paper_quant_density=0.5,
        )
        assert strength == EvidenceStrength.STRONG

    def test_intro_section_without_quant_is_weak(self):
        # Full-text claim from introduction without quant → WEAK
        strength = determine_evidence_strength(
            "Language models have become increasingly important.",
            has_full_text=True,
            has_quantitative=False,
            section="introduction",
        )
        assert strength == EvidenceStrength.WEAK

    def test_low_quant_density_prevents_strong(self):
        # Specific claim but low paper quant density → stays MODERATE
        strength = determine_evidence_strength(
            "We found the model significantly outperforms baselines.",
            has_full_text=True,
            has_quantitative=True,
            paper_quant_density=0.1,
        )
        assert strength == EvidenceStrength.MODERATE


# ---------------------------------------------------------------------------
# TestExtractKeyFindings
# ---------------------------------------------------------------------------


class TestExtractKeyFindings:
    def test_from_abstract_produces_findings(self):
        abstract = (
            "Large language models have shown remarkable performance. "
            "We evaluate GPT-4 on medical benchmarks. "
            "Our results show GPT-4 achieves 86.7% accuracy on MedQA. "
            "The model significantly outperforms prior approaches."
        )
        findings = extract_key_findings(abstract, None, [], "test-paper")
        assert len(findings) >= 1
        # Should find the quantitative result
        quant_findings = [f for f in findings if f.quantitative_result]
        assert len(quant_findings) >= 1
        assert "86.7%" in quant_findings[0].quantitative_result

    def test_minimum_one_finding(self):
        abstract = "This is a short abstract."
        findings = extract_key_findings(abstract, None, [], "test-paper")
        assert len(findings) >= 1

    def test_deduplication(self):
        abstract = (
            "We found X achieves 90% accuracy. "
            "We found X achieves 90% accuracy on the test set. "
            "A completely different finding about Y."
        )
        findings = extract_key_findings(abstract, None, [], "test-paper")
        claims = [f.claim for f in findings]
        # The near-duplicate should be removed
        assert len(claims) <= 3

    def test_paper_id_set(self):
        findings = extract_key_findings("Some result.", None, [], "my-paper-id")
        for f in findings:
            assert f.paper_id == "my-paper-id"


# ---------------------------------------------------------------------------
# TestFindSection
# ---------------------------------------------------------------------------


class TestFindSection:
    def test_finds_methods(self):
        sections = [
            ParsedSection(name="Introduction", start=0, end=100, text="Intro text."),
            ParsedSection(name="Methods", start=100, end=200, text="Method text."),
        ]
        result = _find_section(sections, ["method"])
        assert result is not None
        assert result.name == "Methods"

    def test_returns_none_when_missing(self):
        sections = [
            ParsedSection(name="Introduction", start=0, end=100, text="Intro text."),
        ]
        result = _find_section(sections, ["method"])
        assert result is None

    def test_case_insensitive(self):
        sections = [
            ParsedSection(name="METHODS AND MATERIALS", start=0, end=100, text="Text."),
        ]
        result = _find_section(sections, ["method"])
        assert result is not None

    def test_empty_sections(self):
        result = _find_section([], ["method"])
        assert result is None


# ---------------------------------------------------------------------------
# TestTruncateAtSentenceBoundary
# ---------------------------------------------------------------------------


class TestTruncateAtSentenceBoundary:
    def test_short_text_unchanged(self):
        text = "Short sentence."
        assert _truncate_at_sentence_boundary(text, 100) == text

    def test_truncates_at_period(self):
        text = "First sentence. Second sentence. Third sentence."
        result = _truncate_at_sentence_boundary(text, 20)
        assert result.endswith(".")
        assert len(result) <= 20


# ---------------------------------------------------------------------------
# TestExtractMethodsSummary
# ---------------------------------------------------------------------------


class TestExtractMethodsSummary:
    def test_from_section(self):
        sections = [
            ParsedSection(
                name="Methods",
                start=0,
                end=100,
                text="We used a transformer model trained on 10k samples.",
            ),
        ]
        result = extract_methods_summary(None, sections)
        assert "transformer" in result

    def test_fallback_to_abstract(self):
        abstract = (
            "This paper studies X. We used a novel algorithm for processing. "
            "Results show improvement."
        )
        result = extract_methods_summary(abstract, [])
        assert "algorithm" in result

    def test_no_text_fallback(self):
        result = extract_methods_summary(None, [])
        assert result == "Methods not available."


# ---------------------------------------------------------------------------
# TestExtractLimitations
# ---------------------------------------------------------------------------


class TestExtractLimitations:
    def test_from_section(self):
        sections = [
            ParsedSection(
                name="Limitations",
                start=0,
                end=100,
                text="This study has a small sample size and limited generalizability.",
            ),
        ]
        result = extract_limitations(None, sections)
        assert "small sample" in result

    def test_fallback_to_discussion(self):
        sections = [
            ParsedSection(
                name="Discussion",
                start=0,
                end=200,
                text=(
                    "Our results are promising. "
                    "However, the small sample size limits our conclusions. "
                    "Future work should address this."
                ),
            ),
        ]
        result = extract_limitations(None, sections)
        assert "small sample" in result.lower() or "however" in result.lower()

    def test_final_fallback(self):
        result = extract_limitations(None, [])
        assert "not explicitly stated" in result.lower()

    def test_fallback_to_abstract(self):
        abstract = "We studied X. Although sample sizes were small, results are encouraging."
        result = extract_limitations(abstract, [])
        assert "small" in result.lower() or "although" in result.lower()


# ---------------------------------------------------------------------------
# TestClassifyStudyDesign
# ---------------------------------------------------------------------------


class TestClassifyStudyDesign:
    def test_meta_analysis(self):
        result = classify_study_design("A meta-analysis of interventions", "We performed...", [])
        assert result == StudyDesign.META_ANALYSIS

    def test_computational(self):
        result = classify_study_design(
            "Deep learning for image segmentation",
            "We present a neural network approach.",
            [],
        )
        assert result == StudyDesign.COMPUTATIONAL

    def test_rct(self):
        result = classify_study_design(
            "A randomized controlled trial of drug X",
            "Participants were randomized...",
            [],
        )
        assert result == StudyDesign.RCT

    def test_default_other(self):
        result = classify_study_design(
            "An analysis of trends",
            "We looked at data from various sources.",
            [],
        )
        assert result == StudyDesign.OTHER

    def test_systematic_review(self):
        result = classify_study_design(
            "Systematic review of treatments",
            "Following PRISMA guidelines...",
            [],
        )
        assert result == StudyDesign.SYSTEMATIC_REVIEW


# ---------------------------------------------------------------------------
# TestExtractSampleSize
# ---------------------------------------------------------------------------


class TestExtractSampleSize:
    def test_n_equals(self):
        result = extract_sample_size("We enrolled N=150 patients.", [])
        assert result == 150

    def test_participants(self):
        result = extract_sample_size("The study included 200 participants.", [])
        assert result == 200

    def test_dataset_of(self):
        result = extract_sample_size("We used a dataset of 5000 images.", [])
        assert result == 5000

    def test_no_match(self):
        result = extract_sample_size("No sample size mentioned here.", [])
        assert result is None

    def test_from_methods_section(self):
        sections = [
            ParsedSection(
                name="Methods",
                start=0,
                end=100,
                text="We recruited N=300 subjects for the trial.",
            ),
        ]
        result = extract_sample_size("Abstract without size.", sections)
        assert result == 300

    def test_comma_in_number(self):
        result = extract_sample_size("We analyzed 1,500 participants.", [])
        assert result == 1500

    def test_consisting_of_instances(self):
        result = extract_sample_size(
            "We created ProverQA, consisting of 1,500 instances across three levels.", []
        )
        assert result == 1500

    def test_containing_questions(self):
        result = extract_sample_size("The benchmark containing 5265 questions from Wikipedia.", [])
        assert result == 5265

    def test_evaluate_llms(self):
        result = extract_sample_size(
            "We evaluate 18 state-of-the-art LLMs on medical benchmarks.", []
        )
        assert result == 18

    def test_models_in_abstract(self):
        result = extract_sample_size(
            "We investigate the tool learning ability of 41 prevalent LLMs by reproducing benchmarks.",
            [],
        )
        assert result == 41

    def test_total_of_terms(self):
        result = extract_sample_size(
            "The dataset has a total of 16052 questions across two benchmarks.", []
        )
        assert result == 16052

    def test_rejects_tiny_numbers(self):
        """Numbers < 3 should not be treated as sample sizes."""
        result = extract_sample_size("We used 2 models for comparison.", [])
        assert result is None


# ---------------------------------------------------------------------------
# TestComputeQualityScore
# ---------------------------------------------------------------------------


class TestComputeQualityScore:
    def test_high_quality(self):
        score = compute_quality_score(
            has_full_text=True,
            full_text_length=25000,
            abstract_length=500,
            citation_count=200,
            methods_text_length=2000,
            results_text_length=4000,
            n_quantitative_findings=5,
            n_total_findings=8,
        )
        assert score > 0.5

    def test_low_quality(self):
        score = compute_quality_score(
            has_full_text=False,
            full_text_length=0,
            abstract_length=100,
            citation_count=0,
            methods_text_length=0,
            results_text_length=0,
            n_quantitative_findings=0,
            n_total_findings=1,
        )
        assert score < 0.3

    def test_bounded_zero_one(self):
        score_max = compute_quality_score(
            has_full_text=True,
            full_text_length=100000,
            abstract_length=1000,
            citation_count=10000,
            methods_text_length=5000,
            results_text_length=10000,
            n_quantitative_findings=10,
            n_total_findings=10,
        )
        assert 0.0 <= score_max <= 1.0

        score_min = compute_quality_score(
            has_full_text=False,
            full_text_length=0,
            abstract_length=0,
            citation_count=0,
            methods_text_length=0,
            results_text_length=0,
            n_quantitative_findings=0,
            n_total_findings=0,
        )
        assert 0.0 <= score_min <= 1.0


# ---------------------------------------------------------------------------
# TestProgrammaticExtractor
# ---------------------------------------------------------------------------


class TestProgrammaticExtractor:
    def test_extract_returns_paper_extraction(self):
        sp = _make_screened()
        extractor = ProgrammaticExtractor(config=ExtractionConfig())
        result = extractor.extract(sp)
        assert isinstance(result, PaperExtraction)

    def test_has_key_findings(self):
        sp = _make_screened()
        extractor = ProgrammaticExtractor(config=ExtractionConfig())
        result = extractor.extract(sp)
        assert len(result.key_findings) >= 1

    def test_has_methods(self):
        sp = _make_screened()
        extractor = ProgrammaticExtractor(config=ExtractionConfig())
        result = extractor.extract(sp)
        assert result.methods_summary
        assert len(result.methods_summary) > 0

    def test_has_limitations(self):
        sp = _make_screened()
        extractor = ProgrammaticExtractor(config=ExtractionConfig())
        result = extractor.extract(sp)
        assert result.limitations
        assert len(result.limitations) > 0

    def test_has_study_design(self):
        sp = _make_screened()
        extractor = ProgrammaticExtractor(config=ExtractionConfig())
        result = extractor.extract(sp)
        assert result.study_design is not None

    def test_quality_bounded(self):
        sp = _make_screened()
        extractor = ProgrammaticExtractor(config=ExtractionConfig())
        result = extractor.extract(sp)
        assert result.quality_score is not None
        assert 0.0 <= result.quality_score <= 1.0

    def test_extract_batch_works(self):
        papers = [_make_screened() for _ in range(3)]
        extractor = ProgrammaticExtractor(config=ExtractionConfig())
        successes, failures = extractor.extract_batch(papers)
        assert len(successes) == 3
        assert len(failures) == 0

    def test_handles_failure_gracefully(self):
        """Test that extract_batch catches errors per-paper."""
        sp = _make_screened()
        extractor = ProgrammaticExtractor(config=ExtractionConfig())

        # Monkey-patch extract to fail on this paper
        original_extract = extractor.extract

        def failing_extract(_paper: ScreenedPaper) -> PaperExtraction:  # noqa: ARG001
            msg = "Simulated failure"
            raise RuntimeError(msg)

        extractor.extract = failing_extract  # type: ignore[assignment]
        successes, failures = extractor.extract_batch([sp])
        assert len(successes) == 0
        assert len(failures) == 1
        assert "Simulated failure" in failures[0].error

        # Restore
        extractor.extract = original_extract  # type: ignore[assignment]

    def test_skipped_fields_have_defaults(self):
        sp = _make_screened()
        extractor = ProgrammaticExtractor(config=ExtractionConfig())
        result = extractor.extract(sp)
        assert result.relationships == []
        assert result.methodology_details is None
        assert result.domain_specific_fields == {}

    def test_pydantic_validates(self):
        """Ensure the extraction result passes Pydantic validation."""
        sp = _make_screened()
        extractor = ProgrammaticExtractor(config=ExtractionConfig())
        result = extractor.extract(sp)
        # Re-validate by reconstructing from dict
        validated = PaperExtraction.model_validate(result.model_dump())
        assert validated.paper_id == result.paper_id
        assert len(validated.key_findings) == len(result.key_findings)

    def test_with_full_text(self):
        """Test extraction with full text available."""
        full_text = (
            "Introduction\n"
            "This paper studies machine learning for NLP tasks.\n\n"
            "Methods\n"
            "We trained a deep learning model on 10000 documents.\n\n"
            "Results\n"
            "Our approach achieves 92.5% accuracy on the test set.\n"
            "The model outperforms all baselines.\n\n"
            "Discussion\n"
            "These results demonstrate the effectiveness of our approach.\n"
            "However, the small sample size limits generalizability.\n\n"
            "Conclusion\n"
            "We presented a novel approach achieving state-of-the-art results."
        )
        paper = _make_paper(full_text=full_text)
        sp = _make_screened(paper)
        extractor = ProgrammaticExtractor(config=ExtractionConfig())
        result = extractor.extract(sp)
        assert isinstance(result, PaperExtraction)
        assert len(result.key_findings) >= 1
        assert result.study_design == StudyDesign.COMPUTATIONAL

    def test_paper_id_matches(self):
        sp = _make_screened()
        extractor = ProgrammaticExtractor(config=ExtractionConfig())
        result = extractor.extract(sp)
        assert result.paper_id == sp.paper.id
