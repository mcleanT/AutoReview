# Algorithm Manifest

## ProgrammaticExtractor
- **Location:** `autoreview/extraction/programmatic.py`
- **Purpose:** Zero-token deterministic paper extraction replacing LLM-based extraction
- **Algorithm:** Sentence scoring (position + keywords + quantitative + novelty + title similarity) → top-N selection → Finding construction. Section-based methods/limitations extraction. Keyword-based study design classification. Regex-based sample size extraction. Composite quality score heuristic.
- **Benchmark:** `scripts/benchmark_extractor.py` against 220 LLM ground truth extractions in `data/extraction_corpus/`
- **Scoring:** `autoreview/extraction/scoring.py` — ROUGE-L, word-overlap matching, exact match, Pearson correlation
- **Status:** v3 composite 0.6055 (up from 0.2515 v0). quality_score 0.89, study_design 0.83, evidence_strength 0.70, methods 0.64, key_findings 0.59, limitations 0.53, sample_size 0.46, quantitative_result 0.25. Target: 0.90 all fields.
- **Tests:** `tests/test_extraction/test_programmatic.py` (66 tests), `tests/test_extraction/test_scoring.py` (22 tests)
