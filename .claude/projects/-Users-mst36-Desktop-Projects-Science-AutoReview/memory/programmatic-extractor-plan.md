---
name: programmatic-extractor-plan
description: Plan to build a deterministic Python extractor that replaces LLM-based paper extraction, trained against 173 LLM extractions as ground truth
type: project
---

Build a programmatic (zero-token) paper extraction function to replace the LLM-based extraction stage.

**Why:** The extraction stage is the pipeline's biggest bottleneck — 489 papers × 50K tokens each = 25M input tokens, ~80 min via `claude -p`. A deterministic Python function would reduce this to ~0 tokens and <1 minute.

**How to apply:**
- Use 173 LLM extractions from `output/arise/arise_llm_eval_v2/` as ground truth training data
- Build heuristic extractor: abstract sentence classification, regex for quantitative results, section parsing
- Score against LLM output using embedding similarity for claims, exact match for numbers
- Iterate using auto-research loop until convergence
- Integrate as `extraction_mode: programmatic` config option

**Key data flow insight (from code analysis):**
- `key_findings` (claims + evidence_strength): Used by clustering, citation selection, section writing, evidence chains — HARDEST to replicate programmatically
- `methods_summary`: Used by section writing, passage mining — just extract Methods section text
- `limitations`: Used by section writing only — just extract Limitations section text
- `study_design`, `quality_score`, `sample_size`: Used only by contradiction resolution — keyword/regex/heuristic
- `methodology_details`, `domain_specific_fields`: NEVER USED anywhere — can be eliminated

**Ground truth location:** `output/arise/arise_llm_eval_v2/snapshots/` — extraction_batch_1 through batch_5 contain 173 LLM extractions with source papers in the KnowledgeBase

**Pipeline run status:** Stopped at extraction (173/489 papers). Snapshots saved — can resume if needed. Node timeouts were increased in `autoreview/pipeline/runner.py`.
