## Last Session: 2026-03-27

### Branch
main

### Summary
Added citation evidence stubs to v5 KG extraction pipeline. Fixed ingest parser to handle v5 flat evidence format. Result: evidence-per-claim ratio 0.48→0.92, all attributed_prior claims now have evidence coverage.

### Files Changed
- `autoreview/knowledge_graph/models.py` — added 3 optional fields to KGEvidenceLink: citing_sentence, source_doi, section
- `autoreview/knowledge_graph/ingest.py` — added review_citation/indirect_experimental to valid evidence strengths; fixed _parse_evidence_unit for v5 flat format; added citation stub field parsing
- `Paper Extractor/KnowledgeGraph Extraction/kg_extraction_prompt.md` — citation evidence stub rules: every attributed_prior claim must carry review_citation evidence unit with citing_sentence, source_doi, model_system, organism; updated example c_003; added quality checklist item 14
- `Paper Extractor/KnowledgeGraph Extraction/run_v5_test.py` — fixed paper path (rai14_fulltext.txt, not PIP5K hash); increased --max-turns 1→2

### Key Results
- Evidence-per-claim ratio: 0.48 → 0.92
- attributed_prior claims with evidence: 0/18 → 9/9 (100%)
- 8 citation stubs generated with citing sentences, DOIs, model systems
- 120/120 tests passing

### Next Steps
1. Fix evidence_strength on experimental evidence in kg_extraction_prompt.md example
2. Re-extract 305-paper corpus with updated prompt (citation stubs + context fields)
3. Rebuild KG and rerun NLI scoring
4. Evaluate VOI ranking improvement with populated evidence context
