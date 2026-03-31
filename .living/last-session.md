# Last Session State

Last updated: 2026-03-30

## Current State
- v11.2.1 prompt confirmed as stable standard (reverted from v11.3 experiments)
- Citation supplement system fully implemented and tested:
  - citation_supplement.py: needs_citation_supplement(), extract_sections(), build_claim_summary(), run_citation_supplement(), merge_supplement(), extract_with_supplement()
  - kg_citation_supplement_prompt.md: focused prompt with correct schema (matching main extraction fields)
  - batch_extract_kg.py: --supplement and --poll-supplement flags integrated
- Test results: Large paper 0→25 attributed_prior, medium/small correctly not triggered
- Fixed citation regex to handle (N) parenthetical format
- Ready for large-scale extraction: run main batch, then --supplement pass

## 2026-03-30 — Gastruloid Corpus Quality Audit

Audited 1,023-paper gastruloid corpus. Archived 90 papers across 3 rounds:
- Round 1 (61): reviews, duplicates, short/abstract-only, off-topic, editorial content
- Round 2 (6): deep-scan reviews caught via intro text analysis ("In this review" declarations)
- Round 3 (23): protocols, conference abstracts, theses/dissertations, ethics/law, predatory journals, non-English

Final corpus: 933 papers, all >5000 chars full text. Profile: median 2023, 192 journals, 50/50 human/mouse, Wnt/BMP/Nodal/FGF dominated.

Critical finding: 0% KG extraction coverage on current corpus — 311 existing extractions are orphaned (paper IDs from a prior corpus version no longer match current paper set). KG extraction pipeline needs to be re-run on the 933-paper corpus.

Note: .gitignore and test_mrf_weight_learning.py changes were pre-existing uncommitted files, not from this session.

---
## Session: 2026-03-30 — Gastruloid Corpus Audit and Expansion

### Summary
Three-round corpus audit + comprehensive gap analysis + corpus expansion for the gastruloid KG pipeline.

### What was done
1. **Corpus audit (3 rounds)**: Audited 1,023-paper gastruloid corpus. Archived 90 papers (reviews, duplicates, short text, off-topic, editorial, protocols, conference abstracts, theses, ethics/law, predatory journals, non-English). Down to 933 papers.
2. **Comprehensive gap analysis**: Cross-referenced corpus against PubMed (227), Semantic Scholar (100), OpenAlex (500), Europe PMC (500). Found 135 gastruloid-relevant gaps (73 direct gastruloid, 61 embryo model, 1 micropatterning). Key misses: 2023 ETS special issue almost entirely absent; foundational papers Warmflash 2016 (Dev Cell) and Moris 2020 (Nature) missing.
3. **Corpus expansion**: Ran expand_corpus.py (OpenAlex, 12 search terms, 7798 unique works → 1266 candidates → 319 novel → 134 with full text). Targeted DOI retrieval for 14 priority gaps (recovered 9). retry_inaccessible.py with VPN recovered 94 more.
4. **Final state**: 1,078 papers in corpus, 242 archived, 183 still inaccessible. 5 priority papers remain paywalled (Moris 2020 Nature, retinoid-enhanced gastruloids Nat Cell Bio, 3 others).

### Key decisions
- 1,000+ papers over a single field is sufficient for Phase 2 Bayesian inference ground truth curation; expansion complete.
- Browser-based manual download required for institutional-auth paywalled papers (VPN cannot authenticate automated resolvers).

### Files changed (pre-existing uncommitted from prior sessions)
17 changed files, no new .living/ updates this session prior to stop hook.
