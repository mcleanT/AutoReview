# Learnings — AutoReview

## Format

```
### [YYYY-MM-DD] Learning Title
**Tags**: [tag1, tag2]
**What happened**: Description of the situation
**Lesson**: The transferable insight
```

## 2026-03-04: Evaluation Infrastructure Build

**Tags**: [pydantic, asyncio, dependencies, testing]

**What happened**: While implementing the evaluation infrastructure, multiple challenges emerged:
1. Adding required fields (precision, f1, hallucinated_titles) to the CitationScore Pydantic model broke all existing test fixtures that construct that model
2. Pyright couldn't narrow types from asyncio.gather results when using return_exceptions=True and index-based assignment
3. The textstat dependency was needed for Flesch-Kincaid grade level calculations but wasn't in pyproject.toml
4. rapidfuzz was already declared in dependencies but had never been actively used in the codebase until citation matching

**Lesson**:
- Always update test fixtures when adding required fields to Pydantic models — changes are not backward compatible
- When using asyncio.gather with return_exceptions=True, rely on isinstance checks against concrete types rather than index-based assignment for better type safety
- Review existing dependency declarations before adding new features — check if needed packages are already available
- Keep dependencies synchronized with actual code usage to avoid silent dead imports

## 2026-03-16: Depth Feature Implementation

**Tags**: [prompt-engineering, pydantic, pipeline-design]

**What happened**: Implemented 3-level depth control across the writing pipeline. Key observations:
1. Prompt builders in this codebase use f-string returns, not parts-list assembly — reviewers initially assumed parts-list pattern
2. EvidenceMap.evidence_chains stores serialized dicts (list[Any]), not typed objects — allocator must use dict access
3. Pipeline calls write_all_sections(), not write_section() directly — depth must flow through both
4. outline_critique_loop in critique/revision.py needed depth forwarding to both initial and revision generation paths

**Lesson**:
- Always read the actual function body before assuming a code pattern (f-string vs parts list)
- When threading a parameter through a pipeline, trace the FULL call chain — not just the function you are modifying
- Frozen dataclasses with dict fields allow dict mutation — consider MappingProxyType if true immutability needed

## 2026-03-17: Benchmark Orchestrator Implementation
- Subagent-driven parallel implementation of 9 analysis scripts completed in ~4 min wall-clock
- Pyright diagnostics during parallel agent work are expected (agents writing tests for modules created by other agents)
- Analysis template pattern from depth_comparison.py scaled well to 9 scripts with minimal deviation
- include_depth default should be False in expand_run_matrix (explicit opt-in for depth runs)

## 2026-03-17 — CORE API integration
- CORE API v3 search endpoint uses `q` (not `query`) param and returns `results`/`totalHits` (not `data`/`total`)
- CORE author objects have a `name` field (not `displayName`) but safe to check both
- CORE download URL is stored in `downloadUrl` field in search results; direct download via /outputs/{id}/download requires API key
- respx library works well for mocking httpx calls in async tests

## 2026-03-18: Search & full-text retrieval learnings

- **Unpaywall `best_oa_location` is insufficient**: The API returns an `oa_locations` array with multiple entries. Only checking `best_oa_location` misses papers where the best URL is broken but secondary URLs work. Fix: iterate all locations.
- **External IDs are siloed by search source**: Papers found via OpenAlex don't get `arxiv` or `s2_pdf_url` from Semantic Scholar. The aggregator's dedup must merge `external_ids` dicts, not just keep the winner's IDs.
- **bioRxiv/medRxiv hardcode v1 in URLs**: Preprints at v2+ silently fail. Must query the bioRxiv API for latest version or try v1/v2/v3 sequentially.
- **PLOS URL varies by journal**: `plosone` path only works for PLOS ONE. Must extract journal slug from DOI suffix (e.g., `10.1371/journal.pbio.*` → `plosbiology`).
- **Test env var pollution**: `ELSEVIER_API_KEY` and `CORE_API_KEY` in shell env leak into tests that construct resolvers with `api_key=None`. Must pop env vars BEFORE constructing the object, not after.
- **`structlog.testing.capture_logs()` is fragile in full suite**: Works in isolation but fails when other tests reconfigure structlog. Prefer mock-based assertions over log capture for test robustness.
- **Europe PMC is free, no key needed**: Full JATS XML available for all PMC articles via `/rest/{pmcid}/fullTextXML`. Broader than PubMed for European research.
- **CrossRef abstracts are JATS XML**: Must strip XML tags before storing as plain text abstract.

## 2026-03-17 — Push-only session trigger
Push-only sessions trigger the mycelium stop hook even with no new code changes in the session — the hook detects prior-session work that was committed but not yet registered in .living/. Register significant commit batches even in push-only sessions.

### 2026-03-19: Overnight batch run insights
- macOS revokes Full Disk Access for terminals after extended sessions (~6h). Always implement /tmp fallback for overnight runs. Fix: System Settings > Privacy & Security > Full Disk Access > Terminal
- CORE API (core.ac.uk) is unreliable for programmatic search — consistently empty across 10 diverse topics. Semantic Scholar + PubMed + OpenAlex is the reliable triple.
- Niche topics (e.g., fractal squares topology) yield ~38 papers vs 70-110 for mainstream CS/AI. Pipeline handles gracefully but consider adaptive depth scaling.
- 10-topic batch × 15 stages completed in ~6-8 hours overnight. Sequential execution was reliable; no rate-limit failures.

---

## 2026-03-19 — SearchAggregator Pipeline Run (arise_rag_v2)

**Tags:** search, pipeline, performance, full-text-retrieval

### MCP tools vs Python SearchAggregator: 27x search gap
MCP tools default to `max_results=20/source`, which is a hard cap that severely limits pipeline search. Switching to Python SearchAggregator at 500/source yielded 40 → 1,095 initial candidates (27x) and 47 → 1,302 final corpus (28x), with 113 references cited in the output. This improvement cascades through every downstream stage: screening, extraction, clustering, and citation counts all scale proportionally.

### Programmatic extraction is fast enough for 600+ papers
Regex + keyword matching handled 634 papers in <3 seconds. LLM-based extraction at that scale would be prohibitively expensive. Use programmatic extraction for bulk passes; reserve LLM extraction for targeted high-value papers.

### Full-text retrieval success rate and source breakdown
73.5uccess across 634 papers. Top sources: arXiv (101), Unpaywall (95), S2 API (88), PMC (53), ACL Anthology (45). Source diversity matters — no single source dominates.

### Gap search is an effective corpus multiplier for underrepresented subtopics
Gap search filled the scalability/efficiency gap: 1 → 139 papers via targeted queries for underrepresented subtopics. Similarly, enrichment-derived corpus expansion added 487 papers. The enrichment → expansion pipeline is a significant multiplier and should not be skipped.

### S2 rate limiting without API key adds ~25 min to gap search
Semantic Scholar rate-limits unauthenticated requests aggressively. Obtaining `S2_API_KEY` would meaningfully reduce gap search wall time. Track as a configuration action item.

### PDF generation is mandatory after pipeline completion
- **date**: 2026-03-19
- **tags**: pipeline, output, convention
- **context**: User explicitly stated PDF conversion should always be done after a pipeline run produces a review.md
- **learning**: Every completed pipeline run must automatically generate a PDF from the final review.md. This should be the last step before marking the run complete. Use pandoc with xelatex if available, fall back to other converters. The CLAUDE.md already mentions "Post-Pipeline PDF Generation" as a rule — this reinforces it as non-negotiable for local runs too.
- **action**: The run-local skill's Stage 15 (Final Polish) should include PDF generation as a mandatory substep, not a separate manual action.
