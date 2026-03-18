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
