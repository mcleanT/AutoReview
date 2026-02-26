# Pickup Prompt: Comprehensiveness Checks Implementation

Copy-paste the following into a new Claude Code session from the AutoReview project root:

---

```
Read the implementation plan at docs/plans/2026-02-24-comprehensiveness-checks.md and the design doc at docs/plans/2026-02-24-comprehensiveness-checks-design.md. Execute all 8 tasks in order using TDD (write failing test, verify failure, implement, verify pass, commit). The plan has exact code and file paths for each step.

Summary of what you're building:
- 5 comprehensiveness checks in autoreview/analysis/comprehensiveness.py
- Query coverage prompt in autoreview/llm/prompts/comprehensiveness.py
- Tests in tests/test_analysis/test_comprehensiveness.py
- Modifications to autoreview/models/knowledge_base.py (new field), autoreview/extraction/extractor.py (borderline tracking), and autoreview/pipeline/nodes.py (wiring checks into pipeline)

Task order:
1. ComprehensiveCheckResult model + KnowledgeBase field
2. CoverageAnomalyChecker (pure computation, no LLM)
3. QueryCoverageChecker (LLM-based) + prompt module
4. BorderlineRescreener + PaperScreener modification
5. PostGapRevalidator
6. BenchmarkValidator (Semantic Scholar API)
7. Wire all checks into pipeline nodes
8. Integration tests + full suite verification

Commit after each task. Run the full test suite after tasks 1, 4, 7, and 8 to catch regressions.
```
