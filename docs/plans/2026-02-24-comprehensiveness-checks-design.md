# Comprehensiveness Checks Design

## Problem

The AutoReview pipeline has no quantitative validation that its literature search is comprehensive. Papers can be missed due to narrow queries, aggressive screening, or incomplete gap filling — with no warnings.

## Solution

A dedicated `autoreview/analysis/comprehensiveness.py` module implementing 5 checks that run at natural pipeline checkpoints. Results accumulate in `kb.comprehensiveness_checks`. All checks log warnings via structlog; the pipeline always continues.

## Data Models

```python
class CheckStatus(str, Enum):
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"

class ComprehensiveCheckResult(AutoReviewModel):
    check_name: str
    status: CheckStatus
    score: float              # 0.0-1.0
    details: str
    metrics: dict[str, Any]

class ComprehensivenessReport(TimestampedModel):
    checks: list[ComprehensiveCheckResult]
    overall_passed: bool
```

`KnowledgeBase` gains: `comprehensiveness_checks: list[ComprehensiveCheckResult]`

## Checks

### 1. Query Coverage (after query_expansion)
- LLM evaluates whether generated queries cover all scope document sub-topics
- Returns per-sub-topic coverage assessment
- WARNING if any expected sub-topic lacks a matching query

### 2. Coverage Anomaly Detection (after screening)
- Pure computation:
  - Per-source result count (WARNING if any source returned 0)
  - Screening rejection rate (WARNING if >80% rejected)
  - Papers missing abstracts (WARNING if >50%)
  - Total screened count (WARNING if <10 papers)

### 3. Borderline Re-screening (after screening)
- Captures papers scored exactly 2 (currently discarded)
- Re-screens with enriched context from scope document
- Promotes any scoring >=3 on second pass

### 4. Post-Gap Re-validation (after gap_search)
- Re-runs GapDetector on updated evidence map
- Compares pre/post coverage scores
- WARNING if major gaps remain unfilled

### 5. Benchmark Validation (after gap_search)
- Finds highly-cited review via Semantic Scholar API
- Extracts reference DOIs
- Measures recall against pipeline's discovered papers
- WARNING if recall < 0.5

## Integration

```
query_expansion → query_coverage_check
screening       → coverage_anomaly_check + borderline_rescreening
gap_search      → post_gap_revalidation + benchmark_validation
```

## Files

- `autoreview/analysis/comprehensiveness.py` — all checks
- `autoreview/llm/prompts/comprehensiveness.py` — query coverage prompt
- `tests/test_analysis/test_comprehensiveness.py` — tests
- Modified: `autoreview/models/knowledge_base.py`, `autoreview/pipeline/nodes.py`, `autoreview/extraction/extractor.py`
