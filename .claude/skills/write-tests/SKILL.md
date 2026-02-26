---
name: write-tests
description: Use when writing tests for AutoReview modules - knows project conventions, mock patterns, conftest fixtures, and async setup
---

# Write Tests for AutoReview

## Overview

AutoReview uses pytest with `asyncio_mode = "auto"`. There is no shared StubLLMProvider — each test file defines its own local mock LLM class. All LLM responses must be wrapped in `LLMResponse` or `LLMStructuredResponse`.

## Quick Reference

| Convention | Value |
|---|---|
| Async mode | `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed |
| LLM mocking | Local `Mock*LLM` class per file, not a shared fixture |
| Response wrapping | Always `LLMStructuredResponse(parsed=..., input_tokens=N, output_tokens=N)` |
| Test data builders | Module-level `_make_*()` functions, not fixtures |
| Test organization | Classes with `Test*` prefix; sync + async mixed freely |
| First line | `from __future__ import annotations` |
| File naming | `tests/test_<module>/test_<source_file>.py` |
| KnowledgeBase | `KnowledgeBase(topic="...", domain="biomedical", output_dir="/tmp/autoreview_test")` |
| Paper IDs | Short strings: `"p1"`, `"p2"`, `"p3"` |
| Integration marker | `@pytest.mark.integration` for real API tests |

## Shared Fixtures (tests/conftest.py)

Five fixtures available — do NOT recreate these:
- `sample_paper` → `CandidatePaper`
- `sample_screened_paper(sample_paper)` → `ScreenedPaper`
- `sample_finding` → `Finding`
- `sample_extraction(sample_finding)` → `PaperExtraction`
- `sample_kb` → `KnowledgeBase` with `output_dir="/tmp/autoreview_test"`

## Mock LLM Patterns

### Pattern 1: Static response (simple unit tests)

```python
class MockLLM:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    async def generate_structured(self, prompt, response_model, system="", max_tokens=4096, temperature=0.0):
        self.calls.append({"prompt": prompt, "response_model": response_model, "system": system})
        return LLMStructuredResponse(parsed=self.response, input_tokens=500, output_tokens=300)

    async def generate(self, prompt, system="", max_tokens=4096, temperature=0.3):
        self.calls.append({"prompt": prompt, "system": system})
        return LLMResponse(content="Revised text.", input_tokens=400, output_tokens=200)
```

### Pattern 2: Sequence (loop/iteration tests)

```python
class MockSequenceLLM:
    def __init__(self, responses: list):
        self.responses = list(responses)
        self._idx = 0
        self.calls: list[dict] = []

    async def generate_structured(self, prompt, response_model, system="", max_tokens=4096, temperature=0.0):
        self.calls.append({"prompt": prompt, "response_model": response_model})
        resp = self.responses[min(self._idx, len(self.responses) - 1)]
        self._idx += 1
        return LLMStructuredResponse(parsed=resp, input_tokens=500, output_tokens=300)
```

### Pattern 3: Dispatch (multi-model tests)

```python
class MockDispatchLLM:
    async def generate_structured(self, prompt, response_model, system="", max_tokens=4096, temperature=0.0):
        name = response_model.__name__
        if name == "ClusteringResult":
            return LLMStructuredResponse(parsed=ClusteringResult(...), input_tokens=800, output_tokens=400)
        elif name == "ContradictionResult":
            return LLMStructuredResponse(parsed=ContradictionResult(...), input_tokens=600, output_tokens=300)
        raise ValueError(f"Unexpected model: {name}")
```

## Test Data Builders

Use `_make_*()` helpers at module level:

```python
def _make_extraction(paper_id: str = "p1") -> PaperExtraction:
    return PaperExtraction(
        paper_id=paper_id,
        key_findings=[Finding(claim="Test claim", evidence_strength=EvidenceStrength.MODERATE)],
        methods_summary="Standard methods",
        limitations=["Small sample"],
    )
```

## Assertions

```python
assert len(result) == 2                    # equality
assert "expected" in prompt                # prompt content testing
assert isinstance(obj, ExpectedType)       # type check
assert sorted(items) == ["a", "b"]         # unordered comparison
assert llm.calls[0]["response_model"] is TargetModel  # identity
```

**Prompt content testing** is a first-class pattern — record prompts in `self.calls`, assert keywords appear.

## External API Mocking

```python
from unittest.mock import AsyncMock, patch

@patch("autoreview.search.pubmed.PubMedSearch._sync_search_with_retry")
async def test_search(self, mock_search):
    mock_search.return_value = ["111", "222"]
    ...
```

## Running Tests

```bash
pytest tests/test_<module>/test_<file>.py -x --tb=line    # single file, short output
pytest tests/ -x --tb=line 2>&1 | tail -30                # full suite, truncated
pytest -m "not integration" --tb=line                      # skip real API tests
```

## Common Mistakes

| Mistake | Fix |
|---|---|
| Adding `@pytest.mark.asyncio` | Not needed — `asyncio_mode = "auto"` |
| Returning raw Pydantic model from mock | Wrap in `LLMStructuredResponse(parsed=..., input_tokens=N, output_tokens=N)` |
| Creating a shared StubLLMProvider | Define a local mock class in each test file |
| Using `MagicMock` for LLM | Use a plain class with `async def generate_structured` |
| Missing `from __future__ import annotations` | Always add as first import |
