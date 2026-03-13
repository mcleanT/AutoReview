# Phase 1.5: Year Filtering & Bibliography Injection — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add time-controlled year filtering to the core pipeline and build a standalone bibliography injection script for evaluation experiments.

**Architecture:** Year filtering is a post-filter in `SearchAggregator` applied after each source returns results. It parses `SearchConfig.date_range` into `(year_from, year_to)` bounds, drops papers outside range (and always drops `year=None` with structured logging). Bibliography injection is a standalone script in `paper/analysis/` that extracts references from a PDF, resolves them to `CandidatePaper` records via Semantic Scholar/OpenAlex, and saves a pre-populated `KnowledgeBase` snapshot.

**Tech Stack:** Python 3.11+, Pydantic, structlog, rapidfuzz, httpx, pypdf, pytest, pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-03-13-year-filter-bib-injection-design.md`

**Prerequisites:** Ensure `rapidfuzz` is in `pyproject.toml` dependencies (already declared but verify).

**Deferred:** LLM fallback for unparseable reference lines (spec section 2.3) is deferred to a follow-up task. The regex + heuristic parser handles the common formats; LLM fallback will be added when we have real data on the failure rate.

---

## Chunk 1: Year Filtering in SearchAggregator

### Task 1: Pydantic validator on `SearchConfig.date_range`

**Files:**
- Modify: `autoreview/config/models.py:6-17` (SearchConfig class)
- Test: `tests/test_config/test_models.py` (create)

- [ ] **Step 1: Write failing tests for date_range validation**

```python
# tests/test_config/test_models.py
from __future__ import annotations

import pytest
from pydantic import ValidationError

from autoreview.config.models import SearchConfig


class TestDateRangeValidator:
    def test_standard_range(self):
        cfg = SearchConfig(date_range="2015-2025")
        assert cfg.date_range == "2015-2025"

    def test_whitespace_stripped(self):
        cfg = SearchConfig(date_range=" 2015 - 2025 ")
        assert cfg.date_range == "2015-2025"

    def test_open_start(self):
        cfg = SearchConfig(date_range="-2020")
        assert cfg.date_range == "-2020"

    def test_open_end(self):
        cfg = SearchConfig(date_range="2020-")
        assert cfg.date_range == "2020-"

    def test_same_year(self):
        cfg = SearchConfig(date_range="2020-2020")
        assert cfg.date_range == "2020-2020"

    def test_empty_string(self):
        cfg = SearchConfig(date_range="")
        assert cfg.date_range == ""

    def test_reversed_range_rejected(self):
        with pytest.raises(ValidationError, match="date_range"):
            SearchConfig(date_range="2025-2015")

    def test_malformed_rejected(self):
        with pytest.raises(ValidationError, match="date_range"):
            SearchConfig(date_range="2015-01-2025")

    def test_non_numeric_rejected(self):
        with pytest.raises(ValidationError, match="date_range"):
            SearchConfig(date_range="abc-def")

    def test_bare_dash_rejected(self):
        with pytest.raises(ValidationError, match="date_range"):
            SearchConfig(date_range="-")

    def test_default_unchanged(self):
        cfg = SearchConfig()
        assert cfg.date_range == "2015-2025"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config/test_models.py -v`
Expected: FAIL — no validator exists yet, reversed/malformed inputs accepted

- [ ] **Step 3: Implement the validator**

Add to `autoreview/config/models.py`, inside the `SearchConfig` class after line 16:

```python
import re
from pydantic import field_validator

# Inside SearchConfig:
    @field_validator("date_range", mode="before")
    @classmethod
    def validate_date_range(cls, v: str | None) -> str:
        if v is None:
            return ""
        v = re.sub(r"\s+", "", str(v))  # strip all whitespace
        if not v:
            return ""
        # Match: YYYY-YYYY, -YYYY, YYYY-, or empty
        m = re.fullmatch(r"(\d{4})?-(\d{4})?", v)
        if not m or v == "-":
            raise ValueError(
                f"date_range must be 'YYYY-YYYY', '-YYYY', 'YYYY-', or empty; got '{v}'"
            )
        year_from_str, year_to_str = m.group(1), m.group(2)
        if year_from_str and year_to_str:
            if int(year_from_str) > int(year_to_str):
                raise ValueError(
                    f"date_range start ({year_from_str}) must be <= end ({year_to_str})"
                )
        return v
```

Also add `import re` and `field_validator` to the imports at top of file (re is not yet imported; field_validator is not yet imported).

Additionally, add `validate_assignment=True` to `SearchConfig`'s `model_config` so that direct assignment (e.g., from CLI `--date-range`) also triggers the validator:

```python
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config/test_models.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/config/models.py tests/test_config/__init__.py tests/test_config/test_models.py
git commit -m "feat: add Pydantic validator for SearchConfig.date_range"
```

---

### Task 2: Year filtering methods in SearchAggregator

**Files:**
- Modify: `autoreview/search/aggregator.py:42-119`
- Test: `tests/test_search/test_aggregator.py`

- [ ] **Step 1: Write failing tests for `_parse_date_range` and `_filter_by_year`**

Append to `tests/test_search/test_aggregator.py`:

```python
from autoreview.search.aggregator import _parse_date_range, _filter_by_year


class TestParseDateRange:
    def test_full_range(self):
        assert _parse_date_range("2015-2025") == (2015, 2025)

    def test_open_start(self):
        assert _parse_date_range("-2020") == (None, 2020)

    def test_open_end(self):
        assert _parse_date_range("2020-") == (2020, None)

    def test_none(self):
        assert _parse_date_range(None) == (None, None)

    def test_empty_string(self):
        assert _parse_date_range("") == (None, None)

    def test_same_year(self):
        assert _parse_date_range("2020-2020") == (2020, 2020)


class TestFilterByYear:
    def _make_paper(self, title: str, year: int | None) -> CandidatePaper:
        return CandidatePaper(
            title=title, authors=["A"], source_database="test", year=year,
        )

    def test_filters_below_range(self):
        papers = [self._make_paper("Old", 2010), self._make_paper("In", 2018)]
        result = _filter_by_year(papers, year_from=2015, year_to=2020)
        assert len(result) == 1
        assert result[0].title == "In"

    def test_filters_above_range(self):
        papers = [self._make_paper("Future", 2030), self._make_paper("In", 2018)]
        result = _filter_by_year(papers, year_from=2015, year_to=2020)
        assert len(result) == 1
        assert result[0].title == "In"

    def test_inclusive_bounds(self):
        papers = [
            self._make_paper("Start", 2015),
            self._make_paper("End", 2020),
            self._make_paper("Mid", 2018),
        ]
        result = _filter_by_year(papers, year_from=2015, year_to=2020)
        assert len(result) == 3

    def test_drops_year_none(self):
        papers = [self._make_paper("No Year", None), self._make_paper("Has Year", 2020)]
        result = _filter_by_year(papers, year_from=2015, year_to=2025)
        assert len(result) == 1
        assert result[0].title == "Has Year"

    def test_drops_year_none_even_without_range(self):
        """year=None always dropped when any date_range is set."""
        papers = [self._make_paper("No Year", None), self._make_paper("Has Year", 2020)]
        result = _filter_by_year(papers, year_from=None, year_to=2025)
        assert len(result) == 1

    def test_no_filtering_when_both_none(self):
        """No filtering when both bounds are None (no date_range set).
        Note: year=None papers pass through only when no date_range is active.
        When any bound is set, year=None is always dropped per spec."""
        papers = [self._make_paper("No Year", None), self._make_paper("Has Year", 2020)]
        result = _filter_by_year(papers, year_from=None, year_to=None)
        assert len(result) == 2

    def test_open_end_range(self):
        papers = [self._make_paper("Old", 2010), self._make_paper("New", 2025)]
        result = _filter_by_year(papers, year_from=2020, year_to=None)
        assert len(result) == 1
        assert result[0].title == "New"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_search/test_aggregator.py::TestParseDateRange tests/test_search/test_aggregator.py::TestFilterByYear -v`
Expected: FAIL — functions not defined

- [ ] **Step 3: Implement `_parse_date_range` and `_filter_by_year`**

Add to `autoreview/search/aggregator.py` as module-level functions (before the `SearchAggregator` class, after the existing `_merge_papers` function):

```python
def _parse_date_range(date_range: str | None) -> tuple[int | None, int | None]:
    """Parse 'YYYY-YYYY', '-YYYY', 'YYYY-', or None into (year_from, year_to).

    Range is inclusive on both bounds. Returns (None, None) for empty/None input.
    """
    if not date_range:
        return (None, None)
    parts = date_range.strip().split("-", 1)
    # Handle formats: "YYYY-YYYY", "-YYYY", "YYYY-"
    year_from: int | None = int(parts[0]) if parts[0] else None
    year_to: int | None = int(parts[1]) if len(parts) > 1 and parts[1] else None
    return (year_from, year_to)


def _filter_by_year(
    papers: list[CandidatePaper],
    year_from: int | None,
    year_to: int | None,
) -> list[CandidatePaper]:
    """Drop papers outside the year range. Always drop year=None with logged warning.

    When both year_from and year_to are None (no date_range set), returns all
    papers unfiltered.
    """
    if year_from is None and year_to is None:
        return papers

    filtered: list[CandidatePaper] = []
    for paper in papers:
        if paper.year is None:
            logger.warning(
                "year_filter.dropped_null_year",
                title=paper.title[:80],
                source_database=paper.source_database,
                doi=paper.doi,
            )
            continue
        if year_from is not None and paper.year < year_from:
            continue
        if year_to is not None and paper.year > year_to:
            continue
        filtered.append(paper)

    dropped = len(papers) - len(filtered)
    if dropped:
        logger.info("year_filter.applied", kept=len(filtered), dropped=dropped,
                     year_from=year_from, year_to=year_to)
    return filtered
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_search/test_aggregator.py -v`
Expected: All PASS (existing + new tests)

- [ ] **Step 5: Commit**

```bash
git add autoreview/search/aggregator.py tests/test_search/test_aggregator.py
git commit -m "feat: add year filtering functions to SearchAggregator"
```

---

### Task 3: Wire year filter into `SearchAggregator.search()`

**Files:**
- Modify: `autoreview/search/aggregator.py:42-82` (SearchAggregator class)
- Test: `tests/test_search/test_aggregator.py`

- [ ] **Step 1: Write failing test for filtering in `search()`**

Append to `tests/test_search/test_aggregator.py`:

```python
class TestAggregatorSearchWithYearFilter:
    @pytest.mark.asyncio
    async def test_filters_papers_by_year(self):
        class MockSource:
            source_name = "mock"
            async def search(self, queries, max_results=100):
                return [
                    CandidatePaper(title="Old", authors=["A"], source_database="mock", year=2010),
                    CandidatePaper(title="In Range", authors=["A"], source_database="mock", year=2018),
                    CandidatePaper(title="Future", authors=["A"], source_database="mock", year=2030),
                ]

        agg = SearchAggregator(sources=[MockSource()], date_range="2015-2020")
        result = await agg.search({"mock": ["test"]})
        assert len(result) == 1
        assert result[0].title == "In Range"

    @pytest.mark.asyncio
    async def test_no_filter_without_date_range(self):
        class MockSource:
            source_name = "mock"
            async def search(self, queries, max_results=100):
                return [
                    CandidatePaper(title="A", authors=["A"], source_database="mock", year=2010),
                    CandidatePaper(title="B", authors=["A"], source_database="mock", year=2020),
                ]

        agg = SearchAggregator(sources=[MockSource()])
        result = await agg.search({"mock": ["test"]})
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_drops_null_year_with_date_range(self):
        class MockSource:
            source_name = "mock"
            async def search(self, queries, max_results=100):
                return [
                    CandidatePaper(title="No Year", authors=["A"], source_database="mock", year=None),
                    CandidatePaper(title="Has Year", authors=["A"], source_database="mock", year=2018),
                ]

        agg = SearchAggregator(sources=[MockSource()], date_range="2015-2020")
        result = await agg.search({"mock": ["test"]})
        assert len(result) == 1
        assert result[0].title == "Has Year"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_search/test_aggregator.py::TestAggregatorSearchWithYearFilter -v`
Expected: FAIL — `SearchAggregator.__init__` doesn't accept `date_range`

- [ ] **Step 3: Wire filter into SearchAggregator**

Modify `SearchAggregator.__init__` to accept `date_range` and apply filter in `search()`:

```python
class SearchAggregator:
    """Aggregates results from multiple search sources with deduplication."""

    def __init__(self, sources: list[Any] | None = None, date_range: str | None = None) -> None:
        self.sources: list[Any] = sources or []
        self._year_from, self._year_to = _parse_date_range(date_range)

    # ... add_source stays the same ...

    async def search(
        self,
        queries_by_source: dict[str, list[str]],
        max_results_per_source: int = 500,
    ) -> list[CandidatePaper]:
        tasks = []
        source_names = []
        for source in self.sources:
            name = source.source_name
            qs = queries_by_source.get(name, [])
            if not qs:
                continue
            tasks.append(source.search(qs, max_results_per_source))
            source_names.append(name)

        if not tasks:
            logger.warning("aggregator.no_sources_with_queries")
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_papers: list[CandidatePaper] = []
        for name, result in zip(source_names, results):
            if isinstance(result, Exception):
                logger.error("aggregator.source_failed", source=name, error=str(result))
                continue
            # Apply year filter per-source before aggregation
            filtered = _filter_by_year(result, self._year_from, self._year_to)
            logger.info("aggregator.source_results", source=name,
                        raw=len(result), after_year_filter=len(filtered))
            all_papers.extend(filtered)

        deduplicated = self._deduplicate(all_papers)
        logger.info("aggregator.complete", total_raw=len(all_papers), deduplicated=len(deduplicated))
        return deduplicated
```

- [ ] **Step 4: Run all aggregator tests**

Run: `pytest tests/test_search/test_aggregator.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/search/aggregator.py tests/test_search/test_aggregator.py
git commit -m "feat: wire year filter into SearchAggregator.search()"
```

---

### Task 4: Pass `date_range` at all 5 SearchAggregator sites in `nodes.py`

**Files:**
- Modify: `autoreview/pipeline/nodes.py` — lines 290, 582, 800, 981, 1176

Each `SearchAggregator(sources=sources)` call must become `SearchAggregator(sources=sources, date_range=self.config.search.date_range)`.

- [ ] **Step 1: Write a grep-based verification test**

Create `tests/test_pipeline/test_year_filter_wiring.py`:

```python
"""Verify date_range is passed at all SearchAggregator instantiation sites in nodes.py."""
from __future__ import annotations

from pathlib import Path


def test_all_aggregator_sites_pass_date_range():
    """Every SearchAggregator() call in nodes.py must include date_range=."""
    nodes_path = Path(__file__).parent.parent.parent / "autoreview" / "pipeline" / "nodes.py"
    content = nodes_path.read_text()

    import re
    sites = [m.start() for m in re.finditer(r"SearchAggregator\(", content)]
    assert len(sites) >= 5, f"Expected >=5 SearchAggregator sites, found {len(sites)}"

    for pos in sites:
        # Extract the full constructor call (up to closing paren)
        snippet = content[pos:pos + 200]
        assert "date_range" in snippet, (
            f"SearchAggregator at char {pos} missing date_range=. "
            f"Snippet: {snippet[:100]!r}"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline/test_year_filter_wiring.py -v`
Expected: FAIL — no `date_range` in any site

- [ ] **Step 3: Update all 5 sites in `nodes.py`**

At each of the 5 lines, change:
```python
# Line ~290 (primary search):
agg = SearchAggregator(sources=sources)
# →
agg = SearchAggregator(sources=sources, date_range=self.config.search.date_range)

# Line ~582 (gap search):
agg = SearchAggregator(sources=sources)
# →
agg = SearchAggregator(sources=sources, date_range=self.config.search.date_range)

# Line ~800 (contextual enrichment):
agg = SearchAggregator(sources=sources)
# →
agg = SearchAggregator(sources=sources, date_range=self.config.search.date_range)

# Line ~981 (corpus expansion):
agg = SearchAggregator(sources=sources)
# →
agg = SearchAggregator(sources=sources, date_range=self.config.search.date_range)

# Line ~1176 (passage search):
agg = SearchAggregator(sources=sources)
# →
agg = SearchAggregator(sources=sources, date_range=self.config.search.date_range)
```

- [ ] **Step 4: Run wiring test**

Run: `pytest tests/test_pipeline/test_year_filter_wiring.py -v`
Expected: PASS

- [ ] **Step 5: Run existing pipeline tests to check for regressions**

Run: `pytest tests/test_pipeline/ tests/test_search/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add autoreview/pipeline/nodes.py tests/test_pipeline/test_year_filter_wiring.py
git commit -m "feat: pass date_range to all 5 SearchAggregator sites in pipeline"
```

---

### Task 5: `--date-range` CLI flag

**Files:**
- Modify: `autoreview/cli.py:55-136` (run command) and `autoreview/cli.py:138-191` (resume command)
- Test: `tests/test_cli/test_date_range.py` (create)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cli/test_date_range.py
from __future__ import annotations

from typer.testing import CliRunner
from unittest.mock import patch, MagicMock, AsyncMock

from autoreview.cli import app

runner = CliRunner()


class TestDateRangeCLI:
    @patch("autoreview.cli.asyncio.run")
    @patch("autoreview.cli.create_llm_provider")
    @patch("autoreview.cli.load_config")
    def test_run_passes_date_range_to_config(
        self, mock_load, mock_llm, mock_run
    ):
        mock_config = MagicMock()
        mock_config.llm = MagicMock()
        mock_config.writing.citation_format = "apa"
        mock_load.return_value = mock_config
        mock_llm.return_value = MagicMock()
        mock_kb = MagicMock()
        mock_kb.output_dir = "output"
        mock_run.return_value = mock_kb

        result = runner.invoke(app, [
            "run", "test topic", "--date-range", "-2019"
        ])

        # Verify date_range was set on config
        assert mock_config.search.date_range == "-2019"

    @patch("autoreview.cli.asyncio.run")
    @patch("autoreview.cli.create_llm_provider")
    @patch("autoreview.cli.load_config")
    @patch("autoreview.cli.KnowledgeBase.load_snapshot")
    def test_resume_passes_date_range(self, mock_snap, mock_load, mock_llm, mock_run):
        mock_kb = MagicMock()
        mock_kb.topic = "test"
        mock_kb.domain = "general"
        mock_kb.current_phase = "search"
        mock_kb.candidate_papers = []
        mock_kb.screened_papers = []
        mock_kb.output_dir = "output"
        mock_snap.return_value = mock_kb

        mock_config = MagicMock()
        mock_config.llm = MagicMock()
        mock_config.writing.citation_format = "apa"
        mock_load.return_value = mock_config
        mock_llm.return_value = MagicMock()
        mock_run.return_value = mock_kb

        result = runner.invoke(app, [
            "resume", "snapshot.json", "--date-range", "2015-2019"
        ])

        assert mock_config.search.date_range == "2015-2019"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli/test_date_range.py -v`
Expected: FAIL — `--date-range` option doesn't exist

- [ ] **Step 3: Add `--date-range` to `run` and `resume` commands**

In `autoreview/cli.py`, add the parameter to the `run` function signature (after `fresh`):

```python
    date_range: str | None = typer.Option(
        None, "--date-range", help="Year range filter, e.g. '2015-2020', '-2019', '2020-'"
    ),
```

Then after `config = load_config(...)`, add:

```python
    if date_range is not None:
        config.search.date_range = date_range
```

Similarly for `resume`, add the same `date_range` parameter and apply it after `config = load_config(domain=kb.domain)`:

```python
    if date_range is not None:
        config.search.date_range = date_range
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_cli/test_date_range.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add autoreview/cli.py tests/test_cli/__init__.py tests/test_cli/test_date_range.py
git commit -m "feat: add --date-range CLI flag to run and resume commands"
```

---

## Chunk 2: Bibliography Injection (Evaluation Tooling)

### Task 6: Reference line parser

**Files:**
- Create: `paper/__init__.py` (empty — makes `paper/` importable as a package)
- Create: `paper/analysis/__init__.py` (empty)
- Create: `paper/analysis/reference_parser.py`
- Test: `tests/test_analysis/__init__.py` (empty)
- Test: `tests/test_analysis/test_reference_parser.py`

**Note on Python path:** `paper/` is outside `autoreview/`. For imports to work, either add `paper` to `pyproject.toml`'s `[tool.setuptools.packages.find]` include list, or run tests with `PYTHONPATH=. pytest`. The simplest approach: add `paper/__init__.py` and ensure the project root is on the path (which `pip install -e .` already does for packages, but `paper/` as a standalone namespace needs the `__init__.py` chain).

- [ ] **Step 1: Write failing tests for DOI extraction and title heuristic**

```python
# tests/test_analysis/test_reference_parser.py
from __future__ import annotations

import pytest

from paper.analysis.reference_parser import extract_doi, extract_title_heuristic, parse_reference_line


class TestExtractDoi:
    def test_extracts_doi_from_line(self):
        line = "[1] Smith J et al. Title here. Nature. 2020. doi:10.1038/s41586-020-1234-5"
        assert extract_doi(line) == "10.1038/s41586-020-1234-5"

    def test_extracts_doi_url(self):
        line = "[2] Doe A. Title. https://doi.org/10.1126/science.abc1234"
        assert extract_doi(line) == "10.1126/science.abc1234"

    def test_returns_none_when_no_doi(self):
        line = "[3] Zhang L. Some paper title. Journal of Things. 2019;45:123-130."
        assert extract_doi(line) is None

    def test_handles_parenthesized_doi(self):
        line = "[4] Title (doi: 10.1234/test.5678)"
        assert extract_doi(line) == "10.1234/test.5678"


class TestExtractTitleHeuristic:
    def test_bracket_numbered_format(self):
        line = "[1] Smith J, Doe A. The role of X in Y. Nature. 2020;580:123-130."
        title = extract_title_heuristic(line)
        assert "role of X in Y" in title

    def test_dot_numbered_format(self):
        line = "1. Smith J, Doe A. The role of X in Y. Nature. 2020;580:123-130."
        title = extract_title_heuristic(line)
        assert "role of X in Y" in title

    def test_returns_empty_for_unparseable(self):
        line = ""
        title = extract_title_heuristic(line)
        assert title == ""


class TestParseReferenceLine:
    def test_returns_doi_and_title(self):
        line = "[1] Smith J. Great discovery in science. Nature. 2020. doi:10.1038/test123"
        result = parse_reference_line(line)
        assert result["doi"] == "10.1038/test123"
        assert "Great discovery" in result["title"]

    def test_returns_title_only_when_no_doi(self):
        line = "[2] Doe A. Another finding. Science. 2019;365:100-105."
        result = parse_reference_line(line)
        assert result["doi"] is None
        assert len(result["title"]) > 0

    def test_returns_raw_line(self):
        line = "[1] Smith J. Great discovery. Nature. 2020."
        result = parse_reference_line(line)
        assert result["raw"] == line
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analysis/test_reference_parser.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement reference parser**

Create `paper/analysis/__init__.py` (empty) and `paper/analysis/reference_parser.py`:

```python
"""Parse raw reference lines into structured data (DOI + title)."""
from __future__ import annotations

import re

import structlog

logger = structlog.get_logger()

# DOI regex — matches 10.XXXX/... patterns
# Spec uses [-._;()/:A-Z0-9]+ (strict allowlist); we use [^\s,;)\]]+ (greedy negated class)
# to handle lowercase, unicode, and unusual DOI characters seen in real data.
# Trailing punctuation is stripped by extract_doi().
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s,;)\]]+")

# Common journal name patterns that signal end of title
_JOURNAL_SIGNALS = re.compile(
    r"\b(?:Nature|Science|Cell|JAMA|Lancet|BMJ|PLoS|PNAS|Neuron|"
    r"J\.\s|Journal|Proceedings|Annals|Archives|Reviews|Frontiers)\b",
    re.IGNORECASE,
)

# Year pattern in references (e.g., "2020", "2020;", "(2020)")
_YEAR_RE = re.compile(r"(?:^|\D)((?:19|20)\d{2})(?:\D|$)")


def extract_doi(line: str) -> str | None:
    """Extract a DOI from a reference line.

    Handles formats: doi:10.xxx, https://doi.org/10.xxx, bare 10.xxx/yyy.
    Returns the DOI without URL prefix, or None if not found.
    """
    # Strip URL prefix if present
    cleaned = re.sub(r"https?://(?:dx\.)?doi\.org/", "", line)
    m = _DOI_RE.search(cleaned)
    if not m:
        return None
    doi = m.group(0)
    # Strip trailing punctuation that's not part of DOI
    doi = doi.rstrip(".,;)")
    return doi


def extract_title_heuristic(line: str) -> str:
    """Extract an approximate title from a reference line using heuristics.

    Strategy: strip leading number/bracket prefix, skip author block (up to first period
    after names), take text until the next period that looks like a journal signal or year.
    """
    if not line.strip():
        return ""

    # Strip leading [N] or N. prefix
    text = re.sub(r"^\s*(?:\[\d+\]|\d+\.)\s*", "", line).strip()

    # Split on periods
    segments = text.split(". ")
    if len(segments) < 2:
        return text[:200]  # can't parse, return truncated raw

    # Heuristic: first segment is usually authors, second is title
    # But authors can span multiple segments if they contain initials (e.g., "Smith J. A., Doe B.")
    # Look for the first segment that doesn't look like an author list
    for i, seg in enumerate(segments):
        # Author segments are typically short and contain mostly names/initials
        # Title segments are typically longer and contain more diverse words
        if i == 0:
            continue  # skip first segment (almost always authors)
        # Check if this looks like a title (>4 words, or contains non-name-like words)
        words = seg.split()
        if len(words) >= 3:
            return seg.strip().rstrip(".")
        # If it's short, it might be part of the author block
        if i >= 3:
            # Give up after 3 author segments
            return seg.strip().rstrip(".")

    # Fallback: return second segment
    return segments[1].strip().rstrip(".") if len(segments) > 1 else ""


def parse_reference_line(line: str) -> dict[str, str | None]:
    """Parse a single reference line into structured data.

    Returns:
        dict with keys: 'raw' (original line), 'doi' (str|None), 'title' (str).
    """
    doi = extract_doi(line)
    title = extract_title_heuristic(line)
    return {"raw": line, "doi": doi, "title": title}


def parse_all_references(lines: list[str]) -> list[dict[str, str | None]]:
    """Parse a list of reference lines into structured data.

    Args:
        lines: Raw reference lines from pdf_extractor.extract_bibliography_lines().

    Returns:
        List of dicts, each with 'raw', 'doi', and 'title' keys.
    """
    parsed = []
    for line in lines:
        result = parse_reference_line(line)
        parsed.append(result)
        if result["doi"]:
            logger.debug("reference_parser.doi_found", doi=result["doi"], title=result["title"])
        else:
            logger.debug("reference_parser.no_doi", title=result["title"][:60] if result["title"] else "")
    logger.info(
        "reference_parser.complete",
        total=len(parsed),
        with_doi=sum(1 for p in parsed if p["doi"]),
    )
    return parsed
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_analysis/test_reference_parser.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add paper/__init__.py paper/analysis/__init__.py paper/analysis/reference_parser.py tests/test_analysis/__init__.py tests/test_analysis/test_reference_parser.py
git commit -m "feat: add reference line parser with DOI extraction and title heuristic"
```

---

### Task 7: Reference resolver

**Files:**
- Create: `paper/analysis/reference_resolver.py`
- Test: `tests/test_analysis/test_reference_resolver.py`

- [ ] **Step 1: Write failing tests for resolution logic**

```python
# tests/test_analysis/test_reference_resolver.py
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from autoreview.models.paper import CandidatePaper
from paper.analysis.reference_resolver import (
    resolve_reference,
    ResolvedReference,
    ResolutionConfidence,
)


def _make_paper(title: str = "Test Paper", doi: str = "10.1234/test") -> CandidatePaper:
    return CandidatePaper(
        title=title, authors=["Smith J"], year=2020,
        source_database="semantic_scholar", doi=doi,
    )


class TestResolveReference:
    @pytest.mark.asyncio
    async def test_resolves_by_doi(self):
        mock_s2 = AsyncMock()
        mock_s2.get_paper_details = AsyncMock(return_value=_make_paper())

        result = await resolve_reference(
            parsed={"raw": "[1] ...", "doi": "10.1234/test", "title": "Test Paper"},
            search_sources=[mock_s2],
        )
        assert result is not None
        assert result.confidence == ResolutionConfidence.HIGH
        assert result.paper.doi == "10.1234/test"
        mock_s2.get_paper_details.assert_called_once_with("DOI:10.1234/test")

    @pytest.mark.asyncio
    async def test_falls_back_to_title_search(self):
        mock_s2 = AsyncMock()
        mock_s2.get_paper_details = AsyncMock(return_value=None)
        mock_s2.source_name = "semantic_scholar"
        mock_s2.search = AsyncMock(return_value=[_make_paper(title="Test Paper Exact")])

        result = await resolve_reference(
            parsed={"raw": "[1] ...", "doi": None, "title": "Test Paper Exact"},
            search_sources=[mock_s2],
        )
        assert result is not None
        assert result.confidence in (ResolutionConfidence.MEDIUM, ResolutionConfidence.HIGH)

    @pytest.mark.asyncio
    async def test_returns_none_when_unresolvable(self):
        mock_s2 = AsyncMock()
        mock_s2.get_paper_details = AsyncMock(return_value=None)
        mock_s2.source_name = "semantic_scholar"
        mock_s2.search = AsyncMock(return_value=[])

        result = await resolve_reference(
            parsed={"raw": "[1] ...", "doi": None, "title": "Nonexistent Paper"},
            search_sources=[mock_s2],
        )
        assert result is None


class TestResolutionConfidence:
    def test_high_for_doi_match(self):
        assert ResolutionConfidence.HIGH.value == "high"

    def test_medium_for_fuzzy_match(self):
        assert ResolutionConfidence.MEDIUM.value == "medium"

    def test_low_for_weak_match(self):
        assert ResolutionConfidence.LOW.value == "low"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analysis/test_reference_resolver.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement reference resolver**

Create `paper/analysis/reference_resolver.py`:

```python
"""Resolve parsed reference lines to CandidatePaper records via academic APIs."""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel
from rapidfuzz import fuzz

from autoreview.models.paper import CandidatePaper

logger = structlog.get_logger()


class ResolutionConfidence(str, Enum):
    HIGH = "high"      # DOI exact match
    MEDIUM = "medium"  # Title fuzzy match >= 85%
    LOW = "low"        # Title fuzzy match 70-85%


class ResolvedReference(BaseModel):
    """A reference line that was successfully resolved to a paper."""
    raw_line: str
    paper: CandidatePaper
    confidence: ResolutionConfidence
    match_score: float = 1.0  # 1.0 for DOI, fuzzy ratio for title


async def resolve_reference(
    parsed: dict[str, str | None],
    search_sources: list[Any],
) -> ResolvedReference | None:
    """Resolve a single parsed reference to a CandidatePaper.

    Strategy:
    1. If DOI available, look up via get_paper_details("DOI:xxx")
    2. If no DOI or DOI lookup fails, search by title and fuzzy-match
    3. Return None if unresolvable

    Args:
        parsed: Dict with 'raw', 'doi', 'title' keys from reference_parser.
        search_sources: List of search source instances (S2, OpenAlex, etc.)

    Returns:
        ResolvedReference or None if unresolvable.
    """
    raw = parsed["raw"] or ""
    doi = parsed.get("doi")
    title = parsed.get("title") or ""

    # Strategy 1: DOI lookup
    if doi:
        for source in search_sources:
            try:
                paper = await source.get_paper_details(f"DOI:{doi}")
                if paper is not None:
                    logger.info("resolver.doi_match", doi=doi, source=getattr(source, "source_name", "unknown"))
                    return ResolvedReference(
                        raw_line=raw, paper=paper,
                        confidence=ResolutionConfidence.HIGH, match_score=1.0,
                    )
            except Exception as e:
                logger.warning("resolver.doi_lookup_failed", doi=doi, error=str(e))

    # Strategy 2: Title search + fuzzy match
    # Note: individual search sources use search(queries: list[str], max_results: int),
    # NOT the SearchAggregator dict-based signature.
    if title and len(title) > 10:
        for source in search_sources:
            try:
                source_name = getattr(source, "source_name", "unknown")
                results = await source.search([title], 5)
                if not results:
                    continue

                # Find best fuzzy match
                best_score = 0.0
                best_paper = None
                for candidate in results:
                    score = fuzz.ratio(title.lower(), candidate.title.lower())
                    if score > best_score:
                        best_score = score
                        best_paper = candidate

                if best_paper and best_score >= 70:
                    if best_score >= 85:
                        confidence = ResolutionConfidence.MEDIUM
                    else:
                        confidence = ResolutionConfidence.LOW
                    logger.info(
                        "resolver.title_match",
                        title=title[:60], score=best_score,
                        confidence=confidence.value, source=source_name,
                    )
                    return ResolvedReference(
                        raw_line=raw, paper=best_paper,
                        confidence=confidence, match_score=best_score / 100,
                    )
            except Exception as e:
                logger.warning("resolver.title_search_failed", title=title[:60], error=str(e))

    logger.warning("resolver.unresolvable", raw=raw[:80])
    return None


class ResolutionCache:
    """Cache resolution results keyed by reference line hash."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self._data: dict[str, dict] = {}
        if cache_path.exists():
            self._data = json.loads(cache_path.read_text())

    def _hash(self, line: str) -> str:
        return hashlib.sha256(line.encode()).hexdigest()[:16]

    def get(self, raw_line: str) -> ResolvedReference | None:
        key = self._hash(raw_line)
        entry = self._data.get(key)
        if entry is None:
            return None
        return ResolvedReference.model_validate(entry)

    def put(self, resolved: ResolvedReference) -> None:
        key = self._hash(resolved.raw_line)
        self._data[key] = resolved.model_dump(mode="json")

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._data, indent=2))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_analysis/test_reference_resolver.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add paper/analysis/reference_resolver.py tests/test_analysis/test_reference_resolver.py
git commit -m "feat: add reference resolver with DOI lookup, title fuzzy match, and cache"
```

---

### Task 8: Bibliography injection script

**Files:**
- Create: `paper/analysis/inject_bibliography.py`
- Test: `tests/test_analysis/test_inject_bibliography.py`

- [ ] **Step 1: Write failing integration test**

```python
# tests/test_analysis/test_inject_bibliography.py
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from autoreview.models.paper import CandidatePaper


class TestInjectBibliography:
    @pytest.mark.asyncio
    async def test_builds_knowledge_base_from_resolved_refs(self):
        from paper.analysis.inject_bibliography import build_injected_kb

        papers = [
            CandidatePaper(
                title="Paper A", authors=["Smith"], year=2018,
                doi="10.1234/a", source_database="semantic_scholar",
            ),
            CandidatePaper(
                title="Paper B", authors=["Doe"], year=2019,
                doi="10.1234/b", source_database="openalex",
            ),
        ]

        kb = build_injected_kb(
            papers=papers,
            topic="Test topic",
            domain="biomedical",
            output_dir="/tmp/test_inject",
        )

        assert kb.topic == "Test topic"
        assert kb.domain == "biomedical"
        assert len(kb.candidate_papers) == 2
        assert len(kb.screened_papers) == 2
        assert all(sp.include for sp in kb.screened_papers)
        assert all(sp.relevance_score == 5 for sp in kb.screened_papers)

    def test_resolution_report(self):
        from paper.analysis.inject_bibliography import build_resolution_report
        from paper.analysis.reference_resolver import (
            ResolvedReference,
            ResolutionConfidence,
        )

        resolved = [
            ResolvedReference(
                raw_line="[1] ...", confidence=ResolutionConfidence.HIGH,
                match_score=1.0,
                paper=CandidatePaper(title="A", authors=["X"], source_database="s2", doi="10.1/a"),
            ),
            ResolvedReference(
                raw_line="[2] ...", confidence=ResolutionConfidence.MEDIUM,
                match_score=0.9,
                paper=CandidatePaper(title="B", authors=["Y"], source_database="s2"),
            ),
        ]
        failed = ["[3] Unresolvable reference line"]

        report = build_resolution_report(
            total_extracted=3, resolved=resolved, failed_lines=failed,
        )
        assert report["total_extracted"] == 3
        assert report["resolved_count"] == 2
        assert report["failed_count"] == 1
        assert report["by_confidence"]["high"] == 1
        assert report["by_confidence"]["medium"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analysis/test_inject_bibliography.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement bibliography injection script**

Create `paper/analysis/inject_bibliography.py`:

```python
"""Bibliography injection: extract references from a PDF, resolve, build KnowledgeBase.

Usage:
    python paper/analysis/inject_bibliography.py \
        --pdf paper/references/review.pdf \
        --topic "Topic string" \
        --domain biomedical \
        --output paper/snapshots/injected.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

import structlog

from autoreview.evaluation.pdf_extractor import extract_text_from_pdf, extract_bibliography_lines
from autoreview.models.knowledge_base import KnowledgeBase, PipelinePhase
from autoreview.models.paper import CandidatePaper, ScreenedPaper

from paper.analysis.reference_parser import parse_all_references
from paper.analysis.reference_resolver import (
    ResolutionCache,
    ResolvedReference,
    resolve_reference,
)

logger = structlog.get_logger()


def build_injected_kb(
    papers: list[CandidatePaper],
    topic: str,
    domain: str,
    output_dir: str,
) -> KnowledgeBase:
    """Build a pre-populated KnowledgeBase from resolved papers.

    All papers are added as candidates and auto-screened with include=True,
    relevance_score=5. Full text is NOT populated — the pipeline will fetch
    it when resumed from full_text_retrieval.
    """
    screened = [
        ScreenedPaper(
            paper=paper,
            relevance_score=5,
            rationale="Injected from reference bibliography",
            include=True,
        )
        for paper in papers
    ]

    kb = KnowledgeBase(
        topic=topic,
        domain=domain,
        output_dir=output_dir,
        candidate_papers=papers,
        screened_papers=screened,
        current_phase=PipelinePhase.SCREENING,
    )
    return kb


def build_resolution_report(
    total_extracted: int,
    resolved: list[ResolvedReference],
    failed_lines: list[str],
) -> dict:
    """Build a summary report of the resolution process."""
    confidence_counts = Counter(r.confidence.value for r in resolved)
    return {
        "total_extracted": total_extracted,
        "resolved_count": len(resolved),
        "failed_count": len(failed_lines),
        "by_confidence": {
            "high": confidence_counts.get("high", 0),
            "medium": confidence_counts.get("medium", 0),
            "low": confidence_counts.get("low", 0),
        },
        "failed_references": failed_lines[:20],  # cap for readability
    }


async def run_injection(
    pdf_path: Path,
    topic: str,
    domain: str,
    output_path: Path,
    cache_path: Path | None = None,
) -> dict:
    """Full injection pipeline: extract → parse → resolve → build KB → save.

    Returns the resolution report dict.
    """
    # 1. Extract bibliography from PDF
    text = extract_text_from_pdf(pdf_path)
    ref_lines = extract_bibliography_lines(text)
    logger.info("inject.extracted", ref_count=len(ref_lines))

    if not ref_lines:
        logger.warning("inject.no_references_found", pdf=str(pdf_path))
        return build_resolution_report(0, [], [])

    # 2. Parse reference lines
    parsed = parse_all_references(ref_lines)

    # 3. Set up search sources for resolution
    from autoreview.search.semantic_scholar import SemanticScholarSearch
    from autoreview.search.openalex import OpenAlexSearch

    sources = [SemanticScholarSearch(), OpenAlexSearch()]

    # 4. Resolve with optional caching
    cache = ResolutionCache(cache_path) if cache_path else None
    resolved: list[ResolvedReference] = []
    failed_lines: list[str] = []

    for p in parsed:
        raw = p["raw"] or ""

        # Check cache first
        if cache:
            cached = cache.get(raw)
            if cached is not None:
                resolved.append(cached)
                continue

        result = await resolve_reference(p, sources)
        if result is not None:
            resolved.append(result)
            if cache:
                cache.put(result)
        else:
            failed_lines.append(raw)

    if cache:
        cache.save()

    # 5. Build KB
    papers = [r.paper for r in resolved]
    kb = build_injected_kb(
        papers=papers,
        topic=topic,
        domain=domain,
        output_dir=str(output_path.parent),
    )

    # 6. Save snapshot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_data = kb.model_dump_json(indent=2)
    output_path.write_text(json_data)
    logger.info("inject.saved", path=str(output_path), papers=len(papers))

    # 7. Build and save report
    report = build_resolution_report(len(ref_lines), resolved, failed_lines)
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("inject.report", **{k: v for k, v in report.items() if k != "failed_references"})

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject bibliography from reference PDF into KnowledgeBase")
    parser.add_argument("--pdf", required=True, type=Path, help="Path to reference review PDF")
    parser.add_argument("--topic", required=True, help="Research topic string")
    parser.add_argument("--domain", default="general", help="Domain (biomedical, cs_ai, etc.)")
    parser.add_argument("--output", required=True, type=Path, help="Output snapshot JSON path")
    parser.add_argument("--cache", type=Path, default=None, help="Resolution cache JSON path")
    args = parser.parse_args()

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(structlog.processors.NAME_TO_LEVEL["info"]),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    report = asyncio.run(run_injection(
        pdf_path=args.pdf,
        topic=args.topic,
        domain=args.domain,
        output_path=args.output,
        cache_path=args.cache,
    ))

    print(f"\nResolution Report:")
    print(f"  Total references: {report['total_extracted']}")
    print(f"  Resolved: {report['resolved_count']} "
          f"(high: {report['by_confidence']['high']}, "
          f"medium: {report['by_confidence']['medium']}, "
          f"low: {report['by_confidence']['low']})")
    print(f"  Failed: {report['failed_count']}")
    if report["failed_references"]:
        print(f"\n  Unresolved references:")
        for line in report["failed_references"]:
            print(f"    - {line[:100]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_analysis/test_inject_bibliography.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add paper/analysis/inject_bibliography.py tests/test_analysis/test_inject_bibliography.py
git commit -m "feat: add bibliography injection script for evaluation experiments"
```

---

### Task 9: Final integration test and README

**Files:**
- Create: `paper/analysis/README.md`
- Test: `tests/test_analysis/test_integration.py`

- [ ] **Step 1: Write integration test verifying the full flow with mocks**

```python
# tests/test_analysis/test_integration.py
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from autoreview.models.paper import CandidatePaper


class TestBibliographyInjectionIntegration:
    @pytest.mark.asyncio
    async def test_full_flow_with_mocked_apis(self, tmp_path):
        """Test extract → parse → resolve → KB build with mocked external calls."""
        from paper.analysis.inject_bibliography import run_injection

        # Create a fake PDF text with references
        fake_text = """
Some review paper text here.

References

[1] Smith J, Doe A. Machine learning in biology. Nature. 2020. doi:10.1038/test001
[2] Zhang L. Deep learning approaches. Science. 2019;365:100-105.
[3] Unknown Author. Unparseable reference with no details.
"""
        mock_paper = CandidatePaper(
            title="Machine learning in biology",
            authors=["Smith J", "Doe A"],
            year=2020,
            doi="10.1038/test001",
            source_database="semantic_scholar",
        )

        # Patch at the import sites inside run_injection(). The function imports
        # SemanticScholarSearch/OpenAlexSearch inside the function body, so we patch
        # at the autoreview source module level.
        with patch("paper.analysis.inject_bibliography.extract_text_from_pdf") as mock_extract, \
             patch("paper.analysis.inject_bibliography.extract_bibliography_lines") as mock_bib, \
             patch("autoreview.search.semantic_scholar.SemanticScholarSearch") as mock_s2_cls, \
             patch("autoreview.search.openalex.OpenAlexSearch") as mock_oa_cls:

            mock_extract.return_value = fake_text
            mock_bib.return_value = [
                "[1] Smith J, Doe A. Machine learning in biology. Nature. 2020. doi:10.1038/test001",
                "[2] Zhang L. Deep learning approaches. Science. 2019;365:100-105.",
                "[3] Unknown Author. Unparseable reference with no details.",
            ]

            # S2 resolves DOI lookup, returns None for title searches
            mock_s2 = MagicMock()
            mock_s2.source_name = "semantic_scholar"
            mock_s2.get_paper_details = MagicMock(side_effect=lambda pid: _doi_lookup(pid, mock_paper))
            mock_s2.search = MagicMock(return_value=[])
            mock_s2_cls.return_value = mock_s2

            mock_oa = MagicMock()
            mock_oa.source_name = "openalex"
            mock_oa.get_paper_details = MagicMock(return_value=None)
            mock_oa.search = MagicMock(return_value=[])
            mock_oa_cls.return_value = mock_oa

            # Make mocks async
            import asyncio
            mock_s2.get_paper_details = _async_doi_lookup(mock_paper)
            mock_s2.search = _async_return([])
            mock_oa.get_paper_details = _async_return(None)
            mock_oa.search = _async_return([])

            output_path = tmp_path / "injected.json"
            report = await run_injection(
                pdf_path=Path("fake.pdf"),
                topic="ML in biology",
                domain="biomedical",
                output_path=output_path,
            )

            assert report["total_extracted"] == 3
            assert report["resolved_count"] >= 1  # At least the DOI one
            assert output_path.exists()

            # Verify KB structure
            kb_data = json.loads(output_path.read_text())
            assert kb_data["topic"] == "ML in biology"
            assert kb_data["domain"] == "biomedical"
            assert len(kb_data["candidate_papers"]) >= 1


def _async_return(value):
    async def _inner(*args, **kwargs):
        return value
    return _inner


def _async_doi_lookup(paper):
    async def _inner(paper_id):
        if "10.1038/test001" in paper_id:
            return paper
        return None
    return _inner
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_analysis/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Create README**

Create `paper/analysis/README.md`:

```markdown
# Analysis Scripts

Evaluation-only scripts for the AutoReview benchmark paper. These are NOT part of the core pipeline.

## Bibliography Injection

Injects a human review's bibliography into a KnowledgeBase snapshot for the retrieval-controlled experimental condition.

### Usage

```bash
# Step 1: Inject bibliography from reference PDF
python paper/analysis/inject_bibliography.py \
    --pdf paper/references/car_t_resistance_2019.pdf \
    --topic "CAR-T therapy resistance mechanisms" \
    --domain biomedical \
    --output paper/snapshots/car_t_injected.json \
    --cache paper/cache/resolution_cache.json

# Step 2: Resume pipeline from full_text_retrieval
autoreview resume paper/snapshots/car_t_injected.json \
    --start-from full_text_retrieval \
    --model claude-sonnet-4-6
```

### What it does

1. Extracts bibliography from a reference review PDF
2. Parses each reference line to extract DOIs and approximate titles
3. Resolves references to full paper records via Semantic Scholar and OpenAlex
4. Builds a pre-populated KnowledgeBase with all resolved papers auto-screened
5. Saves as a snapshot JSON that can be resumed with `autoreview resume`

### Resolution confidence tiers

- **High**: DOI exact match
- **Medium**: Title fuzzy match >= 85%
- **Low**: Title fuzzy match 70-85%

Unresolvable references are logged and excluded. The exclusion rate is reported as a metric.
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_analysis/ tests/test_search/ tests/test_config/ tests/test_pipeline/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add paper/analysis/README.md tests/test_analysis/test_integration.py
git commit -m "feat: add integration test and README for bibliography injection"
```

---

## Dependency Summary

```
Task 1 (SearchConfig validator)  ─┐
Task 2 (filter functions)        ─┼─→ Task 3 (wire into aggregator) ─→ Task 4 (nodes.py) ─→ Task 5 (CLI)
                                   │
Task 6 (reference parser)        ─┼─→ Task 7 (resolver) ─→ Task 8 (inject script) ─→ Task 9 (integration)
```

**Batch 1** (no deps): Tasks 1, 2, 6 — in parallel
**Batch 2** (needs batch 1): Tasks 3, 7 — in parallel
**Batch 3** (needs batch 2): Tasks 4, 5, 8 — in parallel
**Batch 4** (needs batch 3): Task 9
