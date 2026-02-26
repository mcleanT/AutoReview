# Passage-Mining Supplemental Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand literature coverage (currently ~36 refs) by mining written section text for undercited claims and running targeted supplemental searches, plus broadening the existing gap-search to use all databases and trigger on coverage score.

**Architecture:** Two complementary improvements: (1) `gap_search` node gets broader triggers (all databases, all gap severities, coverage-score threshold); (2) new `passage_search` node after `section_writing` uses an LLM to identify unsupported/undercited claims in each section, generates targeted queries, retrieves and extracts new papers, then revises any section that gained ≥2 new relevant papers.

**Tech Stack:** Python 3.11, Pydantic v2, asyncio, structlog, existing autoreview LLM/search abstractions.

---

## Task 1: Add `PASSAGE_SEARCH` to `PipelinePhase` enum

**Files:**
- Modify: `autoreview/models/knowledge_base.py:24`
- Test: `tests/test_models.py`

**Step 1: Write failing test**

Add to `tests/test_models.py`:
```python
def test_pipeline_phase_has_passage_search():
    from autoreview.models.knowledge_base import PipelinePhase
    assert PipelinePhase.PASSAGE_SEARCH == "passage_search"
```

**Step 2: Run to verify it fails**
```
pytest tests/test_models.py::test_pipeline_phase_has_passage_search -v
```
Expected: FAIL with `AttributeError: PASSAGE_SEARCH`

**Step 3: Add enum value**

In `autoreview/models/knowledge_base.py`, insert after `SECTION_CRITIQUE`:
```python
    PASSAGE_SEARCH = "passage_search"
```

**Step 4: Run to verify it passes**
```
pytest tests/test_models.py::test_pipeline_phase_has_passage_search -v
```

**Step 5: Commit**
```bash
git add autoreview/models/knowledge_base.py tests/test_models.py
git commit -m "feat: add PASSAGE_SEARCH pipeline phase"
```

---

## Task 2: Create passage-mining Pydantic models and prompts

**Files:**
- Create: `autoreview/llm/prompts/passage_mining.py`

**Step 1: Create the file with models, system prompt, and prompt builder**

```python
"""Prompt models and builders for passage-based evidence mining."""
from __future__ import annotations

from pydantic import Field

from autoreview.models.base import AutoReviewModel


class UndercitedClaim(AutoReviewModel):
    """A claim in a draft section that needs more supporting evidence."""
    text: str
    evidence_weakness: str
    current_citations: list[str] = Field(default_factory=list)
    suggested_queries: list[str] = Field(default_factory=list)
    priority: str  # "high", "medium", or "low"


class SectionMiningResult(AutoReviewModel):
    """Output of passage mining for a single section."""
    section_id: str
    undercited_claims: list[UndercitedClaim] = Field(default_factory=list)
    topic_expansions: list[str] = Field(default_factory=list)


SectionMiningResult.model_rebuild()


PASSAGE_MINING_SYSTEM_PROMPT = """\
You are a scientific editor reviewing draft review-paper sections to identify where \
additional evidence would strengthen the text. Focus on factual claims, mechanistic \
assertions, and quantitative statements. Ignore stylistic issues.
"""


def build_passage_mining_prompt(
    section_id: str,
    section_text: str,
    cited_paper_summaries: str,
) -> str:
    return f"""\
## Section ID: {section_id}

## Draft Text
{section_text}

## Papers Already Cited in This Section
{cited_paper_summaries}

Identify:
1. **Unsupported claims** — factual assertions with no [@paper_id] citation.
2. **Undercited claims** — important claims cited by only 1–2 sources, especially \
if described as "preliminary", "limited", or "conflicting".
3. **Topic expansions** — sub-topics mentioned briefly that could be substantiated \
by additional literature.

For each undercited or unsupported claim, generate 2–3 specific PubMed/Semantic Scholar \
search queries. Prioritize by impact on review quality: "high" (key mechanistic or \
quantitative claims), "medium" (supporting context), "low" (minor details).
Return at most 8 claims total.
"""
```

**Step 2: No runtime test needed yet — models are tested implicitly in Task 3. Verify imports work:**
```
python -c "from autoreview.llm.prompts.passage_mining import SectionMiningResult, UndercitedClaim; print('ok')"
```

**Step 3: Commit**
```bash
git add autoreview/llm/prompts/passage_mining.py
git commit -m "feat: add passage mining prompt models and builders"
```

---

## Task 3: Implement `PassageMiner` class with tests

**Files:**
- Create: `autoreview/analysis/passage_miner.py`
- Create: `tests/test_analysis/test_passage_miner.py`

**Step 1: Write failing tests**

Create `tests/test_analysis/test_passage_miner.py`:
```python
"""Tests for passage mining module."""
from __future__ import annotations

import pytest

from autoreview.analysis.passage_miner import PassageMiner
from autoreview.extraction.models import EvidenceStrength, Finding, PaperExtraction
from autoreview.llm.provider import LLMStructuredResponse
from autoreview.llm.prompts.passage_mining import (
    SectionMiningResult,
    UndercitedClaim,
)


class MockPassageMinerLLM:
    async def generate_structured(
        self, prompt, response_model, system="", max_tokens=4096, temperature=0.0
    ):
        if response_model == SectionMiningResult:
            section_id = "section_1" if "Section ID: section_1" in prompt else "section_2"
            return LLMStructuredResponse(
                parsed=SectionMiningResult(
                    section_id=section_id,
                    undercited_claims=[
                        UndercitedClaim(
                            text="Senescent cells secrete pro-inflammatory cytokines",
                            evidence_weakness="Single preliminary study cited",
                            current_citations=["p1"],
                            suggested_queries=[
                                "cellular senescence SASP interleukin",
                                "senescent cell secretome inflammation",
                            ],
                            priority="high",
                        ),
                    ],
                    topic_expansions=["telomere shortening mechanisms in organ-specific senescence"],
                ),
                input_tokens=400,
                output_tokens=200,
            )
        raise ValueError(f"Unexpected response_model: {response_model}")


def _make_extractions() -> dict[str, PaperExtraction]:
    return {
        "p1": PaperExtraction(
            paper_id="p1",
            key_findings=[
                Finding(
                    claim="SASP promotes inflammation",
                    evidence_strength=EvidenceStrength.PRELIMINARY,
                    paper_id="p1",
                )
            ],
            methods_summary="In vitro",
            limitations="Single cell line",
        ),
    }


class TestPassageMiner:
    async def test_mine_section_returns_result(self):
        miner = PassageMiner(MockPassageMinerLLM())
        result = await miner.mine_section(
            section_id="section_1",
            section_text="Senescent cells secrete pro-inflammatory cytokines [@p1].",
            extractions=_make_extractions(),
        )
        assert result.section_id == "section_1"
        assert len(result.undercited_claims) == 1
        assert result.undercited_claims[0].priority == "high"
        assert len(result.undercited_claims[0].suggested_queries) == 2

    async def test_mine_section_includes_topic_expansions(self):
        miner = PassageMiner(MockPassageMinerLLM())
        result = await miner.mine_section(
            section_id="section_1",
            section_text="Telomere shortening is involved in senescence.",
            extractions=_make_extractions(),
        )
        assert len(result.topic_expansions) >= 1

    async def test_mine_all_sections_processes_each(self):
        miner = PassageMiner(MockPassageMinerLLM())
        sections = {
            "section_1": "Senescent cells secrete cytokines [@p1].",
            "section_2": "Organ-specific senescence patterns differ.",
        }
        results = await miner.mine_all_sections(sections, _make_extractions())
        assert len(results) == 2
        section_ids = {r.section_id for r in results}
        assert "section_1" in section_ids
        assert "section_2" in section_ids

    async def test_collect_queries_high_medium_only(self):
        miner = PassageMiner(MockPassageMinerLLM())
        result = await miner.mine_section(
            section_id="section_1",
            section_text="Senescent cells secrete cytokines [@p1].",
            extractions=_make_extractions(),
        )
        queries = miner.collect_queries([result], priorities={"high", "medium"})
        assert len(queries) > 0
        assert all(isinstance(q, str) for q in queries)

    async def test_empty_section_returns_empty_claims(self):
        """Empty sections should not crash."""
        miner = PassageMiner(MockPassageMinerLLM())
        # MockLLM still returns a result, but real LLM would return empty claims for empty text
        result = await miner.mine_section(
            section_id="section_1",
            section_text="",
            extractions={},
        )
        assert result.section_id == "section_1"
```

**Step 2: Run to verify all fail**
```
pytest tests/test_analysis/test_passage_miner.py -v
```
Expected: all FAIL with `ModuleNotFoundError: autoreview.analysis.passage_miner`

**Step 3: Implement `autoreview/analysis/passage_miner.py`**

```python
"""Passage mining — identifies claims in draft sections that need more evidence."""
from __future__ import annotations

import re
from typing import Any

import structlog

from autoreview.extraction.models import PaperExtraction
from autoreview.llm.prompts.passage_mining import (
    PASSAGE_MINING_SYSTEM_PROMPT,
    SectionMiningResult,
    build_passage_mining_prompt,
)

logger = structlog.get_logger()


def _extract_cited_ids(text: str) -> list[str]:
    """Extract [@paper_id] markers from section text."""
    return re.findall(r"\[@([^\]]+)\]", text)


def _format_cited_paper_summaries(
    cited_ids: list[str],
    extractions: dict[str, PaperExtraction],
) -> str:
    blocks = []
    for pid in cited_ids:
        ext = extractions.get(pid)
        if not ext:
            blocks.append(f"[@{pid}]: (no extraction available)")
            continue
        findings = "; ".join(f.claim for f in ext.key_findings[:3])
        blocks.append(f"[@{pid}]: {findings} | Methods: {ext.methods_summary}")
    return "\n".join(blocks) if blocks else "(none)"


class PassageMiner:
    """Mines draft review sections to find claims that need more evidence."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    async def mine_section(
        self,
        section_id: str,
        section_text: str,
        extractions: dict[str, PaperExtraction],
    ) -> SectionMiningResult:
        """Analyse one section and return undercited claims with search queries."""
        cited_ids = _extract_cited_ids(section_text)
        summaries = _format_cited_paper_summaries(cited_ids, extractions)

        prompt = build_passage_mining_prompt(
            section_id=section_id,
            section_text=section_text or "(empty section)",
            cited_paper_summaries=summaries,
        )

        response = await self.llm.generate_structured(
            prompt=prompt,
            response_model=SectionMiningResult,
            system=PASSAGE_MINING_SYSTEM_PROMPT,
        )
        result: SectionMiningResult = response.parsed

        # Ensure section_id is set correctly (LLM might return wrong id)
        result = result.model_copy(update={"section_id": section_id})

        logger.info(
            "passage_miner.section_complete",
            section_id=section_id,
            undercited_claims=len(result.undercited_claims),
            topic_expansions=len(result.topic_expansions),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        return result

    async def mine_all_sections(
        self,
        sections: dict[str, str],
        extractions: dict[str, PaperExtraction],
    ) -> list[SectionMiningResult]:
        """Mine all sections concurrently and return results."""
        import asyncio

        tasks = [
            self.mine_section(section_id=sid, section_text=text, extractions=extractions)
            for sid, text in sections.items()
        ]
        results = await asyncio.gather(*tasks)
        logger.info("passage_miner.all_complete", sections=len(results))
        return list(results)

    def collect_queries(
        self,
        results: list[SectionMiningResult],
        priorities: set[str] | None = None,
    ) -> list[str]:
        """Collect unique search queries from mining results, filtered by priority."""
        if priorities is None:
            priorities = {"high", "medium"}
        seen: set[str] = set()
        queries: list[str] = []
        for result in results:
            for claim in result.undercited_claims:
                if claim.priority not in priorities:
                    continue
                for q in claim.suggested_queries:
                    if q not in seen:
                        seen.add(q)
                        queries.append(q)
        return queries
```

**Step 4: Run tests**
```
pytest tests/test_analysis/test_passage_miner.py -v
```
Expected: all PASS

**Step 5: Commit**
```bash
git add autoreview/analysis/passage_miner.py tests/test_analysis/test_passage_miner.py
git commit -m "feat: implement PassageMiner for undercited claim detection"
```

---

## Task 4: Add `revise_section_with_evidence()` to `SectionWriter`

**Files:**
- Modify: `autoreview/writing/section_writer.py`
- Create: `tests/test_writing/test_section_revision.py`

**Step 1: Write failing test**

Create `tests/test_writing/test_section_revision.py`:
```python
"""Tests for evidence-based section revision."""
from __future__ import annotations

import pytest

from autoreview.analysis.evidence_map import EvidenceMap
from autoreview.extraction.models import EvidenceStrength, Finding, PaperExtraction
from autoreview.llm.provider import LLMResponse
from autoreview.writing.section_writer import SectionDraft, SectionWriter


class MockRevisionLLM:
    async def generate(self, prompt, system="", max_tokens=4096, temperature=0.3):
        return LLMResponse(
            content=(
                "Senescent cells secrete SASP factors including IL-6 and IL-8 [@p1]. "
                "Recent evidence confirms this in hepatocytes [@p2] and renal tubular cells [@p3]."
            ),
            input_tokens=500,
            output_tokens=120,
        )


def _make_extractions() -> dict[str, PaperExtraction]:
    return {
        "p2": PaperExtraction(
            paper_id="p2",
            key_findings=[
                Finding(
                    claim="Hepatocyte senescence drives liver fibrosis via IL-6",
                    evidence_strength=EvidenceStrength.STRONG,
                    paper_id="p2",
                )
            ],
            methods_summary="Mouse model + human biopsy",
            limitations="Animal model",
        ),
        "p3": PaperExtraction(
            paper_id="p3",
            key_findings=[
                Finding(
                    claim="Renal senescence promotes CKD progression",
                    evidence_strength=EvidenceStrength.MODERATE,
                    paper_id="p3",
                )
            ],
            methods_summary="Cohort study",
            limitations="Observational",
        ),
    }


class TestReviseSection:
    async def test_revise_returns_section_draft(self):
        writer = SectionWriter(MockRevisionLLM())
        draft = await writer.revise_section_with_evidence(
            section_id="1",
            section_title="SASP and Inflammation",
            existing_text="Senescent cells secrete SASP factors [@p1].",
            new_paper_ids=["p2", "p3"],
            extractions=_make_extractions(),
        )
        assert isinstance(draft, SectionDraft)
        assert draft.section_id == "1"
        assert len(draft.text) > 0

    async def test_revise_includes_new_citations(self):
        writer = SectionWriter(MockRevisionLLM())
        draft = await writer.revise_section_with_evidence(
            section_id="1",
            section_title="SASP and Inflammation",
            existing_text="Senescent cells secrete SASP factors [@p1].",
            new_paper_ids=["p2", "p3"],
            extractions=_make_extractions(),
        )
        assert "[@p2]" in draft.text or "[@p3]" in draft.text

    async def test_revise_tracks_citations_used(self):
        writer = SectionWriter(MockRevisionLLM())
        draft = await writer.revise_section_with_evidence(
            section_id="1",
            section_title="SASP and Inflammation",
            existing_text="Senescent cells secrete SASP factors [@p1].",
            new_paper_ids=["p2", "p3"],
            extractions=_make_extractions(),
        )
        assert "p1" in draft.citations_used or "p2" in draft.citations_used
```

**Step 2: Run to verify tests fail**
```
pytest tests/test_writing/test_section_revision.py -v
```
Expected: FAIL with `AttributeError: 'SectionWriter' object has no attribute 'revise_section_with_evidence'`

**Step 3: Add method to `SectionWriter`**

In `autoreview/writing/section_writer.py`, add after the `write_section` method (before `write_all_sections`):
```python
    async def revise_section_with_evidence(
        self,
        section_id: str,
        section_title: str,
        existing_text: str,
        new_paper_ids: list[str],
        extractions: dict[str, PaperExtraction],
    ) -> SectionDraft:
        """Revise an existing section to incorporate newly found papers.

        Args:
            section_id: Section identifier.
            section_title: Human-readable section title.
            existing_text: The current draft text.
            new_paper_ids: IDs of newly retrieved papers to incorporate.
            extractions: All available extractions (including new papers).

        Returns:
            Revised SectionDraft with new evidence incorporated.
        """
        new_evidence = _format_extractions(new_paper_ids, extractions)

        prompt = (
            f"## Section: {section_title}\n\n"
            f"## Existing Draft\n{existing_text}\n\n"
            f"## Newly Available Evidence\n{new_evidence}\n\n"
            "Revise the section to incorporate the new evidence where it strengthens "
            "the text. Add [@paper_id] citations for new claims. Preserve the existing "
            "structure and arguments. Do not pad with unnecessary content."
        )

        response = await self.llm.generate(
            prompt=prompt,
            system=SECTION_WRITING_SYSTEM_PROMPT,
            temperature=0.3,
        )

        citations = _extract_citations(response.content)

        draft = SectionDraft(
            section_id=section_id,
            title=section_title,
            text=response.content,
            citations_used=citations,
        )

        logger.info(
            "section_writer.revised",
            section_id=section_id,
            new_papers=len(new_paper_ids),
            citations=len(citations),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        return draft
```

**Step 4: Run tests**
```
pytest tests/test_writing/test_section_revision.py -v
```
Expected: all PASS

**Step 5: Commit**
```bash
git add autoreview/writing/section_writer.py tests/test_writing/test_section_revision.py
git commit -m "feat: add revise_section_with_evidence to SectionWriter"
```

---

## Task 5: Add `get_references()` to `SemanticScholarSearch` for citation snowballing

**Files:**
- Modify: `autoreview/search/semantic_scholar.py`
- Modify: `tests/test_search/test_aggregator.py` (add a new test file alongside it)

**Step 1: Write failing test**

Create `tests/test_search/test_semantic_scholar.py`:
```python
"""Tests for SemanticScholarSearch citation snowballing."""
from __future__ import annotations

import pytest
import respx
import httpx

from autoreview.search.semantic_scholar import SemanticScholarSearch


S2_BASE = "https://api.semanticscholar.org/graph/v1"


class TestGetReferences:
    @respx.mock
    async def test_get_references_returns_papers(self):
        paper_id = "abc123"
        respx.get(f"{S2_BASE}/paper/{paper_id}/references").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "citedPaper": {
                                "paperId": "ref1",
                                "title": "Reference Paper One",
                                "abstract": "About senescence",
                                "year": 2022,
                                "authors": [{"name": "Alice A"}],
                                "journal": {"name": "Nature"},
                                "externalIds": {"DOI": "10.1/ref1"},
                                "citationCount": 10,
                            }
                        },
                        {
                            "citedPaper": {
                                "paperId": "ref2",
                                "title": None,  # invalid — should be skipped
                                "abstract": None,
                                "year": 2021,
                                "authors": [],
                                "journal": None,
                                "externalIds": {},
                                "citationCount": 0,
                            }
                        },
                    ]
                },
            )
        )
        s2 = SemanticScholarSearch()
        papers = await s2.get_references(paper_id, limit=50)
        assert len(papers) == 1
        assert papers[0].title == "Reference Paper One"
        assert papers[0].source_database == "semantic_scholar"

    @respx.mock
    async def test_get_references_handles_http_error(self):
        paper_id = "bad_id"
        respx.get(f"{S2_BASE}/paper/{paper_id}/references").mock(
            return_value=httpx.Response(404)
        )
        s2 = SemanticScholarSearch()
        papers = await s2.get_references(paper_id, limit=50)
        assert papers == []

    @respx.mock
    async def test_snowball_references_across_papers(self):
        """snowball_references should collect references from multiple papers."""
        for pid, ref_title, ref_id, ref_doi in [
            ("s2_p1", "Ref A", "r1", "10.1/a"),
            ("s2_p2", "Ref B", "r2", "10.1/b"),
        ]:
            respx.get(f"{S2_BASE}/paper/{pid}/references").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "citedPaper": {
                                    "paperId": ref_id,
                                    "title": ref_title,
                                    "abstract": "abstract",
                                    "year": 2020,
                                    "authors": [{"name": "Bob B"}],
                                    "journal": {"name": "Cell"},
                                    "externalIds": {"DOI": ref_doi},
                                    "citationCount": 5,
                                }
                            }
                        ]
                    },
                )
            )
        s2 = SemanticScholarSearch()
        papers = await s2.snowball_references(["s2_p1", "s2_p2"], limit_per_paper=20)
        assert len(papers) == 2
        titles = {p.title for p in papers}
        assert "Ref A" in titles
        assert "Ref B" in titles
```

**Step 2: Run to verify tests fail**
```
pytest tests/test_search/test_semantic_scholar.py -v
```
Expected: FAIL with `AttributeError: get_references`

**Step 3: Add methods to `SemanticScholarSearch`**

In `autoreview/search/semantic_scholar.py`, add after `get_paper_details`:
```python
    async def get_references(
        self,
        paper_id: str,
        limit: int = 50,
    ) -> list[CandidatePaper]:
        """Retrieve papers cited by a given paper (backward snowballing).

        Args:
            paper_id: Semantic Scholar paper ID.
            limit: Maximum references to return.

        Returns:
            List of CandidatePapers found in the reference list.
        """
        await self._limiter.acquire()
        fields = S2_FIELDS
        async with httpx.AsyncClient(timeout=30.0, headers=self._headers) as client:
            try:
                resp = await client.get(
                    f"{S2_API_BASE}/paper/{paper_id}/references",
                    params={"fields": f"citedPaper.{fields}", "limit": limit},
                )
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                logger.warning("s2.references_error", paper_id=paper_id, error=str(e))
                return []

        papers: list[CandidatePaper] = []
        for item in data.get("data", []):
            cited = item.get("citedPaper")
            if not cited:
                continue
            paper = self._parse_paper(cited)
            if paper:
                papers.append(paper)
        return papers

    async def snowball_references(
        self,
        paper_ids: list[str],
        limit_per_paper: int = 30,
    ) -> list[CandidatePaper]:
        """Collect references from multiple papers concurrently (backward snowballing).

        Args:
            paper_ids: Semantic Scholar paper IDs of source papers.
            limit_per_paper: Max references to retrieve per source paper.

        Returns:
            Deduplicated list of CandidatePapers from all reference lists.
        """
        import asyncio

        tasks = [self.get_references(pid, limit=limit_per_paper) for pid in paper_ids]
        results = await asyncio.gather(*tasks)

        seen_dois: set[str] = set()
        seen_titles: set[str] = set()
        unique: list[CandidatePaper] = []
        for batch in results:
            for paper in batch:
                key = (paper.doi or "").lower().strip()
                title_key = paper.title.lower().strip()[:60]
                if key and key in seen_dois:
                    continue
                if title_key in seen_titles:
                    continue
                if key:
                    seen_dois.add(key)
                seen_titles.add(title_key)
                unique.append(paper)

        logger.info("s2.snowball_complete", source_papers=len(paper_ids), new_papers=len(unique))
        return unique
```

**Step 4: Run tests**
```
pytest tests/test_search/test_semantic_scholar.py -v
```
Expected: all PASS

**Step 5: Commit**
```bash
git add autoreview/search/semantic_scholar.py tests/test_search/test_semantic_scholar.py
git commit -m "feat: add get_references and snowball_references to SemanticScholarSearch"
```

---

## Task 6: Improve `gap_search` node (broader triggers, all sources, all gaps)

**Files:**
- Modify: `autoreview/pipeline/nodes.py:193-288`

**Step 1: No new test file needed — we'll extend the existing test suite in Task 7. Instead, read `nodes.py` lines 193–233 carefully before editing.**

**Step 2: Modify the `gap_search` method**

Find this block near line 203:
```python
        major_gaps = [g for g in kb.evidence_map.gaps if g.severity == "major"]
        if not major_gaps:
            await self._run_benchmark_validation(kb)
            return
```
Replace with:
```python
        # Trigger on any gap or when coverage is below threshold
        coverage_score = kb.evidence_map.coverage_score if kb.evidence_map else 0.0
        coverage_threshold = getattr(self.config.search, "min_coverage_threshold", 0.75)
        all_gaps = kb.evidence_map.gaps
        major_gaps = [g for g in all_gaps if g.severity == "major"]

        if not all_gaps and coverage_score >= coverage_threshold:
            await self._run_benchmark_validation(kb)
            return

        if not major_gaps and coverage_score >= coverage_threshold:
            await self._run_benchmark_validation(kb)
            return

        gaps_to_search = all_gaps  # Include minor gaps in query generation
        logger.info(
            "gap_search.triggered",
            major_gaps=len(major_gaps),
            total_gaps=len(gaps_to_search),
            coverage_score=coverage_score,
        )
```

Also update the variable reference from `major_gaps` to `gaps_to_search` a few lines below:
```python
        gap_queries: dict[str, list[str]] = {}
        for db in self.config.databases.get("primary", []):
            gap_queries[db] = []
            for gap in gaps_to_search:            # ← was major_gaps
                gap_queries[db].extend(gap.suggested_queries)
```

Find the section that only searches primary databases (lines ~219-229):
```python
        sources = []
        for db in self.config.databases.get("primary", []):
```
Replace with a version that uses all databases:
```python
        sources = []
        all_dbs = (
            self.config.databases.get("primary", [])
            + self.config.databases.get("secondary", [])
        )
        for db in all_dbs:
```
(Extend the import-and-append block for `pubmed`, `semantic_scholar`, `openalex` to match this expanded list — copy the structure from the `search` node at lines ~98-111 of the same file, but limit to primary + secondary, skipping perplexity for gap search.)

Increase max results from 100 to 200:
```python
        new_papers = await agg.search(gap_queries, max_results_per_source=200)  # ← was 100
```

**Step 3: Run existing tests to confirm nothing broke**
```
pytest tests/test_pipeline/ -v
```
Expected: all PASS (the runner test doesn't test gap_search internals)

**Step 4: Commit**
```bash
git add autoreview/pipeline/nodes.py
git commit -m "feat: broaden gap_search to use all databases, all gaps, and coverage threshold"
```

---

## Task 7: Add `passage_search` node to `nodes.py` with tests

**Files:**
- Modify: `autoreview/pipeline/nodes.py`
- Create: `tests/test_pipeline/test_passage_search_node.py`

**Step 1: Write failing tests**

Create `tests/test_pipeline/test_passage_search_node.py`:
```python
"""Tests for the passage_search pipeline node."""
from __future__ import annotations

import pytest

from autoreview.analysis.evidence_map import EvidenceMap, Theme
from autoreview.analysis.passage_miner import PassageMiner
from autoreview.config import load_config
from autoreview.extraction.models import EvidenceStrength, Finding, PaperExtraction
from autoreview.llm.provider import LLMResponse, LLMStructuredResponse
from autoreview.models.knowledge_base import KnowledgeBase, PipelinePhase
from autoreview.models.paper import CandidatePaper, ScreenedPaper
from autoreview.pipeline.nodes import PipelineNodes


class MockPassageSearchLLM:
    """Minimal mock LLM for passage_search node tests."""

    from autoreview.llm.prompts.passage_mining import SectionMiningResult, UndercitedClaim

    async def generate_structured(self, prompt, response_model, system="", max_tokens=4096, temperature=0.0):
        from autoreview.llm.prompts.passage_mining import SectionMiningResult, UndercitedClaim
        from autoreview.llm.prompts.screening import ScreeningResult, ScreeningDecision
        from autoreview.extraction.models import PaperExtraction, EvidenceStrength, Finding

        if response_model == SectionMiningResult:
            return LLMStructuredResponse(
                parsed=SectionMiningResult(
                    section_id="s1",
                    undercited_claims=[
                        UndercitedClaim(
                            text="Senescent cells accumulate",
                            evidence_weakness="Single source",
                            current_citations=["p1"],
                            suggested_queries=["cellular senescence accumulation aging"],
                            priority="high",
                        )
                    ],
                    topic_expansions=[],
                ),
                input_tokens=300, output_tokens=150,
            )
        # ScreeningResult for new papers
        if hasattr(response_model, "__name__") and "Screening" in response_model.__name__:
            return LLMStructuredResponse(
                parsed=response_model(decisions=[]),
                input_tokens=100, output_tokens=50,
            )
        if response_model == PaperExtraction:
            return LLMStructuredResponse(
                parsed=PaperExtraction(
                    paper_id="new_p1",
                    key_findings=[
                        Finding(claim="New finding", evidence_strength=EvidenceStrength.MODERATE, paper_id="new_p1")
                    ],
                    methods_summary="RCT", limitations="Small n",
                ),
                input_tokens=200, output_tokens=100,
            )
        raise ValueError(f"Unexpected: {response_model}")

    async def generate(self, prompt, system="", max_tokens=4096, temperature=0.3):
        return LLMResponse(
            content="Revised section text with [@p1] and [@new_p1].",
            input_tokens=400, output_tokens=100,
        )


def _make_kb_with_drafts() -> KnowledgeBase:
    kb = KnowledgeBase(
        topic="cellular senescence",
        domain="biomedical",
        output_dir="/tmp/autoreview_test",
    )
    kb.section_drafts = {
        "s1": "Senescent cells accumulate with age [@p1].",
    }
    kb.extractions = {
        "p1": PaperExtraction(
            paper_id="p1",
            key_findings=[
                Finding(claim="Cells senesce with age", evidence_strength=EvidenceStrength.MODERATE, paper_id="p1")
            ],
            methods_summary="In vitro", limitations="Cell lines only",
        )
    }
    kb.evidence_map = EvidenceMap(
        themes=[Theme(name="Senescence", description="Cell senescence", paper_ids=["p1"])],
    )
    kb.scope_document = "Review of cellular senescence across organs."
    return kb


class TestPassageSearchNode:
    async def test_passage_search_sets_phase(self):
        config = load_config(domain="biomedical")
        nodes = PipelineNodes(MockPassageSearchLLM(), config)
        kb = _make_kb_with_drafts()
        await nodes.passage_search(kb)
        assert kb.current_phase == PipelinePhase.PASSAGE_SEARCH

    async def test_passage_search_adds_audit_entry(self):
        config = load_config(domain="biomedical")
        nodes = PipelineNodes(MockPassageSearchLLM(), config)
        kb = _make_kb_with_drafts()
        await nodes.passage_search(kb)
        audit_names = [e.node_name for e in kb.audit_log]
        assert "passage_search" in audit_names

    async def test_passage_search_noop_on_empty_drafts(self):
        """If no section drafts exist, passage_search should return without crashing."""
        config = load_config(domain="biomedical")
        nodes = PipelineNodes(MockPassageSearchLLM(), config)
        kb = KnowledgeBase(topic="test", domain="biomedical", output_dir="/tmp/ar_test")
        # No section_drafts set
        await nodes.passage_search(kb)
        # Should complete without error; phase may or may not be set
```

**Step 2: Run to verify tests fail**
```
pytest tests/test_pipeline/test_passage_search_node.py -v
```
Expected: FAIL with `AttributeError: 'PipelineNodes' object has no attribute 'passage_search'`

**Step 3: Add `passage_search` method to `PipelineNodes` in `nodes.py`**

Add after the `_run_benchmark_validation` method:
```python
    async def passage_search(self, kb: KnowledgeBase) -> None:
        """Node: Mine written sections for undercited claims and retrieve more papers."""
        if not kb.section_drafts:
            return

        from autoreview.analysis.passage_miner import PassageMiner
        from autoreview.search.aggregator import SearchAggregator

        # 1. Mine sections for undercited claims
        miner = PassageMiner(self.llm)
        mining_results = await miner.mine_all_sections(kb.section_drafts, kb.extractions)

        # 2. Collect targeted queries from high and medium priority claims
        queries = miner.collect_queries(mining_results, priorities={"high", "medium"})
        if not queries:
            kb.current_phase = PipelinePhase.PASSAGE_SEARCH
            kb.add_audit_entry("passage_search", "skipped", "No high/medium priority claims found")
            return

        # 3. Build queries dict for all databases
        all_dbs = (
            self.config.databases.get("primary", [])
            + self.config.databases.get("secondary", [])
            + self.config.databases.get("discovery", [])
        )
        queries_by_source: dict[str, list[str]] = {db: queries for db in all_dbs if queries}

        # 4. Search across all databases
        sources = []
        for db in all_dbs:
            try:
                if db == "pubmed":
                    from autoreview.search.pubmed import PubMedSearch
                    sources.append(PubMedSearch())
                elif db == "semantic_scholar":
                    from autoreview.search.semantic_scholar import SemanticScholarSearch
                    sources.append(SemanticScholarSearch())
                elif db == "openalex":
                    from autoreview.search.openalex import OpenAlexSearch
                    sources.append(OpenAlexSearch())
                elif db == "perplexity":
                    from autoreview.search.perplexity import PerplexitySearch
                    sources.append(PerplexitySearch())
            except Exception as e:
                logger.warning("passage_search.source_init_failed", source=db, error=str(e))

        new_papers: list[CandidatePaper] = []
        if sources:
            agg = SearchAggregator(sources=sources)
            new_papers = await agg.search(queries_by_source, max_results_per_source=50)

        # 5. Citation snowballing from S2 papers already in corpus
        try:
            from autoreview.search.semantic_scholar import SemanticScholarSearch as S2
            s2 = S2()
            s2_ids = [
                p.external_ids.get("s2_id", "")
                for p in kb.candidate_papers
                if p.external_ids.get("s2_id")
            ]
            # Snowball from top 10 most-cited papers in corpus
            top_s2_ids = sorted(
                [pid for pid in s2_ids if pid],
                key=lambda pid: next(
                    (p.citation_count or 0 for p in kb.candidate_papers
                     if p.external_ids.get("s2_id") == pid), 0
                ),
                reverse=True,
            )[:10]
            if top_s2_ids:
                snowballed = await s2.snowball_references(top_s2_ids, limit_per_paper=20)
                new_papers.extend(snowballed)
        except Exception as e:
            logger.warning("passage_search.snowball_failed", error=str(e))

        # 6. Screen and extract new papers
        screener = PaperScreener(self.llm)
        new_screened = await screener.screen(
            new_papers,
            scope_document=kb.scope_document or "",
            threshold=self.config.search.relevance_threshold,
        )

        extractor = PaperExtractor(
            self.llm,
            domain_fields=self.config.extraction.domain_fields,
        )
        new_extractions = await extractor.extract_batch(
            [sp.paper for sp in new_screened]
        )

        # 7. Merge into KB
        kb.candidate_papers.extend(new_papers)
        kb.screened_papers.extend(new_screened)
        kb.extractions.update(new_extractions)

        # 8. Revise sections that gained ≥2 new relevant papers
        from autoreview.writing.section_writer import SectionWriter
        writer = SectionWriter(self.llm)
        revised_count = 0

        for result in mining_results:
            # Find new papers relevant to this section's claimed topics
            section_new_ids = [
                pid for pid in new_extractions
                if pid not in (result.undercited_claims[0].current_citations
                               if result.undercited_claims else [])
            ]
            if len(section_new_ids) >= 2:
                draft = await writer.revise_section_with_evidence(
                    section_id=result.section_id,
                    section_title=result.section_id,
                    existing_text=kb.section_drafts[result.section_id],
                    new_paper_ids=section_new_ids[:10],
                    extractions=kb.extractions,
                )
                kb.section_drafts[result.section_id] = draft.text
                revised_count += 1

        kb.current_phase = PipelinePhase.PASSAGE_SEARCH
        kb.add_audit_entry(
            "passage_search",
            "complete",
            f"Queries: {len(queries)}, New screened: {len(new_screened)}, "
            f"New extractions: {len(new_extractions)}, Sections revised: {revised_count}",
        )
```

**Step 4: Run tests**
```
pytest tests/test_pipeline/test_passage_search_node.py -v
```
Expected: all PASS

**Step 5: Run full test suite**
```
pytest tests/ -v --ignore=tests/test_pipeline/test_runner.py
```
Expected: all PASS (runner test is updated in next task)

**Step 6: Commit**
```bash
git add autoreview/pipeline/nodes.py tests/test_pipeline/test_passage_search_node.py
git commit -m "feat: add passage_search node that mines sections for undercited claims"
```

---

## Task 8: Wire `passage_search` into DAG and update runner tests

**Files:**
- Modify: `autoreview/pipeline/runner.py`
- Modify: `tests/test_pipeline/test_runner.py`

**Step 1: Update `runner.py`**

In `autoreview/pipeline/runner.py`, find:
```python
    dag.add_node("section_writing", nodes.section_writing, dependencies=["outline"])
    dag.add_node("assembly", nodes.assembly, dependencies=["section_writing"])
```
Replace with:
```python
    dag.add_node("section_writing", nodes.section_writing, dependencies=["outline"])
    dag.add_node("passage_search", nodes.passage_search, dependencies=["section_writing"])
    dag.add_node("assembly", nodes.assembly, dependencies=["passage_search"])
```

**Step 2: Update `test_runner.py`**

Update `test_pipeline_has_all_nodes` to include `passage_search`:
```python
        expected_nodes = [
            "query_expansion",
            "search",
            "screening",
            "extraction",
            "clustering",
            "gap_search",
            "outline",
            "section_writing",
            "passage_search",   # ← new
            "assembly",
            "final_polish",
        ]
```

Update `test_pipeline_topology_is_valid` flat count from `== 10` to `== 11`:
```python
        assert len(flat) == 11
```

Add a new dependency test:
```python
    def test_passage_search_dependencies(self):
        """passage_search must depend on section_writing, assembly must depend on passage_search."""
        config = load_config(domain="biomedical")
        dag, _ = build_pipeline(llm=None, config=config)
        assert "section_writing" in dag.nodes["passage_search"].dependencies
        assert "passage_search" in dag.nodes["assembly"].dependencies
```

**Step 3: Run runner tests**
```
pytest tests/test_pipeline/test_runner.py -v
```
Expected: all PASS

**Step 4: Run full test suite**
```
pytest tests/ -v
```
Expected: all PASS

**Step 5: Commit**
```bash
git add autoreview/pipeline/runner.py tests/test_pipeline/test_runner.py
git commit -m "feat: wire passage_search node into pipeline DAG between section_writing and assembly"
```

---

## Summary of Changes

| File | Type | What Changed |
|------|------|-------------|
| `autoreview/models/knowledge_base.py` | Modify | Add `PASSAGE_SEARCH` enum value |
| `autoreview/llm/prompts/passage_mining.py` | Create | `UndercitedClaim`, `SectionMiningResult`, prompt builder |
| `autoreview/analysis/passage_miner.py` | Create | `PassageMiner` class with `mine_section`, `mine_all_sections`, `collect_queries` |
| `autoreview/writing/section_writer.py` | Modify | Add `revise_section_with_evidence()` |
| `autoreview/search/semantic_scholar.py` | Modify | Add `get_references()` and `snowball_references()` |
| `autoreview/pipeline/nodes.py` | Modify | Improve `gap_search` + add `passage_search` |
| `autoreview/pipeline/runner.py` | Modify | Wire `passage_search` between `section_writing` and `assembly` |
| `tests/test_models.py` | Modify | Add `PASSAGE_SEARCH` phase test |
| `tests/test_analysis/test_passage_miner.py` | Create | Full test suite for `PassageMiner` |
| `tests/test_writing/test_section_revision.py` | Create | Tests for `revise_section_with_evidence` |
| `tests/test_search/test_semantic_scholar.py` | Create | Tests for citation snowballing |
| `tests/test_pipeline/test_passage_search_node.py` | Create | Tests for `passage_search` node |
| `tests/test_pipeline/test_runner.py` | Modify | Update node count + add dependency test |
