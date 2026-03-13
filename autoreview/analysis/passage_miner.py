"""Passage mining — identifies claims in draft sections that need more evidence."""

from __future__ import annotations

import asyncio
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
        tasks = [
            self.mine_section(section_id=sid, section_text=text, extractions=extractions)
            for sid, text in sections.items()
        ]
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        valid: list[SectionMiningResult] = []
        failed = 0
        for i, r in enumerate(raw):
            if isinstance(r, BaseException):
                section_id = list(sections.keys())[i]
                logger.warning("passage_miner.section_failed", section_id=section_id, error=str(r))
                failed += 1
            else:
                valid.append(r)
        logger.info("passage_miner.all_complete", sections=len(valid), failed=failed)
        return valid

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
