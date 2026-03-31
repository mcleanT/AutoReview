"""Hybrid paper extractor: programmatic draft + LLM refinement.

Runs ProgrammaticExtractor first (zero tokens, instant), then sends a
condensed ~2K-token context to a cheap LLM (Haiku) to refine key_findings,
methods_summary, and limitations. study_design and quality_score are always
kept from the programmatic result (deterministic, not subject to LLM hallucination).

Short-circuit rule: abstract-only papers (no full_text) skip the LLM call
entirely and return the programmatic draft as-is.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import structlog

from autoreview.extraction.extractor import ExtractionFailure
from autoreview.extraction.models import PaperExtraction
from autoreview.extraction.programmatic import ProgrammaticExtractor
from autoreview.extraction.truncation import ParsedSection, parse_sections
from autoreview.llm.prompts.hybrid_extraction import HYBRID_SYSTEM_PROMPT, build_refinement_prompt
from autoreview.models.paper import ScreenedPaper

logger = structlog.get_logger()

# Max characters fed into the ~2K-token condensed context
_ABSTRACT_MAX_CHARS = 800
_SECTION_MAX_CHARS = 600
_TOP_SENTENCES_MAX = 10
_CONTEXT_MAX_CHARS = 2_200


def _extract_json_block(text: str) -> str:
    """Extract JSON from markdown fences or return raw text.

    Handles:
    - ```json ... ```
    - ``` ... ```
    - Raw JSON (returned as-is, stripped)
    """
    # Try ```json ... ``` fence first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: find the first { ... } block spanning the whole response
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return text.strip()


def _find_section_text(
    sections: list[ParsedSection],
    patterns: list[str],
    max_chars: int = _SECTION_MAX_CHARS,
) -> str:
    """Return text from the first section whose name matches any of the given patterns.

    Pattern matching is case-insensitive substring search.
    """
    for section in sections:
        name_lower = section.name.lower()
        if any(p in name_lower for p in patterns):
            return section.text[:max_chars]
    return ""


class HybridExtractor:
    """Programmatic extraction followed by optional LLM refinement.

    Args:
        programmatic: A configured ProgrammaticExtractor instance.
        llm: An LLMProvider used for the refinement step.
        max_concurrent: Maximum number of concurrent LLM refinement calls.
    """

    # Default concurrency by provider type
    _DEFAULT_CONCURRENT_CLAUDE_CODE = 20
    _DEFAULT_CONCURRENT_API = 5

    def __init__(
        self,
        programmatic: ProgrammaticExtractor,
        llm: Any,
        max_concurrent: int | None = None,
    ) -> None:
        self.programmatic = programmatic
        self.llm = llm
        # Auto-detect concurrency from provider type
        if max_concurrent is not None:
            concurrency = max_concurrent
        elif type(llm).__name__ == "ClaudeCodeProvider":
            concurrency = self._DEFAULT_CONCURRENT_CLAUDE_CODE
        else:
            concurrency = self._DEFAULT_CONCURRENT_API
        self._semaphore = asyncio.Semaphore(concurrency)

    async def extract(self, sp: ScreenedPaper) -> PaperExtraction:
        """Extract structured data from a screened paper.

        Runs the programmatic extractor first, then refines with LLM if
        full text is available. Abstract-only papers short-circuit and
        return the programmatic draft immediately.

        Args:
            sp: The screened paper to extract from.

        Returns:
            A PaperExtraction with LLM-refined fields where applicable.
        """
        draft = self.programmatic.extract(sp)

        # Short-circuit: no full text → skip LLM, return programmatic result
        if not sp.paper.full_text:
            logger.debug(
                "hybrid_extraction.short_circuit",
                paper_id=sp.paper.id,
                reason="no_full_text",
            )
            return draft

        async with self._semaphore:
            refined = await self._refine_with_llm(sp, draft)

        # Override study_design and quality_score with authoritative programmatic values
        refined = refined.model_copy(
            update={
                "study_design": draft.study_design,
                "quality_score": draft.quality_score,
            }
        )

        # Post-filter: verify all numbers in LLM claims are grounded in source text
        source_text = (sp.paper.abstract or "") + " " + (sp.paper.full_text or "")
        refined = self._verify_grounding(refined, source_text, sp.paper.id)

        return refined

    @staticmethod
    def _verify_grounding(
        extraction: PaperExtraction,
        source_text: str,
        paper_id: str,
    ) -> PaperExtraction:
        """Verify LLM claims are grounded in source text.

        Only flags TRUE fabrications — numbers that have no close match
        (within 20% tolerance) anywhere in the source. Rounded or
        approximate numbers ("~30%" when source says "30.2%") are fine.
        """
        source_nums: set[float] = set()
        for m in re.finditer(r"\d+\.?\d*", source_text):
            try:
                source_nums.add(float(m.group()))
            except ValueError:
                continue

        def _has_close_match(val: float) -> bool:
            """Check if val is within 20% of any source number."""
            if val < 5:
                return True  # Trivial numbers always pass
            for sn in source_nums:
                if sn == 0:
                    continue
                if abs(val - sn) / max(abs(sn), 1e-9) <= 0.20:
                    return True
            return False

        verified_findings = []
        for f in extraction.key_findings:
            claim_nums = set(re.findall(r"\d+\.?\d*", f.claim))
            quant_nums = set(re.findall(r"\d+\.?\d*", f.quantitative_result or ""))

            fabricated = []
            for n_str in claim_nums | quant_nums:
                try:
                    val = float(n_str)
                except ValueError:
                    continue
                if not _has_close_match(val):
                    fabricated.append(n_str)

            if fabricated:
                logger.warning(
                    "hybrid_extraction.fabricated_numbers",
                    paper_id=paper_id,
                    claim=f.claim[:80],
                    fabricated=fabricated[:5],
                )
                # Keep the claim text but strip the fabricated quant result
                f = f.model_copy(update={"quantitative_result": None})

            verified_findings.append(f)

        return extraction.model_copy(update={"key_findings": verified_findings})

    def _build_refinement_context(self, sp: ScreenedPaper, draft: PaperExtraction) -> str:
        """Build a condensed ~2K-token context from the paper and draft.

        Includes: title, abstract (truncated), top result/conclusion sentences,
        methods excerpt, and limitations excerpt. Stays under _CONTEXT_MAX_CHARS.
        """
        paper = sp.paper
        lines: list[str] = []

        # Title
        lines.append(f"TITLE: {paper.title}")

        # Abstract
        if paper.abstract:
            lines.append(f"\nABSTRACT:\n{paper.abstract[:_ABSTRACT_MAX_CHARS]}")

        # Parse sections from full text for targeted excerpts
        full_text = paper.full_text or ""
        sections: list[ParsedSection] = parse_sections(full_text) if full_text else []

        results_text = _find_section_text(
            sections, ["result", "finding", "experiment", "evaluation", "performance"]
        )
        if results_text:
            lines.append(f"\nRESULTS EXCERPT:\n{results_text}")

        conclusion_text = _find_section_text(sections, ["conclusion", "discussion", "summary"])
        if conclusion_text:
            lines.append(f"\nCONCLUSION EXCERPT:\n{conclusion_text}")

        methods_text = _find_section_text(
            sections, ["method", "approach", "material", "dataset", "implementation"]
        )
        if methods_text:
            lines.append(f"\nMETHODS EXCERPT:\n{methods_text}")

        limitations_text = _find_section_text(sections, ["limitation", "future work", "weakness"])
        if limitations_text:
            lines.append(f"\nLIMITATIONS EXCERPT:\n{limitations_text}")

        # Extract verified numbers from the top programmatic findings
        # These are the ONLY numbers Haiku should use
        verified_quants: list[str] = []
        for f in draft.key_findings[:15]:
            if f.quantitative_result:
                verified_quants.append(f.quantitative_result[:100])
        if verified_quants:
            lines.append(
                "\nVERIFIED NUMBERS (only use these specific values):\n"
                + "\n".join(f"  - {q}" for q in verified_quants[:10])
            )

        context = "\n".join(lines)
        # Hard cap to stay within token budget
        return context[:_CONTEXT_MAX_CHARS]

    async def _refine_with_llm(self, sp: ScreenedPaper, draft: PaperExtraction) -> PaperExtraction:
        """Call the LLM to refine the draft extraction.

        Supports two backends:
        - API mode: llm.generate_structured() → parsed Pydantic model directly
        - claude-p / raw mode: llm.generate() → parse JSON manually

        Falls back to the programmatic draft on any parse failure.

        Args:
            sp: The screened paper (used to build context).
            draft: The programmatic draft extraction.

        Returns:
            Refined PaperExtraction, or the draft on failure.
        """
        context = self._build_refinement_context(sp, draft)
        prompt = build_refinement_prompt(draft, context)

        # API mode: provider supports structured output natively
        if callable(getattr(self.llm, "generate_structured", None)):
            try:
                response = await self.llm.generate_structured(
                    prompt=prompt,
                    response_model=PaperExtraction,
                    system=HYBRID_SYSTEM_PROMPT,
                    max_tokens=2048,
                    temperature=0.0,
                )
                refined: PaperExtraction = response.parsed
                logger.info(
                    "hybrid_extraction.refined",
                    paper_id=draft.paper_id,
                    mode="structured",
                    findings=len(refined.key_findings),
                )
                return refined
            except Exception as exc:
                logger.warning(
                    "hybrid_extraction.structured_failed",
                    paper_id=draft.paper_id,
                    error=str(exc),
                )
                return draft

        # claude-p / raw JSON mode
        try:
            response = await self.llm.generate(
                prompt=prompt,
                system=HYBRID_SYSTEM_PROMPT,
                max_tokens=2048,
                temperature=0.0,
            )
            raw_text: str = response.content
            json_str = _extract_json_block(raw_text)
            refined = PaperExtraction.model_validate_json(json_str)
            logger.info(
                "hybrid_extraction.refined",
                paper_id=draft.paper_id,
                mode="json_parse",
                findings=len(refined.key_findings),
            )
            return refined
        except Exception as exc:
            logger.warning(
                "hybrid_extraction.parse_failed",
                paper_id=draft.paper_id,
                error=str(exc),
            )
            return draft

    async def extract_batch(
        self,
        papers: list[ScreenedPaper],
    ) -> tuple[list[PaperExtraction], list[ExtractionFailure]]:
        """Extract from a batch of papers concurrently.

        Args:
            papers: List of screened papers to extract from.

        Returns:
            A 2-tuple of (successful extractions, failures).
        """
        extractions: list[PaperExtraction] = []
        failures: list[ExtractionFailure] = []

        tasks = [self.extract(sp) for sp in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for sp, result in zip(papers, results, strict=False):
            if isinstance(result, Exception):
                logger.error(
                    "hybrid_extraction.batch_failure",
                    paper_id=sp.paper.id,
                    error=str(result),
                )
                failures.append(ExtractionFailure(paper_id=sp.paper.id, error=str(result)))
            else:
                extractions.append(result)  # type: ignore[arg-type]

        return extractions, failures
