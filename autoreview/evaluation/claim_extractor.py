from __future__ import annotations

import re
from typing import Any

import structlog

from autoreview.evaluation.models import Claim
from autoreview.llm.prompts.claim_extraction import (
    CLAIM_EXTRACTION_SYSTEM_PROMPT,
    ClaimExtractionResult,
    ConceptExtractionResult,
    build_claim_extraction_prompt,
    build_concept_extraction_prompt,
)

logger = structlog.get_logger()

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

_VALID_CATEGORIES = {"empirical", "methodological", "contextual", "synthesis", "limitation"}


class ClaimExtractor:
    """Section-chunked claim and concept extraction from review text."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def split_into_sections(self, text: str) -> list[dict[str, str]]:
        """Split markdown text into sections by headings."""
        matches = list(_HEADING_RE.finditer(text))
        if not matches:
            return [{"id": "full_text", "text": text.strip()}]

        sections: list[dict[str, str]] = []
        for i, match in enumerate(matches):
            title = match.group(2).strip()
            section_id = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append({"id": section_id, "text": body})

        return sections if sections else [{"id": "full_text", "text": text.strip()}]

    async def extract_claims(self, review_text: str) -> list[Claim]:
        """Extract claims from each section of the review, then aggregate."""
        sections = self.split_into_sections(review_text)
        all_claims: list[Claim] = []

        for section in sections:
            prompt = build_claim_extraction_prompt(section["text"])
            resp = await self.llm.generate_structured(
                prompt,
                ClaimExtractionResult,
                system=CLAIM_EXTRACTION_SYSTEM_PROMPT,
            )
            for raw in resp.parsed.claims:
                category = raw.category if raw.category in _VALID_CATEGORIES else "contextual"
                all_claims.append(
                    Claim(
                        text=raw.text,
                        category=category,
                        section_id=section["id"],
                    )
                )

            logger.info(
                "claim_extractor.section_done",
                section_id=section["id"],
                n_claims=len(resp.parsed.claims),
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
            )

        logger.info("claim_extractor.complete", total_claims=len(all_claims))
        return all_claims

    async def extract_concepts(self, review_text: str) -> list[str]:
        """Extract and deduplicate scientific concepts from review text."""
        prompt = build_concept_extraction_prompt(review_text)
        resp = await self.llm.generate_structured(
            prompt,
            ConceptExtractionResult,
            system=CLAIM_EXTRACTION_SYSTEM_PROMPT,
        )
        seen: set[str] = set()
        unique: list[str] = []
        for concept in resp.parsed.concepts:
            normalized = concept.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(normalized)

        logger.info(
            "claim_extractor.concepts_done",
            n_concepts=len(unique),
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
        )
        return unique
