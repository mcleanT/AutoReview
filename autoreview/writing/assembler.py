from __future__ import annotations

import structlog

from autoreview.llm.prompts.outline import ReviewOutline
from autoreview.writing.section_writer import SectionDraft

logger = structlog.get_logger()


class DraftAssembler:
    """Assembles section drafts into a complete review paper draft."""

    def assemble(
        self,
        outline: ReviewOutline,
        section_drafts: dict[str, SectionDraft],
    ) -> str:
        """Assemble sections into a full draft with proper heading levels."""
        parts = [f"# {outline.title}\n"]

        for section in outline.sections:
            self._assemble_section(section, section_drafts, parts, level=2)

        full_draft = "\n\n".join(parts)
        word_count = len(full_draft.split())

        logger.info(
            "assembler.complete",
            sections=len(section_drafts),
            word_count=word_count,
        )

        return full_draft

    def _assemble_section(
        self,
        section: any,
        drafts: dict[str, SectionDraft],
        parts: list[str],
        level: int,
    ) -> None:
        """Recursively assemble a section and its subsections."""
        heading = "#" * level
        draft = drafts.get(section.id)

        if draft:
            parts.append(f"{heading} {section.title}\n\n{draft.text}")
        else:
            parts.append(f"{heading} {section.title}")

        for sub in section.subsections:
            self._assemble_section(sub, drafts, parts, level + 1)
