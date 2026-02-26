from __future__ import annotations

from pydantic import Field

from autoreview.models.base import AutoReviewModel


class OutlineSection(AutoReviewModel):
    """A section in the review outline."""
    id: str  # e.g., "1", "2.1", "3.2.1"
    title: str
    description: str
    theme_refs: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)
    estimated_word_count: int = 500
    subsections: list[OutlineSection] = Field(default_factory=list)

    def flatten(self) -> list[OutlineSection]:
        """Return this section and all subsections in order."""
        result = [self]
        for sub in self.subsections:
            result.extend(sub.flatten())
        return result


OutlineSection.model_rebuild()


class ReviewOutline(AutoReviewModel):
    """Complete review outline."""
    title: str
    sections: list[OutlineSection]

    def flatten(self) -> list[OutlineSection]:
        """Return all sections and subsections in order."""
        result: list[OutlineSection] = []
        for section in self.sections:
            result.extend(section.flatten())
        return result

    def get_section(self, section_id: str) -> OutlineSection | None:
        """Find a section by ID."""
        for s in self.flatten():
            if s.id == section_id:
                return s
        return None


OUTLINE_SYSTEM_PROMPT = """\
You are an expert scientific writer creating the outline for a comprehensive review paper. \
The outline must be hierarchical, with major sections and subsections that map to themes \
in the evidence. Each section should have a clear purpose and be assigned relevant papers.
"""

OUTLINE_CRITIQUE_SYSTEM_PROMPT = """\
You are an expert editor evaluating a review paper outline. Assess completeness against \
the scope, logical ordering, granularity balance, and whether all major themes are covered. \
Identify missing topics, redundant sections, and structural issues.
"""

OUTLINE_REVISION_SYSTEM_PROMPT = """\
You are an expert scientific writer revising the outline for a comprehensive review paper. \
You have received specific critique feedback on the previous outline. Your task is to \
address each issue while preserving the strengths of the existing structure. Do not \
regenerate from scratch — make targeted improvements based on the feedback.
"""


def build_outline_prompt(
    scope_document: str,
    evidence_summary: str,
    required_sections: list[str],
) -> str:
    req = "\n".join(f"- {s}" for s in required_sections)
    return f"""\
## Review Scope
{scope_document}

## Evidence Summary
{evidence_summary}

## Required Sections
{req}

Generate a hierarchical outline for this review paper. Each section needs:
- A unique ID (e.g., "1", "1.1", "1.1.1")
- A descriptive title
- A brief description of what the section covers
- Theme references from the evidence map
- Paper IDs that should be cited in this section
- Estimated word count

Ensure all themes are covered and the outline follows a logical narrative arc.
"""


def build_outline_critique_prompt(outline_text: str, scope_document: str) -> str:
    return f"""\
## Review Scope
{scope_document}

## Current Outline
{outline_text}

Evaluate this outline for:
1. **Completeness**: Does it cover all aspects of the scope?
2. **Logical ordering**: Does the flow make sense for a reader?
3. **Granularity**: Are sections appropriately sized and balanced?
4. **Coverage**: Are all themes represented?

Provide specific issues and suggestions for improvement.
"""


def build_outline_revision_prompt(
    scope_document: str,
    evidence_summary: str,
    required_sections: list[str],
    previous_outline_text: str,
    critique_issues_text: str,
    critique_score: float,
) -> str:
    """Build a prompt that revises an existing outline based on critique feedback.

    Unlike ``build_outline_prompt`` which generates from scratch, this prompt
    includes the previous outline and the specific critique issues so the LLM
    can make targeted improvements.
    """
    req = "\n".join(f"- {s}" for s in required_sections)
    return f"""\
## Review Scope
{scope_document}

## Evidence Summary
{evidence_summary}

## Required Sections
{req}

## Previous Outline
{previous_outline_text}

## Critique Feedback (score: {critique_score:.2f})
The following issues were identified in the previous outline:
{critique_issues_text}

Revise the outline to address each critique issue listed above. Preserve the strengths \
of the existing structure and make targeted changes rather than starting from scratch. \
Each section needs:
- A unique ID (e.g., "1", "1.1", "1.1.1")
- A descriptive title
- A brief description of what the section covers
- Theme references from the evidence map
- Paper IDs that should be cited in this section
- Estimated word count

Focus your revisions on the specific issues raised. Maintain what works well and fix \
what was flagged.
"""
