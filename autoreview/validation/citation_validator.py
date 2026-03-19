"""Citation validation: checks that [@paper_id] markers in text are valid and complete."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import Field

from autoreview.critique.models import CritiqueIssue, CritiqueSeverity
from autoreview.extraction.models import PaperExtraction
from autoreview.models.base import AutoReviewModel

if TYPE_CHECKING:
    from autoreview.writing.citation_selector import SectionCitationPlan


class PlanValidationReport(AutoReviewModel):
    """Report from validating citations against a CitationPlan."""

    section_id: str = ""
    uncited_primary: list[str] = Field(default_factory=list)  # PRIMARY papers not cited
    total_citations: int = 0
    citation_budget: int = 0
    budget_deviation: float = 0.0  # how far off from budget (fraction)
    additive_citations: list[str] = Field(default_factory=list)  # citations not in plan


class SuspiciousAttribution(AutoReviewModel):
    """A citation whose surrounding claim does not closely match any finding."""

    paper_id: str
    claimed_text: str
    closest_finding: str | None = None
    similarity_note: str = ""


class CitationValidationReport(AutoReviewModel):
    """Report from validating citations in a section or full draft."""

    section_id: str = ""
    valid_citations: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    uncited_papers: list[str] = Field(default_factory=list)
    suspicious_attributions: list[SuspiciousAttribution] = Field(default_factory=list)
    total_citation_markers: int = 0
    unique_citations: int = 0


_CITATION_RE = re.compile(r"\[@([^\]]+)\]")


class CitationValidator:
    """Validates citation markers against known paper extractions."""

    def validate_section(
        self,
        text: str,
        section_paper_ids: list[str],
        extractions: dict[str, PaperExtraction],
    ) -> CitationValidationReport:
        """Validate citations in a single section.

        Args:
            text: Section text containing [@paper_id] markers.
            section_paper_ids: Paper IDs assigned to this section by the outline.
            extractions: All available paper extractions.

        Returns:
            CitationValidationReport with valid/invalid/uncited findings.
        """
        cited_ids = _CITATION_RE.findall(text)
        unique_cited = set(cited_ids)
        known_ids = set(extractions.keys())
        set(section_paper_ids)

        valid = [pid for pid in unique_cited if pid in known_ids]
        invalid = [pid for pid in unique_cited if pid not in known_ids]
        uncited = [pid for pid in section_paper_ids if pid not in unique_cited]

        return CitationValidationReport(
            valid_citations=sorted(valid),
            invalid_citations=sorted(invalid),
            uncited_papers=sorted(uncited),
            total_citation_markers=len(cited_ids),
            unique_citations=len(unique_cited),
        )

    def validate_full_draft(
        self,
        full_draft: str,
        extractions: dict[str, PaperExtraction],
    ) -> CitationValidationReport:
        """Validate all citations across the full assembled draft.

        Args:
            full_draft: The complete draft text.
            extractions: All available paper extractions.

        Returns:
            CitationValidationReport for the whole document.
        """
        cited_ids = _CITATION_RE.findall(full_draft)
        unique_cited = set(cited_ids)
        known_ids = set(extractions.keys())

        valid = [pid for pid in unique_cited if pid in known_ids]
        invalid = [pid for pid in unique_cited if pid not in known_ids]

        # All extracted papers that were never cited anywhere
        all_paper_ids = set(extractions.keys())
        uncited = [pid for pid in sorted(all_paper_ids) if pid not in unique_cited]

        return CitationValidationReport(
            section_id="full_draft",
            valid_citations=sorted(valid),
            invalid_citations=sorted(invalid),
            uncited_papers=uncited,
            total_citation_markers=len(cited_ids),
            unique_citations=len(unique_cited),
        )

    def validate_against_plan(
        self,
        text: str,
        plan: SectionCitationPlan,
    ) -> PlanValidationReport:
        """Validate section text against a CitationPlan for compliance.

        Checks that all PRIMARY papers in the plan are cited, compares
        total citation count against the plan budget, and flags citations
        not in the plan (e.g., from passage_search) as additive.

        Args:
            text: Section text containing [@paper_id] markers.
            plan: The SectionCitationPlan produced by citation_selection.

        Returns:
            PlanValidationReport with compliance details.
        """
        cited_ids = set(_CITATION_RE.findall(text))
        total_citations = len(_CITATION_RE.findall(text))

        # Collect all paper IDs in the plan (across all tiers)
        all_plan_paper_ids: set[str] = set()
        for pc in plan.primary_papers:
            all_plan_paper_ids.add(pc.paper_id)
        for pc in plan.supporting_papers:
            all_plan_paper_ids.add(pc.paper_id)
        for pc in plan.contextual_papers:
            all_plan_paper_ids.add(pc.paper_id)

        # Check PRIMARY paper compliance
        primary_paper_ids = {pc.paper_id for pc in plan.primary_papers}
        uncited_primary = sorted(primary_paper_ids - cited_ids)

        # Budget deviation (fraction): (actual - budget) / budget
        budget = plan.citation_budget
        budget_deviation = (total_citations - budget) / budget if budget > 0 else 0.0

        # Additive citations: cited but not in the plan (e.g., from passage_search)
        additive_citations = sorted(cited_ids - all_plan_paper_ids)

        return PlanValidationReport(
            section_id=plan.section_id,
            uncited_primary=uncited_primary,
            total_citations=total_citations,
            citation_budget=budget,
            budget_deviation=round(budget_deviation, 4),
            additive_citations=additive_citations,
        )

    @staticmethod
    def to_critique_issues(report: CitationValidationReport) -> list[CritiqueIssue]:
        """Convert a validation report into CritiqueIssue objects.

        Severity mapping:
        - Invalid citations (phantom references) -> CRITICAL
        - Uncited assigned papers -> MAJOR
        - Suspicious attributions -> MAJOR
        """
        issues: list[CritiqueIssue] = []

        for pid in report.invalid_citations:
            issues.append(
                CritiqueIssue(
                    severity=CritiqueSeverity.CRITICAL,
                    location=report.section_id or "unknown",
                    description=(
                        f"Citation [@{pid}] references a paper not found in the extraction corpus."
                    ),
                    suggested_fix=f"Remove [@{pid}] or replace with a valid paper ID.",
                )
            )

        for pid in report.uncited_papers:
            issues.append(
                CritiqueIssue(
                    severity=CritiqueSeverity.MAJOR,
                    location=report.section_id or "unknown",
                    description=f"Paper '{pid}' was assigned to this section but is never cited.",
                    suggested_fix=(
                        f"Incorporate findings from [@{pid}] or remove from section assignment."
                    ),
                )
            )

        for sa in report.suspicious_attributions:
            issues.append(
                CritiqueIssue(
                    severity=CritiqueSeverity.MAJOR,
                    location=report.section_id or "unknown",
                    description=(
                        f"Citation [@{sa.paper_id}] may be misattributed. "
                        f"Claimed: '{sa.claimed_text}'. {sa.similarity_note}"
                    ),
                    suggested_fix="Verify that the cited paper actually supports this claim.",
                )
            )

        return issues
