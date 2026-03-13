# tests/test_analysis/test_inject_bibliography.py
from __future__ import annotations

import pytest

from autoreview.models.paper import CandidatePaper


class TestInjectBibliography:
    @pytest.mark.asyncio
    async def test_builds_knowledge_base_from_resolved_refs(self):
        from paper.analysis.inject_bibliography import build_injected_kb

        papers = [
            CandidatePaper(
                title="Paper A",
                authors=["Smith"],
                year=2018,
                doi="10.1234/a",
                source_database="semantic_scholar",
            ),
            CandidatePaper(
                title="Paper B",
                authors=["Doe"],
                year=2019,
                doi="10.1234/b",
                source_database="openalex",
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
            ResolutionConfidence,
            ResolvedReference,
        )

        resolved = [
            ResolvedReference(
                raw_line="[1] ...",
                confidence=ResolutionConfidence.HIGH,
                match_score=1.0,
                paper=CandidatePaper(title="A", authors=["X"], source_database="s2", doi="10.1/a"),
            ),
            ResolvedReference(
                raw_line="[2] ...",
                confidence=ResolutionConfidence.MEDIUM,
                match_score=0.9,
                paper=CandidatePaper(title="B", authors=["Y"], source_database="s2"),
            ),
        ]
        failed = ["[3] Unresolvable reference line"]

        report = build_resolution_report(
            total_extracted=3,
            resolved=resolved,
            failed_lines=failed,
        )
        assert report["total_extracted"] == 3
        assert report["resolved_count"] == 2
        assert report["failed_count"] == 1
        assert report["by_confidence"]["high"] == 1
        assert report["by_confidence"]["medium"] == 1
