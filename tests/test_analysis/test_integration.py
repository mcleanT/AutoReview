# tests/test_analysis/test_integration.py
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autoreview.models.paper import CandidatePaper


class TestBibliographyInjectionIntegration:
    @pytest.mark.asyncio
    async def test_full_flow_with_mocked_apis(self, tmp_path):
        """Test extract -> parse -> resolve -> KB build with mocked external calls."""
        from paper.analysis.inject_bibliography import run_injection

        mock_paper = CandidatePaper(
            title="Machine learning in biology",
            authors=["Smith J", "Doe A"],
            year=2020,
            doi="10.1038/test001",
            source_database="semantic_scholar",
        )

        mock_s2 = MagicMock()
        mock_s2.source_name = "semantic_scholar"
        mock_s2.get_paper_details = _async_doi_lookup(mock_paper)
        mock_s2.search = _async_return([])

        mock_oa = MagicMock()
        mock_oa.source_name = "openalex"
        mock_oa.get_paper_details = _async_return(None)
        mock_oa.search = _async_return([])

        # Patch the source-module classes so lazy imports inside run_injection pick up mocks
        with (
            patch("paper.analysis.inject_bibliography.extract_text_from_pdf") as mock_extract,
            patch("paper.analysis.inject_bibliography.extract_bibliography_lines") as mock_bib,
            patch(
                "autoreview.search.semantic_scholar.SemanticScholarSearch",
                return_value=mock_s2,
            ),
            patch(
                "autoreview.search.openalex.OpenAlexSearch",
                return_value=mock_oa,
            ),
        ):
            mock_extract.return_value = "fake text"
            mock_bib.return_value = [
                "[1] Smith J, Doe A. Machine learning in biology. Nature. 2020. doi:10.1038/test001",
                "[2] Zhang L. Deep learning approaches. Science. 2019;365:100-105.",
                "[3] Unknown Author. Unparseable reference with no details.",
            ]

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
