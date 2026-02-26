"""Tests for Unpaywall full-text retrieval."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autoreview.models.paper import CandidatePaper, ScreenedPaper
from autoreview.search.unpaywall import UnpaywallClient, UnpaywallResult


class TestUnpaywallResultModel:
    def test_basic_model(self):
        result = UnpaywallResult(
            doi="10.1234/test",
            is_oa=True,
            best_oa_url="https://example.com/paper",
            oa_status="gold",
            pdf_url="https://example.com/paper.pdf",
            html_url="https://example.com/paper.html",
        )
        assert result.doi == "10.1234/test"
        assert result.is_oa is True
        assert result.oa_status == "gold"

    def test_closed_access(self):
        result = UnpaywallResult(doi="10.1234/closed", is_oa=False, oa_status="closed")
        assert result.is_oa is False
        assert result.pdf_url is None


class TestLookupDoi:
    @pytest.mark.asyncio
    async def test_lookup_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "is_oa": True,
            "oa_status": "gold",
            "best_oa_location": {
                "url": "https://example.com/paper",
                "url_for_pdf": "https://example.com/paper.pdf",
                "url_for_landing_page": "https://example.com/paper.html",
            },
        }

        client = UnpaywallClient(email="test@example.com")
        client._client = MagicMock()
        client._client.get = AsyncMock(return_value=mock_response)

        result = await client.lookup_doi("10.1234/test")
        assert result is not None
        assert result.is_oa is True
        assert result.pdf_url == "https://example.com/paper.pdf"

    @pytest.mark.asyncio
    async def test_lookup_closed_access(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "is_oa": False,
            "oa_status": "closed",
            "best_oa_location": None,
        }

        client = UnpaywallClient(email="test@example.com")
        client._client = MagicMock()
        client._client.get = AsyncMock(return_value=mock_response)

        result = await client.lookup_doi("10.1234/closed")
        assert result is not None
        assert result.is_oa is False

    @pytest.mark.asyncio
    async def test_lookup_404(self):
        mock_response = MagicMock()
        mock_response.status_code = 404

        client = UnpaywallClient(email="test@example.com")
        client._client = MagicMock()
        client._client.get = AsyncMock(return_value=mock_response)

        result = await client.lookup_doi("10.1234/notfound")
        assert result is None


class TestEnrichPapers:
    @pytest.mark.asyncio
    async def test_skips_no_doi(self):
        paper = CandidatePaper(
            title="No DOI Paper",
            authors=["Author A"],
            source_database="test",
            doi=None,
        )
        sp = ScreenedPaper(
            paper=paper,
            relevance_score=4,
            rationale="Relevant",
            include=True,
        )

        client = UnpaywallClient(email="test@example.com")
        client._client = MagicMock()
        attempted, enriched = await client.enrich_papers([sp])

        assert attempted == 0
        assert enriched == 0


class TestNodeWorksWithoutEmail:
    @pytest.mark.asyncio
    async def test_node_runs_without_unpaywall_email(self):
        """Pipeline node should still run (using S2/PMC/arXiv) without UNPAYWALL_EMAIL."""
        from autoreview.config import load_config
        from autoreview.models.knowledge_base import KnowledgeBase, PipelinePhase
        from autoreview.pipeline.nodes import PipelineNodes

        config = load_config(domain="biomedical")
        nodes = PipelineNodes(llm=None, config=config)
        kb = KnowledgeBase(topic="test")

        # Ensure env var is not set
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("UNPAYWALL_EMAIL", None)
            await nodes.full_text_retrieval(kb)

        assert kb.current_phase == PipelinePhase.FULL_TEXT_RETRIEVAL
        assert any("complete" in e.action for e in kb.audit_log)
