"""Tests for Unpaywall full-text retrieval."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autoreview.models.paper import CandidatePaper, ScreenedPaper
from autoreview.search.unpaywall import UnpaywallClient, UnpaywallResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GOOD_TEXT = "a" * 200  # longer than the 100-char threshold


def _make_client() -> UnpaywallClient:
    """Return a client whose HTTP layer is fully mocked."""
    client = UnpaywallClient(email="test@example.com")
    client._client = MagicMock()
    return client


def _ok_pdf_response(content: bytes = b"%PDF" + b"x" * 4000) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.content = content
    return resp


def _ok_html_response(text: str = f"<p>{'word ' * 50}</p>") -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.text = text
    return resp


def _fail_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 404
    resp.content = b""
    return resp


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


class TestLookupDoiOaLocations:
    """Verify that oa_locations is stored on the result."""

    @pytest.mark.asyncio
    async def test_lookup_stores_oa_locations(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "is_oa": True,
            "oa_status": "gold",
            "best_oa_location": {
                "url": "https://best.example.com/paper",
                "url_for_pdf": "https://best.example.com/paper.pdf",
                "url_for_landing_page": "https://best.example.com/paper.html",
            },
            "oa_locations": [
                {
                    "url_for_pdf": "https://best.example.com/paper.pdf",
                    "url_for_landing_page": "https://best.example.com/paper.html",
                },
                {
                    "url_for_pdf": "https://repo.example.com/paper.pdf",
                    "url_for_landing_page": "https://repo.example.com/paper.html",
                },
            ],
        }

        client = _make_client()
        client._client.get = AsyncMock(return_value=mock_response)

        result = await client.lookup_doi("10.1234/test")
        assert result is not None
        assert len(result.oa_locations) == 2
        assert result.oa_locations[1]["url_for_pdf"] == "https://repo.example.com/paper.pdf"

    @pytest.mark.asyncio
    async def test_lookup_empty_oa_locations(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "is_oa": False,
            "oa_status": "closed",
            "best_oa_location": None,
            "oa_locations": [],
        }

        client = _make_client()
        client._client.get = AsyncMock(return_value=mock_response)

        result = await client.lookup_doi("10.1234/closed")
        assert result is not None
        assert result.oa_locations == []


class TestFetchFullTextFallback:
    """Tests for oa_locations fallback iteration in fetch_full_text."""

    @pytest.mark.asyncio
    async def test_secondary_location_tried_when_best_fails(self):
        """When best_oa_location PDF fails, a secondary oa_location should succeed."""
        result = UnpaywallResult(
            doi="10.1234/fallback",
            is_oa=True,
            oa_status="green",
            pdf_url="https://broken.example.com/paper.pdf",
            html_url=None,
            oa_locations=[
                # index 0 — same as best (will be skipped as duplicate)
                {
                    "url_for_pdf": "https://broken.example.com/paper.pdf",
                    "url_for_landing_page": None,
                },
                # index 1 — secondary repo with working PDF
                {
                    "url_for_pdf": "https://repo.example.com/paper.pdf",
                    "url_for_landing_page": None,
                },
            ],
        )

        client = _make_client()

        # Patch the private helpers so we don't need real HTTP or PyMuPDF
        async def _try_pdf(doi: str, url: str) -> str | None:
            if url == "https://repo.example.com/paper.pdf":
                return _GOOD_TEXT
            return None

        client._try_pdf_url = _try_pdf  # type: ignore[method-assign]
        client._try_html_url = AsyncMock(return_value=None)  # type: ignore[method-assign]

        text = await client.fetch_full_text(result)
        assert text == _GOOD_TEXT

    @pytest.mark.asyncio
    async def test_iteration_stops_on_first_success(self):
        """fetch_full_text must not call _try_pdf_url after a success."""
        result = UnpaywallResult(
            doi="10.1234/stops",
            is_oa=True,
            oa_status="green",
            pdf_url=None,
            html_url=None,
            oa_locations=[
                {
                    "url_for_pdf": "https://first.example.com/paper.pdf",
                    "url_for_landing_page": None,
                },
                {
                    "url_for_pdf": "https://second.example.com/paper.pdf",
                    "url_for_landing_page": None,
                },
                {
                    "url_for_pdf": "https://third.example.com/paper.pdf",
                    "url_for_landing_page": None,
                },
            ],
        )

        call_log: list[str] = []

        async def _try_pdf(doi: str, url: str) -> str | None:
            call_log.append(url)
            if url == "https://first.example.com/paper.pdf":
                return _GOOD_TEXT
            return None

        client = _make_client()
        client._try_pdf_url = _try_pdf  # type: ignore[method-assign]
        client._try_html_url = AsyncMock(return_value=None)  # type: ignore[method-assign]

        text = await client.fetch_full_text(result)
        assert text == _GOOD_TEXT
        # Only the first location should have been tried
        assert call_log == ["https://first.example.com/paper.pdf"]

    @pytest.mark.asyncio
    async def test_empty_oa_locations_returns_none(self):
        """With no locations at all, fetch_full_text should return None."""
        result = UnpaywallResult(
            doi="10.1234/empty",
            is_oa=False,
            oa_status="closed",
            pdf_url=None,
            html_url=None,
            oa_locations=[],
        )

        client = _make_client()
        client._try_pdf_url = AsyncMock(return_value=None)  # type: ignore[method-assign]
        client._try_html_url = AsyncMock(return_value=None)  # type: ignore[method-assign]

        text = await client.fetch_full_text(result)
        assert text is None

    @pytest.mark.asyncio
    async def test_all_locations_failing_returns_none(self):
        """When every location (best + all oa_locations) fails, return None."""
        result = UnpaywallResult(
            doi="10.1234/allfail",
            is_oa=True,
            oa_status="green",
            pdf_url="https://broken1.example.com/paper.pdf",
            html_url="https://broken1.example.com/paper.html",
            oa_locations=[
                {
                    "url_for_pdf": "https://broken2.example.com/paper.pdf",
                    "url_for_landing_page": "https://broken2.example.com/paper.html",
                },
                {
                    "url_for_pdf": None,
                    "url_for_landing_page": "https://broken3.example.com/paper.html",
                },
            ],
        )

        client = _make_client()
        client._try_pdf_url = AsyncMock(return_value=None)  # type: ignore[method-assign]
        client._try_html_url = AsyncMock(return_value=None)  # type: ignore[method-assign]

        text = await client.fetch_full_text(result)
        assert text is None

        # Ensure all non-duplicate URLs were attempted
        pdf_calls = [str(c.args[1]) for c in client._try_pdf_url.call_args_list]  # type: ignore[union-attr]
        html_calls = [str(c.args[1]) for c in client._try_html_url.call_args_list]  # type: ignore[union-attr]

        assert "https://broken1.example.com/paper.pdf" in pdf_calls
        assert "https://broken2.example.com/paper.pdf" in pdf_calls
        assert "https://broken1.example.com/paper.html" in html_calls
        assert "https://broken3.example.com/paper.html" in html_calls


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
