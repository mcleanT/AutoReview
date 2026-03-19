"""Tests for multi-source full-text resolver."""

from __future__ import annotations

import os
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autoreview.models.paper import CandidatePaper, ScreenedPaper
from autoreview.search.full_text import (
    FullTextResolver,
    _extract_text_from_elsevier_xml,
    _extract_text_from_jats_xml,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_screened(
    doi: str | None = None,
    external_ids: dict[str, str] | None = None,
    full_text: str | None = None,
) -> ScreenedPaper:
    paper = CandidatePaper(
        title="Test Paper",
        authors=["Author A"],
        source_database="test",
        doi=doi,
        abstract="This is an abstract for a test paper about something important.",
        external_ids=external_ids or {},
        full_text=full_text,
    )
    return ScreenedPaper(
        paper=paper,
        relevance_score=4,
        rationale="Relevant",
        include=True,
    )


_SAMPLE_JATS = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE article PUBLIC "-//NLM//DTD JATS 1.2//EN" "JATS-archivearticle1.dtd">
<article>
  <body>
    <sec>
      <title>Introduction</title>
      <p>This is the introduction paragraph of the paper with enough text
      to pass the 100 character minimum threshold for extraction.</p>
    </sec>
    <sec>
      <title>Methods</title>
      <p>We performed a comprehensive analysis using standard methods
      and approaches that are well established in the field.</p>
    </sec>
    <fig><caption><p>Figure caption</p></caption></fig>
  </body>
</article>
"""

_SAMPLE_PDF_TEXT = "A" * 200  # Simulates extracted PDF text > 100 chars


# ---------------------------------------------------------------------------
# JATS XML extraction
# ---------------------------------------------------------------------------


class TestJatsExtraction:
    def test_extracts_body_text(self):
        text = _extract_text_from_jats_xml(_SAMPLE_JATS)
        assert text is not None
        assert "introduction paragraph" in text.lower()
        assert "comprehensive analysis" in text.lower()

    def test_removes_figures(self):
        text = _extract_text_from_jats_xml(_SAMPLE_JATS)
        assert text is not None
        assert "Figure caption" not in text

    def test_returns_none_for_empty_xml(self):
        text = _extract_text_from_jats_xml(b"<article><body></body></article>")
        assert text is None

    def test_returns_none_for_invalid_xml(self):
        text = _extract_text_from_jats_xml(b"not xml at all")
        # Should not raise, just return None
        assert text is None


# ---------------------------------------------------------------------------
# Elsevier XML extraction
# ---------------------------------------------------------------------------

_SAMPLE_ELSEVIER_XML = """\
<full-text-retrieval-response xmlns:ce="http://www.elsevier.com/xml/common/dtd">
  <originalText>
    <ce:sections>
      <ce:section>
        <ce:section-title>Introduction</ce:section-title>
        <ce:para>During brain development, progenitors generate successive waves
        of neurons that populate distinct cerebral regions. Long enough to
        exceed the minimum threshold.</ce:para>
      </ce:section>
      <ce:section>
        <ce:section-title>Methods</ce:section-title>
        <ce:para>We performed a comprehensive immunohistochemical analysis using standard
        protocols that have been validated across multiple laboratories.</ce:para>
      </ce:section>
    </ce:sections>
  </originalText>
</full-text-retrieval-response>
"""


class TestElsevierXmlExtraction:
    def test_extracts_sections_text(self):
        text = _extract_text_from_elsevier_xml(_SAMPLE_ELSEVIER_XML)
        assert text is not None
        assert "brain development" in text.lower()
        assert "immunohistochemical" in text.lower()

    def test_returns_none_for_empty_xml(self):
        text = _extract_text_from_elsevier_xml("<response></response>")
        assert text is None

    def test_returns_none_for_short_text(self):
        text = _extract_text_from_elsevier_xml(
            "<response><ce:sections><ce:para>Short</ce:para></ce:sections></response>"
        )
        assert text is None


# ---------------------------------------------------------------------------
# Strategy: Elsevier ScienceDirect API
# ---------------------------------------------------------------------------


class TestElsevierApiStrategy:
    @pytest.mark.asyncio
    async def test_fetches_plain_text_for_elsevier_doi(self):
        sp = _make_screened(doi="10.1016/j.tins.2020.10.012")
        resolver = FullTextResolver(elsevier_api_key="test-key")

        plain_text = "Article full text content " * 50  # > 500 chars
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = plain_text
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        text = await resolver._try_elsevier_api(sp.paper)
        assert text == plain_text

        # Verify correct API call
        resolver._client.get.assert_called_once()
        call_args = resolver._client.get.call_args
        assert "api.elsevier.com" in call_args[0][0]
        assert call_args[1]["headers"]["Accept"] == "text/plain"

    @pytest.mark.asyncio
    async def test_falls_back_to_xml_when_plain_fails(self):
        sp = _make_screened(doi="10.1016/j.cell.2020.06.043")
        resolver = FullTextResolver(elsevier_api_key="test-key")

        # First call (plain text) returns empty, second (XML) returns content
        empty_resp = MagicMock()
        empty_resp.status_code = 200
        empty_resp.text = ""

        xml_resp = MagicMock()
        xml_resp.status_code = 200
        xml_resp.text = _SAMPLE_ELSEVIER_XML

        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(side_effect=[empty_resp, xml_resp])

        text = await resolver._try_elsevier_api(sp.paper)
        assert text is not None
        assert "brain development" in text.lower()

    @pytest.mark.asyncio
    async def test_skips_non_elsevier_dois(self):
        sp = _make_screened(doi="10.1038/s41586-020-2649-2")  # Nature DOI
        resolver = FullTextResolver(elsevier_api_key="test-key")
        text = await resolver._try_elsevier_api(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=False)
    async def test_skips_without_api_key(self):
        os.environ.pop("ELSEVIER_API_KEY", None)
        sp = _make_screened(doi="10.1016/j.tins.2020.10.012")
        resolver = FullTextResolver(elsevier_api_key=None)
        text = await resolver._try_elsevier_api(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    async def test_skips_without_doi(self):
        sp = _make_screened(doi=None)
        resolver = FullTextResolver(elsevier_api_key="test-key")
        text = await resolver._try_elsevier_api(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    async def test_handles_auth_failure(self):
        sp = _make_screened(doi="10.1016/j.tins.2020.10.012")
        resolver = FullTextResolver(elsevier_api_key="bad-key")

        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        text = await resolver._try_elsevier_api(sp.paper)
        assert text is None


# ---------------------------------------------------------------------------
# Strategy: Wiley TDM API
# ---------------------------------------------------------------------------


class TestWileyTdmStrategy:
    @pytest.mark.asyncio
    async def test_fetches_wiley_pdf(self):
        sp = _make_screened(doi="10.1002/ana.25430")
        resolver = FullTextResolver(wiley_tdm_token="test-token")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"x" * 2000  # Simulate non-trivial PDF binary
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_wiley_tdm(sp.paper)

        assert text == _SAMPLE_PDF_TEXT
        resolver._client.get.assert_called_once()
        call_args = resolver._client.get.call_args
        assert "api.wiley.com" in call_args[0][0]
        assert "10.1002" in call_args[0][0]
        assert call_args[1]["headers"]["Wiley-TDM-Client-Token"] == "test-token"

    @pytest.mark.asyncio
    async def test_skips_non_wiley_doi(self):
        sp = _make_screened(doi="10.1016/j.cell.2020.06.043")  # Elsevier DOI
        resolver = FullTextResolver(wiley_tdm_token="test-token")
        text = await resolver._try_wiley_tdm(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=False)
    async def test_skips_without_token(self):
        os.environ.pop("WILEY_TDM_TOKEN", None)
        sp = _make_screened(doi="10.1002/ana.25430")
        resolver = FullTextResolver(wiley_tdm_token=None)
        text = await resolver._try_wiley_tdm(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    async def test_returns_none_on_auth_failure(self):
        sp = _make_screened(doi="10.1111/j.1460-9568.2010.07380.x")
        resolver = FullTextResolver(wiley_tdm_token="bad-token")

        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.content = b"Forbidden"
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        text = await resolver._try_wiley_tdm(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    async def test_returns_none_without_doi(self):
        sp = _make_screened(doi=None)
        resolver = FullTextResolver(wiley_tdm_token="test-token")
        text = await resolver._try_wiley_tdm(sp.paper)
        assert text is None


# ---------------------------------------------------------------------------
# Strategy: Springer Nature Open Access API
# ---------------------------------------------------------------------------

_SAMPLE_SPRINGER_JATS = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<response>
  <article>
    <body>
      <sec>
        <title>Introduction</title>
        <p>Cellular senescence is a state of irreversible growth arrest that
        contributes to organismal aging and multiple age-related pathologies.</p>
      </sec>
      <sec>
        <title>Results</title>
        <p>We identified several genetic features shared across tissues including
        p16INK4a expression, SASP components, and telomere dysfunction markers.</p>
      </sec>
    </body>
  </article>
</response>
"""


class TestSpringerOaStrategy:
    @pytest.mark.asyncio
    async def test_fetches_jats_for_springer_doi(self):
        sp = _make_screened(doi="10.1038/s41586-020-2649-2")
        resolver = FullTextResolver(springer_api_key="test-key")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _SAMPLE_SPRINGER_JATS.decode()
        mock_resp.content = _SAMPLE_SPRINGER_JATS
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        text = await resolver._try_springer_oa(sp.paper)
        assert text is not None
        assert "cellular senescence" in text.lower()
        assert "genetic features" in text.lower()

        # Verify correct API call
        resolver._client.get.assert_called_once()
        call_args = resolver._client.get.call_args
        assert "api.springernature.com" in call_args[0][0]
        assert call_args[1]["params"]["api_key"] == "test-key"

    @pytest.mark.asyncio
    async def test_handles_bmc_doi(self):
        sp = _make_screened(doi="10.1186/s12864-023-09275-2")
        resolver = FullTextResolver(springer_api_key="test-key")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _SAMPLE_SPRINGER_JATS.decode()
        mock_resp.content = _SAMPLE_SPRINGER_JATS
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        text = await resolver._try_springer_oa(sp.paper)
        assert text is not None

    @pytest.mark.asyncio
    async def test_handles_springer_journal_doi(self):
        sp = _make_screened(doi="10.1007/s00018-023-04694-y")
        resolver = FullTextResolver(springer_api_key="test-key")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _SAMPLE_SPRINGER_JATS.decode()
        mock_resp.content = _SAMPLE_SPRINGER_JATS
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        text = await resolver._try_springer_oa(sp.paper)
        assert text is not None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_body(self):
        """Articles not in OA collection return empty XML with no <body>."""
        sp = _make_screened(doi="10.1038/s41586-023-06266-9")
        resolver = FullTextResolver(springer_api_key="test-key")

        empty_resp = MagicMock()
        empty_resp.status_code = 200
        empty_resp.text = "<response><result><total>0</total></result><records/></response>"
        empty_resp.content = b"<response><result><total>0</total></result><records/></response>"
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=empty_resp)

        text = await resolver._try_springer_oa(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    async def test_skips_non_springer_dois(self):
        sp = _make_screened(doi="10.1016/j.cell.2020.06.043")  # Elsevier DOI
        resolver = FullTextResolver(springer_api_key="test-key")
        text = await resolver._try_springer_oa(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=False)
    async def test_skips_without_api_key(self):
        os.environ.pop("SPRINGER_API_KEY", None)
        sp = _make_screened(doi="10.1038/s41586-020-2649-2")
        resolver = FullTextResolver(springer_api_key=None)
        text = await resolver._try_springer_oa(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    async def test_skips_without_doi(self):
        sp = _make_screened(doi=None)
        resolver = FullTextResolver(springer_api_key="test-key")
        text = await resolver._try_springer_oa(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    async def test_handles_auth_failure(self):
        sp = _make_screened(doi="10.1038/s41586-020-2649-2")
        resolver = FullTextResolver(springer_api_key="bad-key")

        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        text = await resolver._try_springer_oa(sp.paper)
        assert text is None


# ---------------------------------------------------------------------------
# Strategy: Springer/Nature institutional PDF
# ---------------------------------------------------------------------------


class TestSpringerInstitutionalStrategy:
    @pytest.mark.asyncio
    async def test_fetches_springer_pdf(self):
        sp = _make_screened(doi="10.1007/s10787-025-01807-w")
        resolver = FullTextResolver()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"%PDF" + b"x" * 2000
        mock_resp.headers = {"content-type": "application/pdf"}
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_springer_institutional(sp.paper)

        assert text == _SAMPLE_PDF_TEXT
        call_url = resolver._client.get.call_args[0][0]
        assert "link.springer.com/content/pdf/" in call_url

    @pytest.mark.asyncio
    async def test_uses_nature_url_for_nature_doi(self):
        sp = _make_screened(doi="10.1038/s41572-025-00674-7")
        resolver = FullTextResolver()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"%PDF" + b"x" * 2000
        mock_resp.headers = {"content-type": "application/pdf"}
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_springer_institutional(sp.paper)

        assert text == _SAMPLE_PDF_TEXT
        call_url = resolver._client.get.call_args[0][0]
        assert "nature.com/articles/" in call_url
        assert call_url.endswith(".pdf")

    @pytest.mark.asyncio
    async def test_rejects_html_login_page(self):
        """If not on VPN, Springer returns an HTML login page instead of PDF."""
        sp = _make_screened(doi="10.1007/s12035-024-04421-z")
        resolver = FullTextResolver()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"<html><body>Please log in</body></html>"
        mock_resp.headers = {"content-type": "text/html"}
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        text = await resolver._try_springer_institutional(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    async def test_returns_none_on_403(self):
        sp = _make_screened(doi="10.1007/s10286-025-01107-x")
        resolver = FullTextResolver()

        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.content = b"Forbidden"
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        text = await resolver._try_springer_institutional(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    async def test_skips_non_springer_doi(self):
        sp = _make_screened(doi="10.1002/ana.25430")  # Wiley DOI
        resolver = FullTextResolver()
        text = await resolver._try_springer_institutional(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    async def test_skips_without_doi(self):
        sp = _make_screened(doi=None)
        resolver = FullTextResolver()
        text = await resolver._try_springer_institutional(sp.paper)
        assert text is None


# ---------------------------------------------------------------------------
# Strategy: Semantic Scholar PDF
# ---------------------------------------------------------------------------


class TestS2PdfStrategy:
    @pytest.mark.asyncio
    async def test_fetches_pdf_from_s2_url(self):
        sp = _make_screened(external_ids={"s2_pdf_url": "https://example.com/paper.pdf"})

        resolver = FullTextResolver()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"x" * 2000
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_s2_pdf(sp.paper)

        assert text == _SAMPLE_PDF_TEXT

    @pytest.mark.asyncio
    async def test_returns_none_without_url(self):
        sp = _make_screened(external_ids={})
        resolver = FullTextResolver()
        text = await resolver._try_s2_pdf(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    async def test_returns_none_on_http_error(self):
        sp = _make_screened(external_ids={"s2_pdf_url": "https://example.com/paper.pdf"})

        resolver = FullTextResolver()
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.content = b""
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        text = await resolver._try_s2_pdf(sp.paper)
        assert text is None


# ---------------------------------------------------------------------------
# Strategy: PubMed Central
# ---------------------------------------------------------------------------


class TestPmcStrategy:
    @pytest.mark.asyncio
    async def test_fetches_pmc_xml(self):
        sp = _make_screened(external_ids={"pmid": "12345", "pmcid": "PMC6789"})
        pmid_to_pmcid = {"12345": "PMC6789"}

        resolver = FullTextResolver()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = _SAMPLE_JATS
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        text = await resolver._try_pmc(sp.paper, pmid_to_pmcid)
        assert text is not None
        assert "introduction paragraph" in text.lower()

    @pytest.mark.asyncio
    async def test_uses_pmid_lookup(self):
        """Falls back to pmid_to_pmcid mapping when pmcid not in external_ids."""
        sp = _make_screened(external_ids={"pmid": "12345"})
        pmid_to_pmcid = {"12345": "PMC6789"}

        resolver = FullTextResolver()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = _SAMPLE_JATS
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        text = await resolver._try_pmc(sp.paper, pmid_to_pmcid)
        assert text is not None

    @pytest.mark.asyncio
    async def test_returns_none_without_pmid(self):
        sp = _make_screened(external_ids={})
        resolver = FullTextResolver()
        text = await resolver._try_pmc(sp.paper, {})
        assert text is None

    @pytest.mark.asyncio
    async def test_normalises_pmcid_prefix(self):
        """PMCIDs without 'PMC' prefix should be normalised."""
        sp = _make_screened(external_ids={"pmcid": "6789"})

        resolver = FullTextResolver()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = _SAMPLE_JATS
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        text = await resolver._try_pmc(sp.paper, {})
        assert text is not None

        # Verify the request used the normalised ID
        call_args = resolver._client.get.call_args
        assert "PMC6789" in str(call_args)


# ---------------------------------------------------------------------------
# Strategy: arXiv
# ---------------------------------------------------------------------------


class TestArxivStrategy:
    @pytest.mark.asyncio
    async def test_fetches_arxiv_pdf(self):
        sp = _make_screened(external_ids={"arxiv": "2301.12345"})

        resolver = FullTextResolver()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"x" * 2000
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_arxiv(sp.paper)

        assert text == _SAMPLE_PDF_TEXT
        # Verify correct URL
        call_args = resolver._client.get.call_args
        assert "arxiv.org/pdf/2301.12345" in str(call_args)

    @pytest.mark.asyncio
    async def test_strips_version_suffix(self):
        sp = _make_screened(external_ids={"arxiv": "2301.12345v3"})

        resolver = FullTextResolver()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"x" * 2000
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            await resolver._try_arxiv(sp.paper)

        call_args = resolver._client.get.call_args
        assert "2301.12345v3" not in str(call_args)
        assert "2301.12345" in str(call_args)

    @pytest.mark.asyncio
    async def test_returns_none_without_arxiv_id(self):
        sp = _make_screened(external_ids={})
        resolver = FullTextResolver()
        text = await resolver._try_arxiv(sp.paper)
        assert text is None


# ---------------------------------------------------------------------------
# Strategy: bioRxiv / medRxiv (with version resolution)
# ---------------------------------------------------------------------------


def _make_biorxiv_api_response(doi: str, versions: list[int]) -> MagicMock:
    """Build a mock biorxiv API response with the given version list."""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {"collection": [{"doi": doi, "version": str(v)} for v in versions]}
    return mock


class TestBiorxivStrategy:
    @pytest.mark.asyncio
    async def test_skips_non_biorxiv_doi(self):
        sp = _make_screened(doi="10.1038/s41586-024-12345-6")
        resolver = FullTextResolver()
        text = await resolver._try_biorxiv(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    async def test_fetches_latest_version_from_api(self):
        """With API returning v2, should fetch the v2 PDF."""
        doi = "10.1101/2024.01.01.123456"
        sp = _make_screened(doi=doi)

        resolver = FullTextResolver()

        api_resp = _make_biorxiv_api_response(doi, [1, 2])
        pdf_resp = MagicMock()
        pdf_resp.status_code = 200
        pdf_resp.content = b"x" * 2000

        resolver._client = MagicMock()
        # First call: version API; second call: PDF download
        resolver._client.get = AsyncMock(side_effect=[api_resp, pdf_resp])

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_biorxiv(sp.paper)

        assert text == _SAMPLE_PDF_TEXT
        # The PDF URL should contain v2
        pdf_call = resolver._client.get.call_args_list[1]
        assert "v2" in str(pdf_call)
        assert "biorxiv.org" in str(pdf_call)

    @pytest.mark.asyncio
    async def test_fallback_when_api_fails(self):
        """When the version API fails (non-200), tries v1, v2, v3 sequentially."""
        doi = "10.1101/2024.01.01.123456"
        sp = _make_screened(doi=doi)

        resolver = FullTextResolver()

        # biorxiv API → fail; v1 → fail; v2 → success PDF
        api_fail = MagicMock()
        api_fail.status_code = 500
        api_fail.json.return_value = {}

        v1_fail = MagicMock()
        v1_fail.status_code = 404
        v1_fail.content = b""

        v2_success = MagicMock()
        v2_success.status_code = 200
        v2_success.content = b"x" * 2000

        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(side_effect=[api_fail, v1_fail, v2_success])

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_biorxiv(sp.paper)

        assert text == _SAMPLE_PDF_TEXT
        calls = resolver._client.get.call_args_list
        # First call is version API
        assert "api.biorxiv.org" in str(calls[0])
        # v1 tried and failed
        assert "v1" in str(calls[1])
        # v2 succeeded
        assert "v2" in str(calls[2])

    @pytest.mark.asyncio
    async def test_falls_back_to_medrxiv_when_biorxiv_all_fail(self):
        """After all biorxiv attempts fail, tries medrxiv."""
        doi = "10.1101/2024.01.01.123456"
        sp = _make_screened(doi=doi)

        resolver = FullTextResolver()

        # biorxiv API → empty collection → fallback v1/v2/v3 all fail →
        # medrxiv API → v1 success
        biorxiv_api_resp = MagicMock()
        biorxiv_api_resp.status_code = 200
        biorxiv_api_resp.json.return_value = {"collection": []}

        fail_resp = MagicMock()
        fail_resp.status_code = 404
        fail_resp.content = b""

        medrxiv_api_resp = MagicMock()
        medrxiv_api_resp.status_code = 200
        medrxiv_api_resp.json.return_value = {"collection": [{"doi": doi, "version": "1"}]}

        medrxiv_pdf = MagicMock()
        medrxiv_pdf.status_code = 200
        medrxiv_pdf.content = b"x" * 2000

        # Call order:
        # [0] biorxiv API (empty) → [1] biorxiv v1 fail → [2] v2 fail → [3] v3 fail →
        # [4] medrxiv API (v1) → [5] medrxiv v1 PDF success
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(
            side_effect=[
                biorxiv_api_resp,
                fail_resp,
                fail_resp,
                fail_resp,
                medrxiv_api_resp,
                medrxiv_pdf,
            ]
        )

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_biorxiv(sp.paper)

        assert text == _SAMPLE_PDF_TEXT
        pdf_call = resolver._client.get.call_args_list[5]
        assert "medrxiv.org" in str(pdf_call)


# ---------------------------------------------------------------------------
# Batch PMID -> PMCID conversion
# ---------------------------------------------------------------------------


class TestBatchPmidConversion:
    @pytest.mark.asyncio
    async def test_batch_conversion(self):
        sp1 = _make_screened(external_ids={"pmid": "111"})
        sp2 = _make_screened(external_ids={"pmid": "222", "pmcid": "PMC999"})
        sp3 = _make_screened(external_ids={})

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "records": [
                {"pmid": "111", "pmcid": "PMC555"},
            ]
        }

        resolver = FullTextResolver()
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        result = await resolver._batch_pmid_to_pmcid([sp1, sp2, sp3])

        # sp2's PMCID should be included directly (no API call needed)
        assert result["222"] == "PMC999"
        # sp1's PMID should be converted via API
        assert result["111"] == "PMC555"
        # sp3 has no PMID — not in result
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_skips_papers_with_full_text(self):
        sp = _make_screened(
            external_ids={"pmid": "111"},
            full_text="Already have full text for this paper",
        )

        resolver = FullTextResolver()
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock()

        result = await resolver._batch_pmid_to_pmcid([sp])

        # Should not have called the API
        resolver._client.get.assert_not_called()
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Full resolve() integration
# ---------------------------------------------------------------------------


class TestResolveIntegration:
    @pytest.mark.asyncio
    async def test_skips_already_enriched(self):
        sp = _make_screened(
            doi="10.1234/test",
            external_ids={"s2_pdf_url": "https://example.com/paper.pdf"},
            full_text="Already present",
        )

        resolver = FullTextResolver()
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock()
        resolver._client.aclose = AsyncMock()

        source_counts = await resolver.resolve([sp])
        assert sum(source_counts.values()) == 0
        assert sp.paper.full_text == "Already present"

    @pytest.mark.asyncio
    async def test_tries_strategies_in_order(self):
        sp = _make_screened(
            doi="10.1234/test",
            external_ids={"s2_pdf_url": "https://example.com/paper.pdf"},
        )

        resolver = FullTextResolver()

        # Mock batch PMID conversion to return empty
        resolver._batch_pmid_to_pmcid = AsyncMock(return_value={})

        # Elsevier skipped (no matching DOI), S2 PDF succeeds
        resolver._try_elsevier_api = AsyncMock(return_value=None)
        resolver._try_s2_pdf = AsyncMock(return_value=_SAMPLE_PDF_TEXT)
        resolver._try_core = AsyncMock(return_value=None)
        resolver._try_crossref = AsyncMock(return_value=None)
        resolver._try_pmc = AsyncMock(return_value=None)
        resolver._try_europe_pmc = AsyncMock(return_value=None)
        resolver._try_arxiv = AsyncMock(return_value=None)
        resolver._try_biorxiv = AsyncMock(return_value=None)
        resolver._try_plos = AsyncMock(return_value=None)
        resolver._try_mdpi = AsyncMock(return_value=None)
        resolver._try_frontiers = AsyncMock(return_value=None)
        resolver._try_unpaywall = AsyncMock(return_value=None)
        resolver._try_springer_oa = AsyncMock(return_value=None)

        source_counts = await resolver.resolve([sp])

        assert source_counts.get("s2_pdf") == 1
        assert sp.paper.full_text == _SAMPLE_PDF_TEXT
        assert sp.paper.full_text_source == "s2_pdf"

        # Later strategies should NOT have been called since S2 succeeded
        resolver._try_core.assert_not_called()
        resolver._try_pmc.assert_not_called()
        resolver._try_arxiv.assert_not_called()
        resolver._try_plos.assert_not_called()
        resolver._try_mdpi.assert_not_called()
        resolver._try_frontiers.assert_not_called()
        resolver._try_springer_oa.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_through_to_next_strategy(self):
        sp = _make_screened(
            doi="10.1234/test",
            external_ids={"pmid": "12345", "pmcid": "PMC6789"},
        )

        resolver = FullTextResolver()
        resolver._batch_pmid_to_pmcid = AsyncMock(return_value={"12345": "PMC6789"})

        # Elsevier, S2, CORE, CrossRef fail — PMC succeeds
        resolver._try_elsevier_api = AsyncMock(return_value=None)
        resolver._try_s2_pdf = AsyncMock(return_value=None)
        resolver._try_core = AsyncMock(return_value=None)
        resolver._try_crossref = AsyncMock(return_value=None)
        resolver._try_pmc = AsyncMock(return_value="PMC full text content here " * 10)
        resolver._try_europe_pmc = AsyncMock(return_value=None)
        resolver._try_arxiv = AsyncMock(return_value=None)
        resolver._try_biorxiv = AsyncMock(return_value=None)
        resolver._try_plos = AsyncMock(return_value=None)
        resolver._try_mdpi = AsyncMock(return_value=None)
        resolver._try_frontiers = AsyncMock(return_value=None)
        resolver._try_unpaywall = AsyncMock(return_value=None)
        resolver._try_springer_oa = AsyncMock(return_value=None)

        source_counts = await resolver.resolve([sp])

        assert source_counts.get("pmc") == 1
        assert sp.paper.full_text_source == "pmc"


# ---------------------------------------------------------------------------
# Pipeline node integration
# ---------------------------------------------------------------------------


class TestPipelineNodeIntegration:
    @pytest.mark.asyncio
    async def test_node_uses_full_text_resolver(self):
        """The pipeline node should use FullTextResolver instead of UnpaywallClient."""
        from autoreview.config import load_config
        from autoreview.models.knowledge_base import KnowledgeBase, PipelinePhase
        from autoreview.pipeline.nodes import PipelineNodes

        config = load_config(domain="biomedical")
        nodes = PipelineNodes(llm=None, config=config)
        kb = KnowledgeBase(topic="test")

        sp = _make_screened(doi="10.1234/test")
        kb.screened_papers = [sp]

        with (
            patch("autoreview.search.full_text.FullTextResolver") as mock_resolver_cls,
            patch("autoreview.search.full_text_cache.CachedFullTextResolver") as mock_cached_cls,
        ):
            mock_instance = AsyncMock()
            mock_instance.resolve = AsyncMock(return_value={"s2_pdf": 1})
            mock_instance.close = AsyncMock()
            mock_resolver_cls.return_value = AsyncMock()
            mock_cached_cls.return_value = mock_instance

            await nodes.full_text_retrieval(kb)

        assert kb.current_phase == PipelinePhase.FULL_TEXT_RETRIEVAL
        assert any("complete" in e.action for e in kb.audit_log)
        mock_instance.resolve.assert_called_once()
        mock_instance.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_node_works_without_unpaywall_email(self):
        """Node should still work for S2/PMC/arXiv even without UNPAYWALL_EMAIL."""
        from autoreview.config import load_config
        from autoreview.models.knowledge_base import KnowledgeBase, PipelinePhase
        from autoreview.pipeline.nodes import PipelineNodes

        config = load_config(domain="biomedical")
        nodes = PipelineNodes(llm=None, config=config)
        kb = KnowledgeBase(topic="test")
        kb.screened_papers = []

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("UNPAYWALL_EMAIL", None)

            with patch("autoreview.search.full_text.FullTextResolver") as mock_resolver:
                mock_instance = AsyncMock()
                mock_instance.resolve = AsyncMock(return_value={})
                mock_instance.close = AsyncMock()
                mock_resolver.return_value = mock_instance

                await nodes.full_text_retrieval(kb)

        assert kb.current_phase == PipelinePhase.FULL_TEXT_RETRIEVAL
        # The resolver should still be created (it handles None email gracefully)
        mock_resolver.assert_called_once()


# ---------------------------------------------------------------------------
# Search source changes: S2 openAccessPdf and PubMed PMCID
# ---------------------------------------------------------------------------


class TestSemanticScholarOpenAccessPdf:
    def test_parse_paper_with_open_access_pdf(self):
        from autoreview.search.semantic_scholar import SemanticScholarSearch

        s2 = SemanticScholarSearch()
        data = {
            "paperId": "abc123",
            "title": "Test Paper",
            "abstract": "An abstract",
            "year": 2024,
            "authors": [{"name": "Author A"}],
            "journal": {"name": "Test Journal"},
            "externalIds": {"DOI": "10.1234/test", "ArXiv": "2301.12345"},
            "citationCount": 10,
            "openAccessPdf": {"url": "https://example.com/paper.pdf", "status": "GREEN"},
        }

        paper = s2._parse_paper(data)
        assert paper is not None
        assert paper.external_ids.get("s2_pdf_url") == "https://example.com/paper.pdf"
        assert paper.external_ids.get("arxiv") == "2301.12345"

    def test_parse_paper_without_open_access_pdf(self):
        from autoreview.search.semantic_scholar import SemanticScholarSearch

        s2 = SemanticScholarSearch()
        data = {
            "paperId": "abc123",
            "title": "Paywalled Paper",
            "abstract": "Abstract",
            "year": 2024,
            "authors": [{"name": "Author B"}],
            "journal": {"name": "Expensive Journal"},
            "externalIds": {"DOI": "10.1234/paywalled"},
            "citationCount": 5,
            "openAccessPdf": None,
        }

        paper = s2._parse_paper(data)
        assert paper is not None
        assert "s2_pdf_url" not in paper.external_ids


class TestPubMedPmcid:
    def test_parse_article_with_pmcid(self):
        from autoreview.search.pubmed import PubMedSearch

        pm = PubMedSearch()

        # Simulate a PubMed XML article dict with PMCID
        class MockId:
            def __init__(self, value: str, id_type: str):
                self._value = value
                self.attributes = {"IdType": id_type}

            def __str__(self):
                return self._value

        article = {
            "MedlineCitation": {
                "PMID": "12345678",
                "Article": {
                    "ArticleTitle": "Test Article with PMC",
                    "AuthorList": [{"LastName": "Smith", "ForeName": "John"}],
                    "Journal": {
                        "Title": "Test Journal",
                        "JournalIssue": {"PubDate": {"Year": "2024"}},
                    },
                    "Abstract": {"AbstractText": ["An abstract."]},
                    "ELocationID": [],
                },
            },
            "PubmedData": {
                "ArticleIdList": [
                    MockId("10.1234/test", "doi"),
                    MockId("PMC9876543", "pmc"),
                ],
            },
        }

        paper = pm._parse_article(article)
        assert paper is not None
        assert paper.external_ids.get("pmid") == "12345678"
        assert paper.external_ids.get("pmcid") == "PMC9876543"
        assert paper.doi == "10.1234/test"


# ---------------------------------------------------------------------------
# Strategy: PLOS direct PDF
# ---------------------------------------------------------------------------


class TestPlosStrategy:
    @pytest.mark.asyncio
    async def test_fetches_plos_pdf(self):
        doi = "10.1371/journal.pbio.3001234"
        sp = _make_screened(doi=doi)

        resolver = FullTextResolver()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"x" * 2000
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_plos(sp.paper)

        assert text == _SAMPLE_PDF_TEXT
        call_args = resolver._client.get.call_args
        assert "journals.plos.org" in str(call_args)
        assert doi in str(call_args)
        assert "printable" in str(call_args)

    @pytest.mark.asyncio
    async def test_skips_non_plos_doi(self):
        sp = _make_screened(doi="10.1016/j.cell.2020.06.043")
        resolver = FullTextResolver()
        text = await resolver._try_plos(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    async def test_returns_none_on_http_error(self):
        sp = _make_screened(doi="10.1371/journal.pone.0123456")

        resolver = FullTextResolver()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.content = b""
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        text = await resolver._try_plos(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    async def test_returns_none_without_doi(self):
        sp = _make_screened(doi=None)
        resolver = FullTextResolver()
        text = await resolver._try_plos(sp.paper)
        assert text is None


# ---------------------------------------------------------------------------
# Strategy: MDPI direct PDF
# ---------------------------------------------------------------------------


class TestMdpiStrategy:
    @pytest.mark.asyncio
    async def test_fetches_mdpi_pdf_via_doi_resolve(self):
        doi = "10.3390/ijms24010123"
        sp = _make_screened(doi=doi)

        resolver = FullTextResolver()
        # Mock HEAD response (DOI resolution) with a redirect URL
        mock_head_resp = MagicMock()
        mock_head_resp.status_code = 200
        mock_head_resp.url = "https://www.mdpi.com/1422-0067/24/1/123"
        # Mock GET response (PDF download)
        mock_get_resp = MagicMock()
        mock_get_resp.status_code = 200
        mock_get_resp.content = b"x" * 2000
        resolver._client = MagicMock()
        resolver._client.head = AsyncMock(return_value=mock_head_resp)
        resolver._client.get = AsyncMock(return_value=mock_get_resp)

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_mdpi(sp.paper)

        assert text == _SAMPLE_PDF_TEXT
        # PDF URL should be the resolved article URL + /pdf
        get_call_args = resolver._client.get.call_args
        assert "www.mdpi.com" in str(get_call_args)
        assert "/pdf" in str(get_call_args)

    @pytest.mark.asyncio
    async def test_returns_none_when_doi_resolution_fails(self):
        doi = "10.3390/ijms24010123"
        sp = _make_screened(doi=doi)

        resolver = FullTextResolver()
        # HEAD fails (DOI resolution error) — should return None gracefully
        resolver._client = MagicMock()
        resolver._client.head = AsyncMock(side_effect=Exception("timeout"))

        text = await resolver._try_mdpi(sp.paper)

        assert text is None

    @pytest.mark.asyncio
    async def test_skips_non_mdpi_doi(self):
        sp = _make_screened(doi="10.1038/s41586-020-2649-2")
        resolver = FullTextResolver()
        text = await resolver._try_mdpi(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    async def test_returns_none_on_http_error(self):
        sp = _make_screened(doi="10.3390/cells11010001")

        resolver = FullTextResolver()
        # HEAD resolves DOI
        mock_head_resp = MagicMock()
        mock_head_resp.status_code = 200
        mock_head_resp.url = "https://www.mdpi.com/2073-4409/11/1/1"
        # GET returns 403 (Akamai block) for both PDF and XML
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.content = b""
        resolver._client = MagicMock()
        resolver._client.head = AsyncMock(return_value=mock_head_resp)
        resolver._client.get = AsyncMock(return_value=mock_resp)

        text = await resolver._try_mdpi(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    async def test_returns_none_without_doi(self):
        sp = _make_screened(doi=None)
        resolver = FullTextResolver()
        text = await resolver._try_mdpi(sp.paper)
        assert text is None


# ---------------------------------------------------------------------------
# Strategy: Frontiers direct PDF
# ---------------------------------------------------------------------------


class TestFrontiersStrategy:
    @pytest.mark.asyncio
    async def test_fetches_frontiers_pdf(self):
        doi = "10.3389/fnins.2023.1234567"
        sp = _make_screened(doi=doi)

        resolver = FullTextResolver()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"x" * 2000
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_frontiers(sp.paper)

        assert text == _SAMPLE_PDF_TEXT
        call_args = resolver._client.get.call_args
        assert "www.frontiersin.org" in str(call_args)
        assert doi in str(call_args)
        assert "/pdf" in str(call_args)

    @pytest.mark.asyncio
    async def test_skips_non_frontiers_doi(self):
        sp = _make_screened(doi="10.1371/journal.pone.0123456")
        resolver = FullTextResolver()
        text = await resolver._try_frontiers(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    async def test_returns_none_on_http_error(self):
        sp = _make_screened(doi="10.3389/fimmu.2023.1100000")

        resolver = FullTextResolver()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.content = b""
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        text = await resolver._try_frontiers(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    async def test_returns_none_without_doi(self):
        sp = _make_screened(doi=None)
        resolver = FullTextResolver()
        text = await resolver._try_frontiers(sp.paper)
        assert text is None


# ---------------------------------------------------------------------------
# Structured failure logging in _try_all
# ---------------------------------------------------------------------------


class TestFailureLogging:
    @pytest.mark.asyncio
    async def test_logs_warning_when_all_strategies_fail(self):
        """When all strategies return None, a WARNING event is emitted."""
        sp = _make_screened(doi="10.9999/unknown")

        resolver = FullTextResolver()
        resolver._batch_pmid_to_pmcid = AsyncMock(return_value={})
        # All strategies return None (skipped/failed)
        for attr in (
            "_try_elsevier_api",
            "_try_s2_pdf",
            "_try_pmc",
            "_try_arxiv",
            "_try_biorxiv",
            "_try_plos",
            "_try_mdpi",
            "_try_frontiers",
            "_try_unpaywall",
            "_try_springer_oa",
        ):
            setattr(resolver, attr, AsyncMock(return_value=None))

        import structlog.testing

        with structlog.testing.capture_logs() as cap_logs:
            source, text = await resolver._try_all(sp.paper, {})

        assert text is None
        assert source == ""
        warning_events = [e for e in cap_logs if e.get("log_level") == "warning"]
        assert any("all_strategies_failed" in e.get("event", "") for e in warning_events)

    @pytest.mark.asyncio
    async def test_logs_info_on_success(self):
        """A successful strategy emits an INFO event."""
        sp = _make_screened(doi="10.1371/journal.pbio.3001234")

        resolver = FullTextResolver()
        for attr in (
            "_try_elsevier_api",
            "_try_wiley_tdm",
            "_try_s2_pdf",
            "_try_s2_api_lookup",
            "_try_core",
            "_try_crossref",
            "_try_pmc",
            "_try_europe_pmc",
            "_try_arxiv",
            "_try_biorxiv",
        ):
            setattr(resolver, attr, AsyncMock(return_value=None))
        resolver._try_plos = AsyncMock(return_value=_SAMPLE_PDF_TEXT)

        import structlog.testing

        with structlog.testing.capture_logs() as cap_logs:
            source, text = await resolver._try_all(sp.paper, {})

        assert source == "plos"
        assert text == _SAMPLE_PDF_TEXT
        info_events = [e for e in cap_logs if e.get("log_level") == "info"]
        assert any(
            "strategy_succeeded" in e.get("event", "") and e.get("strategy") == "plos"
            for e in info_events
        )

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=False)
    async def test_logs_debug_skips_for_inapplicable_strategies(self):
        """Strategies that return None are logged at DEBUG as skipped.

        Verifies via return value and mock call counts rather than structlog
        capture, which is fragile across the full test suite.
        """
        os.environ.pop("ELSEVIER_API_KEY", None)
        os.environ.pop("SPRINGER_API_KEY", None)
        sp = _make_screened(doi="10.1234/generic")

        resolver = FullTextResolver()
        resolver._try_elsevier_api = AsyncMock(return_value=None)  # wrong prefix
        resolver._try_s2_pdf = AsyncMock(return_value=_SAMPLE_PDF_TEXT)
        for attr in (
            "_try_pmc",
            "_try_arxiv",
            "_try_biorxiv",
            "_try_plos",
            "_try_mdpi",
            "_try_frontiers",
            "_try_unpaywall",
            "_try_springer_oa",
        ):
            setattr(resolver, attr, AsyncMock(return_value=None))

        source, text = await resolver._try_all(sp.paper, {})

        # s2_pdf is the second strategy and returns text, so it should win
        assert source == "s2_pdf"
        assert text == _SAMPLE_PDF_TEXT
        # elsevier_api was called (returned None = skipped)
        resolver._try_elsevier_api.assert_called_once()
        # strategies after s2_pdf should NOT be called (early return on success)
        cast(AsyncMock, resolver._try_pmc).assert_not_called()
        cast(AsyncMock, resolver._try_arxiv).assert_not_called()


# ---------------------------------------------------------------------------
# Exception logging from asyncio.gather in resolve()
# ---------------------------------------------------------------------------


class TestGatherExceptionLogging:
    @pytest.mark.asyncio
    async def test_logs_error_for_exception_in_gather(self):
        """Exceptions raised during paper processing are logged at ERROR level."""
        sp = _make_screened(doi="10.1234/test")

        resolver = FullTextResolver()

        # Make _batch_pmid_to_pmcid succeed
        resolver._batch_pmid_to_pmcid = AsyncMock(return_value={})

        # Make _try_all raise an unexpected exception
        async def _raise(*args, **kwargs):
            raise RuntimeError("unexpected failure")

        resolver._try_all = _raise

        import structlog.testing

        with structlog.testing.capture_logs() as cap_logs:
            await resolver.resolve([sp])

        error_events = [e for e in cap_logs if e.get("log_level") == "error"]
        assert any("resolve_paper_exception" in e.get("event", "") for e in error_events), (
            f"No error event found; got: {cap_logs}"
        )

    @pytest.mark.asyncio
    async def test_doi_included_in_exception_log(self):
        """The paper DOI is included in the exception log event."""
        doi = "10.1234/exception-test"
        sp = _make_screened(doi=doi)

        resolver = FullTextResolver()
        resolver._batch_pmid_to_pmcid = AsyncMock(return_value={})

        async def _raise(*args, **kwargs):
            raise ValueError("boom")

        resolver._try_all = _raise

        import structlog.testing

        with structlog.testing.capture_logs() as cap_logs:
            await resolver.resolve([sp])

        error_events = [e for e in cap_logs if "resolve_paper_exception" in e.get("event", "")]
        assert len(error_events) >= 1
        assert error_events[0].get("doi") == doi


# ---------------------------------------------------------------------------
# Strategy: arXiv DOI prefix extraction
# ---------------------------------------------------------------------------


class TestArxivDoiPrefix:
    @pytest.mark.asyncio
    async def test_arxiv_from_doi_prefix(self):
        """DOI 10.48550/arXiv.XXXX with no 'arxiv' in external_ids → uses DOI."""
        sp = _make_screened(
            doi="10.48550/arXiv.2312.10997",
            external_ids={},
        )

        resolver = FullTextResolver()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"x" * 2000
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_arxiv(sp.paper)

        assert text == _SAMPLE_PDF_TEXT
        call_args = resolver._client.get.call_args
        assert "arxiv.org/pdf/2312.10997" in str(call_args)

    @pytest.mark.asyncio
    async def test_arxiv_from_doi_prefix_lowercase(self):
        """Lowercase DOI prefix 10.48550/arxiv.XXXX is handled case-insensitively."""
        sp = _make_screened(
            doi="10.48550/arxiv.2312.10997",
            external_ids={},
        )

        resolver = FullTextResolver()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"x" * 2000
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_arxiv(sp.paper)

        assert text == _SAMPLE_PDF_TEXT
        call_args = resolver._client.get.call_args
        assert "arxiv.org/pdf/2312.10997" in str(call_args)

    @pytest.mark.asyncio
    async def test_arxiv_prefers_external_id(self):
        """When external_ids has 'arxiv', it takes priority over DOI extraction."""
        sp = _make_screened(
            doi="10.48550/arXiv.9999.00000",
            external_ids={"arxiv": "2312.10997"},
        )

        resolver = FullTextResolver()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"x" * 2000
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_arxiv(sp.paper)

        assert text == _SAMPLE_PDF_TEXT
        call_args = resolver._client.get.call_args
        # Should use the external_ids value, not the one embedded in the DOI
        assert "arxiv.org/pdf/2312.10997" in str(call_args)
        assert "9999.00000" not in str(call_args)


# ---------------------------------------------------------------------------
# Strategy: ACL Anthology
# ---------------------------------------------------------------------------


class TestAclAnthologyStrategy:
    @pytest.mark.asyncio
    async def test_fetches_acl_anthology_pdf(self):
        """DOI 10.18653/v1/XXXX → fetches https://aclanthology.org/XXXX.pdf"""
        doi = "10.18653/v1/2024.emnlp-main.519"
        sp = _make_screened(doi=doi)

        resolver = FullTextResolver()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"x" * 2000
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_acl_anthology(sp.paper)

        assert text == _SAMPLE_PDF_TEXT
        call_args = resolver._client.get.call_args
        assert "aclanthology.org/2024.emnlp-main.519.pdf" in str(call_args)

    @pytest.mark.asyncio
    async def test_skips_non_acl_doi(self):
        """Non-ACL DOI → returns None without making any HTTP call."""
        sp = _make_screened(doi="10.1038/s41586-020-2649-2")
        resolver = FullTextResolver()
        text = await resolver._try_acl_anthology(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    async def test_skips_acl_without_v1_prefix(self):
        """ACL DOI with prefix other than 10.18653/v1/ → returns None."""
        sp = _make_screened(doi="10.18653/other/xxx")
        resolver = FullTextResolver()
        text = await resolver._try_acl_anthology(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    async def test_returns_none_without_doi(self):
        sp = _make_screened(doi=None)
        resolver = FullTextResolver()
        text = await resolver._try_acl_anthology(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    async def test_returns_none_on_http_error(self):
        doi = "10.18653/v1/2023.acl-long.100"
        sp = _make_screened(doi=doi)

        resolver = FullTextResolver()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.content = b""
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        text = await resolver._try_acl_anthology(sp.paper)
        assert text is None


# ---------------------------------------------------------------------------
# Strategy: JMIR
# ---------------------------------------------------------------------------


class TestJmirStrategy:
    @pytest.mark.asyncio
    async def test_fetches_jmir_xml(self):
        """DOI 10.2196/38052 → fetches XML from https://www.jmir.org/api/download-xml/38052"""
        doi = "10.2196/38052"
        sp = _make_screened(doi=doi)

        resolver = FullTextResolver()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = _SAMPLE_JATS  # len > 500
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        text = await resolver._try_jmir(sp.paper)

        assert text is not None
        call_args = resolver._client.get.call_args
        assert "www.jmir.org/api/download-xml/38052" in str(call_args)

    @pytest.mark.asyncio
    async def test_jmir_pdf_fallback(self):
        """When XML fetch fails, falls back to DOI resolution + PDF."""
        doi = "10.2196/38052"
        sp = _make_screened(doi=doi)

        resolver = FullTextResolver()

        # XML fetch fails (short/empty response)
        xml_fail_resp = MagicMock()
        xml_fail_resp.status_code = 200
        xml_fail_resp.content = b"<err/>"  # len < 500, skipped

        # HEAD (DOI resolution) redirects to a jmir.org URL
        head_resp = MagicMock()
        head_resp.status_code = 200
        head_resp.url = "https://www.jmir.org/2022/1/e38052"

        # GET (PDF download) succeeds
        pdf_resp = MagicMock()
        pdf_resp.status_code = 200
        pdf_resp.content = b"x" * 2000

        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(side_effect=[xml_fail_resp, pdf_resp])
        resolver._client.head = AsyncMock(return_value=head_resp)

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_jmir(sp.paper)

        assert text == _SAMPLE_PDF_TEXT
        # HEAD should have been called for DOI resolution
        resolver._client.head.assert_called_once()
        head_call = resolver._client.head.call_args
        assert f"doi.org/{doi}" in str(head_call)

    @pytest.mark.asyncio
    async def test_skips_non_jmir_doi(self):
        """Non-JMIR DOI → returns None without any HTTP call."""
        sp = _make_screened(doi="10.1038/s41586-020-2649-2")
        resolver = FullTextResolver()
        text = await resolver._try_jmir(sp.paper)
        assert text is None

    @pytest.mark.asyncio
    async def test_returns_none_without_doi(self):
        sp = _make_screened(doi=None)
        resolver = FullTextResolver()
        text = await resolver._try_jmir(sp.paper)
        assert text is None


# ---------------------------------------------------------------------------
# Strategy: CORE API full-text
# ---------------------------------------------------------------------------


class TestCoreStrategy:
    @pytest.mark.asyncio
    async def test_fetches_pdf_when_core_pdf_url_exists(self):
        pdf_url = "https://core.ac.uk/download/pdf/12345678.pdf"
        sp = _make_screened(external_ids={"core": "12345678", "core_pdf_url": pdf_url})

        resolver = FullTextResolver(core_api_key="test-core-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"x" * 2000
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_core(sp.paper)

        assert text == _SAMPLE_PDF_TEXT
        call_args = resolver._client.get.call_args_list[0]
        assert pdf_url in str(call_args)

    @pytest.mark.asyncio
    async def test_falls_through_to_download_endpoint_when_pdf_url_missing(self):
        sp = _make_screened(external_ids={"core": "12345678"})

        resolver = FullTextResolver(core_api_key="test-core-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"x" * 2000
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_core(sp.paper)

        assert text == _SAMPLE_PDF_TEXT
        call_args = resolver._client.get.call_args_list[0]
        assert "outputs/12345678/download" in str(call_args)

    @pytest.mark.asyncio
    async def test_skips_when_no_core_id_and_no_pdf_url(self):
        sp = _make_screened(external_ids={})

        resolver = FullTextResolver(core_api_key="test-core-key")
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock()

        text = await resolver._try_core(sp.paper)

        assert text is None
        resolver._client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_download_endpoint_without_api_key(self):
        """Without a CORE API key, the download endpoint strategy is skipped."""
        sp = _make_screened(external_ids={"core": "12345678"})

        resolver = FullTextResolver(core_api_key=None)
        import os

        env_backup = os.environ.pop("CORE_API_KEY", None)
        try:
            resolver._core_api_key = None
            resolver._client = MagicMock()
            resolver._client.get = AsyncMock()

            text = await resolver._try_core(sp.paper)
        finally:
            if env_backup is not None:
                os.environ["CORE_API_KEY"] = env_backup

        assert text is None
        resolver._client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_http_error_on_pdf_url_gracefully(self):
        pdf_url = "https://core.ac.uk/download/pdf/12345678.pdf"
        sp = _make_screened(external_ids={"core": "12345678", "core_pdf_url": pdf_url})

        resolver = FullTextResolver(core_api_key="test-core-key")
        # First call (pdf_url) fails with 404, second call (download endpoint) also fails
        fail_resp = MagicMock()
        fail_resp.status_code = 404
        fail_resp.content = b""
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=fail_resp)

        text = await resolver._try_core(sp.paper)

        assert text is None

    @pytest.mark.asyncio
    async def test_handles_request_exception_gracefully(self):
        pdf_url = "https://core.ac.uk/download/pdf/12345678.pdf"
        sp = _make_screened(external_ids={"core": "12345678", "core_pdf_url": pdf_url})

        resolver = FullTextResolver(core_api_key="test-core-key")
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(side_effect=Exception("Connection timed out"))

        text = await resolver._try_core(sp.paper)

        assert text is None

    @pytest.mark.asyncio
    async def test_core_is_in_strategy_chain(self):
        """CORE strategy appears after s2_pdf but before pmc in _try_all."""
        sp = _make_screened(doi="10.1234/test")

        resolver = FullTextResolver(core_api_key="test-core-key")
        resolver._batch_pmid_to_pmcid = AsyncMock(return_value={})

        call_order: list[str] = []

        async def make_strategy(name: str, return_val):
            async def strategy(*args, **kwargs):
                call_order.append(name)
                return return_val

            return strategy

        resolver._try_elsevier_api = await make_strategy("elsevier_api", None)
        resolver._try_s2_pdf = await make_strategy("s2_pdf", None)
        resolver._try_core = await make_strategy("core", _SAMPLE_PDF_TEXT)
        resolver._try_crossref = await make_strategy("crossref", None)
        resolver._try_pmc = await make_strategy("pmc", None)
        resolver._try_europe_pmc = await make_strategy("europe_pmc", None)
        resolver._try_arxiv = await make_strategy("arxiv", None)
        resolver._try_biorxiv = await make_strategy("biorxiv", None)
        resolver._try_plos = await make_strategy("plos", None)
        resolver._try_mdpi = await make_strategy("mdpi", None)
        resolver._try_frontiers = await make_strategy("frontiers", None)
        resolver._try_unpaywall = await make_strategy("unpaywall", None)
        resolver._try_springer_oa = await make_strategy("springer_oa", None)

        source, text = await resolver._try_all(sp.paper, {})

        assert source == "core"
        assert text == _SAMPLE_PDF_TEXT
        # core should come after s2_pdf
        assert call_order.index("core") > call_order.index("s2_pdf")
        # pmc should NOT have been called (core succeeded first)
        assert "pmc" not in call_order


# ---------------------------------------------------------------------------
# Strategy: CrossRef links full-text
# ---------------------------------------------------------------------------


class TestCrossRefStrategy:
    @pytest.mark.asyncio
    async def test_fetches_pdf_from_crossref_link(self):
        """Fetches PDF when a crossref_links entry with content-type application/pdf is present."""
        import json

        links = [{"content-type": "application/pdf", "URL": "https://example.com/paper.pdf"}]
        sp = _make_screened(external_ids={"crossref_links": json.dumps(links)})

        resolver = FullTextResolver()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"x" * 2000
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_crossref(sp.paper)

        assert text == _SAMPLE_PDF_TEXT
        call_args = resolver._client.get.call_args
        assert "example.com/paper.pdf" in str(call_args)

    @pytest.mark.asyncio
    async def test_tries_xml_when_no_pdf_link(self):
        """Falls back to JATS XML when no PDF link is available."""
        import json

        jats_body = b"<article><body><p>" + b"A" * 200 + b"</p></body></article>"

        links = [{"content-type": "text/xml", "URL": "https://example.com/paper.xml"}]
        sp = _make_screened(external_ids={"crossref_links": json.dumps(links)})

        resolver = FullTextResolver()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = jats_body
        mock_resp.text = jats_body.decode()
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "autoreview.search.full_text._extract_text_from_jats_xml",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_crossref(sp.paper)

        assert text == _SAMPLE_PDF_TEXT

    @pytest.mark.asyncio
    async def test_skips_when_no_crossref_links(self):
        """Returns None immediately when crossref_links is not in external_ids."""
        sp = _make_screened(external_ids={})

        resolver = FullTextResolver()
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock()

        text = await resolver._try_crossref(sp.paper)

        assert text is None
        resolver._client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_on_http_error(self):
        """Returns None when the HTTP request fails."""
        import json

        links = [{"content-type": "application/pdf", "URL": "https://example.com/paper.pdf"}]
        sp = _make_screened(external_ids={"crossref_links": json.dumps(links)})

        resolver = FullTextResolver()
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(side_effect=Exception("Connection error"))

        text = await resolver._try_crossref(sp.paper)

        assert text is None

    @pytest.mark.asyncio
    async def test_prefers_pdf_over_xml(self):
        """PDF link is tried before XML link (sorted by priority)."""
        import json

        pdf_url = "https://example.com/paper.pdf"
        xml_url = "https://example.com/paper.xml"
        # Put XML first in list — PDF should still be tried first
        links = [
            {"content-type": "text/xml", "URL": xml_url},
            {"content-type": "application/pdf", "URL": pdf_url},
        ]
        sp = _make_screened(external_ids={"crossref_links": json.dumps(links)})

        resolver = FullTextResolver()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"x" * 2000
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_crossref(sp.paper)

        assert text == _SAMPLE_PDF_TEXT
        # PDF URL should have been the first (and only) call
        first_call_url = str(resolver._client.get.call_args_list[0])
        assert pdf_url in first_call_url

    @pytest.mark.asyncio
    async def test_crossref_is_in_strategy_chain_after_core(self):
        """CrossRef strategy appears after CORE but before PMC in _try_all."""
        sp = _make_screened(doi="10.1234/test")

        resolver = FullTextResolver()
        resolver._batch_pmid_to_pmcid = AsyncMock(return_value={})

        call_order: list[str] = []

        async def make_strategy(name: str, return_val):
            async def strategy(*args, **kwargs):
                call_order.append(name)
                return return_val

            return strategy

        resolver._try_elsevier_api = await make_strategy("elsevier_api", None)
        resolver._try_s2_pdf = await make_strategy("s2_pdf", None)
        resolver._try_core = await make_strategy("core", None)
        resolver._try_crossref = await make_strategy("crossref", _SAMPLE_PDF_TEXT)
        resolver._try_pmc = await make_strategy("pmc", None)
        resolver._try_europe_pmc = await make_strategy("europe_pmc", None)
        resolver._try_arxiv = await make_strategy("arxiv", None)
        resolver._try_biorxiv = await make_strategy("biorxiv", None)
        resolver._try_plos = await make_strategy("plos", None)
        resolver._try_mdpi = await make_strategy("mdpi", None)
        resolver._try_frontiers = await make_strategy("frontiers", None)
        resolver._try_unpaywall = await make_strategy("unpaywall", None)
        resolver._try_springer_oa = await make_strategy("springer_oa", None)

        source, text = await resolver._try_all(sp.paper, {})

        assert source == "crossref"
        assert text == _SAMPLE_PDF_TEXT
        # crossref must come after core
        assert call_order.index("crossref") > call_order.index("core")
        # pmc should NOT have been called (crossref succeeded first)
        assert "pmc" not in call_order


# ---------------------------------------------------------------------------
# Strategy: Europe PMC full-text XML
# ---------------------------------------------------------------------------


class TestEuropePmcStrategy:
    @pytest.mark.asyncio
    async def test_fetches_jats_xml_when_pmcid_present(self):
        sp = _make_screened(external_ids={"pmcid": "PMC8765432"})
        resolver = FullTextResolver()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = _SAMPLE_JATS
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        text = await resolver._try_europe_pmc(sp.paper)

        assert text is not None
        assert "introduction paragraph" in text.lower()
        resolver._client.get.assert_called_once()
        call_args = resolver._client.get.call_args
        assert "PMC8765432" in call_args[0][0]
        assert "ebi.ac.uk" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_normalises_pmcid_without_prefix(self):
        """PMCIDs stored without 'PMC' prefix should be normalised before use."""
        sp = _make_screened(external_ids={"pmcid": "8765432"})
        resolver = FullTextResolver()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = _SAMPLE_JATS
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        text = await resolver._try_europe_pmc(sp.paper)

        assert text is not None
        call_args = resolver._client.get.call_args
        assert "PMC8765432" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_skips_when_no_pmcid(self):
        sp = _make_screened(external_ids={"pmid": "33456789"})
        resolver = FullTextResolver()
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock()

        text = await resolver._try_europe_pmc(sp.paper)

        assert text is None
        resolver._client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_none_on_http_error(self):
        sp = _make_screened(external_ids={"pmcid": "PMC8765432"})
        resolver = FullTextResolver()

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        text = await resolver._try_europe_pmc(sp.paper)

        assert text is None

    @pytest.mark.asyncio
    async def test_returns_none_on_request_exception(self):
        import httpx as _httpx

        sp = _make_screened(external_ids={"pmcid": "PMC8765432"})
        resolver = FullTextResolver()
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(side_effect=_httpx.ConnectError("Connection refused"))

        text = await resolver._try_europe_pmc(sp.paper)

        assert text is None

    @pytest.mark.asyncio
    async def test_strategy_position_after_pmc_before_arxiv(self):
        """europe_pmc must appear after pmc and before arxiv in the strategy chain."""
        sp = _make_screened(
            doi="10.1101/12345",
            external_ids={"pmcid": "PMC8765432"},
        )
        resolver = FullTextResolver()
        resolver._batch_pmid_to_pmcid = AsyncMock(return_value={})

        call_order: list[str] = []

        async def make_strategy(name: str, return_val):
            async def strategy(*args, **kwargs):
                call_order.append(name)
                return return_val

            return strategy

        resolver._try_elsevier_api = await make_strategy("elsevier_api", None)
        resolver._try_s2_pdf = await make_strategy("s2_pdf", None)
        resolver._try_core = await make_strategy("core", None)
        resolver._try_crossref = await make_strategy("crossref", None)
        resolver._try_pmc = await make_strategy("pmc", None)
        resolver._try_europe_pmc = await make_strategy("europe_pmc", _SAMPLE_PDF_TEXT)
        resolver._try_arxiv = await make_strategy("arxiv", None)
        resolver._try_biorxiv = await make_strategy("biorxiv", None)
        resolver._try_plos = await make_strategy("plos", None)
        resolver._try_mdpi = await make_strategy("mdpi", None)
        resolver._try_frontiers = await make_strategy("frontiers", None)
        resolver._try_unpaywall = await make_strategy("unpaywall", None)
        resolver._try_springer_oa = await make_strategy("springer_oa", None)

        source, text = await resolver._try_all(sp.paper, {})

        assert source == "europe_pmc"
        assert text == _SAMPLE_PDF_TEXT
        # europe_pmc must come after pmc
        assert call_order.index("europe_pmc") > call_order.index("pmc")
        # arxiv should NOT have been called (europe_pmc succeeded first)
        assert "arxiv" not in call_order


# ---------------------------------------------------------------------------
# Strategy: Semantic Scholar API lookup (_try_s2_api_lookup)
# ---------------------------------------------------------------------------


class TestS2ApiLookupStrategy:
    @pytest.mark.asyncio
    async def test_discovers_oa_pdf(self):
        """S2 API returns openAccessPdf.url → fetches PDF and returns text."""
        sp = _make_screened(doi="10.1109/CVPR.2020.12345")
        resolver = FullTextResolver()

        s2_api_resp = MagicMock()
        s2_api_resp.status_code = 200
        s2_api_resp.json.return_value = {
            "openAccessPdf": {"url": "http://example.com/paper.pdf"},
            "externalIds": {},
        }

        pdf_resp = MagicMock()
        pdf_resp.status_code = 200
        pdf_resp.content = b"x" * 2000

        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(side_effect=[s2_api_resp, pdf_resp])

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_s2_api_lookup(sp.paper)

        assert text == _SAMPLE_PDF_TEXT

    @pytest.mark.asyncio
    async def test_discovers_arxiv_mirror(self):
        """S2 API returns externalIds.ArXiv but no openAccessPdf → fetches from arxiv.org."""
        sp = _make_screened(doi="10.1109/CVPR.2020.12345")
        resolver = FullTextResolver()

        s2_api_resp = MagicMock()
        s2_api_resp.status_code = 200
        s2_api_resp.json.return_value = {
            "openAccessPdf": None,
            "externalIds": {"ArXiv": "2310.00259"},
        }

        arxiv_resp = MagicMock()
        arxiv_resp.status_code = 200
        arxiv_resp.content = b"x" * 2000

        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(side_effect=[s2_api_resp, arxiv_resp])

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_s2_api_lookup(sp.paper)

        assert text == _SAMPLE_PDF_TEXT
        # Verify the arXiv URL was used for the second call
        second_call_url = resolver._client.get.call_args_list[1][0][0]
        assert second_call_url == "https://arxiv.org/pdf/2310.00259"

    @pytest.mark.asyncio
    async def test_skips_without_doi(self):
        """Paper with no DOI → returns None immediately without any HTTP call."""
        sp = _make_screened(doi=None)
        resolver = FullTextResolver()
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock()

        text = await resolver._try_s2_api_lookup(sp.paper)

        assert text is None
        resolver._client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_429_with_retry(self):
        """S2 API returns 429 rate-limit → retries with backoff, returns None after exhausting."""
        sp = _make_screened(doi="10.1109/CVPR.2020.12345")
        resolver = FullTextResolver()

        rate_limit_resp = MagicMock()
        rate_limit_resp.status_code = 429

        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=rate_limit_resp)

        text = await resolver._try_s2_api_lookup(sp.paper)

        assert text is None
        # 3 calls: 1 initial + 2 retries (default max_retries=2)
        assert resolver._client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_handles_404(self):
        """S2 API returns 404 (paper not in S2) → returns None."""
        sp = _make_screened(doi="10.9999/does-not-exist")
        resolver = FullTextResolver()

        not_found_resp = MagicMock()
        not_found_resp.status_code = 404

        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=not_found_resp)

        text = await resolver._try_s2_api_lookup(sp.paper)

        assert text is None

    @pytest.mark.asyncio
    async def test_oa_pdf_fallback_to_arxiv(self):
        """S2 returns both openAccessPdf.url and ArXiv; OA PDF download fails → falls back to arXiv."""
        sp = _make_screened(doi="10.1109/CVPR.2020.12345")
        resolver = FullTextResolver()

        s2_api_resp = MagicMock()
        s2_api_resp.status_code = 200
        s2_api_resp.json.return_value = {
            "openAccessPdf": {"url": "http://example.com/paper.pdf"},
            "externalIds": {"ArXiv": "2310.00259"},
        }

        # OA PDF fetch fails (non-200)
        oa_pdf_resp = MagicMock()
        oa_pdf_resp.status_code = 403
        oa_pdf_resp.content = b""

        # arXiv fetch succeeds
        arxiv_resp = MagicMock()
        arxiv_resp.status_code = 200
        arxiv_resp.content = b"x" * 2000

        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(side_effect=[s2_api_resp, oa_pdf_resp, arxiv_resp])

        with patch(
            "autoreview.search.full_text._extract_text_from_pdf",
            return_value=_SAMPLE_PDF_TEXT,
        ):
            text = await resolver._try_s2_api_lookup(sp.paper)

        assert text == _SAMPLE_PDF_TEXT
        # Three calls: S2 API + OA PDF attempt + arXiv fallback
        assert resolver._client.get.call_count == 3
        # arXiv URL used for the third call
        third_call_url = resolver._client.get.call_args_list[2][0][0]
        assert third_call_url == "https://arxiv.org/pdf/2310.00259"


# ---------------------------------------------------------------------------
# DOI → PMID/PMCID enrichment
# ---------------------------------------------------------------------------


_NCBI_IDCONV_RESPONSE = {
    "records": [
        {"doi": "10.1234/test.001", "pmid": "11111111", "pmcid": "PMC1111111"},
        {"doi": "10.1234/test.002", "pmid": "22222222", "pmcid": ""},
        {"doi": "10.1234/test.003", "pmid": "", "pmcid": ""},  # Not in PubMed
    ]
}


class TestDOIToPMIDEnrichment:
    """Tests for _enrich_doi_to_pmid bulk DOI→PMID/PMCID step."""

    @pytest.mark.asyncio
    async def test_enriches_papers_with_pmids(self):
        """Papers with DOIs get PMID/PMCID written into external_ids."""
        papers = [
            _make_screened(doi="10.1234/test.001"),
            _make_screened(doi="10.1234/test.002"),
            _make_screened(doi="10.1234/test.003"),
        ]
        resolver = FullTextResolver()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _NCBI_IDCONV_RESPONSE
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        await resolver._enrich_doi_to_pmid(papers)

        p0 = papers[0].paper
        assert p0.external_ids["pmid"] == "11111111"
        assert p0.external_ids["pmcid"] == "PMC1111111"

        p1 = papers[1].paper
        assert p1.external_ids["pmid"] == "22222222"
        assert "pmcid" not in p1.external_ids  # empty string not stored

        p2 = papers[2].paper
        assert "pmid" not in p2.external_ids
        assert "pmcid" not in p2.external_ids

    @pytest.mark.asyncio
    async def test_skips_papers_with_existing_pmid(self):
        """Papers that already have a pmid in external_ids are not re-queried."""
        papers = [
            _make_screened(doi="10.1234/test.001", external_ids={"pmid": "99999999"}),
        ]
        resolver = FullTextResolver()
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock()

        await resolver._enrich_doi_to_pmid(papers)

        # Should not make any HTTP call — all candidates were filtered out
        resolver._client.get.assert_not_called()
        # Existing PMID unchanged
        assert papers[0].paper.external_ids["pmid"] == "99999999"

    @pytest.mark.asyncio
    async def test_handles_api_failure_gracefully(self):
        """If the NCBI API returns a non-200 status, the method continues without error."""
        papers = [_make_screened(doi="10.1234/test.001")]
        resolver = FullTextResolver()

        mock_resp = MagicMock()
        mock_resp.status_code = 503
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        # Should not raise
        await resolver._enrich_doi_to_pmid(papers)

        # No IDs written
        assert "pmid" not in papers[0].paper.external_ids

    @pytest.mark.asyncio
    async def test_handles_network_exception_gracefully(self):
        """If the HTTP call raises an exception, the method continues without error."""
        papers = [_make_screened(doi="10.1234/test.001")]
        resolver = FullTextResolver()

        import httpx

        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(side_effect=httpx.ConnectError("timeout"))

        # Should not raise
        await resolver._enrich_doi_to_pmid(papers)

        assert "pmid" not in papers[0].paper.external_ids

    @pytest.mark.asyncio
    async def test_batches_large_sets(self):
        """More than 200 DOIs are split into multiple API calls."""
        # 450 papers → 3 batches (200 + 200 + 50)
        papers = [_make_screened(doi=f"10.9999/paper.{i:04d}") for i in range(450)]
        resolver = FullTextResolver()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"records": []}  # Empty — just testing batch count
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        await resolver._enrich_doi_to_pmid(papers)

        # 3 batches × 1 call each, plus the rate-limiter acquire calls
        assert resolver._client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_skips_papers_without_doi(self):
        """Papers with no DOI are silently skipped."""
        papers = [
            _make_screened(doi=None),
            _make_screened(doi="10.1234/test.001"),
        ]
        resolver = FullTextResolver()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "records": [
                {"doi": "10.1234/test.001", "pmid": "11111111", "pmcid": "PMC1111111"},
            ]
        }
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=mock_resp)

        await resolver._enrich_doi_to_pmid(papers)

        # Only one HTTP call for the paper that has a DOI
        assert resolver._client.get.call_count == 1
        # Paper with DOI got enriched
        assert papers[1].paper.external_ids["pmid"] == "11111111"


# ---------------------------------------------------------------------------
# Europe PMC: DOI-based fallback lookup
# ---------------------------------------------------------------------------


class TestEuropePMCDOIFallback:
    """Tests for the DOI-based search path in _try_europe_pmc."""

    @pytest.mark.asyncio
    async def test_doi_lookup_fetches_full_text_when_no_pmcid(self):
        """When PMCID is absent, Europe PMC search API is used to discover it."""
        sp = _make_screened(doi="10.1234/test.001")
        resolver = FullTextResolver()

        # First call: Europe PMC search returns a hit with a PMCID
        search_resp = MagicMock()
        search_resp.status_code = 200
        search_resp.json.return_value = {"resultList": {"result": [{"pmcid": "PMC9999999"}]}}

        # Second call: full-text XML endpoint
        xml_resp = MagicMock()
        xml_resp.status_code = 200
        xml_resp.content = _SAMPLE_JATS

        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(side_effect=[search_resp, xml_resp])

        text = await resolver._try_europe_pmc(sp.paper)

        assert text is not None
        assert "introduction paragraph" in text.lower()
        # PMCID stored for future pipeline steps
        assert sp.paper.external_ids["pmcid"] == "PMC9999999"

    @pytest.mark.asyncio
    async def test_doi_lookup_returns_none_when_not_in_europe_pmc(self):
        """When DOI search returns no results, returns None without error."""
        sp = _make_screened(doi="10.9999/not-in-pmc")
        resolver = FullTextResolver()

        search_resp = MagicMock()
        search_resp.status_code = 200
        search_resp.json.return_value = {"resultList": {"result": []}}

        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=search_resp)

        text = await resolver._try_europe_pmc(sp.paper)

        assert text is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_pmcid_and_no_doi(self):
        """Paper with neither PMCID nor DOI is immediately skipped."""
        sp = _make_screened(doi=None)
        resolver = FullTextResolver()
        resolver._client = MagicMock()
        resolver._client.get = AsyncMock()

        text = await resolver._try_europe_pmc(sp.paper)

        assert text is None
        resolver._client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_pmcid_path_still_works(self):
        """Existing PMCID path continues to function correctly."""
        sp = _make_screened(doi="10.1234/test.001", external_ids={"pmcid": "PMC1234567"})
        resolver = FullTextResolver()

        xml_resp = MagicMock()
        xml_resp.status_code = 200
        xml_resp.content = _SAMPLE_JATS

        resolver._client = MagicMock()
        resolver._client.get = AsyncMock(return_value=xml_resp)

        text = await resolver._try_europe_pmc(sp.paper)

        assert text is not None
        assert "introduction paragraph" in text.lower()
        # Only one call — straight to the full-text XML endpoint
        resolver._client.get.assert_called_once()
        call_url = resolver._client.get.call_args[0][0]
        assert "PMC1234567" in call_url
