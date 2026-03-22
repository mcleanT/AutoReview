"""Integration test: run hybrid extractor on corpus papers with mocked LLM."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from autoreview.config.models import ExtractionConfig
from autoreview.extraction.hybrid import HybridExtractor
from autoreview.extraction.models import EvidenceStrength, Finding, PaperExtraction
from autoreview.extraction.programmatic import ProgrammaticExtractor
from scripts.benchmark_extractor import build_candidate_paper, build_screened_paper, load_corpus

CORPUS_DIR = Path(__file__).parent.parent.parent / "data" / "extraction_corpus"


@pytest.mark.asyncio
@pytest.mark.skipif(not CORPUS_DIR.exists(), reason="corpus not available")
async def test_hybrid_on_corpus_papers():
    """Run hybrid extractor on 3 real corpus papers with mocked LLM."""
    corpus = load_corpus(CORPUS_DIR)[:3]
    config = ExtractionConfig()
    programmatic = ProgrammaticExtractor(config)

    # Track the current paper_id so the mock can return a matching paper_id.
    # HybridExtractor calls generate_structured with the draft's paper_id embedded
    # in the prompt; we capture the real ID via a mutable cell updated per iteration.
    current_paper_id: list[str] = [""]

    def make_response(*args, **kwargs):
        resp = MagicMock()
        pid = current_paper_id[0]
        resp.parsed = PaperExtraction(
            paper_id=pid,
            key_findings=[
                Finding(
                    claim="Mocked finding",
                    evidence_strength=EvidenceStrength.MODERATE,
                    paper_id=pid,
                )
            ],
            methods_summary="Mocked methods.",
            limitations="Mocked limitations.",
        )
        return resp

    mock_llm = AsyncMock()
    mock_llm.generate_structured = AsyncMock(side_effect=make_response)
    hybrid = HybridExtractor(programmatic, mock_llm)

    for entry in corpus:
        candidate = build_candidate_paper(entry)
        screened = build_screened_paper(candidate)
        current_paper_id[0] = candidate.id
        result = await hybrid.extract(screened)
        assert isinstance(result, PaperExtraction)
        assert result.paper_id == candidate.id
        assert result.study_design is not None
        assert result.quality_score is not None
