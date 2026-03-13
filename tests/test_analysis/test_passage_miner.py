"""Tests for passage mining module."""

from __future__ import annotations

from autoreview.analysis.passage_miner import PassageMiner
from autoreview.extraction.models import EvidenceStrength, Finding, PaperExtraction
from autoreview.llm.prompts.passage_mining import (
    SectionMiningResult,
    UndercitedClaim,
)
from autoreview.llm.provider import LLMStructuredResponse


class MockPassageMinerLLM:
    async def generate_structured(
        self, prompt, response_model, system="", max_tokens=4096, temperature=0.0
    ):
        if response_model == SectionMiningResult:
            # Determine section_id from the prompt
            section_id = "section_1"
            for sid in ["section_1", "section_2"]:
                if f"Section ID: {sid}" in prompt:
                    section_id = sid
                    break
            return LLMStructuredResponse(
                parsed=SectionMiningResult(
                    section_id=section_id,
                    undercited_claims=[
                        UndercitedClaim(
                            text="Senescent cells secrete pro-inflammatory cytokines",
                            evidence_weakness="Single preliminary study cited",
                            current_citations=["p1"],
                            suggested_queries=[
                                "cellular senescence SASP interleukin",
                                "senescent cell secretome inflammation",
                            ],
                            priority="high",
                        ),
                    ],
                    topic_expansions=[
                        "telomere shortening mechanisms in organ-specific senescence",
                    ],
                ),
                input_tokens=400,
                output_tokens=200,
            )
        raise ValueError(f"Unexpected response_model: {response_model}")


def _make_extractions() -> dict[str, PaperExtraction]:
    return {
        "p1": PaperExtraction(
            paper_id="p1",
            key_findings=[
                Finding(
                    claim="SASP promotes inflammation",
                    evidence_strength=EvidenceStrength.PRELIMINARY,
                    paper_id="p1",
                )
            ],
            methods_summary="In vitro",
            limitations="Single cell line",
        ),
    }


class TestPassageMiner:
    async def test_mine_section_returns_result(self):
        miner = PassageMiner(MockPassageMinerLLM())
        result = await miner.mine_section(
            section_id="section_1",
            section_text="Senescent cells secrete pro-inflammatory cytokines [@p1].",
            extractions=_make_extractions(),
        )
        assert result.section_id == "section_1"
        assert len(result.undercited_claims) == 1
        assert result.undercited_claims[0].priority == "high"
        assert len(result.undercited_claims[0].suggested_queries) == 2

    async def test_mine_section_includes_topic_expansions(self):
        miner = PassageMiner(MockPassageMinerLLM())
        result = await miner.mine_section(
            section_id="section_1",
            section_text="Telomere shortening is involved in senescence.",
            extractions=_make_extractions(),
        )
        assert len(result.topic_expansions) >= 1

    async def test_mine_all_sections_processes_each(self):
        miner = PassageMiner(MockPassageMinerLLM())
        sections = {
            "section_1": "Senescent cells secrete cytokines [@p1].",
            "section_2": "Organ-specific senescence patterns differ.",
        }
        results = await miner.mine_all_sections(sections, _make_extractions())
        assert len(results) == 2
        section_ids = {r.section_id for r in results}
        assert "section_1" in section_ids
        assert "section_2" in section_ids

    async def test_collect_queries_high_medium_only(self):
        miner = PassageMiner(MockPassageMinerLLM())
        result = await miner.mine_section(
            section_id="section_1",
            section_text="Senescent cells secrete cytokines [@p1].",
            extractions=_make_extractions(),
        )
        queries = miner.collect_queries([result], priorities={"high", "medium"})
        assert len(queries) > 0
        assert all(isinstance(q, str) for q in queries)

    async def test_collect_queries_excludes_low_priority(self):
        miner = PassageMiner(MockPassageMinerLLM())
        # Mock returns "high" priority claims; filtering for only "low" should return empty
        result = await miner.mine_section(
            section_id="section_1",
            section_text="Senescent cells secrete cytokines [@p1].",
            extractions=_make_extractions(),
        )
        queries = miner.collect_queries([result], priorities={"low"})
        # The mock returns high priority claims, so filtering for "low" returns nothing
        assert queries == []
