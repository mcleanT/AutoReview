"""Tests for corpus expansion prompt building."""

from __future__ import annotations

from autoreview.llm.prompts.corpus_expansion import (
    CORPUS_EXPANSION_SYSTEM_PROMPT,
    CorpusExpansionQuery,
    CorpusExpansionQueryResult,
    build_corpus_expansion_query_prompt,
)


class TestBuildCorpusExpansionQueryPrompt:
    def test_contains_section_info(self):
        prompt = build_corpus_expansion_query_prompt(
            section_id="1",
            section_title="Introduction",
            section_description="Overview of the topic",
            key_concepts=["microglia", "cytokines"],
            cross_field_connections=["immunology"],
            existing_paper_ids=["p1", "p2"],
            scope_document="Review of neurodegeneration.",
        )
        assert "**ID:** 1" in prompt
        assert "**Title:** Introduction" in prompt
        assert "Overview of the topic" in prompt

    def test_contains_key_concepts(self):
        prompt = build_corpus_expansion_query_prompt(
            section_id="1",
            section_title="Mechanisms",
            section_description="Mechanistic pathways",
            key_concepts=["microglia", "cytokines", "blood-brain barrier"],
            cross_field_connections=[],
            existing_paper_ids=[],
            scope_document="Scope doc.",
        )
        assert "microglia" in prompt
        assert "cytokines" in prompt
        assert "blood-brain barrier" in prompt

    def test_contains_cross_field_connections(self):
        prompt = build_corpus_expansion_query_prompt(
            section_id="2",
            section_title="Cross-field",
            section_description="Connections",
            key_concepts=[],
            cross_field_connections=["immunology", "metabolomics"],
            existing_paper_ids=[],
            scope_document="Scope.",
        )
        assert "immunology" in prompt
        assert "metabolomics" in prompt

    def test_contains_scope_document(self):
        scope = "This review covers gut-brain axis in neurodegeneration."
        prompt = build_corpus_expansion_query_prompt(
            section_id="1",
            section_title="Intro",
            section_description="Desc",
            key_concepts=["concept"],
            cross_field_connections=[],
            existing_paper_ids=[],
            scope_document=scope,
        )
        assert scope in prompt

    def test_shows_existing_paper_count(self):
        prompt = build_corpus_expansion_query_prompt(
            section_id="1",
            section_title="Intro",
            section_description="Desc",
            key_concepts=["concept"],
            cross_field_connections=[],
            existing_paper_ids=["p1", "p2", "p3"],
            scope_document="Scope.",
        )
        assert "3 papers already assigned" in prompt

    def test_empty_concepts_shows_none(self):
        prompt = build_corpus_expansion_query_prompt(
            section_id="1",
            section_title="Intro",
            section_description="Desc",
            key_concepts=[],
            cross_field_connections=[],
            existing_paper_ids=[],
            scope_document="Scope.",
        )
        assert "(none)" in prompt


class TestCorpusExpansionModels:
    def test_query_model(self):
        q = CorpusExpansionQuery(
            query="test query",
            source_section_id="1",
            rationale="test rationale",
            target_concepts=["concept1", "concept2"],
        )
        assert q.query == "test query"
        assert q.source_section_id == "1"
        assert len(q.target_concepts) == 2

    def test_query_result_model(self):
        result = CorpusExpansionQueryResult(
            section_id="1",
            queries=[
                CorpusExpansionQuery(
                    query="test",
                    source_section_id="1",
                    rationale="reason",
                ),
            ],
        )
        assert result.section_id == "1"
        assert len(result.queries) == 1

    def test_system_prompt_mentions_primary_research(self):
        assert "PRIMARY RESEARCH" in CORPUS_EXPANSION_SYSTEM_PROMPT
        assert "citable" in CORPUS_EXPANSION_SYSTEM_PROMPT.lower()
