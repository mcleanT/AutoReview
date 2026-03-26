"""Tests for extraction JSON ingestion."""

from __future__ import annotations

from pathlib import Path


class TestIngestSingleExtraction:
    def test_parses_entities(self, sample_extraction_json: dict):
        from autoreview.knowledge_graph.ingest import ingest_extraction

        result = ingest_extraction(sample_extraction_json, paper_hash="aaa111")
        # Should extract subject + object = 2 entities
        assert len(result.entities) == 2
        assert result.entities[0]["canonical_name"] == "Wnt signaling pathway"
        assert result.entities[0]["entity_type"] == "pathway"
        assert result.entities[0]["ontology_id"] == "GO:0016055"

    def test_parses_assertions(self, sample_extraction_json: dict):
        from autoreview.knowledge_graph.ingest import ingest_extraction

        result = ingest_extraction(sample_extraction_json, paper_hash="aaa111")
        assert len(result.assertions) == 1
        assert result.assertions[0]["predicate"] == "is_required_for"
        assert result.assertions[0]["direction"] == "positive"

    def test_parses_evidence(self, sample_extraction_json: dict):
        from autoreview.knowledge_graph.ingest import ingest_extraction

        result = ingest_extraction(sample_extraction_json, paper_hash="aaa111")
        assert len(result.evidence_units) == 1
        assert result.evidence_units[0]["evidence_strength"] == "direct_experimental"
        assert result.evidence_units[0]["evidence_direction"] == "supports"

    def test_parses_citations(self, sample_extraction_json: dict):
        from autoreview.knowledge_graph.ingest import ingest_extraction

        result = ingest_extraction(sample_extraction_json, paper_hash="aaa111")
        assert len(result.citations) == 1

    def test_coerces_null_string_direction(self, sample_extraction_json: dict):
        from autoreview.knowledge_graph.ingest import ingest_extraction

        sample_extraction_json["assertion_drafts"][0]["direction"] = "null"
        result = ingest_extraction(sample_extraction_json, paper_hash="aaa111")
        assert result.assertions[0]["direction"] is None

    def test_assigns_related_to_for_empty_predicate(self, sample_extraction_json: dict):
        from autoreview.knowledge_graph.ingest import ingest_extraction

        sample_extraction_json["assertion_drafts"][0]["predicate"] = ""
        result = ingest_extraction(sample_extraction_json, paper_hash="aaa111")
        assert result.assertions[0]["predicate"] == "related_to"

    def test_normalizes_ontology_source(self, sample_extraction_json: dict):
        from autoreview.knowledge_graph.ingest import ingest_extraction

        sample_extraction_json["assertion_drafts"][0]["subject_entity"]["ontology_source"] = (
            "go; UniProt"
        )
        result = ingest_extraction(sample_extraction_json, paper_hash="aaa111")
        ent = result.entities[0]
        assert ent["ontology_source"] == "GO"  # First source after normalization

    def test_unknown_evidence_strength_maps_to_expert_opinion(self, sample_extraction_json: dict):
        from autoreview.knowledge_graph.ingest import ingest_extraction

        sample_extraction_json["evidence_units"][0]["evidence_strength"] = "anecdotal"
        result = ingest_extraction(sample_extraction_json, paper_hash="aaa111")
        assert result.evidence_units[0]["evidence_strength"] == "expert_opinion"


class TestIngestDirectory:
    def test_ingests_all_jsons(self, sample_extraction_dir: Path):
        from autoreview.knowledge_graph.ingest import ingest_directory

        result = ingest_directory(sample_extraction_dir)
        assert result.paper_count == 3
        assert len(result.all_entities) >= 4  # 2 per paper, some shared
        assert len(result.all_assertions) == 3
        assert len(result.all_evidence_units) == 3

    def test_skips_non_json_files(self, sample_extraction_dir: Path):
        from autoreview.knowledge_graph.ingest import ingest_directory

        result = ingest_directory(sample_extraction_dir)
        assert result.paper_count == 3

    def test_handles_malformed_json(self, sample_extraction_dir: Path):
        from autoreview.knowledge_graph.ingest import ingest_directory

        (sample_extraction_dir / "bad.json").write_text("{invalid json")
        result = ingest_directory(sample_extraction_dir)
        assert result.paper_count == 3  # Still processes the valid ones
        assert len(result.parse_errors) == 1
