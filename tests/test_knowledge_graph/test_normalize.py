"""Tests for post-extraction claim normalization."""

from __future__ import annotations

import asyncio


class TestTextCleaning:
    """Tests for entity name text cleaning transforms."""

    def test_strip_parenthetical_synonym(self):
        from autoreview.knowledge_graph.normalize import clean_entity_name

        name, aliases = clean_entity_name("Invariant chain (CD74)")
        assert name == "Invariant chain"
        assert "CD74" in aliases

    def test_strip_multiple_parentheticals(self):
        from autoreview.knowledge_graph.normalize import clean_entity_name

        name, aliases = clean_entity_name("T (Brachyury) (TBXT)")
        assert name == "T"
        assert "Brachyury" in aliases
        assert "TBXT" in aliases

    def test_strip_leading_article(self):
        from autoreview.knowledge_graph.normalize import clean_entity_name

        name, _ = clean_entity_name("the Wnt signaling pathway")
        assert name == "Wnt signaling"

    def test_strip_multiple_leading_articles(self):
        from autoreview.knowledge_graph.normalize import clean_entity_name

        name, _ = clean_entity_name("these cell populations")
        assert name == "cell populations"

    def test_collapse_whitespace(self):
        from autoreview.knowledge_graph.normalize import clean_entity_name

        name, _ = clean_entity_name("mesoderm   differentiation\t process")
        assert name == "mesoderm differentiation"

    def test_strip_trailing_descriptor(self):
        from autoreview.knowledge_graph.normalize import clean_entity_name

        name, _ = clean_entity_name("Wnt signaling pathway")
        assert name == "Wnt signaling"

    def test_no_strip_trailing_descriptor_if_too_short(self):
        from autoreview.knowledge_graph.normalize import clean_entity_name

        name, _ = clean_entity_name("Wnt pathway")
        assert name == "Wnt pathway"

    def test_preserves_clean_name(self):
        from autoreview.knowledge_graph.normalize import clean_entity_name

        name, aliases = clean_entity_name("BMP4")
        assert name == "BMP4"
        assert aliases == []

    def test_combined_transforms(self):
        from autoreview.knowledge_graph.normalize import clean_entity_name

        name, aliases = clean_entity_name(
            "the BMP4 (bone morphogenetic protein 4) signaling cascade"
        )
        assert name == "BMP4 signaling"
        assert "bone morphogenetic protein 4" in aliases

    def test_empty_string(self):
        from autoreview.knowledge_graph.normalize import clean_entity_name

        name, aliases = clean_entity_name("")
        assert name == ""
        assert aliases == []


class TestPredicateCleaning:
    """Tests for predicate string cleaning transforms."""

    def test_strip_trailing_punctuation(self):
        from autoreview.knowledge_graph.normalize import clean_predicate

        assert clean_predicate("induces.") == "induces"
        assert clean_predicate("inhibits;") == "inhibits"

    def test_collapse_internal_whitespace(self):
        from autoreview.knowledge_graph.normalize import clean_predicate

        assert clean_predicate("activ ates") == "activates"

    def test_underscore_normalization(self):
        from autoreview.knowledge_graph.normalize import clean_predicate

        assert clean_predicate("is required for") == "is_required_for"
        assert clean_predicate("is located in") == "is_located_in"
        assert clean_predicate("interacts with") == "interacts_with"

    def test_tense_past(self):
        from autoreview.knowledge_graph.normalize import clean_predicate

        assert clean_predicate("inhibited") == "inhibits"
        assert clean_predicate("induced") == "induces"
        assert clean_predicate("promoted") == "promotes"

    def test_tense_gerund(self):
        from autoreview.knowledge_graph.normalize import clean_predicate

        assert clean_predicate("inhibiting") == "inhibits"
        assert clean_predicate("inducing") == "induces"

    def test_already_canonical(self):
        from autoreview.knowledge_graph.normalize import clean_predicate

        assert clean_predicate("induces") == "induces"
        assert clean_predicate("is_required_for") == "is_required_for"

    def test_combined_cleanup_then_tense(self):
        from autoreview.knowledge_graph.normalize import clean_predicate

        assert clean_predicate("inhibited.") == "inhibits"

    def test_unknown_predicate_passthrough(self):
        from autoreview.knowledge_graph.normalize import clean_predicate

        assert clean_predicate("some_unknown_pred") == "some_unknown_pred"


class TestCompoundDecomposition:
    """Tests for rule-based compound object decomposition."""

    def test_conjunction_with_head_noun(self):
        from autoreview.knowledge_graph.normalize import decompose_object

        result = decompose_object("endoderm and mesoderm differentiation")
        assert len(result) == 2
        assert "endoderm differentiation" in result
        assert "mesoderm differentiation" in result

    def test_comma_and_list_with_head_noun(self):
        from autoreview.knowledge_graph.normalize import decompose_object

        result = decompose_object("T, Sox2, and Nanog expression levels")
        assert len(result) == 3
        assert any("T" in r and "expression" in r for r in result)
        assert any("Sox2" in r and "expression" in r for r in result)
        assert any("Nanog" in r and "expression" in r for r in result)

    def test_slash_split_with_tail(self):
        from autoreview.knowledge_graph.normalize import decompose_object

        result = decompose_object("BMP4/WNT signaling in gastruloids")
        assert len(result) == 2
        assert any("BMP4" in r for r in result)
        assert any("WNT" in r for r in result)

    def test_prepositional_compound(self):
        from autoreview.knowledge_graph.normalize import decompose_object

        result = decompose_object("self-organization into endoderm and mesoderm")
        assert len(result) == 2
        assert "endoderm" in result
        assert "mesoderm" in result

    def test_short_object_not_decomposed(self):
        from autoreview.knowledge_graph.normalize import decompose_object

        result = decompose_object("mesoderm differentiation")
        assert result == ["mesoderm differentiation"]

    def test_three_word_object_not_decomposed(self):
        from autoreview.knowledge_graph.normalize import decompose_object

        result = decompose_object("lateral plate mesoderm")
        assert result == ["lateral plate mesoderm"]

    def test_no_pattern_match_returns_original(self):
        from autoreview.knowledge_graph.normalize import decompose_object

        result = decompose_object("mesoderm differentiation in mouse gastruloids")
        assert result == ["mesoderm differentiation in mouse gastruloids"]

    def test_conjunction_without_head_noun(self):
        from autoreview.knowledge_graph.normalize import decompose_object

        result = decompose_object("lateral mesoderm and neural crest")
        assert len(result) == 2
        assert "lateral mesoderm" in result
        assert "neural crest" in result

    def test_flag_for_llm(self):
        from autoreview.knowledge_graph.normalize import flag_for_llm_decomposition

        long_obj = (
            "self-organization of human gastruloids into homogenous"
            " subpopulations of endoderm and mesoderm"
        )
        assert flag_for_llm_decomposition(long_obj) is True
        assert flag_for_llm_decomposition("mesoderm differentiation") is False

    def test_slash_not_in_units(self):
        from autoreview.knowledge_graph.normalize import decompose_object

        # "ng/mL" should not be split
        result = decompose_object("10 ng/mL BMP4 treatment effect")
        assert len(result) == 1


class TestLLMDecomposition:
    """Tests for LLM fallback decomposition with mocked LLM."""

    def test_llm_fallback_decomposes_verbose_object(self):
        from autoreview.knowledge_graph.normalize import llm_decompose_objects

        async def mock_llm(objects: list[str]) -> list[list[str]]:
            return [["endoderm differentiation", "mesoderm differentiation"]]

        long_obj = (
            "self-organization of human gastruloids into homogenous"
            " subpopulations of endoderm and mesoderm"
        )
        result = asyncio.get_event_loop().run_until_complete(
            llm_decompose_objects([long_obj], mock_llm)
        )
        assert result == [["endoderm differentiation", "mesoderm differentiation"]]

    def test_llm_fallback_atomic_passthrough(self):
        from autoreview.knowledge_graph.normalize import llm_decompose_objects

        async def mock_llm(objects: list[str]) -> list[list[str]]:
            return [["mesoderm differentiation"]]

        result = asyncio.get_event_loop().run_until_complete(
            llm_decompose_objects(["mesoderm differentiation"], mock_llm)
        )
        assert result == [["mesoderm differentiation"]]

    def test_llm_fallback_batch(self):
        from autoreview.knowledge_graph.normalize import llm_decompose_objects

        async def mock_llm(objects: list[str]) -> list[list[str]]:
            return [
                ["endoderm", "mesoderm"],
                ["neural crest migration", "neural tube closure"],
            ]

        result = asyncio.get_event_loop().run_until_complete(
            llm_decompose_objects(["obj1 long enough words", "obj2 long enough words"], mock_llm)
        )
        assert len(result) == 2
        assert result[0] == ["endoderm", "mesoderm"]
        assert result[1] == ["neural crest migration", "neural tube closure"]

    def test_llm_fallback_none_fn_returns_originals(self):
        from autoreview.knowledge_graph.normalize import llm_decompose_objects

        result = asyncio.get_event_loop().run_until_complete(
            llm_decompose_objects(["some verbose object name here"], None)
        )
        assert result == [["some verbose object name here"]]


class TestQuantitativeBackfill:
    """Tests for extracting quantitative context from natural language text."""

    def test_extract_concentration(self):
        from autoreview.knowledge_graph.normalize import backfill_quantitative_context

        assertion = {
            "natural_language": "BMP4 at 10 ng/mL induces mesoderm differentiation",
            "quantitative_context": None,
        }
        changed = backfill_quantitative_context(assertion)
        assert changed is True
        assert assertion["quantitative_context"]["concentration"] == "10 ng/mL"

    def test_extract_timepoint(self):
        from autoreview.knowledge_graph.normalize import backfill_quantitative_context

        assertion = {
            "natural_language": "Mesoderm markers appear at 48h of culture",
            "quantitative_context": None,
        }
        changed = backfill_quantitative_context(assertion)
        assert changed is True
        assert assertion["quantitative_context"]["timepoint"] == "48h"

    def test_normalize_time_units(self):
        from autoreview.knowledge_graph.normalize import backfill_quantitative_context

        assertion = {
            "natural_language": "Expression peaks at 72 hours post-treatment",
            "quantitative_context": None,
        }
        changed = backfill_quantitative_context(assertion)
        assert changed is True
        assert assertion["quantitative_context"]["timepoint"] == "72h"

    def test_extract_dose(self):
        from autoreview.knowledge_graph.normalize import backfill_quantitative_context

        assertion = {
            "natural_language": "Animals received 5 mg/kg of the compound",
            "quantitative_context": None,
        }
        changed = backfill_quantitative_context(assertion)
        assert changed is True
        assert assertion["quantitative_context"]["dose"] == "5 mg/kg"

    def test_extract_multiple_fields(self):
        from autoreview.knowledge_graph.normalize import backfill_quantitative_context

        assertion = {
            "natural_language": "BMP4 at 10 ng/mL induces T expression at 48h",
            "quantitative_context": None,
        }
        changed = backfill_quantitative_context(assertion)
        assert changed is True
        assert assertion["quantitative_context"]["concentration"] == "10 ng/mL"
        assert assertion["quantitative_context"]["timepoint"] == "48h"

    def test_no_overwrite_existing(self):
        from autoreview.knowledge_graph.normalize import backfill_quantitative_context

        assertion = {
            "natural_language": "BMP4 at 10 ng/mL induces mesoderm at 48h",
            "quantitative_context": {
                "concentration": "5 ng/mL",
                "timepoint": None,
                "dose": None,
            },
        }
        changed = backfill_quantitative_context(assertion)
        assert changed is True
        assert assertion["quantitative_context"]["concentration"] == "5 ng/mL"  # preserved
        assert assertion["quantitative_context"]["timepoint"] == "48h"  # backfilled

    def test_no_match_returns_false(self):
        from autoreview.knowledge_graph.normalize import backfill_quantitative_context

        assertion = {
            "natural_language": "BMP4 induces mesoderm differentiation",
            "quantitative_context": None,
        }
        changed = backfill_quantitative_context(assertion)
        assert changed is False
        assert assertion["quantitative_context"] is None

    def test_fallback_to_treatment(self):
        from autoreview.knowledge_graph.normalize import backfill_quantitative_context

        assertion = {
            "natural_language": "BMP4 induces mesoderm",
            "quantitative_context": None,
            "conditions": {"treatment": ["10 ng/mL BMP4"]},
        }
        changed = backfill_quantitative_context(assertion)
        assert changed is True
        assert assertion["quantitative_context"]["concentration"] == "10 ng/mL"

    def test_day_timepoint(self):
        from autoreview.knowledge_graph.normalize import backfill_quantitative_context

        assertion = {
            "natural_language": "At day 5 gastruloids show elongation",
            "quantitative_context": None,
        }
        changed = backfill_quantitative_context(assertion)
        assert changed is True
        assert assertion["quantitative_context"]["timepoint"] == "5d"


class TestClaimNormalizer:
    """Tests for the ClaimNormalizer orchestrator."""

    def _make_entity(
        self, name: str, entity_type: str = "biological_process", paper_id: str = "p1"
    ) -> dict:
        return {
            "canonical_name": name,
            "entity_type": entity_type,
            "ontology_id": None,
            "ontology_source": None,
            "aliases": [],
            "paper_ids": [paper_id],
        }

    def _make_assertion(
        self,
        subject: str,
        obj: str,
        predicate: str = "induces",
        draft_id: str = "a_001",
        natural_language: str = "",
    ) -> dict:
        return {
            "draft_id": draft_id,
            "subject_canonical_name": subject,
            "object_canonical_name": obj,
            "predicate": predicate,
            "direction": "positive",
            "assertion_type": "mechanistic_causal",
            "evidence_unit_ids": ["e_001"],
            "paper_id": "p1",
            "publication_date": "2023-01-15",
            "natural_language": natural_language,
            "quantitative_context": None,
            "conditions": None,
            "model_system": None,
            "organism": None,
            "in_vitro": None,
        }

    def test_pre_dedup_text_cleaning(self):
        from autoreview.knowledge_graph.normalize import ClaimNormalizer

        normalizer = ClaimNormalizer(llm_decompose=False)
        entities = [self._make_entity("the Wnt signaling pathway")]
        assertions = [self._make_assertion("BMP4", "the Wnt signaling pathway")]
        new_ents, new_asserts, report = asyncio.get_event_loop().run_until_complete(
            normalizer.pre_dedup(entities, assertions)
        )
        assert new_ents[0]["canonical_name"] == "Wnt signaling"
        assert new_asserts[0]["object_canonical_name"] == "Wnt signaling"
        assert report.text_cleaned >= 1

    def test_pre_dedup_predicate_cleaning(self):
        from autoreview.knowledge_graph.normalize import ClaimNormalizer

        normalizer = ClaimNormalizer(llm_decompose=False)
        entities = [
            self._make_entity("BMP4", "protein"),
            self._make_entity("mesoderm differentiation"),
        ]
        assertions = [
            self._make_assertion("BMP4", "mesoderm differentiation", predicate="promoted.")
        ]
        _, new_asserts, report = asyncio.get_event_loop().run_until_complete(
            normalizer.pre_dedup(entities, assertions)
        )
        assert new_asserts[0]["predicate"] == "promotes"
        assert report.predicates_cleaned == 1

    def test_pre_dedup_decomposition(self):
        from autoreview.knowledge_graph.normalize import ClaimNormalizer

        normalizer = ClaimNormalizer(llm_decompose=False)
        entities = [
            self._make_entity("BMP4", "protein"),
            self._make_entity("endoderm and mesoderm differentiation"),
        ]
        assertions = [
            self._make_assertion("BMP4", "endoderm and mesoderm differentiation", draft_id="a_001"),
        ]
        new_ents, new_asserts, report = asyncio.get_event_loop().run_until_complete(
            normalizer.pre_dedup(entities, assertions)
        )
        assert len(new_asserts) == 2
        obj_names = {a["object_canonical_name"] for a in new_asserts}
        assert "endoderm differentiation" in obj_names
        assert "mesoderm differentiation" in obj_names
        ent_names = {e["canonical_name"] for e in new_ents}
        assert "endoderm differentiation" in ent_names
        assert "mesoderm differentiation" in ent_names
        assert report.claims_decomposed == 1
        assert report.claims_produced == 2

    def test_pre_dedup_decomposed_claim_audit_trail(self):
        from autoreview.knowledge_graph.normalize import ClaimNormalizer

        normalizer = ClaimNormalizer(llm_decompose=False)
        entities = [
            self._make_entity("BMP4", "protein"),
            self._make_entity("endoderm and mesoderm differentiation"),
        ]
        assertions = [
            self._make_assertion("BMP4", "endoderm and mesoderm differentiation", draft_id="a_001"),
        ]
        _, new_asserts, _ = asyncio.get_event_loop().run_until_complete(
            normalizer.pre_dedup(entities, assertions)
        )
        for a in new_asserts:
            assert a["_decomposed_from"] == "a_001"
            assert a["draft_id"].startswith("a_001_d")

    def test_post_dedup_quantitative_backfill(self):
        from autoreview.knowledge_graph.normalize import ClaimNormalizer

        normalizer = ClaimNormalizer(llm_decompose=False)
        assertions = [
            self._make_assertion(
                "BMP4",
                "mesoderm differentiation",
                natural_language="BMP4 at 10 ng/mL induces mesoderm at 48h",
            ),
        ]
        new_asserts, report = asyncio.get_event_loop().run_until_complete(
            normalizer.post_dedup(assertions)
        )
        assert new_asserts[0]["quantitative_context"]["concentration"] == "10 ng/mL"
        assert new_asserts[0]["quantitative_context"]["timepoint"] == "48h"
        assert report.quant_backfilled == 1

    def test_normalization_report_fields(self):
        from autoreview.knowledge_graph.normalize import NormalizationReport

        report = NormalizationReport()
        assert report.text_cleaned == 0
        assert report.predicates_cleaned == 0
        assert report.claims_decomposed == 0
        assert report.claims_produced == 0
        assert report.quant_backfilled == 0
        assert report.llm_calls == 0


class TestPipelineIntegration:
    """Integration tests for normalization in the build_graph pipeline."""

    def test_normalize_false_unchanged(self, sample_v5_extraction_dir):
        """Regression: normalize=False produces identical output to current code."""
        from autoreview.knowledge_graph import build_graph

        graph_without = build_graph(sample_v5_extraction_dir, version=2)
        graph_with = build_graph(sample_v5_extraction_dir, version=2, normalize=False)
        assert graph_without.number_of_edges() == graph_with.number_of_edges()
        assert graph_without.number_of_nodes() == graph_with.number_of_nodes()

    def test_normalize_true_accepted(self, sample_v5_extraction_dir):
        """normalize=True runs without error and produces a valid graph."""
        from autoreview.knowledge_graph import build_graph

        graph = build_graph(
            sample_v5_extraction_dir, version=2, normalize=True, llm_decompose=False
        )
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

    def test_normalization_report_on_graph(self, sample_v5_extraction_dir):
        """NormalizationReport is stored on the graph object."""
        from autoreview.knowledge_graph import build_graph

        graph = build_graph(
            sample_v5_extraction_dir, version=2, normalize=True, llm_decompose=False
        )
        report = graph.graph.get("normalization_report")
        assert report is not None
        assert hasattr(report, "text_cleaned")
        assert hasattr(report, "quant_backfilled")

    def test_normalize_with_v1_version(self, sample_v5_extraction_dir):
        """Normalization works with v1 merge strategy too."""
        from autoreview.knowledge_graph import build_graph

        graph = build_graph(
            sample_v5_extraction_dir, version=1, normalize=True, llm_decompose=False
        )
        assert graph.number_of_nodes() > 0
