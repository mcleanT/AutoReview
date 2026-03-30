"""Tests for entity deduplication, predicate normalization, and assertion merging."""

from __future__ import annotations


class TestEntityDedup:
    """Test the three-pass entity resolution strategy."""

    def test_pass1_ontology_id_merge(self):
        """Entities sharing ontology_id merge regardless of name."""
        from autoreview.knowledge_graph.dedup import deduplicate_entities

        entities = [
            {
                "canonical_name": "Wnt signaling pathway",
                "entity_type": "pathway",
                "ontology_id": "GO:0016055",
                "ontology_source": "GO",
                "aliases": ["Wnt pathway"],
                "paper_ids": ["p1"],
            },
            {
                "canonical_name": "Wnt signalling",
                "entity_type": "pathway",
                "ontology_id": "GO:0016055",
                "ontology_source": "GO",
                "aliases": ["canonical Wnt"],
                "paper_ids": ["p2"],
            },
        ]
        registry = deduplicate_entities(entities)
        assert len(registry.entities) == 1
        ent = list(registry.entities.values())[0]
        assert ent.paper_count == 2
        assert "Wnt pathway" in ent.aliases
        assert "canonical Wnt" in ent.aliases

    def test_pass2_canonical_name_normalization(self):
        """Entities with same normalized name merge."""
        from autoreview.knowledge_graph.dedup import deduplicate_entities

        entities = [
            {
                "canonical_name": "Anterior-Posterior Axis",
                "entity_type": "biological_process",
                "ontology_id": None,
                "ontology_source": None,
                "aliases": [],
                "paper_ids": ["p1"],
            },
            {
                "canonical_name": "anterior-posterior axis",
                "entity_type": "biological_process",
                "ontology_id": None,
                "ontology_source": None,
                "aliases": [],
                "paper_ids": ["p2"],
            },
        ]
        registry = deduplicate_entities(entities)
        assert len(registry.entities) == 1

    def test_pass3_fuzzy_match_within_type(self):
        """Similar names within same entity type merge above threshold."""
        from autoreview.knowledge_graph.dedup import deduplicate_entities

        entities = [
            {
                "canonical_name": "mesoderm specification",
                "entity_type": "biological_process",
                "ontology_id": None,
                "ontology_source": None,
                "aliases": [],
                "paper_ids": ["p1"],
            },
            {
                "canonical_name": "mesoderm specifications",
                "entity_type": "biological_process",
                "ontology_id": None,
                "ontology_source": None,
                "aliases": [],
                "paper_ids": ["p2"],
            },
        ]
        registry = deduplicate_entities(entities)
        assert len(registry.entities) == 1

    def test_no_cross_type_fuzzy_merge(self):
        """Similar names across different entity types do NOT merge."""
        from autoreview.knowledge_graph.dedup import deduplicate_entities

        entities = [
            {
                "canonical_name": "BMP4",
                "entity_type": "gene",
                "ontology_id": None,
                "ontology_source": None,
                "aliases": [],
                "paper_ids": ["p1"],
            },
            {
                "canonical_name": "BMP4",
                "entity_type": "protein",
                "ontology_id": None,
                "ontology_source": None,
                "aliases": [],
                "paper_ids": ["p2"],
            },
        ]
        registry = deduplicate_entities(entities)
        assert len(registry.entities) == 2

    def test_ontology_id_entity_wins_canonical(self):
        """Entity with ontology ID becomes canonical in merge."""
        from autoreview.knowledge_graph.dedup import deduplicate_entities

        entities = [
            {
                "canonical_name": "wnt signaling",
                "entity_type": "pathway",
                "ontology_id": None,
                "ontology_source": None,
                "aliases": [],
                "paper_ids": ["p1"],
            },
            {
                "canonical_name": "Wnt signaling pathway",
                "entity_type": "pathway",
                "ontology_id": "GO:0016055",
                "ontology_source": "GO",
                "aliases": [],
                "paper_ids": ["p2"],
            },
        ]
        registry = deduplicate_entities(entities)
        ent = list(registry.entities.values())[0]
        assert ent.ontology_id == "GO:0016055"
        assert ent.canonical_name == "Wnt signaling pathway"

    def test_surface_form_reverse_index(self):
        """Registry provides surface_form → entity_id lookup."""
        from autoreview.knowledge_graph.dedup import deduplicate_entities

        entities = [
            {
                "canonical_name": "Wnt signaling pathway",
                "entity_type": "pathway",
                "ontology_id": "GO:0016055",
                "ontology_source": "GO",
                "aliases": ["Wnt pathway"],
                "paper_ids": ["p1"],
            },
        ]
        registry = deduplicate_entities(entities)
        assert "wnt signaling pathway" in registry.surface_to_id
        assert "wnt pathway" in registry.surface_to_id

    def test_merge_log_records_decisions(self):
        """All merge decisions are logged for auditability."""
        from autoreview.knowledge_graph.dedup import deduplicate_entities

        entities = [
            {
                "canonical_name": "Wnt signaling pathway",
                "entity_type": "pathway",
                "ontology_id": "GO:0016055",
                "ontology_source": "GO",
                "aliases": [],
                "paper_ids": ["p1"],
            },
            {
                "canonical_name": "Wnt signalling",
                "entity_type": "pathway",
                "ontology_id": "GO:0016055",
                "ontology_source": "GO",
                "aliases": [],
                "paper_ids": ["p2"],
            },
        ]
        registry = deduplicate_entities(entities)
        assert len(registry.merge_log) >= 1
        assert registry.merge_log[0]["pass"] == "ontology_id"

    def test_other_type_uses_token_blocking(self):
        """'other' entity type uses token-based blocking to avoid O(n^2) comparisons."""
        from autoreview.knowledge_graph.dedup import deduplicate_entities

        entities = [
            {
                "canonical_name": "gastruloid elongation morphogenesis",
                "entity_type": "other",
                "ontology_id": None,
                "ontology_source": None,
                "aliases": [],
                "paper_ids": ["p1"],
            },
            {
                "canonical_name": "gastruloid elongation process",
                "entity_type": "other",
                "ontology_id": None,
                "ontology_source": None,
                "aliases": [],
                "paper_ids": ["p2"],
            },
            {
                "canonical_name": "culture medium composition",
                "entity_type": "other",
                "ontology_id": None,
                "ontology_source": None,
                "aliases": [],
                "paper_ids": ["p3"],
            },
        ]
        registry = deduplicate_entities(entities)
        # First two share "gastruloid" + "elongation" tokens and are similar → merge
        # Third shares no significant tokens with the others → stays separate
        assert len(registry.entities) == 2


class TestPredicateNormalization:
    """Test predicate synonym resolution."""

    def test_exact_synonym_match(self):
        from autoreview.knowledge_graph.dedup import normalize_predicate

        assert normalize_predicate("is_necessary_for") == "is_required_for"
        assert normalize_predicate("activates") == "induces"
        assert normalize_predicate("suppresses") == "inhibits"
        assert normalize_predicate("binds_to") == "interacts_with"

    def test_canonical_predicate_unchanged(self):
        from autoreview.knowledge_graph.dedup import normalize_predicate

        assert normalize_predicate("is_required_for") == "is_required_for"
        assert normalize_predicate("induces") == "induces"

    def test_unknown_predicate_kept_as_is(self):
        from autoreview.knowledge_graph.dedup import normalize_predicate

        assert normalize_predicate("totally_novel_predicate") == "totally_novel_predicate"

    def test_related_to_default_unchanged(self):
        from autoreview.knowledge_graph.dedup import normalize_predicate

        assert normalize_predicate("related_to") == "related_to"

    def test_normalization_log_records_mappings(self):
        """All predicate normalizations are logged for auditability."""
        from autoreview.knowledge_graph.dedup import PredicateNormalizer

        normalizer = PredicateNormalizer()
        normalizer.normalize("is_necessary_for")
        normalizer.normalize("activates")
        normalizer.normalize("totally_novel_predicate")
        assert len(normalizer.log) == 3
        assert normalizer.log[0]["original"] == "is_necessary_for"
        assert normalizer.log[0]["canonical"] == "is_required_for"
        assert normalizer.log[2]["original"] == "totally_novel_predicate"
        assert normalizer.log[2]["canonical"] == "totally_novel_predicate"


class TestAssertionMerging:
    """Test merging of assertions with same (subject, predicate, object) after dedup."""

    def test_same_triple_merges(self):
        from autoreview.knowledge_graph.dedup import merge_assertions

        assertions = [
            {
                "subject_id": "ent1",
                "object_id": "ent2",
                "predicate": "is_required_for",
                "direction": "positive",
                "assertion_type": "mechanistic_causal",
                "draft_id": "a_001",
                "evidence_unit_ids": ["e_001"],
                "paper_id": "p1",
                "publication_date": "2023-01-15",
            },
            {
                "subject_id": "ent1",
                "object_id": "ent2",
                "predicate": "is_required_for",
                "direction": "positive",
                "assertion_type": "mechanistic_causal",
                "draft_id": "a_002",
                "evidence_unit_ids": ["e_002"],
                "paper_id": "p2",
                "publication_date": "2022-06-01",
            },
        ]
        result = merge_assertions(assertions)
        assert len(result.assertions) == 1
        assert len(result.assertions[0]["source_assertions"]) == 2
        assert result.assertions[0]["publication_date"] == "2022-06-01"  # earliest

    def test_different_triples_stay_separate(self):
        from autoreview.knowledge_graph.dedup import merge_assertions

        assertions = [
            {
                "subject_id": "ent1",
                "object_id": "ent2",
                "predicate": "is_required_for",
                "direction": "positive",
                "assertion_type": "mechanistic_causal",
                "draft_id": "a_001",
                "evidence_unit_ids": ["e_001"],
                "paper_id": "p1",
                "publication_date": "2023-01-15",
            },
            {
                "subject_id": "ent3",
                "object_id": "ent4",
                "predicate": "induces",
                "direction": "positive",
                "assertion_type": "mechanistic_causal",
                "draft_id": "a_002",
                "evidence_unit_ids": ["e_002"],
                "paper_id": "p2",
                "publication_date": "2022-06-01",
            },
        ]
        result = merge_assertions(assertions)
        assert len(result.assertions) == 2

    def test_direction_conflict_sets_none(self):
        from autoreview.knowledge_graph.dedup import merge_assertions

        assertions = [
            {
                "subject_id": "ent1",
                "object_id": "ent2",
                "predicate": "induces",
                "direction": "positive",
                "assertion_type": "mechanistic_causal",
                "draft_id": "a_001",
                "evidence_unit_ids": ["e_001"],
                "paper_id": "p1",
                "publication_date": "2023-01-15",
            },
            {
                "subject_id": "ent1",
                "object_id": "ent2",
                "predicate": "induces",
                "direction": "negative",
                "assertion_type": "mechanistic_causal",
                "draft_id": "a_002",
                "evidence_unit_ids": ["e_002"],
                "paper_id": "p2",
                "publication_date": "2022-06-01",
            },
        ]
        result = merge_assertions(assertions)
        assert len(result.assertions) == 1
        assert result.assertions[0]["direction"] is None  # conflict → None
        assert result.assertions[0]["direction_conflict"] is True

    def test_merge_log_records_decisions(self):
        """All assertion merge decisions are logged for auditability."""
        from autoreview.knowledge_graph.dedup import merge_assertions

        assertions = [
            {
                "subject_id": "ent1",
                "object_id": "ent2",
                "predicate": "is_required_for",
                "direction": "positive",
                "assertion_type": "mechanistic_causal",
                "draft_id": "a_001",
                "evidence_unit_ids": ["e_001"],
                "paper_id": "p1",
                "publication_date": "2023-01-15",
            },
            {
                "subject_id": "ent1",
                "object_id": "ent2",
                "predicate": "is_required_for",
                "direction": "positive",
                "assertion_type": "mechanistic_causal",
                "draft_id": "a_002",
                "evidence_unit_ids": ["e_002"],
                "paper_id": "p2",
                "publication_date": "2022-06-01",
            },
        ]
        result = merge_assertions(assertions)
        assert len(result.merge_log) >= 1
        assert result.merge_log[0]["merged_draft_ids"] == ["a_001", "a_002"]
        assert result.merge_log[0]["papers"] == ["p1", "p2"]

    def test_self_loop_allowed(self):
        """Assertions where subject == object are valid (e.g., autoregulation)."""
        from autoreview.knowledge_graph.dedup import merge_assertions

        assertions = [
            {
                "subject_id": "ent1",
                "object_id": "ent1",
                "predicate": "regulates",
                "direction": "positive",
                "assertion_type": "mechanistic_causal",
                "draft_id": "a_001",
                "evidence_unit_ids": ["e_001"],
                "paper_id": "p1",
                "publication_date": "2023-01-15",
            },
        ]
        result = merge_assertions(assertions)
        assert len(result.assertions) == 1
        assert result.assertions[0]["subject_id"] == result.assertions[0]["object_id"]


class TestConditionSignature:
    """Test condition signature computation for v2 merging."""

    def test_deterministic(self):
        from autoreview.knowledge_graph.dedup import compute_condition_signature

        sig1 = compute_condition_signature("Mus musculus", True, "mouse esc gastruloids")
        sig2 = compute_condition_signature("Mus musculus", True, "mouse esc gastruloids")
        assert sig1 == sig2

    def test_different_organism_different_signature(self):
        from autoreview.knowledge_graph.dedup import compute_condition_signature

        sig_mouse = compute_condition_signature("Mus musculus", True, "esc gastruloids")
        sig_human = compute_condition_signature("Homo sapiens", True, "ipsc organoids")
        assert sig_mouse != sig_human

    def test_different_in_vitro_different_signature(self):
        from autoreview.knowledge_graph.dedup import compute_condition_signature

        sig_vitro = compute_condition_signature("Mus musculus", True, "esc gastruloids")
        sig_vivo = compute_condition_signature("Mus musculus", False, "mouse embryo")
        assert sig_vitro != sig_vivo

    def test_case_insensitive(self):
        from autoreview.knowledge_graph.dedup import compute_condition_signature

        sig1 = compute_condition_signature("Mus musculus", True, "Mouse ESC Gastruloids")
        sig2 = compute_condition_signature("mus musculus", True, "mouse esc gastruloids")
        assert sig1 == sig2

    def test_none_values_hash_consistently(self):
        from autoreview.knowledge_graph.dedup import compute_condition_signature

        sig1 = compute_condition_signature(None, None, None)
        sig2 = compute_condition_signature(None, None, None)
        assert sig1 == sig2

    def test_none_vs_populated_different(self):
        from autoreview.knowledge_graph.dedup import compute_condition_signature

        sig_none = compute_condition_signature(None, None, None)
        sig_mouse = compute_condition_signature("Mus musculus", True, "gastruloids")
        assert sig_none != sig_mouse


class TestModelSystemNormalization:
    """Test fuzzy bucketing of model system strings."""

    def test_identical_strings(self):
        from autoreview.knowledge_graph.dedup import ModelSystemRegistry

        reg = ModelSystemRegistry()
        cls1 = reg.normalize("mouse ESC gastruloids")
        cls2 = reg.normalize("mouse ESC gastruloids")
        assert cls1 == cls2

    def test_synonym_merge(self):
        from autoreview.knowledge_graph.dedup import ModelSystemRegistry

        reg = ModelSystemRegistry()
        cls1 = reg.normalize("mouse ESC gastruloids")
        cls2 = reg.normalize("mESC-derived gastruloids")
        assert cls1 == cls2

    def test_different_systems_separate(self):
        from autoreview.knowledge_graph.dedup import ModelSystemRegistry

        reg = ModelSystemRegistry()
        cls1 = reg.normalize("mouse ESC gastruloids")
        cls2 = reg.normalize("zebrafish embryo")
        assert cls1 != cls2

    def test_none_returns_empty(self):
        from autoreview.knowledge_graph.dedup import ModelSystemRegistry

        reg = ModelSystemRegistry()
        assert reg.normalize(None) == ""
        assert reg.normalize("") == ""

    def test_case_insensitive(self):
        from autoreview.knowledge_graph.dedup import ModelSystemRegistry

        reg = ModelSystemRegistry()
        cls1 = reg.normalize("Mouse Embryo")
        cls2 = reg.normalize("mouse embryo")
        assert cls1 == cls2


class TestMergeAssertionsV2:
    """Test condition-aware assertion merging."""

    def _make_assertion(
        self,
        subject_id: str = "ent1",
        object_id: str = "ent2",
        predicate: str = "induces",
        direction: str = "positive",
        draft_id: str = "a_001",
        paper_id: str = "p1",
        organism: str | None = "Mus musculus",
        model_system: str | None = "mouse ESC gastruloids",
        in_vitro: bool | None = True,
        conditions: dict | None = None,
    ) -> dict:
        return {
            "subject_id": subject_id,
            "object_id": object_id,
            "predicate": predicate,
            "direction": direction,
            "assertion_type": "mechanistic_causal",
            "draft_id": draft_id,
            "evidence_unit_ids": [f"e_{draft_id}"],
            "paper_id": paper_id,
            "publication_date": "2023-01-15",
            "organism": organism,
            "model_system": model_system,
            "in_vitro": in_vitro,
            "conditions": conditions or {},
            "certainty": "high",
            "section_source": "primary_empirical",
        }

    def test_same_spo_same_conditions_merge(self):
        from autoreview.knowledge_graph.dedup import merge_assertions_v2

        assertions = [
            self._make_assertion(draft_id="a_001", paper_id="p1"),
            self._make_assertion(draft_id="a_002", paper_id="p2"),
        ]
        result = merge_assertions_v2(assertions)
        assert len(result.assertions) == 1
        assert len(result.assertions[0]["source_assertions"]) == 2

    def test_same_spo_different_organism_separate(self):
        from autoreview.knowledge_graph.dedup import merge_assertions_v2

        assertions = [
            self._make_assertion(draft_id="a_001", paper_id="p1", organism="Mus musculus"),
            self._make_assertion(
                draft_id="a_002",
                paper_id="p2",
                organism="Homo sapiens",
                model_system="human iPSC organoids",
            ),
        ]
        result = merge_assertions_v2(assertions)
        assert len(result.assertions) == 2

    def test_same_spo_different_in_vitro_separate(self):
        from autoreview.knowledge_graph.dedup import merge_assertions_v2

        assertions = [
            self._make_assertion(draft_id="a_001", paper_id="p1", in_vitro=True),
            self._make_assertion(
                draft_id="a_002",
                paper_id="p2",
                in_vitro=False,
                model_system="mouse embryo",
            ),
        ]
        result = merge_assertions_v2(assertions)
        assert len(result.assertions) == 2

    def test_condition_signature_on_merged(self):
        from autoreview.knowledge_graph.dedup import merge_assertions_v2

        assertions = [
            self._make_assertion(draft_id="a_001", paper_id="p1"),
        ]
        result = merge_assertions_v2(assertions)
        assert result.assertions[0].get("condition_signature") is not None
        assert len(result.assertions[0]["condition_signature"]) == 12

    def test_condition_context_accumulated(self):
        from autoreview.knowledge_graph.dedup import merge_assertions_v2

        assertions = [
            self._make_assertion(
                draft_id="a_001",
                paper_id="p1",
                conditions={"cell_type": ["mESC"], "treatment": ["10 ng/mL BMP4"]},
            ),
            self._make_assertion(
                draft_id="a_002",
                paper_id="p2",
                conditions={
                    "cell_type": ["E14Tg2a"],
                    "treatment": ["3 µM CHIR99021"],
                },
            ),
        ]
        result = merge_assertions_v2(assertions)
        assert len(result.assertions) == 1
        ctx = result.assertions[0]["condition_context"]
        assert "mESC" in ctx["cell_types"]
        assert "E14Tg2a" in ctx["cell_types"]
        assert "10 ng/mL BMP4" in ctx["treatments"]
        assert "3 µM CHIR99021" in ctx["treatments"]

    def test_direction_conflict_within_context(self):
        from autoreview.knowledge_graph.dedup import merge_assertions_v2

        assertions = [
            self._make_assertion(draft_id="a_001", paper_id="p1", direction="positive"),
            self._make_assertion(draft_id="a_002", paper_id="p2", direction="negative"),
        ]
        result = merge_assertions_v2(assertions)
        assert len(result.assertions) == 1
        assert result.assertions[0]["direction_conflict"] is True

    def test_v4_data_without_conditions(self):
        from autoreview.knowledge_graph.dedup import merge_assertions_v2

        assertions = [
            self._make_assertion(
                draft_id="a_001",
                paper_id="p1",
                organism=None,
                model_system=None,
                in_vitro=None,
            ),
            self._make_assertion(
                draft_id="a_002",
                paper_id="p2",
                organism=None,
                model_system=None,
                in_vitro=None,
            ),
        ]
        result = merge_assertions_v2(assertions)
        assert len(result.assertions) == 1

    def test_v1_merge_still_works(self):
        from autoreview.knowledge_graph.dedup import merge_assertions

        assertions = [
            {
                "subject_id": "ent1",
                "object_id": "ent2",
                "predicate": "induces",
                "direction": "positive",
                "assertion_type": "mechanistic_causal",
                "draft_id": "a_001",
                "evidence_unit_ids": ["e_001"],
                "paper_id": "p1",
                "publication_date": "2023-01-15",
            },
        ]
        result = merge_assertions(assertions)
        assert len(result.assertions) == 1
