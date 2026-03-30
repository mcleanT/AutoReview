"""Tests for post-extraction claim normalization."""

from __future__ import annotations


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

        assert (
            flag_for_llm_decomposition(
                "self-organization of human gastruloids into homogenous subpopulations of endoderm and mesoderm"
            )
            is True
        )
        assert flag_for_llm_decomposition("mesoderm differentiation") is False

    def test_slash_not_in_units(self):
        from autoreview.knowledge_graph.normalize import decompose_object

        # "ng/mL" should not be split
        result = decompose_object("10 ng/mL BMP4 treatment effect")
        assert len(result) == 1
