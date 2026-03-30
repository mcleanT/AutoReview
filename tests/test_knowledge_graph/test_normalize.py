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
