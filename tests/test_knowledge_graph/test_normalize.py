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
