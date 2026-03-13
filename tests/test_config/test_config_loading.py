from __future__ import annotations

from autoreview.config import deep_merge, load_config


class TestDeepMerge:
    def test_simple_override(self):
        result = deep_merge({"a": 1, "b": 2}, {"b": 3})
        assert result == {"a": 1, "b": 3}

    def test_nested_merge(self):
        base = {"search": {"date_range": "2015-2025", "max_results": 500}}
        override = {"search": {"date_range": "2020-2025"}}
        result = deep_merge(base, override)
        assert result["search"]["date_range"] == "2020-2025"
        assert result["search"]["max_results"] == 500

    def test_new_keys_added(self):
        result = deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}


class TestConfigLoading:
    def test_load_biomedical(self):
        config = load_config("biomedical")
        assert config.domain == "biomedical"
        assert "pubmed" in config.databases["primary"]
        assert config.writing.citation_format == "vancouver"

    def test_load_cs_ai(self):
        config = load_config("cs_ai")
        assert config.domain == "cs_ai"
        assert config.search.date_range == "2018-2025"

    def test_load_chemistry(self):
        config = load_config("chemistry")
        assert config.domain == "chemistry"
        assert config.search.max_results_per_source == 400

    def test_load_unknown_domain_uses_defaults(self):
        config = load_config("unknown_domain")
        assert config.domain == "unknown_domain"

    def test_explicit_overrides(self):
        config = load_config("biomedical", overrides={"search": {"date_range": "2020-2025"}})
        assert config.search.date_range == "2020-2025"
        # Other values preserved
        assert config.search.relevance_threshold == 3

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("AUTOREVIEW_SEARCH__DATE_RANGE", "2022-2025")
        config = load_config("biomedical")
        assert config.search.date_range == "2022-2025"
