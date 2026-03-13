from __future__ import annotations

import pytest
from pydantic import ValidationError

from autoreview.config.models import SearchConfig


class TestDateRangeValidator:
    def test_standard_range(self):
        cfg = SearchConfig(date_range="2015-2025")
        assert cfg.date_range == "2015-2025"

    def test_whitespace_stripped(self):
        cfg = SearchConfig(date_range=" 2015 - 2025 ")
        assert cfg.date_range == "2015-2025"

    def test_open_start(self):
        cfg = SearchConfig(date_range="-2020")
        assert cfg.date_range == "-2020"

    def test_open_end(self):
        cfg = SearchConfig(date_range="2020-")
        assert cfg.date_range == "2020-"

    def test_same_year(self):
        cfg = SearchConfig(date_range="2020-2020")
        assert cfg.date_range == "2020-2020"

    def test_empty_string(self):
        cfg = SearchConfig(date_range="")
        assert cfg.date_range == ""

    def test_reversed_range_rejected(self):
        with pytest.raises(ValidationError, match="date_range"):
            SearchConfig(date_range="2025-2015")

    def test_malformed_rejected(self):
        with pytest.raises(ValidationError, match="date_range"):
            SearchConfig(date_range="2015-01-2025")

    def test_non_numeric_rejected(self):
        with pytest.raises(ValidationError, match="date_range"):
            SearchConfig(date_range="abc-def")

    def test_bare_dash_rejected(self):
        with pytest.raises(ValidationError, match="date_range"):
            SearchConfig(date_range="-")

    def test_default_unchanged(self):
        cfg = SearchConfig()
        assert cfg.date_range == "2015-2025"
