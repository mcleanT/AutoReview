"""Tests for pre-flight validation checks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from autoreview.pipeline.preflight import check_api_keys, check_output_dir


class TestCheckApiKeys:
    def test_present_key_passes(self) -> None:
        with patch.dict("os.environ", {"MY_API_KEY": "secret"}, clear=False):
            result = check_api_keys(["MY_API_KEY"])

        assert result.ok is True
        assert result.missing == []

    def test_missing_key_fails(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = check_api_keys(["NONEXISTENT_KEY"])

        assert result.ok is False
        assert "NONEXISTENT_KEY" in result.missing

    def test_multiple_keys_partial_failure(self) -> None:
        with patch.dict("os.environ", {"PRESENT_KEY": "value"}, clear=True):
            result = check_api_keys(["PRESENT_KEY", "ABSENT_KEY"])

        assert result.ok is False
        assert "ABSENT_KEY" in result.missing
        assert "PRESENT_KEY" not in result.missing


class TestCheckOutputDir:
    def test_writable_dir_passes(self, tmp_path: Path) -> None:
        result = check_output_dir(tmp_path)

        assert result.ok is True
        assert result.missing == []

    def test_nonexistent_dir_fails(self) -> None:
        result = check_output_dir(Path("/nonexistent/path/surely"))

        assert result.ok is False
        assert len(result.missing) > 0
