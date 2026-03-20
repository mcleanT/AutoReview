"""Tests for snapshot integrity: schema version and checksum validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoreview.models.knowledge_base import KnowledgeBase, SnapshotIntegrityError


@pytest.fixture
def kb(tmp_path: Path) -> KnowledgeBase:
    """Return a minimal KnowledgeBase pointed at tmp_path."""
    instance = KnowledgeBase(topic="integrity test topic", domain="biomedical")
    instance.output_dir = str(tmp_path)
    return instance


def _latest_path(tmp_path: Path) -> Path:
    return tmp_path / "snapshots" / "latest.json"


class TestSaveSnapshot:
    def test_save_includes_schema_version(self, kb: KnowledgeBase, tmp_path: Path) -> None:
        kb.save_snapshot("test_node")
        data = json.loads(_latest_path(tmp_path).read_text())
        assert "_schema_version" in data
        assert isinstance(data["_schema_version"], int)

    def test_save_includes_checksum(self, kb: KnowledgeBase, tmp_path: Path) -> None:
        kb.save_snapshot("test_node")
        data = json.loads(_latest_path(tmp_path).read_text())
        assert "_checksum" in data
        assert isinstance(data["_checksum"], str)
        assert len(data["_checksum"]) == 64  # SHA256 hex digest length


class TestLoadSnapshot:
    def test_load_accepts_valid_snapshot(self, kb: KnowledgeBase, tmp_path: Path) -> None:
        kb.save_snapshot("test_node")
        loaded = KnowledgeBase.load_snapshot(str(_latest_path(tmp_path)))
        assert loaded.topic == "integrity test topic"

    def test_load_validates_checksum(self, kb: KnowledgeBase, tmp_path: Path) -> None:
        kb.save_snapshot("test_node")
        path = _latest_path(tmp_path)

        # Corrupt the topic field in the saved JSON
        data = json.loads(path.read_text())
        data["topic"] = "tampered topic — this should fail"
        path.write_text(json.dumps(data, indent=2))

        with pytest.raises(SnapshotIntegrityError):
            KnowledgeBase.load_snapshot(str(path))

    def test_load_legacy_snapshot_no_checksum(self, kb: KnowledgeBase, tmp_path: Path) -> None:
        """Legacy snapshots without _checksum should load with a warning, not raise."""
        kb.save_snapshot("test_node")
        path = _latest_path(tmp_path)

        data = json.loads(path.read_text())
        data.pop("_checksum", None)
        data.pop("_schema_version", None)
        path.write_text(json.dumps(data, indent=2))

        with pytest.warns(UserWarning, match="no checksum"):
            loaded = KnowledgeBase.load_snapshot(str(path))

        assert loaded.topic == "integrity test topic"
