"""Tests for cost analysis utilities."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from autoreview.evaluation.cost_analyzer import compare_costs, compute_cost
from autoreview.pipeline.nodes import _GlobalTokenAccumulator


def test_compute_cost_sonnet() -> None:
    usage = {"total": {"input_tokens": 1_000_000, "output_tokens": 100_000}}
    result = compute_cost(usage, "claude-sonnet-4-6")
    assert result.total_input_tokens == 1_000_000
    assert result.total_output_tokens == 100_000
    assert result.input_cost_usd == 3.0  # 1M * 3.0 / 1M
    assert result.output_cost_usd == 1.5  # 100K * 15.0 / 1M
    assert result.total_cost_usd == 4.5


def test_compute_cost_unknown_model() -> None:
    usage = {"total": {"input_tokens": 500_000, "output_tokens": 50_000}}
    result = compute_cost(usage, "unknown-model-xyz")
    assert result.total_cost_usd == 0.0
    assert result.model == "unknown-model-xyz"


def test_compare_costs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_a = Path(tmpdir) / "run_a"
        dir_b = Path(tmpdir) / "run_b"
        dir_a.mkdir()
        dir_b.mkdir()

        data_a = {
            "model": "claude-sonnet-4-6",
            "total": {"input_tokens": 1_000_000, "output_tokens": 100_000},
        }
        data_b = {
            "model": "claude-haiku-4-5",
            "total": {"input_tokens": 1_000_000, "output_tokens": 100_000},
        }

        file_a = dir_a / "token_usage.json"
        file_b = dir_b / "token_usage.json"
        file_a.write_text(json.dumps(data_a))
        file_b.write_text(json.dumps(data_b))

        rows = compare_costs([file_a, file_b])
        assert len(rows) == 2
        assert rows[0].label == "run_a"
        assert rows[0].cost.total_cost_usd > rows[1].cost.total_cost_usd


def test_token_summary_structure() -> None:
    acc = _GlobalTokenAccumulator(budget=None)
    acc.add_node("search", 1000, 500, cache_read=200, cache_creation=100)
    acc.add_node("search", 500, 250, cache_read=100, cache_creation=50)
    acc.add_node("extraction", 2000, 1000)

    summary = acc.token_summary()
    assert "per_node" in summary
    assert "total" in summary
    assert summary["per_node"]["search"]["input_tokens"] == 1500
    assert summary["per_node"]["search"]["output_tokens"] == 750
    assert summary["per_node"]["extraction"]["input_tokens"] == 2000
    assert summary["total"]["input_tokens"] == 3500
    assert summary["total"]["output_tokens"] == 1750
