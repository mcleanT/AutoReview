"""Cost analysis utilities for token usage data."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path

from autoreview.models.base import AutoReviewModel

logger = structlog.get_logger()

# Pricing per 1M tokens: (input_price_usd, output_price_usd)
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-20250514": (15.0, 75.0),
    "claude-opus-4-6": (15.0, 75.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.8, 4.0),
    "claude-haiku-4-5": (0.8, 4.0),
}


class CostSummary(AutoReviewModel):
    model: str
    total_input_tokens: int
    total_output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


class CostComparisonRow(AutoReviewModel):
    label: str
    model: str
    cost: CostSummary


def compute_cost(token_usage: dict, model: str) -> CostSummary:
    """Compute cost from a token_usage dict (as written by runner) and model name."""
    total = token_usage.get("total", {})
    input_tokens = total.get("input_tokens", 0)
    output_tokens = total.get("output_tokens", 0)

    input_price, output_price = PRICING.get(model, (0.0, 0.0))
    if model not in PRICING:
        logger.warning("cost_analyzer.unknown_model", model=model)

    input_cost = input_tokens * input_price / 1_000_000
    output_cost = output_tokens * output_price / 1_000_000

    return CostSummary(
        model=model,
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        input_cost_usd=round(input_cost, 4),
        output_cost_usd=round(output_cost, 4),
        total_cost_usd=round(input_cost + output_cost, 4),
    )


def compare_costs(usage_files: list[Path]) -> list[CostComparisonRow]:
    """Load multiple token_usage.json files and produce a cost comparison."""
    rows: list[CostComparisonRow] = []
    for path in usage_files:
        data = json.loads(path.read_text())
        model = data.get("model", "unknown")
        cost = compute_cost(data, model)
        rows.append(CostComparisonRow(label=path.parent.name, model=model, cost=cost))
    return rows
