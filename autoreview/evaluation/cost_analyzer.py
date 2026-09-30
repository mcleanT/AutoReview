"""Cost analysis utilities for token usage data."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from pathlib import Path

from autoreview.models.base import AutoReviewModel

logger = structlog.get_logger()

# Pricing per 1M tokens: (input_price_usd, output_price_usd). Exact model ids only;
# a model absent here is UNPRICED (cost fields None), never priced at $0.
# Current-model prices from the Claude API model catalog (2026-09): Opus 5.5 $4/$20,
# Opus 5 and Opus 4.8 $5/$25, Sonnet 5 and 5.5 $2/$10, Fable 5 and 5.1 $10/$50,
# Haiku 4.5 $1/$5 (previously listed here at $0.80/$4, the Haiku 3.5 price).
PRICING: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-4-20250514": (15.0, 75.0),
    "claude-opus-4-6": (15.0, 75.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


class CostSummary(AutoReviewModel):
    model: str
    total_input_tokens: int
    total_output_tokens: int
    priced: bool
    # None when the model has no known price: unknown cost, not zero cost.
    input_cost_usd: float | None
    output_cost_usd: float | None
    total_cost_usd: float | None


class CostComparisonRow(AutoReviewModel):
    label: str
    model: str
    cost: CostSummary


def compute_cost(token_usage: dict[str, Any], model: str) -> CostSummary:
    """Compute cost from a token_usage dict (as written by runner) and model name."""
    total = token_usage.get("total", {})
    input_tokens = total.get("input_tokens", 0)
    output_tokens = total.get("output_tokens", 0)

    prices = PRICING.get(model)
    if prices is None:
        logger.warning(
            "cost_analyzer.unknown_model", model=model, reason="no price; cost left None"
        )
        return CostSummary(
            model=model,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            priced=False,
            input_cost_usd=None,
            output_cost_usd=None,
            total_cost_usd=None,
        )

    input_price, output_price = prices
    input_cost = input_tokens * input_price / 1_000_000
    output_cost = output_tokens * output_price / 1_000_000

    return CostSummary(
        model=model,
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        priced=True,
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
