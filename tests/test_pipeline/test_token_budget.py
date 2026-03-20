"""Tests for TokenBudgetMonitor and BudgetAction."""

from __future__ import annotations

from autoreview.pipeline.dag import BudgetAction, TokenBudgetMonitor


def test_below_80_percent_continues() -> None:
    monitor = TokenBudgetMonitor(budget=1000)
    assert monitor.check(500) == BudgetAction.CONTINUE


def test_at_80_percent_warns() -> None:
    monitor = TokenBudgetMonitor(budget=1000)
    assert monitor.check(800) == BudgetAction.WARN


def test_at_95_percent_degrades() -> None:
    monitor = TokenBudgetMonitor(budget=1000)
    assert monitor.check(950) == BudgetAction.DEGRADE


def test_at_100_percent_saves_and_stops() -> None:
    monitor = TokenBudgetMonitor(budget=1000)
    assert monitor.check(1000) == BudgetAction.SAVE_AND_STOP


def test_no_budget_always_continues() -> None:
    monitor = TokenBudgetMonitor(budget=None)
    assert monitor.check(999999) == BudgetAction.CONTINUE


def test_warn_fires_only_once() -> None:
    """WARN should only fire on the first crossing of the 80% threshold."""
    monitor = TokenBudgetMonitor(budget=1000)
    first = monitor.check(800)
    second = monitor.check(850)
    assert first == BudgetAction.WARN
    assert second == BudgetAction.CONTINUE


def test_degrade_supersedes_warn_flag() -> None:
    """After WARN is emitted, crossing 95% should still return DEGRADE."""
    monitor = TokenBudgetMonitor(budget=1000)
    monitor.check(800)  # triggers WARN, sets _warned=True
    assert monitor.check(950) == BudgetAction.DEGRADE


def test_save_and_stop_on_over_budget() -> None:
    """Tokens exceeding the budget should also return SAVE_AND_STOP."""
    monitor = TokenBudgetMonitor(budget=1000)
    assert monitor.check(1500) == BudgetAction.SAVE_AND_STOP


def test_exactly_below_80_continues() -> None:
    """799 tokens out of 1000 (79.9%) should still be CONTINUE."""
    monitor = TokenBudgetMonitor(budget=1000)
    assert monitor.check(799) == BudgetAction.CONTINUE


def test_exactly_below_95_warns_not_degrades() -> None:
    """949 tokens out of 1000 (94.9%) with prior warn already set → CONTINUE."""
    monitor = TokenBudgetMonitor(budget=1000)
    monitor.check(800)  # WARN + set _warned=True
    assert monitor.check(949) == BudgetAction.CONTINUE


def test_zero_budget_continues() -> None:
    """A budget of 0 should never divide by zero — always return CONTINUE."""
    monitor = TokenBudgetMonitor(budget=0)
    action = monitor.check(tokens_used=100)
    assert action == BudgetAction.CONTINUE
