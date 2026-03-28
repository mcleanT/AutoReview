"""Tests for the HL-MRF inference engine (hlmrf.py)."""

from __future__ import annotations

import pytest

from autoreview.knowledge_graph.hlmrf import GroundRule, HLMRFEngine, RuleTemplate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_engine(*rules: GroundRule, **inits: float) -> HLMRFEngine:
    """Convenience: build an engine with given rules and variable inits."""
    engine = HLMRFEngine(max_iter=500, tol=1e-8)
    # Collect all variables referenced by rules
    all_vars: set[str] = set()
    for rule in rules:
        all_vars.add(rule.head_var)
        all_vars.update(rule.body_vars)
    for var in sorted(all_vars):
        engine.add_variable(var, init=inits.get(var, 0.5))
    for rule in rules:
        engine.add_ground_rule(rule)
    return engine


# ---------------------------------------------------------------------------
# Test 1: Single unary rule pushes variable toward target
# ---------------------------------------------------------------------------


def test_unary_rule_pushes_toward_target() -> None:
    """Unary rule with target 1.0 and weight 10 should push init-0.5 var above 0.9."""
    rule = GroundRule(
        head_var="x",
        body_vars=[],
        body_coeffs=[],
        target=1.0,
        weight=10.0,
        rule_type="unary",
    )
    engine = HLMRFEngine()
    engine.add_variable("x", init=0.5)
    engine.add_ground_rule(rule)

    result = engine.solve()

    assert "x" in result
    assert result["x"] > 0.9, f"Expected > 0.9, got {result['x']:.4f}"
    assert 0.0 <= result["x"] <= 1.0


def test_unary_rule_pushes_toward_low_target() -> None:
    """Unary rule with target 0.0 should push variable below 0.1."""
    rule = GroundRule(
        head_var="y",
        body_vars=[],
        body_coeffs=[],
        target=0.0,
        weight=10.0,
        rule_type="unary",
    )
    engine = HLMRFEngine()
    engine.add_variable("y", init=0.5)
    engine.add_ground_rule(rule)

    result = engine.solve()

    assert result["y"] < 0.1, f"Expected < 0.1, got {result['y']:.4f}"


# ---------------------------------------------------------------------------
# Test 2: Contradiction rule — two vars + contradiction → weighted compromise
# ---------------------------------------------------------------------------


def test_contradiction_rule_resolves_conflict() -> None:
    """Strong unary rules + contradiction constraint → sum < 1.8 and high-weight var wins."""
    # x strongly pulled to 1.0 (weight 20), y strongly pulled to 1.0 (weight 5)
    # Contradiction: x + y <= 1.8, enforced with weight 50
    rule_x = GroundRule(
        head_var="x",
        body_vars=[],
        body_coeffs=[],
        target=1.0,
        weight=20.0,
        rule_type="unary",
    )
    rule_y = GroundRule(
        head_var="y",
        body_vars=[],
        body_coeffs=[],
        target=1.0,
        weight=5.0,
        rule_type="unary",
    )
    contradiction = GroundRule(
        head_var="x",
        body_vars=["y"],
        body_coeffs=[1.0],
        target=1.8,
        weight=500.0,
        rule_type="contradiction",
    )

    engine = make_engine(rule_x, rule_y, contradiction, x=0.5, y=0.5)
    result = engine.solve()

    x_val = result["x"]
    y_val = result["y"]

    # Squared hinge is a soft penalty, so the sum can slightly exceed 1.8 at equilibrium.
    # The key checks: sum is well below unconstrained 2.0, and the constraint pushes it
    # toward 1.8 (within 0.05 with weight 500).
    assert x_val + y_val <= 1.85, f"Sum {x_val + y_val:.4f} not constrained toward 1.8"
    # Higher-weight variable (x, weight 20) should beat lower-weight (y, weight 5)
    assert x_val > y_val, f"Expected x ({x_val:.4f}) > y ({y_val:.4f})"
    assert 0.0 <= x_val <= 1.0
    assert 0.0 <= y_val <= 1.0


# ---------------------------------------------------------------------------
# Test 3: Composition rule — chain should boost sparse direct edge
# ---------------------------------------------------------------------------


def test_composition_rule_boosts_transitive() -> None:
    """Composition via strong chain A->B and B->C should boost A->C above its unary prior."""
    # Unary rules: A_B=1.0, B_C=1.0, A_C=0.3 (sparse/weak direct)
    rule_ab = GroundRule(
        head_var="A_B",
        body_vars=[],
        body_coeffs=[],
        target=1.0,
        weight=20.0,
        rule_type="unary",
    )
    rule_bc = GroundRule(
        head_var="B_C",
        body_vars=[],
        body_coeffs=[],
        target=1.0,
        weight=20.0,
        rule_type="unary",
    )
    rule_ac_weak = GroundRule(
        head_var="A_C",
        body_vars=[],
        body_coeffs=[],
        target=0.3,
        weight=2.0,
        rule_type="unary",
    )
    # Composition: if A->B and B->C are strong, A->C should be high
    # target=1.0 means body_strength * 1.0 is the expected floor
    composition = GroundRule(
        head_var="A_C",
        body_vars=["A_B", "B_C"],
        body_coeffs=[],
        target=1.0,
        weight=15.0,
        rule_type="composition",
    )

    engine = HLMRFEngine()
    for var in ["A_B", "B_C", "A_C"]:
        engine.add_variable(var, init=0.5)
    for rule in [rule_ab, rule_bc, rule_ac_weak, composition]:
        engine.add_ground_rule(rule)

    result = engine.solve()

    a_c = result["A_C"]
    assert a_c > 0.4, f"A_C should be boosted above unary prior 0.3, got {a_c:.4f}"
    assert 0.0 <= a_c <= 1.0
    assert 0.0 <= result["A_B"] <= 1.0
    assert 0.0 <= result["B_C"] <= 1.0


# ---------------------------------------------------------------------------
# Test 4: All results remain in [0, 1] even with out-of-range targets
# ---------------------------------------------------------------------------


def test_bounded_values_with_extreme_targets() -> None:
    """All result values must be within [0, 1] regardless of target values."""
    rules = [
        GroundRule(
            head_var="a",
            body_vars=[],
            body_coeffs=[],
            target=5.0,
            weight=100.0,
            rule_type="unary",
        ),
        GroundRule(
            head_var="b",
            body_vars=[],
            body_coeffs=[],
            target=-3.0,
            weight=100.0,
            rule_type="unary",
        ),
        GroundRule(
            head_var="c",
            body_vars=["a"],
            body_coeffs=[2.0],
            target=0.5,
            weight=50.0,
            rule_type="contradiction",
        ),
    ]

    engine = HLMRFEngine()
    for var in ["a", "b", "c"]:
        engine.add_variable(var, init=0.5)
    for rule in rules:
        engine.add_ground_rule(rule)

    result = engine.solve()

    for var, val in result.items():
        assert 0.0 <= val <= 1.0, f"Variable {var} = {val:.4f} out of [0, 1]"


# ---------------------------------------------------------------------------
# Test 5: Empty engine returns init values
# ---------------------------------------------------------------------------


def test_empty_engine_returns_no_variables() -> None:
    """Engine with no variables at all returns empty dict."""
    engine = HLMRFEngine()
    result = engine.solve()
    assert result == {}


def test_no_rules_returns_init_values() -> None:
    """Engine with variables but no rules returns the initial values unchanged."""
    engine = HLMRFEngine()
    engine.add_variable("x", init=0.3)
    engine.add_variable("y", init=0.7)

    result = engine.solve()

    assert abs(result["x"] - 0.3) < 1e-9, f"Expected 0.3, got {result['x']}"
    assert abs(result["y"] - 0.7) < 1e-9, f"Expected 0.7, got {result['y']}"


# ---------------------------------------------------------------------------
# Test 6: n_variables and n_rules properties count correctly
# ---------------------------------------------------------------------------


def test_properties_count_correctly() -> None:
    """n_variables and n_rules return the correct counts."""
    engine = HLMRFEngine()
    assert engine.n_variables == 0
    assert engine.n_rules == 0

    engine.add_variable("a")
    engine.add_variable("b")
    engine.add_variable("c")
    assert engine.n_variables == 3
    assert engine.n_rules == 0

    rule1 = GroundRule(
        head_var="a",
        body_vars=["b"],
        body_coeffs=[1.0],
        target=1.0,
        weight=1.0,
        rule_type="contradiction",
    )
    rule2 = GroundRule(
        head_var="c",
        body_vars=[],
        body_coeffs=[],
        target=0.5,
        weight=2.0,
        rule_type="unary",
    )
    engine.add_ground_rule(rule1)
    engine.add_ground_rule(rule2)
    assert engine.n_variables == 3
    assert engine.n_rules == 2


def test_duplicate_variable_not_added_twice() -> None:
    """Adding the same variable name twice does not increase n_variables."""
    engine = HLMRFEngine()
    engine.add_variable("x", init=0.3)
    engine.add_variable("x", init=0.9)  # duplicate — should be ignored
    assert engine.n_variables == 1


# ---------------------------------------------------------------------------
# Test 7: RuleTemplate dataclass
# ---------------------------------------------------------------------------


def test_rule_template_dataclass() -> None:
    """RuleTemplate stores its fields correctly."""
    tmpl = RuleTemplate(
        name="chain_rule",
        weight=5.0,
        rule_type="composition",
        description="Transitive chain inference",
    )
    assert tmpl.name == "chain_rule"
    assert tmpl.weight == 5.0
    assert tmpl.rule_type == "composition"
    assert tmpl.description == "Transitive chain inference"


def test_rule_template_default_description() -> None:
    """RuleTemplate description defaults to empty string."""
    tmpl = RuleTemplate(name="t", weight=1.0, rule_type="unary")
    assert tmpl.description == ""


# ---------------------------------------------------------------------------
# Test 8: GroundRule is frozen (immutable)
# ---------------------------------------------------------------------------


def test_ground_rule_is_frozen() -> None:
    """GroundRule is a frozen dataclass — mutation should raise FrozenInstanceError."""
    rule = GroundRule(
        head_var="x",
        body_vars=[],
        body_coeffs=[],
        target=1.0,
        weight=1.0,
        rule_type="unary",
    )
    with pytest.raises(Exception):  # FrozenInstanceError is a subclass of AttributeError
        rule.weight = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test 9: Multiple unary rules on same variable reach weighted average
# ---------------------------------------------------------------------------


def test_multiple_unary_rules_weighted_average() -> None:
    """Two unary rules pulling in opposite directions settle near the weighted average."""
    # weight 1 pulling to 0, weight 3 pulling to 1 → expected ~0.75
    rule_low = GroundRule(
        head_var="x",
        body_vars=[],
        body_coeffs=[],
        target=0.0,
        weight=1.0,
        rule_type="unary",
    )
    rule_high = GroundRule(
        head_var="x",
        body_vars=[],
        body_coeffs=[],
        target=1.0,
        weight=3.0,
        rule_type="unary",
    )
    engine = HLMRFEngine()
    engine.add_variable("x", init=0.5)
    engine.add_ground_rule(rule_low)
    engine.add_ground_rule(rule_high)

    result = engine.solve()

    # Weighted minimum: d/dx [1*(x-0)^2 + 3*(x-1)^2] = 0 → 2x + 6x - 6 = 0 → x = 6/8 = 0.75
    assert abs(result["x"] - 0.75) < 1e-4, f"Expected ~0.75, got {result['x']:.4f}"
