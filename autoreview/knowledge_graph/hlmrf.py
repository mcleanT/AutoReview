"""Hinge-Loss Markov Random Field (HL-MRF) inference engine.

Implements convex optimization over continuous [0, 1] truth values.
Variables are soft-truth assignments; rules are weighted soft constraints.
Inference minimizes the weighted sum of squared hinge losses using L-BFGS-B.

No domain dependencies — pure math module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from scipy.optimize import minimize

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class GroundRule:
    """A grounded (instantiated) rule in the HL-MRF.

    Args:
        head_var: Name of the head variable (what is being constrained).
        body_vars: Names of body variables (antecedents).
        body_coeffs: Coefficients for each body variable.
        target: Target value that the head should satisfy given the body.
        weight: Weight of this rule's penalty in the objective.
        rule_type: One of "unary", "contradiction", "composition".
    """

    head_var: str
    body_vars: list[str]
    body_coeffs: list[float]
    target: float
    weight: float
    rule_type: str  # "unary", "contradiction", "composition"


@dataclass
class RuleTemplate:
    """A template for generating ground rules from graph structure.

    Args:
        name: Identifying name for this template.
        weight: Weight to assign ground rules generated from this template.
        rule_type: One of "unary", "contradiction", "composition".
        description: Human-readable description of what this rule encodes.
    """

    name: str
    weight: float
    rule_type: str
    description: str = ""


class HLMRFEngine:
    """Hinge-Loss MRF inference engine via convex optimization.

    Variables are continuous truth values in [0, 1].
    Rules are weighted soft constraints penalizing violations with squared hinge loss.
    Inference calls scipy L-BFGS-B to find the MAP truth assignment.

    Example::

        engine = HLMRFEngine()
        engine.add_variable("x", init=0.5)
        engine.add_ground_rule(GroundRule(
            head_var="x", body_vars=[], body_coeffs=[],
            target=1.0, weight=10.0, rule_type="unary"
        ))
        result = engine.solve()
        # result["x"] will be close to 1.0
    """

    def __init__(self, max_iter: int = 200, tol: float = 1e-6) -> None:
        self._max_iter = max_iter
        self._tol = tol
        # Ordered dict: var_name -> (index, init_value)
        self._variables: dict[str, tuple[int, float]] = {}
        self._rules: list[GroundRule] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_variable(self, name: str, init: float = 0.5) -> None:
        """Register a variable with an initial truth value.

        Args:
            name: Unique variable name.
            init: Initial value in [0, 1].
        """
        if name not in self._variables:
            idx = len(self._variables)
            self._variables[name] = (idx, float(init))

    def add_ground_rule(self, rule: GroundRule) -> None:
        """Add a grounded rule to the engine.

        All variables referenced by the rule must already be registered
        via add_variable before calling solve().

        Args:
            rule: The grounded rule to add.
        """
        self._rules.append(rule)

    @property
    def n_variables(self) -> int:
        """Number of registered variables."""
        return len(self._variables)

    @property
    def n_rules(self) -> int:
        """Number of registered ground rules."""
        return len(self._rules)

    def solve(self) -> dict[str, float]:
        """Run MAP inference and return optimal truth values.

        Returns:
            Mapping from variable name to optimal truth value in [0, 1].
        """
        if self.n_variables == 0:
            log.info("hlmrf.solve.empty", n_rules=self.n_rules)
            return {}

        # Build initial value vector
        var_names = sorted(self._variables, key=lambda v: self._variables[v][0])
        x0 = np.array([self._variables[v][1] for v in var_names], dtype=np.float64)

        # Map variable name -> index in x0
        var_idx: dict[str, int] = {v: i for i, v in enumerate(var_names)}

        bounds = [(0.0, 1.0)] * len(var_names)

        if self.n_rules == 0:
            log.info("hlmrf.solve.no_rules", n_variables=self.n_variables)
            return dict(zip(var_names, x0.tolist()))

        def objective_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
            obj = 0.0
            grad = np.zeros_like(x)

            for rule in self._rules:
                hi = var_idx[rule.head_var]

                if rule.rule_type == "unary":
                    # Squared distance to target
                    diff = x[hi] - rule.target
                    obj += rule.weight * diff * diff
                    grad[hi] += 2.0 * rule.weight * diff

                elif rule.rule_type == "contradiction":
                    # Hinge penalty when head + Σ(c_i * body_i) > target
                    val = x[hi]
                    for bv, bc in zip(rule.body_vars, rule.body_coeffs):
                        val += bc * x[var_idx[bv]]
                    violation = max(0.0, val - rule.target)
                    if violation > 0.0:
                        obj += rule.weight * violation * violation
                        grad[hi] += 2.0 * rule.weight * violation
                        for bv, bc in zip(rule.body_vars, rule.body_coeffs):
                            grad[var_idx[bv]] += 2.0 * rule.weight * violation * bc

                elif rule.rule_type == "composition":
                    # AND semantics: weakest body link
                    if rule.body_vars:
                        body_strength = min(x[var_idx[bv]] for bv in rule.body_vars)
                    else:
                        body_strength = 1.0
                    expected = rule.target * body_strength
                    shortfall = max(0.0, expected - x[hi])
                    if shortfall > 0.0:
                        obj += rule.weight * shortfall * shortfall
                        grad[hi] -= 2.0 * rule.weight * shortfall

            return obj, grad

        log.info(
            "hlmrf.solve.start",
            n_variables=self.n_variables,
            n_rules=self.n_rules,
            max_iter=self._max_iter,
        )

        result = minimize(
            objective_and_grad,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": self._max_iter, "ftol": self._tol, "gtol": self._tol},
        )

        log.info(
            "hlmrf.solve.done",
            converged=result.success,
            message=result.message,
            n_iter=result.nit,
            final_obj=float(result.fun),
        )

        return dict(zip(var_names, result.x.tolist()))
