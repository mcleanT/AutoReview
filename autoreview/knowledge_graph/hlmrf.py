"""Hinge-Loss Markov Random Field (HL-MRF) inference engine.

Implements convex optimization over continuous [0, 1] truth values.
Variables are soft-truth assignments; rules are weighted soft constraints.
Inference minimizes the weighted sum of squared hinge losses using L-BFGS-B.

No domain dependencies — pure math module.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

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
        # Convergence metadata — updated after each solve()
        self._last_converged: bool = True
        self._last_n_iterations: int = 0
        self._last_final_objective: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_variable(self, name: str, init: float = 0.5) -> None:
        """Register a variable with an initial truth value.

        Args:
            name: Unique variable name.
            init: Initial value in [0, 1].
        """
        if name in self._variables:
            log.debug("hlmrf.duplicate_variable", name=name)
            return
        idx = len(self._variables)
        self._variables[name] = (idx, max(0.0, min(1.0, float(init))))

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

    @property
    def last_converged(self) -> bool:
        """Whether the last solve() call converged."""
        return self._last_converged

    @property
    def last_n_iterations(self) -> int:
        """Number of iterations in the last solve() call."""
        return self._last_n_iterations

    @property
    def last_final_objective(self) -> float:
        """Final objective value from the last solve() call."""
        return self._last_final_objective

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
                        body_indices = [var_idx[bv] for bv in rule.body_vars]
                        body_vals = [x[bi] for bi in body_indices]
                        argmin_local = int(np.argmin(body_vals))
                        body_strength = body_vals[argmin_local]
                        argmin_idx = body_indices[argmin_local]
                    else:
                        body_strength = 1.0
                        argmin_idx = None
                    expected = rule.target * body_strength
                    shortfall = max(0.0, expected - x[hi])
                    if shortfall > 0.0:
                        obj += rule.weight * shortfall * shortfall
                        grad[hi] -= 2.0 * rule.weight * shortfall
                        if argmin_idx is not None:
                            grad[argmin_idx] += 2.0 * rule.weight * shortfall * rule.target

                elif rule.rule_type == "aggregation":
                    # Squared distance to weighted mean of body vars
                    # body_coeffs are weights (should sum to 1 for a true mean)
                    if rule.body_vars:
                        body_mean = sum(
                            bc * x[var_idx[bv]] for bv, bc in zip(rule.body_vars, rule.body_coeffs)
                        )
                    else:
                        body_mean = rule.target
                    diff = x[hi] - body_mean
                    obj += rule.weight * diff * diff
                    grad[hi] += 2.0 * rule.weight * diff
                    for bv, bc in zip(rule.body_vars, rule.body_coeffs):
                        grad[var_idx[bv]] -= 2.0 * rule.weight * diff * bc

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

        self._last_converged = result.success
        self._last_n_iterations = int(result.nit)
        self._last_final_objective = float(result.fun)

        log.info(
            "hlmrf.solve.done",
            converged=result.success,
            message=result.message,
            n_iter=result.nit,
            final_obj=float(result.fun),
        )

        return dict(zip(var_names, result.x.tolist()))

    def solve_incremental(
        self,
        prior_solution: dict[str, float],
        affected_vars: set[str],
        hop_radius: int = 2,
    ) -> dict[str, float]:
        """Re-solve the MRF using a prior solution as a warm start.

        Only variables within ``hop_radius`` hops of ``affected_vars`` in the
        rule adjacency graph are optimized; all others are frozen at their
        ``prior_solution`` values.  This is substantially cheaper than a full
        re-solve when the number of new variables is small relative to the graph.

        Args:
            prior_solution: Variable name -> value mapping from a previous solve().
                Variables not present in prior_solution fall back to their registered
                init value.
            affected_vars: Set of variable names that have changed (e.g. new edges).
            hop_radius: BFS depth limit from affected_vars when determining which
                additional variables to free.  ``hop_radius=0`` frees only the
                affected vars themselves; ``hop_radius=2`` (default) frees variables
                up to two rule-hops away.

        Returns:
            Mapping from variable name to optimal truth value in [0, 1].
        """
        if self.n_variables == 0:
            return {}

        # Build rule adjacency graph: each variable is adjacent to all other
        # variables that share a rule with it.
        adjacency: dict[str, set[str]] = defaultdict(set)
        for rule in self._rules:
            all_rule_vars = [rule.head_var] + list(rule.body_vars)
            for i, vi in enumerate(all_rule_vars):
                for j, vj in enumerate(all_rule_vars):
                    if i != j:
                        adjacency[vi].add(vj)

        # BFS from affected_vars to find free_vars within hop_radius hops.
        free_vars: set[str] = set()
        frontier = set(affected_vars) & set(self._variables)
        for _ in range(hop_radius + 1):
            free_vars |= frontier
            next_frontier: set[str] = set()
            for v in frontier:
                for neighbor in adjacency.get(v, set()):
                    if neighbor in self._variables and neighbor not in free_vars:
                        next_frontier.add(neighbor)
            frontier = next_frontier

        # If no free vars, return prior solution (everything frozen).
        if not free_vars:
            var_names = sorted(self._variables, key=lambda v: self._variables[v][0])
            result: dict[str, float] = {}
            for v in var_names:
                result[v] = prior_solution.get(v, self._variables[v][1])
            return result

        # Build full variable list in index order.
        var_names = sorted(self._variables, key=lambda v: self._variables[v][0])
        var_idx: dict[str, int] = {v: i for i, v in enumerate(var_names)}

        # Build frozen values vector (used for non-free vars throughout optimization).
        frozen_vals = np.array(
            [prior_solution.get(v, self._variables[v][1]) for v in var_names],
            dtype=np.float64,
        )

        # Free variable indices and initial values.
        free_list = sorted(free_vars, key=lambda v: var_idx[v])
        free_indices = np.array([var_idx[v] for v in free_list], dtype=np.intp)
        x0_free = frozen_vals[free_indices].copy()

        bounds = [(0.0, 1.0)] * len(free_list)

        def objective_and_grad(x_free: np.ndarray) -> tuple[float, np.ndarray]:
            # Reconstruct full x vector: frozen vars hold prior, free vars use x_free.
            x = frozen_vals.copy()
            x[free_indices] = x_free

            obj = 0.0
            grad_full = np.zeros(len(var_names), dtype=np.float64)

            for rule in self._rules:
                hi = var_idx[rule.head_var]

                if rule.rule_type == "unary":
                    diff = x[hi] - rule.target
                    obj += rule.weight * diff * diff
                    grad_full[hi] += 2.0 * rule.weight * diff

                elif rule.rule_type == "contradiction":
                    val = x[hi]
                    for bv, bc in zip(rule.body_vars, rule.body_coeffs):
                        val += bc * x[var_idx[bv]]
                    violation = max(0.0, val - rule.target)
                    if violation > 0.0:
                        obj += rule.weight * violation * violation
                        grad_full[hi] += 2.0 * rule.weight * violation
                        for bv, bc in zip(rule.body_vars, rule.body_coeffs):
                            grad_full[var_idx[bv]] += 2.0 * rule.weight * violation * bc

                elif rule.rule_type == "composition":
                    if rule.body_vars:
                        body_indices = [var_idx[bv] for bv in rule.body_vars]
                        body_vals = [x[bi] for bi in body_indices]
                        argmin_local = int(np.argmin(body_vals))
                        body_strength = body_vals[argmin_local]
                        argmin_idx = body_indices[argmin_local]
                    else:
                        body_strength = 1.0
                        argmin_idx = None
                    expected = rule.target * body_strength
                    shortfall = max(0.0, expected - x[hi])
                    if shortfall > 0.0:
                        obj += rule.weight * shortfall * shortfall
                        grad_full[hi] -= 2.0 * rule.weight * shortfall
                        if argmin_idx is not None:
                            grad_full[argmin_idx] += 2.0 * rule.weight * shortfall * rule.target

                elif rule.rule_type == "aggregation":
                    # Squared distance to weighted mean of body vars
                    # body_coeffs are weights (should sum to 1 for a true mean)
                    if rule.body_vars:
                        body_mean = sum(
                            bc * x[var_idx[bv]] for bv, bc in zip(rule.body_vars, rule.body_coeffs)
                        )
                    else:
                        body_mean = rule.target
                    diff = x[hi] - body_mean
                    obj += rule.weight * diff * diff
                    grad_full[hi] += 2.0 * rule.weight * diff
                    for bv, bc in zip(rule.body_vars, rule.body_coeffs):
                        grad_full[var_idx[bv]] -= 2.0 * rule.weight * diff * bc

            # Return only the gradients for free variables.
            grad_free = grad_full[free_indices]
            return obj, grad_free

        log.info(
            "hlmrf.solve_incremental.start",
            n_variables=self.n_variables,
            n_free=len(free_list),
            n_frozen=self.n_variables - len(free_list),
            n_rules=self.n_rules,
        )

        result_opt = minimize(
            objective_and_grad,
            x0_free,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": self._max_iter, "ftol": self._tol, "gtol": self._tol},
        )

        self._last_converged = result_opt.success
        self._last_n_iterations = int(result_opt.nit)
        self._last_final_objective = float(result_opt.fun)

        log.info(
            "hlmrf.solve_incremental.done",
            converged=result_opt.success,
            n_iter=result_opt.nit,
            final_obj=float(result_opt.fun),
        )

        # Reconstruct the full solution dict.
        x_final = frozen_vals.copy()
        x_final[free_indices] = result_opt.x
        return dict(zip(var_names, x_final.tolist()))

    def compute_diagnostics(self, solution: dict[str, float]) -> list[dict[str, Any]]:
        """Compute per-rule residuals given a solution.

        Args:
            solution: Variable name -> value mapping (output of solve()).

        Returns:
            List of dicts, one per rule, each with keys:
                rule_index, rule_type, head_var, violation, target, actual.
        """
        if not self._rules:
            return []

        var_names = sorted(self._variables, key=lambda v: self._variables[v][0])
        var_idx: dict[str, int] = {v: i for i, v in enumerate(var_names)}
        x = np.array([solution.get(v, self._variables[v][1]) for v in var_names], dtype=np.float64)

        results: list[dict[str, Any]] = []
        for i, rule in enumerate(self._rules):
            hi = var_idx[rule.head_var]

            if rule.rule_type == "unary":
                diff = x[hi] - rule.target
                violation = rule.weight * diff * diff
                actual = float(x[hi])

            elif rule.rule_type == "contradiction":
                val = x[hi]
                for bv, bc in zip(rule.body_vars, rule.body_coeffs):
                    val += bc * x[var_idx[bv]]
                hinge = max(0.0, val - rule.target)
                violation = rule.weight * hinge * hinge
                actual = float(val)

            elif rule.rule_type == "composition":
                if rule.body_vars:
                    body_vals = [x[var_idx[bv]] for bv in rule.body_vars]
                    body_strength = min(body_vals)
                else:
                    body_strength = 1.0
                expected = rule.target * body_strength
                shortfall = max(0.0, expected - x[hi])
                violation = rule.weight * shortfall * shortfall
                actual = float(x[hi])

            elif rule.rule_type == "aggregation":
                if rule.body_vars:
                    body_mean = sum(
                        bc * x[var_idx[bv]] for bv, bc in zip(rule.body_vars, rule.body_coeffs)
                    )
                else:
                    body_mean = rule.target
                diff = x[hi] - body_mean
                violation = rule.weight * diff * diff
                actual = float(x[hi])

            else:
                violation = 0.0
                actual = float(x[hi])

            results.append(
                {
                    "rule_index": i,
                    "rule_type": rule.rule_type,
                    "head_var": rule.head_var,
                    "violation": float(violation),
                    "target": float(rule.target),
                    "actual": actual,
                }
            )

        return results
