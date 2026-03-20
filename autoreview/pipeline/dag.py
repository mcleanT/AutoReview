from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import structlog

logger = structlog.get_logger()


class BudgetAction(StrEnum):
    """Action to take based on current token budget consumption."""

    CONTINUE = "continue"
    WARN = "warn"
    DEGRADE = "degrade"
    SAVE_AND_STOP = "save_and_stop"


class TokenBudgetMonitor:
    """Monitors token consumption against a budget and recommends actions.

    Thresholds (checked in order, highest first):
      - ratio >= 1.00 → SAVE_AND_STOP (budget exhausted)
      - ratio >= 0.95 → DEGRADE      (near-exhausted, reduce quality)
      - ratio >= 0.80 → WARN         (once per instance)
      - else          → CONTINUE
    """

    def __init__(self, budget: int | None = None) -> None:
        self._budget = budget
        self._warned = False

    def check(self, tokens_used: int) -> BudgetAction:
        """Return the recommended action given current token consumption.

        Args:
            tokens_used: Total tokens consumed so far.

        Returns:
            BudgetAction indicating what the caller should do.
        """
        if self._budget is None:
            return BudgetAction.CONTINUE

        ratio = tokens_used / self._budget

        if ratio >= 1.0:
            logger.error(
                "token_budget.exhausted",
                tokens_used=tokens_used,
                budget=self._budget,
                ratio=round(ratio, 3),
            )
            return BudgetAction.SAVE_AND_STOP

        if ratio >= 0.95:
            logger.warning(
                "token_budget.near_exhausted",
                tokens_used=tokens_used,
                budget=self._budget,
                ratio=round(ratio, 3),
            )
            return BudgetAction.DEGRADE

        if ratio >= 0.80 and not self._warned:
            logger.warning(
                "token_budget.high_usage",
                tokens_used=tokens_used,
                budget=self._budget,
                ratio=round(ratio, 3),
            )
            self._warned = True
            return BudgetAction.WARN

        return BudgetAction.CONTINUE


@dataclass
class DAGNode:
    """A single node in the pipeline DAG."""

    name: str
    func: Callable[..., Coroutine[Any, Any, Any]]
    dependencies: list[str] = field(default_factory=list)
    timeout_seconds: float = 300.0


class DAGExecutionError(Exception):
    """Raised when a DAG node fails."""

    def __init__(self, node_name: str, original_error: Exception) -> None:
        self.node_name = node_name
        self.original_error = original_error
        super().__init__(f"Node '{node_name}' failed: {original_error}")


class DAGRunner:
    """Lightweight async DAG runner with topological execution.

    Executes nodes in topological order, running independent nodes
    in parallel within each execution level. Supports starting from
    a specific node for resume/restart scenarios.
    """

    def __init__(self, budget_monitor: TokenBudgetMonitor | None = None) -> None:
        self.nodes: dict[str, DAGNode] = {}
        self._results: dict[str, Any] = {}
        self._budget_monitor = budget_monitor

    def add_node(
        self,
        name: str,
        func: Callable[..., Coroutine[Any, Any, Any]],
        dependencies: list[str] | None = None,
        timeout_seconds: float = 300.0,
    ) -> None:
        """Register a node in the DAG."""
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already registered")
        self.nodes[name] = DAGNode(
            name=name,
            func=func,
            dependencies=dependencies or [],
            timeout_seconds=timeout_seconds,
        )

    def _topological_levels(self, start_from: str | None = None) -> list[list[str]]:
        """Group nodes into execution levels via topological sort.

        Nodes in the same level have no dependencies on each other
        and can run in parallel.

        If start_from is specified, only include that node and its
        downstream dependents.
        """
        # Validate all dependencies exist
        for node in self.nodes.values():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    raise ValueError(f"Node '{node.name}' depends on unknown node '{dep}'")

        # Determine which nodes to execute
        if start_from:
            if start_from not in self.nodes:
                raise ValueError(f"Start node '{start_from}' not found")
            # Find start_from and all downstream nodes
            to_execute = self._downstream_nodes(start_from)
        else:
            to_execute = set(self.nodes.keys())

        # Build in-degree map for nodes to execute
        in_degree: dict[str, int] = {}
        for name in to_execute:
            node = self.nodes[name]
            # Only count deps that are also in to_execute
            in_degree[name] = sum(1 for dep in node.dependencies if dep in to_execute)

        # Kahn's algorithm grouped by levels
        levels: list[list[str]] = []
        remaining = dict(in_degree)

        while remaining:
            # Find all nodes with no remaining dependencies
            level = [name for name, deg in remaining.items() if deg == 0]
            if not level:
                raise ValueError("Cycle detected in DAG")

            levels.append(sorted(level))  # Sort for determinism

            # Remove this level's nodes and update in-degrees
            for name in level:
                del remaining[name]
                # Reduce in-degree for dependents
                for other_name in remaining:
                    other_node = self.nodes[other_name]
                    if name in other_node.dependencies:
                        remaining[other_name] -= 1

        return levels

    def _downstream_nodes(self, start: str) -> set[str]:
        """Find a node and all its downstream dependents (BFS)."""
        # Build reverse adjacency: node -> list of nodes that depend on it
        dependents: dict[str, list[str]] = {name: [] for name in self.nodes}
        for node in self.nodes.values():
            for dep in node.dependencies:
                dependents[dep].append(node.name)

        visited: set[str] = set()
        queue = [start]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for dependent in dependents.get(current, []):
                if dependent not in visited:
                    queue.append(dependent)

        return visited

    async def execute(
        self,
        context: Any,
        start_from: str | None = None,
        on_node_complete: Callable[[str, Any], Coroutine[Any, Any, None]] | None = None,
        on_node_error: Callable[[str, Exception], Coroutine[Any, Any, None]] | None = None,
        on_node_start: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        skip_nodes: set[str] | None = None,
        get_token_count: Callable[[], int] | None = None,
    ) -> dict[str, Any]:
        """Execute the DAG.

        Args:
            context: Shared state passed to each node function.
                     Node functions receive (context,) as their argument.
            start_from: Optional node name to start execution from.
                       Nodes before this point are skipped (assumed already done).
            on_node_complete: Optional async callback after each node completes.
            on_node_error: Optional async callback when a node fails.
            on_node_start: Optional async callback before each node starts.
            skip_nodes: Optional set of node names to skip (bypass execution).
                        Skipped nodes are not executed but their dependents still run.
            get_token_count: Optional callable returning current total tokens used.
                             Used by the budget monitor (if set) after each node completes.

        Returns:
            Dict mapping node names to their return values.
        """
        levels = self._topological_levels(start_from)
        self._results = {}

        total_nodes = sum(len(level) for level in levels)
        completed = 0

        logger.info("dag.execute.start", total_nodes=total_nodes, start_from=start_from)

        for level_idx, level in enumerate(levels):
            logger.info(
                "dag.execute.level",
                level=level_idx,
                nodes=level,
            )

            async def _run_node(name: str) -> None:
                nonlocal completed
                node = self.nodes[name]
                start_time = time.monotonic()

                if name in (skip_nodes or set()):
                    logger.warning("dag.node.skipped", node=name)
                    return

                if on_node_start:
                    await on_node_start(name)
                logger.info("dag.node.start", node=name)
                try:
                    try:
                        result = await asyncio.wait_for(
                            node.func(context), timeout=node.timeout_seconds
                        )
                    except TimeoutError:
                        timeout_err = TimeoutError(f"timed out after {node.timeout_seconds}s")
                        raise DAGExecutionError(node.name, timeout_err)
                    elapsed = time.monotonic() - start_time
                    self._results[name] = result
                    completed += 1

                    logger.info(
                        "dag.node.complete",
                        node=name,
                        elapsed_seconds=round(elapsed, 2),
                        progress=f"{completed}/{total_nodes}",
                    )

                    if on_node_complete:
                        await on_node_complete(name, result)

                    # Token budget monitoring — runs after every node completion
                    if self._budget_monitor is not None and get_token_count is not None:
                        tokens_used = get_token_count()
                        action = self._budget_monitor.check(tokens_used)
                        if action == BudgetAction.SAVE_AND_STOP:
                            raise DAGExecutionError(
                                name,
                                RuntimeError("Token budget exhausted"),
                            )
                        elif action == BudgetAction.DEGRADE:
                            logger.warning(
                                "dag.budget.degrade",
                                node=name,
                                tokens_used=tokens_used,
                            )
                            if isinstance(context, dict):
                                context["depth_override"] = "low"
                        # BudgetAction.WARN is handled (logged) inside TokenBudgetMonitor.check()

                except DAGExecutionError:
                    raise
                except Exception as e:
                    elapsed = time.monotonic() - start_time
                    logger.error(
                        "dag.node.error",
                        node=name,
                        error=str(e),
                        elapsed_seconds=round(elapsed, 2),
                    )
                    if on_node_error:
                        await on_node_error(name, e)
                    raise DAGExecutionError(name, e) from e

            # Execute all nodes in this level concurrently
            if len(level) == 1:
                await _run_node(level[0])
            else:
                await asyncio.gather(*[_run_node(name) for name in level])

        logger.info("dag.execute.complete", total_nodes=total_nodes)
        return self._results

    @property
    def results(self) -> dict[str, Any]:
        """Access results from the last execution."""
        return dict(self._results)
