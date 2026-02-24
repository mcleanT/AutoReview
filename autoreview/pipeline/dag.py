from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

import structlog

logger = structlog.get_logger()


@dataclass
class DAGNode:
    """A single node in the pipeline DAG."""
    name: str
    func: Callable[..., Coroutine[Any, Any, Any]]
    dependencies: list[str] = field(default_factory=list)


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

    def __init__(self) -> None:
        self.nodes: dict[str, DAGNode] = {}
        self._results: dict[str, Any] = {}

    def add_node(
        self,
        name: str,
        func: Callable[..., Coroutine[Any, Any, Any]],
        dependencies: list[str] | None = None,
    ) -> None:
        """Register a node in the DAG."""
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already registered")
        self.nodes[name] = DAGNode(
            name=name,
            func=func,
            dependencies=dependencies or [],
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
                    raise ValueError(
                        f"Node '{node.name}' depends on unknown node '{dep}'"
                    )

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
            in_degree[name] = sum(
                1 for dep in node.dependencies if dep in to_execute
            )

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
    ) -> dict[str, Any]:
        """Execute the DAG.

        Args:
            context: Shared state passed to each node function.
                     Node functions receive (context,) as their argument.
            start_from: Optional node name to start execution from.
                       Nodes before this point are skipped (assumed already done).
            on_node_complete: Optional async callback after each node completes.
            on_node_error: Optional async callback when a node fails.

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

                logger.info("dag.node.start", node=name)
                try:
                    result = await node.func(context)
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
