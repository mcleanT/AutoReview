"""Tests for skip_nodes functionality."""

from __future__ import annotations

import pytest

from autoreview.pipeline.dag import DAGRunner


@pytest.mark.asyncio
async def test_skip_nodes_bypasses_node() -> None:
    dag = DAGRunner()
    call_log: list[str] = []

    async def node_a(ctx: dict) -> None:
        call_log.append("a")

    async def node_b(ctx: dict) -> None:
        call_log.append("b")

    async def node_c(ctx: dict) -> None:
        call_log.append("c")

    dag.add_node("a", node_a)
    dag.add_node("b", node_b, dependencies=["a"])
    dag.add_node("c", node_c, dependencies=["b"])

    await dag.execute({}, skip_nodes={"b"})
    assert "a" in call_log
    assert "b" not in call_log
    assert "c" in call_log
