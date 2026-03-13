from __future__ import annotations

import pytest

from autoreview.pipeline.dag import DAGExecutionError, DAGRunner


class TestDAGRunner:
    def test_add_node(self):
        dag = DAGRunner()
        dag.add_node("a", self._noop)
        assert "a" in dag.nodes

    def test_duplicate_node_raises(self):
        dag = DAGRunner()
        dag.add_node("a", self._noop)
        with pytest.raises(ValueError, match="already registered"):
            dag.add_node("a", self._noop)

    def test_topological_levels_linear(self):
        dag = DAGRunner()
        dag.add_node("a", self._noop)
        dag.add_node("b", self._noop, dependencies=["a"])
        dag.add_node("c", self._noop, dependencies=["b"])
        levels = dag._topological_levels()
        assert levels == [["a"], ["b"], ["c"]]

    def test_topological_levels_parallel(self):
        dag = DAGRunner()
        dag.add_node("a", self._noop)
        dag.add_node("b", self._noop, dependencies=["a"])
        dag.add_node("c", self._noop, dependencies=["a"])
        dag.add_node("d", self._noop, dependencies=["b", "c"])
        levels = dag._topological_levels()
        assert levels[0] == ["a"]
        assert set(levels[1]) == {"b", "c"}
        assert levels[2] == ["d"]

    def test_cycle_detection(self):
        dag = DAGRunner()
        dag.add_node("a", self._noop, dependencies=["b"])
        dag.add_node("b", self._noop, dependencies=["a"])
        with pytest.raises(ValueError, match="Cycle"):
            dag._topological_levels()

    def test_unknown_dependency(self):
        dag = DAGRunner()
        dag.add_node("a", self._noop, dependencies=["nonexistent"])
        with pytest.raises(ValueError, match="unknown node"):
            dag._topological_levels()

    def test_start_from(self):
        dag = DAGRunner()
        dag.add_node("a", self._noop)
        dag.add_node("b", self._noop, dependencies=["a"])
        dag.add_node("c", self._noop, dependencies=["b"])
        levels = dag._topological_levels(start_from="b")
        node_names = [n for level in levels for n in level]
        assert "a" not in node_names
        assert "b" in node_names
        assert "c" in node_names

    async def test_execute_order(self):
        dag = DAGRunner()
        execution_order = []

        async def make_node(name):
            async def node(ctx):
                execution_order.append(name)
                return name
            return node

        dag.add_node("a", await make_node("a"))
        dag.add_node("b", await make_node("b"), dependencies=["a"])
        dag.add_node("c", await make_node("c"), dependencies=["a"])
        dag.add_node("d", await make_node("d"), dependencies=["b", "c"])

        results = await dag.execute(context=None)
        assert execution_order[0] == "a"
        assert execution_order[-1] == "d"
        assert set(execution_order[1:3]) == {"b", "c"}
        assert results["d"] == "d"

    async def test_execute_error_handling(self):
        dag = DAGRunner()

        async def failing_node(ctx):
            raise RuntimeError("boom")

        dag.add_node("a", failing_node)
        with pytest.raises(DAGExecutionError, match="boom"):
            await dag.execute(context=None)

    async def test_on_node_complete_callback(self):
        dag = DAGRunner()
        completed = []

        async def node_a(ctx):
            return "result_a"

        async def on_complete(name, result):
            completed.append((name, result))

        dag.add_node("a", node_a)
        await dag.execute(context=None, on_node_complete=on_complete)
        assert completed == [("a", "result_a")]

    @staticmethod
    async def _noop(ctx):
        pass
