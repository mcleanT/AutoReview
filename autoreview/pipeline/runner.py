"""End-to-end pipeline execution."""
from __future__ import annotations

from typing import Any

import structlog

from autoreview.config.models import DomainConfig
from autoreview.models.knowledge_base import KnowledgeBase, PipelinePhase
from autoreview.pipeline.dag import DAGRunner
from autoreview.pipeline.nodes import PipelineNodes

logger = structlog.get_logger()


def build_pipeline(llm: Any, config: DomainConfig) -> tuple[DAGRunner, PipelineNodes]:
    """Build the full pipeline DAG.

    Returns:
        Tuple of (DAGRunner, PipelineNodes).
    """
    nodes = PipelineNodes(llm, config)
    dag = DAGRunner()

    dag.add_node("query_expansion", nodes.query_expansion)
    dag.add_node("search", nodes.search, dependencies=["query_expansion"])
    dag.add_node("screening", nodes.screening, dependencies=["search"])
    dag.add_node("extraction", nodes.extraction, dependencies=["screening"])
    dag.add_node("clustering", nodes.clustering, dependencies=["extraction"])
    dag.add_node("gap_search", nodes.gap_search, dependencies=["clustering"])
    dag.add_node("outline", nodes.outline, dependencies=["gap_search"])
    dag.add_node("section_writing", nodes.section_writing, dependencies=["outline"])
    dag.add_node("assembly", nodes.assembly, dependencies=["section_writing"])
    dag.add_node("final_polish", nodes.final_polish, dependencies=["assembly"])

    return dag, nodes


async def run_pipeline(
    llm: Any,
    config: DomainConfig,
    kb: KnowledgeBase,
    start_from: str | None = None,
) -> KnowledgeBase:
    """Execute the full pipeline.

    Args:
        llm: LLM provider instance.
        config: Domain configuration.
        kb: KnowledgeBase (initial or restored from snapshot).
        start_from: Optional node name to resume from.

    Returns:
        Updated KnowledgeBase with all results.
    """
    dag, nodes = build_pipeline(llm, config)

    async def on_node_complete(name: str, result: Any) -> None:
        kb.save_snapshot(name)
        logger.info("pipeline.snapshot_saved", node=name)

    async def on_node_error(name: str, error: Exception) -> None:
        kb.current_phase = PipelinePhase.FAILED
        kb.save_snapshot(f"{name}_FAILED")
        logger.error("pipeline.node_failed", node=name, error=str(error))

    try:
        await dag.execute(
            context=kb,
            start_from=start_from,
            on_node_complete=on_node_complete,
            on_node_error=on_node_error,
        )
        kb.current_phase = PipelinePhase.COMPLETE
        kb.save_snapshot("complete")
    except Exception as e:
        logger.error("pipeline.failed", error=str(e))
        raise

    return kb
