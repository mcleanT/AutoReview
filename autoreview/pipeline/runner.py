"""End-to-end pipeline execution."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from autoreview.models.knowledge_base import KnowledgeBase, PipelinePhase
from autoreview.pipeline.dag import DAGRunner
from autoreview.pipeline.nodes import PipelineNodes

if TYPE_CHECKING:
    from autoreview.config.models import DomainConfig

logger = structlog.get_logger()


def _node_summary(name: str, kb: KnowledgeBase) -> str:
    """Extract a 1-line summary from KB state for a completed node."""
    summaries: dict[str, str] = {
        "query_expansion": f"{len(kb.search_queries)} queries generated",
        "search": f"{len(kb.candidate_papers)} candidates found",
        "screening": f"{len(kb.screened_papers)} papers passed screening",
        "full_text_retrieval": (
            f"{sum(1 for sp in kb.screened_papers if sp.paper.full_text)} full texts retrieved"
        ),
        "extraction": f"{len(kb.extractions)} papers extracted",
        "clustering": f"{len(kb.evidence_map.themes) if kb.evidence_map else 0} themes, "
        f"{len(kb.evidence_map.contradictions) if kb.evidence_map else 0} contradictions",
        "gap_search": f"{len(kb.evidence_map.gaps) if kb.evidence_map else 0} gaps identified",
        "outline": f"{len(kb.outline.sections) if kb.outline else 0} sections planned",
        "narrative_planning": "narrative arc planned",
        "contextual_enrichment": "context enriched",
        "corpus_expansion": "corpus expanded",
        "section_writing": f"{len(kb.section_drafts)} sections drafted",
        "passage_search": "supporting passages found",
        "assembly": "draft assembled",
        "final_polish": "final polish complete",
    }
    return summaries.get(name, "done")


def _write_progress(
    output_dir: str,
    topic: str,
    started: str,
    node_records: dict[str, dict[str, Any]],
    current_node: str | None = None,
) -> None:
    """Write progress.json to the output directory."""
    path = Path(output_dir) / "progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    progress = {
        "topic": topic,
        "started": started,
        "current_node": current_node,
        "nodes": node_records,
    }
    path.write_text(json.dumps(progress, indent=2))


def _write_node_output(output_dir: str, name: str, summary: str, kb: KnowledgeBase) -> Path:
    """Write a per-node markdown summary file."""
    node_dir = Path(output_dir) / "node_outputs"
    node_dir.mkdir(parents=True, exist_ok=True)
    path = node_dir / f"{name}.md"

    lines = [f"# {name}", "", f"**Summary**: {summary}", ""]

    # Add node-specific details
    if name == "search":
        lines.append(f"- Candidates: {len(kb.candidate_papers)}")
        sources: dict[str, int] = {}
        for p in kb.candidate_papers:
            src = p.source_database
            sources[src] = sources.get(src, 0) + 1
        for src, count in sorted(sources.items()):
            lines.append(f"  - {src}: {count}")
    elif name == "screening":
        lines.append(f"- Passed: {len(kb.screened_papers)}")
    elif name == "extraction":
        lines.append(f"- Extracted: {len(kb.extractions)}")
    elif name == "section_writing":
        for title in kb.section_drafts:
            lines.append(f"- {title}")

    path.write_text("\n".join(lines) + "\n")
    return path


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
    dag.add_node("full_text_retrieval", nodes.full_text_retrieval, dependencies=["screening"])
    dag.add_node("extraction", nodes.extraction, dependencies=["full_text_retrieval"])
    dag.add_node("clustering", nodes.clustering, dependencies=["extraction"])
    dag.add_node("gap_search", nodes.gap_search, dependencies=["clustering"])
    dag.add_node("outline", nodes.outline, dependencies=["gap_search"])
    dag.add_node("narrative_planning", nodes.narrative_planning, dependencies=["outline"])
    dag.add_node(
        "contextual_enrichment", nodes.contextual_enrichment, dependencies=["narrative_planning"]
    )
    dag.add_node("corpus_expansion", nodes.corpus_expansion, dependencies=["contextual_enrichment"])
    dag.add_node("section_writing", nodes.section_writing, dependencies=["corpus_expansion"])
    dag.add_node("passage_search", nodes.passage_search, dependencies=["section_writing"])
    dag.add_node("assembly", nodes.assembly, dependencies=["passage_search"])
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

    pipeline_started = datetime.now(UTC).isoformat()
    node_records: dict[str, dict[str, Any]] = {}
    node_start_times: dict[str, float] = {}

    async def on_node_start(name: str) -> None:
        node_start_times[name] = time.monotonic()
        node_records[name] = {
            "status": "running",
            "started": datetime.now(UTC).isoformat(),
        }
        _write_progress(kb.output_dir, kb.topic, pipeline_started, node_records, current_node=name)

    async def on_node_complete(name: str, result: Any) -> None:
        kb.save_snapshot(name)

        elapsed = time.monotonic() - node_start_times.get(name, time.monotonic())
        summary = _node_summary(name, kb)

        node_records[name] = {
            "status": "completed",
            "duration_s": round(elapsed, 1),
            "summary": summary,
        }
        _write_progress(kb.output_dir, kb.topic, pipeline_started, node_records)

        output_path = _write_node_output(kb.output_dir, name, summary, kb)
        print(f"  [{name}] Done — {summary} → {output_path}")

    async def on_node_error(name: str, error: Exception) -> None:
        kb.current_phase = PipelinePhase.FAILED
        kb.save_snapshot(f"{name}_FAILED")

        elapsed = time.monotonic() - node_start_times.get(name, time.monotonic())
        node_records[name] = {
            "status": "failed",
            "duration_s": round(elapsed, 1),
            "error": str(error),
        }
        _write_progress(kb.output_dir, kb.topic, pipeline_started, node_records, current_node=name)
        logger.error("pipeline.node_failed", node=name, error=str(error))

    try:
        await dag.execute(
            context=kb,
            start_from=start_from,
            on_node_complete=on_node_complete,
            on_node_error=on_node_error,
            on_node_start=on_node_start,
        )
        kb.current_phase = PipelinePhase.COMPLETE
        kb.save_snapshot("complete")
    except Exception as e:
        logger.error("pipeline.failed", error=str(e))
        raise

    # Write token usage summary
    token_summary = nodes._global_tokens.token_summary()
    token_summary["model"] = config.llm.model
    token_usage_path = Path(kb.output_dir) / "token_usage.json"
    token_usage_path.write_text(json.dumps(token_summary, indent=2))
    logger.info("pipeline.token_usage_saved", path=str(token_usage_path))

    return kb
