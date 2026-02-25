"""Central pipeline state model."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field

from autoreview.analysis.comprehensiveness import ComprehensiveCheckResult
from autoreview.analysis.evidence_map import EvidenceMap
from autoreview.critique.models import CritiqueReport
from autoreview.extraction.models import PaperExtraction
from autoreview.models.base import AutoReviewModel, TimestampedModel
from autoreview.models.paper import CandidatePaper, ScreenedPaper


class PipelinePhase(str, Enum):
    """Current phase of the review generation pipeline."""

    INITIALIZED = "initialized"
    QUERY_EXPANSION = "query_expansion"
    SEARCH = "search"
    SCREENING = "screening"
    EXTRACTION = "extraction"
    CLUSTERING = "clustering"
    OUTLINE = "outline"
    GAP_SEARCH = "gap_search"
    SECTION_WRITING = "section_writing"
    SECTION_CRITIQUE = "section_critique"
    PASSAGE_SEARCH = "passage_search"
    HOLISTIC_CRITIQUE = "holistic_critique"
    FINAL_POLISH = "final_polish"
    COMPLETE = "complete"
    FAILED = "failed"


class AuditEntry(AutoReviewModel):
    """A single entry in the pipeline audit log.

    Tracks actions taken by each pipeline node, including
    LLM token usage for cost monitoring.
    """

    timestamp: datetime
    node_name: str
    action: str
    details: str = ""
    token_usage: dict[str, int] = Field(default_factory=dict)


class KnowledgeBase(TimestampedModel):
    """Central state container for the entire pipeline.

    Holds all data produced by every pipeline stage, from search
    queries through to the final draft. Serialized to JSON after
    every DAG node for crash recovery and debugging.
    """

    topic: str
    domain: str = "general"
    scope_document: str | None = None
    search_queries: dict[str, list[str]] = Field(default_factory=dict)
    candidate_papers: list[CandidatePaper] = Field(default_factory=list)
    screened_papers: list[ScreenedPaper] = Field(default_factory=list)
    extractions: dict[str, PaperExtraction] = Field(default_factory=dict)
    evidence_map: EvidenceMap | None = None
    outline: Any = None  # Will be ReviewOutline, use Any to avoid circular import
    section_drafts: dict[str, str] = Field(default_factory=dict)
    full_draft: str | None = None
    critique_history: list[CritiqueReport] = Field(default_factory=list)
    comprehensiveness_checks: list[ComprehensiveCheckResult] = Field(default_factory=list)
    current_phase: PipelinePhase = PipelinePhase.INITIALIZED
    iteration_counts: dict[str, int] = Field(default_factory=dict)
    audit_log: list[AuditEntry] = Field(default_factory=list)
    output_dir: str = "output"

    def add_audit_entry(
        self,
        node_name: str,
        action: str,
        details: str = "",
        token_usage: dict[str, int] | None = None,
    ) -> None:
        """Append a new entry to the audit log.

        Args:
            node_name: Name of the pipeline node generating the entry.
            action: Description of the action taken.
            details: Optional additional details.
            token_usage: Optional dict with 'input_tokens' and 'output_tokens' counts.
        """
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc),
            node_name=node_name,
            action=action,
            details=details,
            token_usage=token_usage or {},
        )
        self.audit_log.append(entry)

    def save_snapshot(self, node_name: str) -> None:
        """Write the current state to a JSON snapshot file.

        Creates two files:
        - A timestamped snapshot: ``{output_dir}/snapshots/{timestamp}_{node_name}.json``
        - A latest pointer: ``{output_dir}/snapshots/latest.json``

        Directories are created automatically if they do not exist.

        Args:
            node_name: Name of the pipeline node that just completed.
        """
        snapshots_dir = Path(self.output_dir) / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        snapshot_path = snapshots_dir / f"{timestamp}_{node_name}.json"
        latest_path = snapshots_dir / "latest.json"

        json_data = self.model_dump_json(indent=2)
        snapshot_path.write_text(json_data)
        latest_path.write_text(json_data)

    @classmethod
    def load_snapshot(cls, path: str) -> KnowledgeBase:
        """Load a KnowledgeBase from a JSON snapshot file.

        Args:
            path: Filesystem path to the snapshot JSON file.

        Returns:
            A KnowledgeBase instance reconstructed from the snapshot.
        """
        json_data = Path(path).read_text()
        return cls.model_validate_json(json_data)

    def total_tokens(self) -> dict[str, int]:
        """Sum token usage across all audit log entries.

        Returns:
            A dict with 'input_tokens' and 'output_tokens' totals.
        """
        totals: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        for entry in self.audit_log:
            totals["input_tokens"] += entry.token_usage.get("input_tokens", 0)
            totals["output_tokens"] += entry.token_usage.get("output_tokens", 0)
        return totals
