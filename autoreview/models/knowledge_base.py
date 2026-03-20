"""Central pipeline state model."""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field

from autoreview.analysis.comprehensiveness import ComprehensiveCheckResult
from autoreview.analysis.evidence_map import EvidenceMap
from autoreview.critique.models import CritiqueReport
from autoreview.extraction.models import PaperExtraction
from autoreview.models.base import AutoReviewModel, TimestampedModel
from autoreview.models.enrichment import CorpusExpansionResult, SectionEnrichment
from autoreview.models.narrative import NarrativePlan
from autoreview.models.paper import CandidatePaper, ScreenedPaper

logger = logging.getLogger(__name__)


class SnapshotIntegrityError(Exception):
    """Raised when a snapshot's checksum does not match its content."""


class PipelinePhase(StrEnum):
    """Current phase of the review generation pipeline."""

    INITIALIZED = "initialized"
    QUERY_EXPANSION = "query_expansion"
    SEARCH = "search"
    SCREENING = "screening"
    FULL_TEXT_RETRIEVAL = "full_text_retrieval"
    EXTRACTION = "extraction"
    CLUSTERING = "clustering"
    OUTLINE = "outline"
    NARRATIVE_PLANNING = "narrative_planning"
    CONTEXTUAL_ENRICHMENT = "contextual_enrichment"
    CORPUS_EXPANSION = "corpus_expansion"
    GAP_SEARCH = "gap_search"
    DRAFT_OUTLINE = "draft_outline"
    FINAL_OUTLINE = "final_outline"
    CITATION_SELECTION = "citation_selection"
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


def _compute_checksum(data: dict[str, Any]) -> str:
    """Compute SHA256 checksum of a dict, excluding the ``_checksum`` key.

    The dict is serialised to JSON with sorted keys for determinism.

    Args:
        data: Parsed JSON dict.  The ``_checksum`` key, if present, is ignored.

    Returns:
        Hex-encoded SHA256 digest.
    """
    payload = {k: v for k, v in data.items() if k != "_checksum"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


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
    draft_outline: Any = None
    citation_plan: Any = None
    narrative_plan: NarrativePlan | None = None
    contextual_enrichment: dict[str, SectionEnrichment] = Field(default_factory=dict)
    corpus_expansion_results: dict[str, CorpusExpansionResult] = Field(default_factory=dict)
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
            timestamp=datetime.now(UTC),
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

        The saved JSON includes two metadata fields injected at the top level:

        - ``_schema_version`` (int): version marker for migration logic.
        - ``_checksum`` (str): SHA256 of the remaining content for integrity checks.

        Directories are created automatically if they do not exist.

        Args:
            node_name: Name of the pipeline node that just completed.
        """
        snapshots_dir = Path(self.output_dir) / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        snapshot_path = snapshots_dir / f"{timestamp}_{node_name}.json"
        latest_path = snapshots_dir / "latest.json"

        raw_json = self.model_dump_json(
            indent=2,
            exclude={
                "candidate_papers": {"__all__": {"full_text"}},
                "screened_papers": {"__all__": {"paper": {"full_text"}}},
            },
        )
        data: dict[str, Any] = json.loads(raw_json)
        data["_schema_version"] = 1
        checksum = _compute_checksum(data)
        data["_checksum"] = checksum

        json_data = json.dumps(data, indent=2)
        snapshot_path.write_text(json_data)
        latest_path.write_text(json_data)

    @classmethod
    def load_snapshot(cls, path: str) -> KnowledgeBase:
        """Load a KnowledgeBase from a JSON snapshot file.

        Validates the ``_checksum`` field if present.  Legacy snapshots without
        a checksum are loaded with a warning.

        Args:
            path: Filesystem path to the snapshot JSON file.

        Returns:
            A KnowledgeBase instance reconstructed from the snapshot.

        Raises:
            SnapshotIntegrityError: If ``_checksum`` is present but does not
                match the recomputed digest of the file content.
        """
        raw = Path(path).read_text()
        data: dict[str, Any] = json.loads(raw)

        stored_checksum = data.get("_checksum")
        if stored_checksum is None:
            warnings.warn(
                f"Snapshot at '{path}' has no checksum — skipping integrity check "
                "(legacy snapshot).",
                stacklevel=2,
            )
        else:
            recomputed = _compute_checksum(data)
            if recomputed != stored_checksum:
                raise SnapshotIntegrityError(
                    f"Snapshot integrity check failed for '{path}': "
                    f"stored checksum {stored_checksum!r} does not match "
                    f"recomputed checksum {recomputed!r}."
                )

        # Strip metadata keys before deserialisation — KnowledgeBase does not
        # define these fields (extra="ignore" handles them, but being explicit
        # avoids any future config change breaking loads).
        for key in ("_schema_version", "_checksum"):
            data.pop(key, None)

        return cls.model_validate(data)

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
