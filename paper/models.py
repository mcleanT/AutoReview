# paper/models.py
"""Pydantic models for benchmark orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import structlog
import yaml  # type: ignore[import-untyped]

from autoreview.models.base import AutoReviewModel

logger = structlog.get_logger()

ABLATION_CONDITIONS = [
    "no_evidence_chains",
    "no_critique_loops",
    "no_passage_mining",
    "no_comprehensiveness",
]

MODELS_DEFAULT = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
]

# Only Sonnet gets depth and ablation runs
SONNET_MODEL = "claude-sonnet-4-6"

RunKey = tuple[str, str, str, str]  # (topic_id, model, depth, condition)


class ReferenceInfo(AutoReviewModel):
    doi: str
    title: str
    year: int
    citation_count: int
    pdf_path: str


class TopicEntry(AutoReviewModel):
    id: str
    title: str
    domain: str
    tier: Literal["A", "B"]
    reference: ReferenceInfo
    conditions: list[str]
    date_range: str | None = None
    ablation: bool = False


class TopicsConfig(AutoReviewModel):
    metadata: dict[str, Any] = {}
    topics: list[TopicEntry] = []


class RunRegistryEntry(AutoReviewModel):
    status: Literal["running", "completed", "failed", "permanently_failed"] = "running"
    output_dir: str = ""
    review_path: str = ""
    snapshot_path: str = ""
    evaluation_path: str = ""
    started_at: str = ""
    completed_at: str = ""
    cost_usd: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    error: str = ""
    failure_count: int = 0


class RunRegistry(AutoReviewModel):
    runs: dict[str, RunRegistryEntry] = {}
    last_updated: str = ""

    def is_completed(self, key: str) -> bool:
        entry = self.runs.get(key)
        return entry is not None and entry.status == "completed"

    def is_evaluated(self, key: str) -> bool:
        entry = self.runs.get(key)
        return entry is not None and bool(entry.evaluation_path)

    def register_start(self, key: str) -> None:
        self.runs[key] = RunRegistryEntry(
            status="running",
            started_at=datetime.now(UTC).isoformat(),
        )
        self.last_updated = datetime.now(UTC).isoformat()

    def register_complete(
        self,
        key: str,
        output_dir: str,
        review_path: str,
        snapshot_path: str = "",
        cost_usd: float = 0.0,
        tokens_input: int = 0,
        tokens_output: int = 0,
    ) -> None:
        entry = self.runs.get(key, RunRegistryEntry())
        entry.status = "completed"
        entry.output_dir = output_dir
        entry.review_path = review_path
        entry.snapshot_path = snapshot_path
        entry.completed_at = datetime.now(UTC).isoformat()
        entry.cost_usd = cost_usd
        entry.tokens_input = tokens_input
        entry.tokens_output = tokens_output
        self.runs[key] = entry
        self.last_updated = datetime.now(UTC).isoformat()

    def register_failure(self, key: str, error: str) -> None:
        entry = self.runs.get(key, RunRegistryEntry())
        entry.status = "failed"
        entry.error = error
        entry.failure_count += 1
        if entry.failure_count >= 3:
            entry.status = "permanently_failed"
        self.runs[key] = entry
        self.last_updated = datetime.now(UTC).isoformat()

    def register_evaluation(self, key: str, evaluation_path: str) -> None:
        entry = self.runs.get(key)
        if entry:
            entry.evaluation_path = evaluation_path
            self.last_updated = datetime.now(UTC).isoformat()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self.model_dump_json(indent=2))
        tmp.rename(path)

    @classmethod
    def load(cls, path: Path) -> RunRegistry:
        if not path.exists():
            return cls()
        return cls.model_validate_json(path.read_text())

    @property
    def completed_count(self) -> int:
        return sum(1 for e in self.runs.values() if e.status == "completed")

    @property
    def failed_count(self) -> int:
        return sum(1 for e in self.runs.values() if e.status in ("failed", "permanently_failed"))


def load_topics(path: Path) -> TopicsConfig:
    """Load topics from a YAML file."""
    data = yaml.safe_load(path.read_text())
    return TopicsConfig.model_validate(data)


def make_run_key(topic_id: str, model: str, depth: str, condition: str) -> str:
    """Create a registry key string from components."""
    return f"{topic_id}|{model}|{depth}|{condition}"


def parse_run_key(key: str) -> tuple[str, str, str, str]:
    """Parse a registry key back into (topic_id, model, depth, condition)."""
    parts = key.split("|")
    return (parts[0], parts[1], parts[2], parts[3])


def expand_run_matrix(
    topics: list[TopicEntry],
    models: list[str] | None = None,
    include_depth: bool = False,
    include_ablation: bool = True,
) -> list[RunKey]:
    """Expand all unique (topic_id, model, depth, condition) tuples with dedup.

    Rules:
    - All topics x all models x medium x each condition
    - Depth runs (low, deep): Sonnet only, all topics, end_to_end only
    - Ablation runs: Sonnet only, topics with ablation=True, medium only
    """
    if models is None:
        models = MODELS_DEFAULT

    seen: set[RunKey] = set()
    result: list[RunKey] = []

    def _add(key: RunKey) -> None:
        if key not in seen:
            seen.add(key)
            result.append(key)

    for topic in topics:
        for model in models:
            for condition in topic.conditions:
                _add((topic.id, model, "medium", condition))

        # Depth runs: Sonnet only, end_to_end only
        if include_depth and SONNET_MODEL in models:
            for depth in ("low", "deep"):
                _add((topic.id, SONNET_MODEL, depth, "end_to_end"))

        # Ablation runs: Sonnet only, ablation topics, medium
        if include_ablation and topic.ablation and SONNET_MODEL in models:
            for ablation_cond in ABLATION_CONDITIONS:
                _add((topic.id, SONNET_MODEL, "medium", ablation_cond))

    return result
