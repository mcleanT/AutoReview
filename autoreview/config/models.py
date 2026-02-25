from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field


class SearchConfig(BaseModel):
    """Configuration for literature search."""
    model_config = ConfigDict(extra="forbid")

    date_range: str = "2015-2025"
    max_results_per_source: int = 500
    relevance_threshold: int = 3
    screening_batch_size: int = 20
    max_query_expansion_rounds: int = 2
    max_gap_search_rounds: int = 2
    min_coverage_threshold: float = 0.75


class ExtractionConfig(BaseModel):
    """Configuration for paper extraction."""
    model_config = ConfigDict(extra="forbid")

    domain_fields: dict[str, bool] = {}
    max_concurrent: int = 10


class CritiqueConfig(BaseModel):
    """Configuration for the critique system."""
    model_config = ConfigDict(extra="forbid")

    rubric_weights: dict[str, float] = Field(default_factory=lambda: {
        "coverage": 0.20,
        "synthesis": 0.25,
        "accuracy": 0.20,
        "balance": 0.15,
        "narrative": 0.10,
        "gaps": 0.10,
    })
    score_threshold: float = 0.80
    max_revision_cycles: int = 3
    convergence_delta: float = 0.02


class WritingConfig(BaseModel):
    """Configuration for the writing system."""
    model_config = ConfigDict(extra="forbid")

    style: str = "academic"
    citation_format: str = "apa"
    writing_temperature: float = 0.3
    analysis_temperature: float = 0.0


class OutlineConfig(BaseModel):
    """Configuration for outline generation."""
    model_config = ConfigDict(extra="forbid")

    required_sections: list[str] = Field(default_factory=lambda: [
        "Introduction",
        "Methods of Review",
        "Results",
        "Discussion",
        "Future Directions",
    ])
    max_critique_cycles: int = 2


class LLMConfig(BaseModel):
    """Configuration for the LLM provider."""
    model_config = ConfigDict(extra="forbid")

    model: str = "claude-sonnet-4-20250514"
    api_key: str | None = None
    max_tokens_generate: int = 4096
    max_tokens_structured: int = 4096


class DomainConfig(BaseModel):
    """Top-level domain configuration."""
    model_config = ConfigDict(extra="forbid")

    domain: str = "general"
    databases: dict[str, list[str]] = Field(default_factory=lambda: {
        "primary": ["semantic_scholar"],
        "secondary": ["openalex"],
        "discovery": ["perplexity"],
    })
    search: SearchConfig = Field(default_factory=SearchConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    critique: CritiqueConfig = Field(default_factory=CritiqueConfig)
    writing: WritingConfig = Field(default_factory=WritingConfig)
    outline: OutlineConfig = Field(default_factory=OutlineConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
