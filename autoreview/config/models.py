from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SearchConfig(BaseModel):
    """Configuration for literature search."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    date_range: str = "2015-2025"
    max_results_per_source: int = 500
    relevance_threshold: int = 3
    screening_batch_size: int = 20
    max_query_expansion_rounds: int = 2
    max_gap_search_rounds: int = 2
    min_coverage_threshold: float = 0.75

    @field_validator("date_range", mode="before")
    @classmethod
    def validate_date_range(cls, v: str | None) -> str:
        if v is None:
            return ""
        v = re.sub(r"\s+", "", str(v))  # strip all whitespace
        if not v:
            return ""
        m = re.fullmatch(r"(\d{4})?-(\d{4})?", v)
        if not m or v == "-":
            raise ValueError(
                f"date_range must be 'YYYY-YYYY', '-YYYY', 'YYYY-', or empty; got '{v}'"
            )
        year_from_str, year_to_str = m.group(1), m.group(2)
        if year_from_str and year_to_str:
            if int(year_from_str) > int(year_to_str):
                raise ValueError(
                    f"date_range start ({year_from_str}) must be <= end ({year_to_str})"
                )
        return v


class TieredModelConfig(BaseModel):
    """Configuration for tiered model selection during extraction."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False  # Opt-in, existing behavior unchanged
    high_relevance_model: str = "claude-sonnet-4-20250514"
    moderate_relevance_model: str = "claude-haiku-4-20250514"
    high_relevance_scores: list[int] = [4, 5]
    moderate_relevance_scores: list[int] = [3]


class SectionTruncationConfig(BaseModel):
    """Configuration for section-aware text truncation."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True  # On by default — falls back to head/tail if parsing fails
    keep_sections: list[str] = Field(
        default_factory=lambda: [
            "abstract",
            "introduction",
            "results",
            "discussion",
            "conclusion",
        ]
    )
    drop_sections: list[str] = Field(
        default_factory=lambda: [
            "references",
            "acknowledgment",
            "acknowledgement",
            "supplementary",
            "supporting information",
            "funding",
            "author contributions",
            "competing interests",
            "conflict of interest",
            "data availability",
        ]
    )
    intro_max_chars: int = 3000
    methods_max_chars: int = 2000  # 0 = drop entirely


class ExtractionConfig(BaseModel):
    """Configuration for paper extraction."""

    model_config = ConfigDict(extra="forbid")

    domain_fields: dict[str, bool] = {}
    max_concurrent: int = 10
    ollama_max_concurrent: int = 2
    full_text_max_chars: int = 80_000
    extraction_batch_size: int = 20
    tiered_models: TieredModelConfig = Field(default_factory=TieredModelConfig)
    section_truncation: SectionTruncationConfig = Field(default_factory=SectionTruncationConfig)


class CritiqueConfig(BaseModel):
    """Configuration for the critique system."""

    model_config = ConfigDict(extra="forbid")

    rubric_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "coverage": 0.20,
            "synthesis": 0.25,
            "accuracy": 0.20,
            "balance": 0.15,
            "narrative": 0.10,
            "gaps": 0.10,
        }
    )
    score_threshold: float = 0.80
    max_revision_cycles: int = 3
    convergence_delta: float = 0.05


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

    required_sections: list[str] = Field(
        default_factory=lambda: [
            "Introduction",
            "Methods of Review",
            "Results",
            "Discussion",
            "Future Directions",
        ]
    )
    max_critique_cycles: int = 2


class LLMConfig(BaseModel):
    """Configuration for the LLM provider."""

    model_config = ConfigDict(extra="forbid")

    model: str = "claude-sonnet-4-20250514"
    api_key: str | None = None
    max_tokens_generate: int = 4096
    max_tokens_structured: int = 4096
    token_budget: int | None = None
    provider: str | None = None  # "claude", "ollama", or None (auto-detect)
    ollama_base_url: str = "http://localhost:11434"
    ollama_keep_alive: str = "1h"
    ollama_num_ctx: int | None = None


class DomainConfig(BaseModel):
    """Top-level domain configuration."""

    model_config = ConfigDict(extra="forbid")

    domain: str = "general"
    databases: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "primary": ["semantic_scholar"],
            "secondary": ["openalex"],
            "discovery": ["perplexity"],
        }
    )
    search: SearchConfig = Field(default_factory=SearchConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    critique: CritiqueConfig = Field(default_factory=CritiqueConfig)
    writing: WritingConfig = Field(default_factory=WritingConfig)
    outline: OutlineConfig = Field(default_factory=OutlineConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
