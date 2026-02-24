"""Base models for the AutoReview system."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class AutoReviewModel(BaseModel):
    """Base model for all AutoReview data structures.

    Configured with strict validation: no extra fields allowed,
    enum values used directly, and defaults validated on creation.
    """

    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
        validate_default=True,
    )


class TimestampedModel(AutoReviewModel):
    """Base model with automatic ID generation and timestamp tracking."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
