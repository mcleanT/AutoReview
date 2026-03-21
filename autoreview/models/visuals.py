"""Data models for pipeline-generated figures and tables."""

from __future__ import annotations

from typing import Literal

from autoreview.models.base import AutoReviewModel


class VisualInsertionAnchor(AutoReviewModel):
    """Specifies where a visual element should be inserted in the assembled draft."""

    section_id: str
    position: Literal["before", "after"] = "after"


class FigureMetadata(AutoReviewModel):
    """Metadata for a pipeline-generated figure."""

    key: str
    path: str
    caption: str
    anchor: VisualInsertionAnchor
    data_driven: bool = False


class TableMetadata(AutoReviewModel):
    """Metadata for a pipeline-generated markdown table."""

    key: str
    markdown: str
    caption: str
    anchor: VisualInsertionAnchor
