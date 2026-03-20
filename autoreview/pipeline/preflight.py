"""Pre-flight validation checks for the AutoReview pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import structlog

logger = structlog.get_logger()


@dataclass(frozen=True)
class PreflightResult:
    """Result of a pre-flight validation check."""

    ok: bool
    missing: list[str] = field(default_factory=list)


def check_api_keys(required: list[str]) -> PreflightResult:
    """Check that all required API keys are present in the environment.

    Args:
        required: List of environment variable names that must be set.

    Returns:
        PreflightResult with ok=True if all keys are present, ok=False with
        the missing key names otherwise.
    """
    missing = [key for key in required if key not in os.environ]

    if missing:
        logger.warning(
            "pre_flight.api_keys.missing",
            missing_keys=missing,
            required_keys=required,
        )
        return PreflightResult(ok=False, missing=missing)

    logger.info(
        "pre_flight.api_keys.ok",
        required_keys=required,
    )
    return PreflightResult(ok=True, missing=[])


def check_output_dir(path: Path) -> PreflightResult:
    """Check that the output directory exists and is writable.

    Args:
        path: Directory path to validate.

    Returns:
        PreflightResult with ok=True if the directory is writable, ok=False
        otherwise.
    """
    if not path.exists():
        logger.warning(
            "pre_flight.output_dir.missing",
            path=str(path),
        )
        return PreflightResult(ok=False, missing=[str(path)])

    if not os.access(path, os.W_OK):
        logger.warning(
            "pre_flight.output_dir.not_writable",
            path=str(path),
        )
        return PreflightResult(ok=False, missing=[str(path)])

    logger.info(
        "pre_flight.output_dir.ok",
        path=str(path),
    )
    return PreflightResult(ok=True, missing=[])
