"""Per-model request capabilities for the Anthropic Messages API.

Newer Claude models return HTTP 400 when sampling parameters are sent:
Opus 4.7 / 4.8 / 5 / 5.5 and Fable 5 / 5.1 reject ``temperature``, ``top_p`` and
``top_k`` outright; Sonnet 5 / 5.5 reject any non-default value (omitting them is
accepted). Sonnet <= 4.6, Opus <= 4.6 and Haiku 4.5 still accept ``temperature``.

An unrecognised model id is treated as not accepting sampling parameters: omitting
them can only lose a tuning knob, while sending them can fail the whole request.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class SamplingSupport(Enum):
    ACCEPTED = "accepted"  # temperature/top_p/top_k are sent as requested
    DEFAULT_ONLY = "default_only"  # non-default values 400; omit them
    REJECTED = "rejected"  # any value 400s; omit them
    UNKNOWN = "unknown"  # not in the table; omit them (safe choice) and warn


_ACCEPTED = SamplingSupport.ACCEPTED
_DEFAULT_ONLY = SamplingSupport.DEFAULT_ONLY
_REJECTED = SamplingSupport.REJECTED

# Keys are model ids with any ``-YYYYMMDD`` snapshot suffix and ``[1m]`` suffix removed.
SAMPLING_SUPPORT: dict[str, SamplingSupport] = {
    # Fable / Mythos: no sampling parameters.
    "claude-fable-5-1": _REJECTED,
    "claude-fable-5": _REJECTED,
    "claude-mythos-5-1": _REJECTED,
    "claude-mythos-5": _REJECTED,
    # Opus 4.7 and later: sampling parameters removed.
    "claude-opus-5-5": _REJECTED,
    "claude-opus-5": _REJECTED,
    "claude-opus-4-8": _REJECTED,
    "claude-opus-4-7": _REJECTED,
    # Sonnet 5.x: only default values accepted.
    "claude-sonnet-5-5": _DEFAULT_ONLY,
    "claude-sonnet-5": _DEFAULT_ONLY,
    # Older models that accept temperature.
    "claude-opus-4-6": _ACCEPTED,
    "claude-opus-4-5": _ACCEPTED,
    "claude-opus-4-1": _ACCEPTED,
    "claude-opus-4-0": _ACCEPTED,
    "claude-opus-4": _ACCEPTED,
    "claude-sonnet-4-6": _ACCEPTED,
    "claude-sonnet-4-5": _ACCEPTED,
    "claude-sonnet-4-0": _ACCEPTED,
    "claude-sonnet-4": _ACCEPTED,
    "claude-haiku-4-5": _ACCEPTED,
    "claude-3-haiku": _ACCEPTED,
}

_SNAPSHOT_SUFFIX = re.compile(r"-\d{8}$")
_warned_unknown: set[str] = set()


def normalize_model_id(model: str) -> str:
    """Strip the ``[1m]`` context suffix and a ``-YYYYMMDD`` snapshot suffix."""
    base = model.strip()
    if base.endswith("[1m]"):
        base = base[: -len("[1m]")]
    return _SNAPSHOT_SUFFIX.sub("", base)


def sampling_support(model: str) -> SamplingSupport:
    return SAMPLING_SUPPORT.get(normalize_model_id(model), SamplingSupport.UNKNOWN)


def sampling_params(
    model: str,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Return the sampling kwargs that may be sent to ``model``.

    Parameters left as ``None`` are never sent. For any model not known to accept
    sampling parameters, all of them are omitted.
    """
    requested = {
        name: value
        for name, value in (("temperature", temperature), ("top_p", top_p), ("top_k", top_k))
        if value is not None
    }
    support = sampling_support(model)
    if support is SamplingSupport.ACCEPTED:
        return requested
    if support is SamplingSupport.UNKNOWN and model not in _warned_unknown:
        _warned_unknown.add(model)
        logger.warning(
            "llm.sampling.unknown_model",
            model=model,
            omitted=sorted(requested),
            reason="model not in sampling capability table; omitting sampling parameters",
        )
    elif requested:
        logger.debug(
            "llm.sampling.omitted", model=model, support=support.value, omitted=sorted(requested)
        )
    return {}
