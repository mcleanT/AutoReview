"""Experiment runner: I/O layer that runs KG extractions via the claude CLI."""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR: Path = Path(__file__).resolve().parent.parent

_SKIP_INDICES: set[int] = {0, 1, 3, 4, 5, 9}
_MIN_TEXT_LENGTH: int = 5_000

_FENCE_RE: re.Pattern[str] = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

# ---------------------------------------------------------------------------
# Paper loading
# ---------------------------------------------------------------------------


def load_test_papers(
    micro_indices: list[int] | None = None,
    include_rai14: bool = True,
) -> list[dict[str, str]]:
    """Load the test paper corpus.

    Args:
        micro_indices: Indices from micro_sample.json to include.
            Defaults to [2, 6].
        include_rai14: When True (default), always prepend the rai14 full-text
            paper (67 K experimental paper) to the returned list.

    Returns:
        List of dicts with keys ``id``, ``title``, and ``text``.
    """
    if micro_indices is None:
        micro_indices = [2, 6]

    papers: list[dict[str, str]] = []

    # --- rai14 paper ---------------------------------------------------------
    if include_rai14:
        rai14_path = SCRIPT_DIR / "rai14_fulltext.txt"
        rai14_text = rai14_path.read_text(encoding="utf-8")
        papers.append(
            {
                "id": "rai14",
                "title": "Rai et al. 2014 (full text)",
                "text": rai14_text,
            }
        )

    # --- micro_sample papers -------------------------------------------------
    micro_path = SCRIPT_DIR / "gastruloid_run" / "micro_sample.json"
    with micro_path.open(encoding="utf-8") as fh:
        corpus: list[dict[str, Any]] = json.load(fh)

    for idx in micro_indices:
        if idx in _SKIP_INDICES:
            continue
        if idx >= len(corpus):
            continue
        entry = corpus[idx]
        full_text: str = entry.get("full_text", "")
        if len(full_text) < _MIN_TEXT_LENGTH:
            continue
        papers.append(
            {
                "id": f"micro_{idx}",
                "title": entry.get("title", f"Paper {idx}"),
                "text": full_text,
            }
        )

    return papers


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _parse_json(raw: str) -> dict[str, Any]:
    """Parse JSON from raw LLM output with 3-tier fallback.

    Tier 1 — direct json.loads.
    Tier 2 — extract from markdown ```json ... ``` fence.
    Tier 3 — brace slice (first ``{`` to last ``}``).

    Raises:
        ValueError: If all three tiers fail to produce valid JSON.
    """
    # Tier 1: direct parse
    stripped = raw.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Tier 2: markdown fence
    match = _FENCE_RE.search(stripped)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Tier 3: brace slice
    brace_start = stripped.find("{")
    brace_end = stripped.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            return json.loads(stripped[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from model output (first 200 chars): {raw[:200]!r}")


# ---------------------------------------------------------------------------
# Single extraction
# ---------------------------------------------------------------------------


def run_extraction(
    prompt_text: str,
    paper_text: str,
    timeout: int = 300,
) -> dict[str, Any]:
    """Run a single KG extraction via the claude CLI.

    The prompt_text must contain a ``{PAPER_TEXT}`` marker.  Everything before
    that marker is used as the system prompt; the user message is a fixed
    instruction prepended to the paper text.

    Args:
        prompt_text: Full prompt file contents (system + ``{PAPER_TEXT}``
            placeholder).
        paper_text: Full text of the paper to extract from.
        timeout: Subprocess timeout in seconds.

    Returns:
        Parsed extraction dict from the model.

    Raises:
        RuntimeError: If the subprocess exits with a non-zero return code.
        ValueError: If JSON parsing fails on the model output.
    """
    marker = "{PAPER_TEXT}"
    marker_pos = prompt_text.find(marker)
    if marker_pos == -1:
        system_prompt = prompt_text.strip()
    else:
        system_prompt = prompt_text[:marker_pos].strip()

    user_prompt = (
        "Extract all falsifiable claims, entities, and relationships from "
        "the following paper text.\n\n" + paper_text
    )

    cmd = [
        "claude",
        "-p",
        "--model",
        "haiku",
        "--output-format",
        "text",
        "--max-turns",
        "3",
        "--system-prompt",
        system_prompt,
    ]

    result = subprocess.run(
        cmd,
        input=user_prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"claude CLI exited with code {result.returncode}.\nstderr: {result.stderr[:500]}"
        )

    return _parse_json(result.stdout)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def run_all_extractions(
    prompt_text: str,
    papers: list[dict[str, str]],
    timeout: int = 300,
) -> list[dict[str, Any]]:
    """Run extractions on all papers sequentially and collect results.

    Args:
        prompt_text: Full prompt file contents with ``{PAPER_TEXT}`` marker.
        papers: List of paper dicts as returned by :func:`load_test_papers`.
        timeout: Per-paper subprocess timeout in seconds.

    Returns:
        List of extraction result dicts.  Each dict gets two extra keys:
        ``_paper_id`` (str) and ``_elapsed`` (float seconds).  Failed papers
        get ``_error`` (str) and empty ``claims``/``evidence`` lists.
    """
    results: list[dict[str, Any]] = []

    for paper in papers:
        title = paper["title"]
        paper_id = paper["id"]
        print(f"  Extracting: {title[:60]} ...", end=" ", flush=True)
        t0 = time.monotonic()

        try:
            extraction = run_extraction(prompt_text, paper["text"], timeout=timeout)
            elapsed = time.monotonic() - t0
            claim_count = len(extraction.get("claims", []))
            print(f"{claim_count} claims, {elapsed:.1f}s")
            extraction["_paper_id"] = paper_id
            extraction["_elapsed"] = elapsed
            results.append(extraction)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            print(f"ERROR after {elapsed:.1f}s: {exc}")
            results.append(
                {
                    "_paper_id": paper_id,
                    "_elapsed": elapsed,
                    "_error": str(exc),
                    "claims": [],
                    "evidence": [],
                }
            )

    return results
