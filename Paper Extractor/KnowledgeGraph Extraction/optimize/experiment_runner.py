"""Experiment runner: I/O layer that runs KG extractions via the local claude CLI."""

from __future__ import annotations

import importlib
import json
import subprocess  # noqa: F401 — used in run_extraction()
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: F401
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR: Path = Path(__file__).resolve().parent.parent
_AUTOREVIEW_ROOT: Path = SCRIPT_DIR.parents[1]

_SKIP_INDICES: set[int] = {0, 1, 9}  # 0=abstract-only, 1=review, 9=review
_SKIP_CORPUS_IDS: set[str] = {"corpus_0"}  # Reviews / non-experimental papers
_MIN_TEXT_LENGTH: int = 5_000
_TRUNCATION_LIMIT: int = 0  # Disabled — Haiku 4.5 200K context fits all papers
_RAPID_TRUNCATION_LIMIT: int = 20_000  # Aggressive cap for rapid optimizer mode

# Rough chars-to-tokens ratio for English text + JSON
_CHARS_PER_TOKEN = 4.0
_MAX_OUTPUT_TOKENS: int = 64_000  # Match production batch_extract_kg.py
_MODEL: str = "haiku"

# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------


def _coerce_kg_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Import and apply coerce_kg_dict from the parent directory at call time."""
    kg_dir = str(SCRIPT_DIR)
    if kg_dir not in sys.path:
        sys.path.insert(0, kg_dir)
    kg_coerce = importlib.import_module("kg_coerce")
    return kg_coerce.coerce_kg_dict(raw)  # type: ignore[no-any-return]


def _truncate_paper(text: str) -> str:
    """Apply section-aware truncation matching production batch_extract_kg behaviour."""
    if _TRUNCATION_LIMIT <= 0 or len(text) <= _TRUNCATION_LIMIT:
        return text
    autoreview_root = str(_AUTOREVIEW_ROOT)
    if autoreview_root not in sys.path:
        sys.path.insert(0, autoreview_root)
    from autoreview.config.models import SectionTruncationConfig
    from autoreview.extraction.truncation import section_aware_truncate

    config = SectionTruncationConfig(
        enabled=True,
        keep_sections=[
            "results",
            "methods",
            "discussion",
            "introduction",
            "references",
            "conclusion",
        ],
        drop_sections=[
            "acknowledgments",
            "acknowledgements",
            "supplementary",
            "abstract",
            "funding",
            "conflict of interest",
            "data availability",
            "author contributions",
            "competing interests",
            "ethics",
            "supporting information",
        ],
        intro_max_chars=4000,
        methods_max_chars=5000,
    )
    return section_aware_truncate(text, _TRUNCATION_LIMIT, config)


def _truncate_paper_rapid(text: str) -> str:
    """Aggressive section-aware truncation for rapid optimizer mode.

    Keeps only Results + Methods + References (the claim-dense sections).
    Caps total at 20K chars. Drops Intro/Discussion entirely.
    """
    if len(text) <= _RAPID_TRUNCATION_LIMIT:
        return text
    autoreview_root = str(_AUTOREVIEW_ROOT)
    if autoreview_root not in sys.path:
        sys.path.insert(0, autoreview_root)
    from autoreview.config.models import SectionTruncationConfig
    from autoreview.extraction.truncation import section_aware_truncate

    config = SectionTruncationConfig(
        enabled=True,
        keep_sections=[
            "results",
            "methods",
            "references",
        ],
        drop_sections=[
            "introduction",
            "discussion",
            "conclusion",
            "acknowledgments",
            "acknowledgements",
            "supplementary",
            "abstract",
            "funding",
            "conflict of interest",
            "data availability",
            "author contributions",
            "competing interests",
            "ethics",
            "supporting information",
        ],
        intro_max_chars=0,
        methods_max_chars=5000,
    )
    return section_aware_truncate(text, _RAPID_TRUNCATION_LIMIT, config)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

# Haiku 4.5 pricing (USD per million tokens)
_HAIKU_INPUT_RATE = 0.80  # per M input tokens
_HAIKU_OUTPUT_RATE = 4.00  # per M output tokens
_HAIKU_CACHE_READ_RATE = 0.08  # per M cached input tokens
_BATCH_DISCOUNT = 0.50  # 50% off for batch API


def estimate_cost(
    system_chars: int,
    paper_chars: int,
    output_chars: int,
) -> dict[str, float]:
    """Estimate extraction cost for a single paper."""
    system_tokens = system_chars / _CHARS_PER_TOKEN
    paper_tokens = paper_chars / _CHARS_PER_TOKEN
    output_tokens = output_chars / _CHARS_PER_TOKEN
    input_tokens = system_tokens + paper_tokens

    regular_cost = (
        input_tokens * _HAIKU_INPUT_RATE / 1_000_000
        + output_tokens * _HAIKU_OUTPUT_RATE / 1_000_000
    )

    batch_cost = (
        system_tokens * _HAIKU_CACHE_READ_RATE * _BATCH_DISCOUNT / 1_000_000
        + paper_tokens * _HAIKU_INPUT_RATE * _BATCH_DISCOUNT / 1_000_000
        + output_tokens * _HAIKU_OUTPUT_RATE * _BATCH_DISCOUNT / 1_000_000
    )

    return {
        "input_tokens_est": int(input_tokens),
        "output_tokens_est": int(output_tokens),
        "cost_per_paper": round(regular_cost, 4),
        "batch_cost_per_paper": round(batch_cost, 4),
    }


# ---------------------------------------------------------------------------
# Paper loading
# ---------------------------------------------------------------------------


def load_test_papers(
    micro_indices: list[int] | None = None,
    include_rai14: bool = True,
    extra_corpus_path: Path | None = None,
    max_text_length: int = 0,
) -> list[dict[str, str]]:
    """Load the test paper corpus.

    Args:
        micro_indices: Indices from micro_sample.json to include.
            Defaults to [2, 6].
        include_rai14: When True (default), always prepend the rai14 full-text
            paper (67 K experimental paper) to the returned list.
        extra_corpus_path: Optional path to a JSON file containing additional
            papers. Each entry must have "id", "title", and "full_text" keys.
        max_text_length: When > 0, skip papers whose text exceeds this many
            characters. Use to prefer shorter papers for rapid optimizer iterations.

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
        if not max_text_length or len(rai14_text) <= max_text_length:
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
        if max_text_length and len(full_text) > max_text_length:
            continue
        papers.append(
            {
                "id": f"micro_{idx}",
                "title": entry.get("title", f"Paper {idx}"),
                "text": full_text,
            }
        )

    # --- extra corpus papers -------------------------------------------------
    if extra_corpus_path and extra_corpus_path.exists():
        with extra_corpus_path.open(encoding="utf-8") as fh:
            extra: list[dict[str, Any]] = json.load(fh)
        for entry in extra:
            entry_id = entry.get("id", "extra")
            if entry_id in _SKIP_CORPUS_IDS:
                continue
            full_text = entry.get("full_text", "")
            if len(full_text) < _MIN_TEXT_LENGTH:
                continue
            if max_text_length and len(full_text) > max_text_length:
                continue
            papers.append(
                {
                    "id": entry.get("id", "extra"),
                    "title": entry.get("title", "Unknown"),
                    "text": full_text,
                }
            )

    return papers


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _repair_truncated_json(raw: str) -> str:
    """Repair JSON truncated mid-output by closing unclosed structures."""
    start = raw.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", raw[:200], 0)

    text = raw[start:]
    in_string = False
    escape_next = False
    stack: list[str] = []

    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{" or ch == "]" and stack and stack[-1] == "[":
            stack.pop()

    if not stack:
        return text

    result = text
    if in_string:
        result += '"'

    result = result.rstrip()
    while result and result[-1] in (",", ":", " ", "\n", "\r", "\t"):
        result = result[:-1]

    for bracket in reversed(stack):
        result += "}" if bracket == "{" else "]"

    return result


def _parse_json(raw: str) -> dict[str, Any]:
    """Parse JSON from raw LLM output with 4-tier fallback.

    Tier 1 — direct json.loads.
    Tier 2 — brace slice (first ``{`` to last ``}``).
    Tier 3 — truncation repair (close unclosed brackets/braces).

    Raises:
        ValueError: If all tiers fail to produce valid JSON.
    """
    stripped = raw.strip()

    # Tier 1: direct parse
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Tier 2: brace slice (handles markdown fences, preamble, trailing text)
    brace_start = stripped.find("{")
    brace_end = stripped.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            return json.loads(stripped[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    # Tier 3: truncation repair — close unclosed brackets/braces
    try:
        repaired = _repair_truncated_json(stripped)
        return json.loads(repaired)
    except (json.JSONDecodeError, Exception):
        pass

    raise ValueError(f"Could not parse JSON from model output (first 200 chars): {raw[:200]!r}")


# ---------------------------------------------------------------------------
# Single extraction
# ---------------------------------------------------------------------------


def run_extraction(
    prompt_text: str,
    paper_text: str,
    timeout: int = 600,
    rapid: bool = False,
) -> dict[str, Any]:
    """Run a single KG extraction via the local ``claude`` CLI."""
    marker = "{PAPER_TEXT}"
    marker_pos = prompt_text.find(marker)
    if marker_pos == -1:
        system_prompt = prompt_text.strip()
    else:
        system_prompt = prompt_text[:marker_pos].strip()

    paper_text = _truncate_paper_rapid(paper_text) if rapid else _truncate_paper(paper_text)

    user_prompt = (
        "Extract all falsifiable claims from the following paper as structured JSON "
        "according to the schema in your system prompt.\n"
        "IMPORTANT: Output ONLY valid JSON. No markdown fences. No commentary.\n\n"
        "---\n\n" + paper_text
    )

    cmd = [
        "claude",
        "-p",
        "--model",
        _MODEL,
        "--output-format",
        "text",
        "--max-turns",
        "1",
        "--tools",
        "",
        "--system-prompt",
        system_prompt,
    ]

    import os

    env = {**os.environ, "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(_MAX_OUTPUT_TOKENS)}
    result = subprocess.run(
        cmd,
        input=user_prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"claude CLI exited with code {result.returncode}.\nstderr: {result.stderr[:500]}"
        )

    model_output = result.stdout.strip()

    if not model_output:
        raise RuntimeError("Empty response from claude CLI.")

    raw = _parse_json(model_output)
    coerced = _coerce_kg_dict(raw)

    coerced["_cost"] = estimate_cost(len(system_prompt), len(user_prompt), len(model_output))

    return coerced


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def _extract_one(
    prompt_text: str,
    paper: dict[str, str],
    timeout: int,
    rapid: bool = False,
) -> dict[str, Any]:
    """Extract a single paper and return a result dict with metadata.

    Internal helper for :func:`run_all_extractions`. Catches all exceptions
    so that failures never crash the parallel executor.
    """
    paper_id = paper["id"]
    t0 = time.monotonic()
    try:
        extraction = run_extraction(prompt_text, paper["text"], timeout=timeout, rapid=rapid)
        elapsed = time.monotonic() - t0
        extraction["_paper_id"] = paper_id
        extraction["_elapsed"] = elapsed
        return extraction
    except Exception as exc:
        elapsed = time.monotonic() - t0
        err_msg = str(exc)[:200]
        return {
            "_paper_id": paper_id,
            "_elapsed": elapsed,
            "_error": err_msg,
            "_cost": {
                "input_tokens_est": 0,
                "output_tokens_est": 0,
                "cost_per_paper": 0,
                "batch_cost_per_paper": 0,
            },
            "claims": [],
            "evidence": [],
        }


def run_all_extractions(
    prompt_text: str,
    papers: list[dict[str, str]],
    timeout: int = 600,
    max_workers: int | None = None,
    rapid: bool = False,
) -> list[dict[str, Any]]:
    """Run extractions on all papers in parallel and collect results.

    Uses a thread pool to launch ``claude -p`` subprocesses concurrently.
    Each subprocess is independent so threads simply wait on I/O.

    Args:
        prompt_text: Full prompt file contents with ``{PAPER_TEXT}`` marker.
        papers: List of paper dicts as returned by :func:`load_test_papers`.
        timeout: Per-paper subprocess timeout in seconds.
        max_workers: Maximum concurrent extractions. Defaults to
            ``min(len(papers), 5)`` to avoid rate limiting.

    Returns:
        List of extraction result dicts.  Each dict gets two extra keys:
        ``_paper_id`` (str) and ``_elapsed`` (float seconds).  Failed papers
        get ``_error`` (str) and empty ``claims``/``evidence`` lists.
        All results also carry a ``_cost`` dict with token and cost estimates.
    """
    if not papers:
        return []

    if max_workers is None:
        max_workers = min(len(papers), 5)

    print(f"  Launching {len(papers)} extractions ({max_workers} parallel)...")
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_paper = {
            executor.submit(_extract_one, prompt_text, paper, timeout, rapid): paper
            for paper in papers
        }
        for future in as_completed(future_to_paper):
            paper = future_to_paper[future]
            result = future.result()
            paper_id = result["_paper_id"]
            elapsed = result.get("_elapsed", 0)
            if "_error" in result:
                print(
                    f"  [{paper_id}] ERROR after {elapsed:.1f}s: {result['_error'][:100]}",
                    flush=True,
                )
            else:
                cc = len(result.get("claims", []))
                ec = len(result.get("evidence", []))
                cost = result.get("_cost", {})
                bc = cost.get("batch_cost_per_paper", 0)
                print(f"  [{paper_id}] {cc}c {ec}e, {elapsed:.1f}s, ~${bc:.3f}/paper(batch)")
            results.append(result)

    return results
