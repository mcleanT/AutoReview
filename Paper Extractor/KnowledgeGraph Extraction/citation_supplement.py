"""Second-pass citation supplement extraction for KG pipelines.

Detects papers where the main extraction missed ``attributed_prior`` claims
and re-runs a focused extraction over the Introduction, Discussion, and
References sections to recover citation-backed claims.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCRIPT_DIR: Path = Path(__file__).resolve().parent
_SUPPLEMENT_PROMPT_PATH: Path = _SCRIPT_DIR / "kg_citation_supplement_prompt.md"

_SUPPLEMENT_MODEL: str = "haiku"
_SUPPLEMENT_MAX_OUTPUT_TOKENS: int = 32_000

# Regex patterns for citation detection in text
_NUMERIC_CITATION_RE = re.compile(r"(?:\[\d+\]|\(\d+(?:,\s*\d+)*\))")
_AUTHOR_CITATION_RE = re.compile(r"\b\w[\w\-']+\s+et\s+al\.")

# Section headers — capture common academic paper formats
_SECTION_HEADER_RE = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*\.?\s+)|(?:#{1,3}\s+))?("
    r"introduction|discussion|references|bibliography"
    r")\b",
    re.IGNORECASE | re.MULTILINE,
)

# Broad header pattern to detect the *next* section boundary
_ANY_SECTION_HEADER_RE = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*\.?\s+)|(?:#{1,3}\s+))?"
    r"(?:abstract|introduction|background|methods?|materials?|results?|"
    r"discussion|conclusion|acknowledgements?|acknowledgments?|references?|"
    r"bibliography|supplementary|appendix|funding|author\s+contributions?|"
    r"competing\s+interests?|data\s+availability)\b",
    re.IGNORECASE | re.MULTILINE,
)

# Sections targeted for citation counting and supplement input
_TARGET_SECTIONS = {"introduction", "discussion"}
_SUPPLEMENT_SECTIONS = ["introduction", "discussion", "references", "bibliography"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_kg_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Lazy import and apply coerce_kg_dict from kg_coerce in the same directory."""
    kg_dir = str(_SCRIPT_DIR)
    if kg_dir not in sys.path:
        sys.path.insert(0, kg_dir)
    kg_coerce = importlib.import_module("kg_coerce")
    return kg_coerce.coerce_kg_dict(raw)  # type: ignore[no-any-return]


def _repair_truncated_json(raw: str) -> str:
    """Close unclosed JSON structures caused by output truncation."""
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
    """Parse JSON from raw LLM output with 3-tier fallback.

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

    raise ValueError(
        f"Could not parse JSON from supplement output (first 200 chars): {raw[:200]!r}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def needs_citation_supplement(extraction: dict[str, Any], paper_text: str) -> bool:
    """Return True if the extraction should be supplemented with a citation pass.

    Conservative trigger: fires only when pass-1 produced *zero* attributed_prior
    claims AND the paper text contains a meaningful number of citations (>5) in
    lines that plausibly belong to an Introduction or Discussion.

    Args:
        extraction: Coerced KG extraction dict from the main pass.
        paper_text: Full paper text used for citation counting.

    Returns:
        True when a citation supplement is clearly warranted, False otherwise.
    """
    # Count attributed_prior claims from pass 1
    attributed_prior_count = sum(
        1
        for claim in extraction.get("claims", [])
        if isinstance(claim, dict) and claim.get("section_source") == "attributed_prior"
    )

    if attributed_prior_count > 0:
        return False

    # Estimate citations in intro/discussion lines only (avoid Results counting)
    lines = paper_text.splitlines()
    in_target = False
    citation_count = 0

    for line in lines:
        stripped_line = line.strip()
        # Check if we're entering/leaving a target section
        header_match = _ANY_SECTION_HEADER_RE.match(stripped_line)
        if header_match:
            section_name = header_match.group(0).lower()
            # Resolve to canonical name by checking keyword presence
            in_target = any(kw in section_name for kw in ("introduction", "discussion"))

        if in_target:
            citation_count += len(_NUMERIC_CITATION_RE.findall(line))
            citation_count += len(_AUTHOR_CITATION_RE.findall(line))

    return citation_count > 5


def extract_sections(paper_text: str, sections: list[str]) -> str:
    """Extract named sections from paper text using regex-based header detection.

    Handles common academic paper section header formats:
    - "Introduction"
    - "INTRODUCTION"
    - "1. Introduction" / "1 Introduction"
    - "## Introduction"
    - "2.1 Introduction"

    A section runs from its header line to the next recognised section header
    at the same or higher level.  The References/Bibliography section is always
    included when requested, regardless of level.

    Args:
        paper_text: Full paper text.
        sections: List of canonical section names to extract, lower-cased
            (e.g. ``["introduction", "discussion", "references"]``).

    Returns:
        Concatenated text of all matched sections, separated by blank lines.
        Returns empty string if none of the requested sections were found.
    """
    target_names = {s.lower() for s in sections}

    # Build a list of (start_pos, section_name, header_line) for all headers
    headers: list[tuple[int, str]] = []
    for match in _ANY_SECTION_HEADER_RE.finditer(paper_text):
        raw_header = match.group(0)
        # Resolve the canonical section keyword from the raw header text
        for keyword in (
            "introduction",
            "discussion",
            "references",
            "bibliography",
            "abstract",
            "background",
            "methods",
            "materials",
            "results",
            "conclusion",
            "acknowledgements",
            "acknowledgments",
            "supplementary",
            "appendix",
            "funding",
        ):
            if keyword in raw_header.lower():
                headers.append((match.start(), keyword))
                break

    if not headers:
        return ""

    # Extract content for each targeted section
    extracted_parts: list[str] = []
    for i, (start_pos, section_name) in enumerate(headers):
        if section_name not in target_names:
            continue

        # Section content runs from this header to the next header
        if i + 1 < len(headers):
            end_pos = headers[i + 1][0]
        else:
            end_pos = len(paper_text)

        section_text = paper_text[start_pos:end_pos].strip()
        if section_text:
            extracted_parts.append(section_text)

    return "\n\n".join(extracted_parts)


def build_claim_summary(extraction: dict[str, Any]) -> str:
    """Build a compact one-line-per-claim summary of pass-1 claims.

    Format: ``c_001: [predicate] subject -> object (section_source)``

    This compact form lets the supplement prompt link newly-found
    citation_contexts to relevant existing claims without bloating the prompt.

    Args:
        extraction: Coerced KG extraction dict from the main pass.

    Returns:
        Multi-line string with one claim per line.  Empty string if no claims.
    """
    lines: list[str] = []
    for claim in extraction.get("claims", []):
        if not isinstance(claim, dict):
            continue
        claim_id = claim.get("claim_id", "c_???")
        predicate = claim.get("predicate", "unknown")
        subject_name = (claim.get("subject") or {}).get("name", "?")
        object_name = (claim.get("object") or {}).get("name", "?")
        section_source = claim.get("section_source", "?")
        lines.append(
            f"{claim_id}: [{predicate}] {subject_name} -> {object_name} ({section_source})"
        )

    return "\n".join(lines)


def run_citation_supplement(
    paper_text: str,
    extraction: dict[str, Any],
    timeout: int = 300,
) -> dict[str, Any]:
    """Run a focused citation-supplement extraction over Intro/Discussion/References.

    Loads the supplement prompt from ``kg_citation_supplement_prompt.md``, constructs
    a user message containing only the relevant sections and a compact pass-1 claim
    summary, then calls the ``claude`` CLI.

    Args:
        paper_text: Full paper text (used for section extraction).
        extraction: Coerced KG extraction dict from the main pass.
        timeout: Subprocess timeout in seconds.

    Returns:
        Coerced dict containing only the supplementary claims, evidence, and
        citation_contexts.  Returns an empty dict on any failure.

    Raises:
        FileNotFoundError: If the supplement prompt file is missing.
        RuntimeError: If the claude CLI exits with a non-zero return code.
        ValueError: If JSON parsing of the CLI output fails.
    """
    if not _SUPPLEMENT_PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Supplement prompt not found: {_SUPPLEMENT_PROMPT_PATH}. "
            "Create kg_citation_supplement_prompt.md in the KnowledgeGraph Extraction directory."
        )

    system_prompt = _SUPPLEMENT_PROMPT_PATH.read_text(encoding="utf-8").strip()

    # Extract only the citation-dense sections
    sections_text = extract_sections(paper_text, _SUPPLEMENT_SECTIONS)
    if not sections_text:
        # Fall back to a truncated version of the full text if section extraction failed
        sections_text = paper_text[:40_000]

    claim_summary = build_claim_summary(extraction)

    user_prompt = (
        "Below are the Introduction, Discussion, and References sections of a paper, "
        "followed by a summary of claims already extracted in pass 1.\n\n"
        "Your task: extract ONLY attributed_prior claims — claims that cite prior literature "
        "as evidence, not claims about this paper's own results.\n\n"
        "IMPORTANT: Output ONLY valid JSON. No markdown fences. No commentary.\n\n"
        "--- SECTIONS ---\n\n"
        + sections_text
        + "\n\n--- PASS-1 CLAIM SUMMARY (for cross-referencing only) ---\n\n"
        + (claim_summary if claim_summary else "(no claims extracted in pass 1)")
    )

    cmd = [
        "claude",
        "-p",
        "--model",
        _SUPPLEMENT_MODEL,
        "--output-format",
        "text",
        "--max-turns",
        "1",
        "--tools",
        "",
        "--system-prompt",
        system_prompt,
    ]

    env = {**os.environ, "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(_SUPPLEMENT_MAX_OUTPUT_TOKENS)}
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
            f"claude CLI (supplement) exited with code {result.returncode}.\n"
            f"stderr: {result.stderr[:500]}"
        )

    model_output = result.stdout.strip()
    if not model_output:
        raise RuntimeError("Empty response from claude CLI (supplement pass).")

    raw = _parse_json(model_output)
    return _coerce_kg_dict(raw)


def merge_supplement(extraction: dict[str, Any], supplement: dict[str, Any]) -> dict[str, Any]:
    """Merge supplementary extraction into the main extraction in-place.

    Appends claims, evidence, and citation_contexts from the supplement to
    the corresponding lists in the main extraction dict.

    Args:
        extraction: Main extraction dict (modified in place).
        supplement: Supplementary extraction dict returned by
            :func:`run_citation_supplement`.

    Returns:
        The mutated ``extraction`` dict.
    """
    if not supplement:
        return extraction

    extraction.setdefault("claims", [])
    extraction.setdefault("evidence", [])
    extraction.setdefault("citation_contexts", [])

    new_claims = supplement.get("claims") or []
    new_evidence = supplement.get("evidence") or []
    new_contexts = supplement.get("citation_contexts") or []

    extraction["claims"].extend(new_claims)
    extraction["evidence"].extend(new_evidence)
    extraction["citation_contexts"].extend(new_contexts)

    return extraction


def extract_with_supplement(
    prompt_text: str,
    paper_text: str,
    timeout: int = 600,
) -> dict[str, Any]:
    """Run the full two-step extraction: main pass followed by optional citation supplement.

    Step 1: Call ``experiment_runner.run_extraction()`` with the given prompt and paper.
    Step 2: Check whether a citation supplement is needed via
            :func:`needs_citation_supplement`.
    Step 3: If needed, run :func:`run_citation_supplement` and merge the result.

    The import of ``experiment_runner`` is deferred to call time to avoid
    circular import issues when this module is imported from experiment_runner's
    own callers.

    Args:
        prompt_text: Full KG extraction prompt (passed to ``run_extraction``).
        paper_text: Full paper text.
        timeout: Timeout in seconds for each individual subprocess call.
            The supplement pass uses ``min(timeout, 300)`` seconds.

    Returns:
        Coerced and (optionally) supplemented KG extraction dict.
    """
    # Lazy import to avoid circular dependency
    optimize_dir = str(_SCRIPT_DIR / "optimize")
    if optimize_dir not in sys.path:
        sys.path.insert(0, optimize_dir)
    experiment_runner = importlib.import_module("experiment_runner")
    run_extraction = experiment_runner.run_extraction

    extraction = run_extraction(prompt_text, paper_text, timeout=timeout)

    if needs_citation_supplement(extraction, paper_text):
        supplement_timeout = min(timeout, 300)
        try:
            supplement = run_citation_supplement(paper_text, extraction, timeout=supplement_timeout)
            extraction = merge_supplement(extraction, supplement)
        except (FileNotFoundError, RuntimeError, ValueError):
            # Supplement is best-effort; do not fail the whole extraction
            pass

    return extraction
