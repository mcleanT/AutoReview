"""Main CLI entrypoint for the KG extraction prompt optimizer.

Runs a two-phase iterative optimization loop:
  Phase 1 — Quality optimization: maximize composite score > 0.95
  Phase 2 — Cost reduction: minimize per-paper cost while quality > 0.93

Each iteration uses a tournament selection strategy: N optimizer candidates are
generated in parallel with different lenses, screened on 1 paper, and the best
is validated on 3 additional papers before acceptance.  Per-iteration extractions
use a random 4-paper pool (1 screen + 3 validation) to reduce cost.  Full
25-paper evaluations run every 5 accepted iterations and at phase transitions.

Usage:
    python optimize_extraction_prompt.py [--max-iterations N] [--sample-size 3]
                                          [--no-rai14] [--extra-papers PATH]
                                          [--rapid] [--candidates N]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

# Force unbuffered stdout so log files update in real-time
os.environ["PYTHONUNBUFFERED"] = "1"

# Override print to always flush (covers cases where env var isn't inherited)
_builtin_print = print


def print(*args, **kwargs):  # noqa: A001
    kwargs.setdefault("flush", True)
    _builtin_print(*args, **kwargs)


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from optimize.error_analyzer import analyze_errors  # noqa: E402
from optimize.experiment_runner import load_test_papers, run_all_extractions  # noqa: E402
from optimize.scoring import METRIC_WEIGHTS, RAPID_EXCLUDE, score_extraction  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROMPT_PATH = SCRIPT_DIR / "kg_extraction_prompt.md"
VERSIONS_DIR = SCRIPT_DIR / "prompt_versions"
LOG_PATH = SCRIPT_DIR / "optimize" / "optimization_log.json"
PROGRAM_PATH = SCRIPT_DIR / "optimize" / "program.md"

CONVERGENCE_THRESHOLD = 0.002  # Tighter convergence for overnight runs
QUALITY_REGRESSION_THRESHOLD = 0.12  # Legacy — no longer used for accept/reject
MAX_CHANGE_RATIO = 1.0  # Disabled — structured edits are inherently bounded (1-3 per iteration)
MAX_CONSECUTIVE_REJECTS = 25  # More patience for overnight operation
PHASE1_TARGET = 0.95  # Quality target for phase transition
PHASE2_QUALITY_FLOOR = 0.93  # Minimum quality during cost optimization

# ---------------------------------------------------------------------------
# Diversity strategies — rotated each iteration to prevent the optimizer
# from getting stuck proposing the same edits repeatedly.
# ---------------------------------------------------------------------------
DIVERSITY_STRATEGIES = [
    {
        "lens": "structural_rewrite",
        "hint": (
            "Try restructuring how the prompt organizes its instructions — "
            "reorder sections, merge related rules, or split overloaded rules "
            "into clearer sub-rules. Focus on making the instructions easier "
            "to follow rather than adding more content."
        ),
    },
    {
        "lens": "example_driven",
        "hint": (
            "Add or improve concrete input→output EXAMPLES rather than adding "
            "more rules. Show the model exactly what a good extraction looks "
            "like for the weakest metrics. One good example beats three rules."
        ),
    },
    {
        "lens": "counter_example",
        "hint": (
            "Add COUNTER-EXAMPLES showing common mistakes and how to avoid "
            "them. Focus on the error patterns — show what wrong output looks "
            "like and contrast it with the correct version."
        ),
    },
    {
        "lens": "constraint_tightening",
        "hint": (
            "Make vague instructions more specific and actionable. Replace "
            "'should' with 'MUST', add exact thresholds, specify minimum "
            "counts. Turn guidelines into hard constraints with clear pass/fail."
        ),
    },
    {
        "lens": "checklist_approach",
        "hint": (
            "Add a pre-output self-check or checklist that the model must run "
            "before producing its final JSON. Target the weakest metrics with "
            "specific verification steps."
        ),
    },
    {
        "lens": "negative_space",
        "hint": (
            "Focus on what the prompt does NOT say. Identify implicit "
            "assumptions the extraction model might make that lead to errors. "
            "Add explicit disambiguation for ambiguous cases."
        ),
    },
    {
        "lens": "cross_metric_synergy",
        "hint": (
            "Find edits that improve MULTIPLE weak metrics simultaneously. "
            "For example, better evidence instructions can improve both "
            "evidence_depth and evidence_completeness at once."
        ),
    },
    {
        "lens": "simplification",
        "hint": (
            "The prompt may be too complex or contradictory. Try SIMPLIFYING "
            "or REMOVING instructions that may be confusing the extraction "
            "model. Sometimes less is more — conflicting rules cause errors."
        ),
    },
    {
        "lens": "workflow_reframing",
        "hint": (
            "Restructure the prompt to guide the model through a specific "
            "extraction WORKFLOW — e.g., 'First scan all figures, then extract "
            "claims, then link evidence.' A clear process can be more effective "
            "than a list of rules."
        ),
    },
    {
        "lens": "weakest_link_focus",
        "hint": (
            "Ignore the top error patterns — they've likely been targeted "
            "already. Focus on the LOWEST-SCORING metric that hasn't been "
            "the target of recent edits. Sometimes gains come from unexpected "
            "places."
        ),
    },
]

_FENCE_RE = re.compile(r"```markdown\s*\n(.*?)```", re.DOTALL)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def score_all(
    extractions: list[dict],
    rapid: bool = False,
) -> tuple[float, dict[str, float]]:
    """Average composite and per-metric scores across successful extractions.

    Extractions with ``_error`` set are excluded from scoring so that
    transient API failures do not poison the composite score.
    """
    if not extractions:
        return 0.0, {name: 0.0 for name in METRIC_WEIGHTS}

    successful = [ex for ex in extractions if not ex.get("_error")]

    if not successful:
        return 0.0, {name: 0.0 for name in METRIC_WEIGHTS}

    composites: list[float] = []
    per_metric_accum: dict[str, list[float]] = {name: [] for name in METRIC_WEIGHTS}

    for extraction in successful:
        composite, metrics = score_extraction(extraction, rapid=rapid)
        composites.append(composite)
        for name, value in metrics.items():
            per_metric_accum[name].append(value)

    avg_composite = sum(composites) / len(composites)
    avg_metrics = {name: sum(vals) / len(vals) for name, vals in per_metric_accum.items()}

    return avg_composite, avg_metrics


def sum_costs(extractions: list[dict]) -> dict[str, float]:
    """Sum cost estimates across all extractions in a round.

    Returns dict with:
        total_regular: Total regular API cost for this round
        total_batch: Total batch API cost for this round
        avg_batch_per_paper: Average batch cost per paper
        total_input_tokens: Total estimated input tokens
        total_output_tokens: Total estimated output tokens
    """
    total_regular = 0.0
    total_batch = 0.0
    total_input = 0
    total_output = 0
    n_papers = 0

    for ex in extractions:
        cost = ex.get("_cost", {})
        total_regular += cost.get("cost_per_paper", 0)
        total_batch += cost.get("batch_cost_per_paper", 0)
        total_input += cost.get("input_tokens_est", 0)
        total_output += cost.get("output_tokens_est", 0)
        if not ex.get("_error"):
            n_papers += 1

    return {
        "total_regular": round(total_regular, 4),
        "total_batch": round(total_batch, 4),
        "avg_batch_per_paper": round(total_batch / max(n_papers, 1), 4),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def check_regression(
    old_metrics: dict[str, float],
    new_metrics: dict[str, float],
    threshold: float = 0.05,
) -> list[str]:
    """Return list of metric names that dropped more than threshold.

    Args:
        old_metrics: Per-metric scores from the previous iteration.
        new_metrics: Per-metric scores from the new prompt.
        threshold: Drop threshold; defaults to 0.05.

    Returns:
        List of metric names that regressed beyond the threshold.
    """
    regressed: list[str] = []
    for name, old_val in old_metrics.items():
        new_val = new_metrics.get(name, 0.0)
        if old_val - new_val > threshold:
            regressed.append(name)
    return regressed


def compute_change_ratio(old_prompt: str, new_prompt: str) -> float:
    """Fraction of lines that differ between old and new prompt.

    Uses a simple line-by-line comparison.  Ratio is computed as:
        changed_lines / max(len(old_lines), len(new_lines))

    Args:
        old_prompt: Text of the old prompt.
        new_prompt: Text of the new prompt.

    Returns:
        Float in [0, 1] representing the fraction of lines that changed.
    """
    old_lines = old_prompt.splitlines()
    new_lines = new_prompt.splitlines()
    max_len = max(len(old_lines), len(new_lines), 1)

    # Pad shorter list with empty strings
    old_padded = old_lines + [""] * (max_len - len(old_lines))
    new_padded = new_lines + [""] * (max_len - len(new_lines))

    changed = sum(1 for a, b in zip(old_padded, new_padded) if a != b)
    return changed / max_len


# ---------------------------------------------------------------------------
# Optimizer call
# ---------------------------------------------------------------------------


def call_optimizer(
    current_prompt: str,
    metrics: dict[str, float],
    composite: float,
    errors: list,
    history: list[dict],
    phase: str = "quality",
    current_batch_cost: float = 0.0,
    iteration: int = 0,
    consecutive_rejects: int = 0,
    strategy_index: int | None = None,
) -> tuple[str, str]:
    """Call the Sonnet optimizer agent to propose structured edits.

    The optimizer returns a JSON object with find/replace edits which are
    applied to the current prompt to produce the new version.

    Args:
        current_prompt: Full text of the current extraction prompt.
        metrics: Per-metric scores for the current prompt.
        composite: Composite score for the current prompt.
        errors: List of ErrorPattern objects from analyze_errors().
        history: List of previous iteration result dicts (last 5 used).
        phase: Current optimization phase ("quality" or "cost").
        current_batch_cost: Current average batch cost per paper (used in cost phase).
        iteration: Current iteration number (used for strategy rotation).
        consecutive_rejects: Number of consecutive rejects (triggers stronger diversity hints).
        strategy_index: If provided, use this index for lens selection instead of iteration.

    Returns:
        Tuple of (new_prompt, optimizer_summary).

    Raises:
        ValueError: If the optimizer output cannot be parsed or edits fail.
        RuntimeError: If the claude CLI exits with a non-zero return code.
    """
    system_prompt = PROGRAM_PATH.read_text(encoding="utf-8")

    # Serialize error patterns (top 8)
    error_dicts = [
        {
            "category": e.category,
            "description": e.description,
            "severity": round(e.severity, 3),
            "frequency": e.frequency,
            "examples": e.examples[:2],
        }
        for e in errors[:8]
    ]

    # Build context JSON
    context = {
        "composite_score": round(composite, 4),
        "per_metric_scores": {k: round(v, 4) for k, v in metrics.items()},
        "metric_weights": METRIC_WEIGHTS,
        "error_patterns": error_dicts,
        "edit_history": history[-5:],
        "phase": phase,
    }

    if phase == "cost":
        context["cost_per_paper"] = round(current_batch_cost, 6)
        context["cost_target"] = "reduce while maintaining quality > 0.93"

    # Select diversity strategy — rotate through strategies, with randomization
    strat_idx = strategy_index if strategy_index is not None else iteration
    strategy = DIVERSITY_STRATEGIES[strat_idx % len(DIVERSITY_STRATEGIES)]
    diversity_block = f"\n## Optimization Lens: {strategy['lens']}\n\n{strategy['hint']}\n"

    # If we've had many consecutive rejects, add a stronger diversity push
    if consecutive_rejects >= 5:
        # Extract summaries of recent failed attempts from history
        recent_failures = [
            h.get("optimizer_summary", "unknown approach")
            for h in history[-consecutive_rejects:]
            if h.get("outcome", "").startswith("rejected")
        ]
        failure_list = "\n".join(f"  - {s}" for s in recent_failures[-5:])
        diversity_block += (
            f"\n## IMPORTANT: {consecutive_rejects} consecutive rejects\n\n"
            "The following recent approaches ALL FAILED — do NOT repeat them:\n"
            f"{failure_list}\n\n"
            "You MUST try a fundamentally different strategy. Consider:\n"
            "- Editing a completely different section of the prompt\n"
            "- Using a different technique (examples vs rules vs checklists)\n"
            "- Targeting a different metric than the obvious choice\n"
            "- Simplifying or removing instructions instead of adding more\n"
        )

    user_message = (
        "## Current Extraction Prompt\n\n"
        "```markdown\n"
        f"{current_prompt}\n"
        "```\n\n"
        "## Current Scores and Error Analysis\n\n"
        "```json\n"
        f"{json.dumps(context, indent=2)}\n"
        "```\n"
        f"{diversity_block}\n"
        "Analyze the error patterns and propose 1-3 targeted find/replace edits. "
        "Output ONLY the JSON object with your edits — no explanation, no markdown fences."
    )

    cmd = [
        "claude",
        "-p",
        "--model",
        "sonnet",
        "--output-format",
        "text",
        "--max-turns",
        "5",
        "--system-prompt",
        system_prompt,
    ]

    result = subprocess.run(
        cmd,
        input=user_message,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"claude CLI exited with code {result.returncode}.\nstderr: {result.stderr[:500]}"
        )

    output = result.stdout.strip()

    # Parse the JSON response (with fallback for markdown fences)
    try:
        edits_obj = json.loads(output)
    except json.JSONDecodeError:
        # Try extracting from JSON fence
        fence = re.search(r"```(?:json)?\s*\n(.*?)```", output, re.DOTALL)
        if fence:
            edits_obj = json.loads(fence.group(1).strip())
        else:
            # Brace slice
            first = output.find("{")
            last = output.rfind("}")
            if first != -1 and last > first:
                edits_obj = json.loads(output[first : last + 1])
            else:
                raise ValueError(
                    f"Could not parse optimizer JSON. First 300 chars: {output[:300]!r}"
                )

    edits = edits_obj.get("edits", [])
    summary = edits_obj.get("summary", "")
    if summary:
        print(f"  Optimizer summary: {summary}")

    if not edits:
        raise ValueError("Optimizer returned no edits.")

    # Apply edits to prompt
    new_prompt = current_prompt
    applied = 0
    for i, edit in enumerate(edits):
        find_str = edit.get("find", "")
        replace_str = edit.get("replace", "")
        rationale = edit.get("rationale", "")

        if not find_str:
            print(f"  Edit {i + 1}: SKIP (empty find string)")
            continue

        if find_str not in new_prompt:
            print(f"  Edit {i + 1}: SKIP (find string not found in prompt)")
            print(f"    Find: {find_str[:80]!r}...")
            continue

        new_prompt = new_prompt.replace(find_str, replace_str, 1)
        applied += 1
        print(f"  Edit {i + 1}: APPLIED — {rationale[:80]}")

    if applied == 0:
        raise ValueError("No edits could be applied (all find strings missing).")

    print(f"  Applied {applied}/{len(edits)} edits")
    return new_prompt, summary


# ---------------------------------------------------------------------------
# Tournament optimizer
# ---------------------------------------------------------------------------


def run_optimizer_tournament(
    current_prompt: str,
    metrics: dict[str, float],
    composite: float,
    errors: list,
    history: list[dict],
    phase: str = "quality",
    current_batch_cost: float = 0.0,
    iteration: int = 0,
    consecutive_rejects: int = 0,
    n_candidates: int = 5,
) -> list[dict]:
    """Run N optimizer calls in parallel with different lenses.

    Args:
        current_prompt: Full text of the current extraction prompt.
        metrics: Per-metric scores for the current prompt.
        composite: Composite score for the current prompt.
        errors: List of ErrorPattern objects from analyze_errors().
        history: List of previous iteration result dicts (last 5 used).
        phase: Current optimization phase ("quality" or "cost").
        current_batch_cost: Current average batch cost per paper.
        iteration: Current iteration number.
        consecutive_rejects: Number of consecutive rejects.
        n_candidates: Number of parallel optimizer candidates to run.

    Returns:
        List of dicts with keys: new_prompt, summary, lens, strategy_index.
        Only includes successful candidates (failures are printed and skipped).
    """
    n_strats = len(DIVERSITY_STRATEGIES)
    # Pick N different lenses — sample without replacement if possible
    if n_candidates <= n_strats:
        indices = random.sample(range(n_strats), n_candidates)
    else:
        indices = list(range(n_strats)) + random.choices(range(n_strats), k=n_candidates - n_strats)

    print(f"  Launching {n_candidates} optimizer candidates in parallel...")
    lenses_str = ", ".join(DIVERSITY_STRATEGIES[i % n_strats]["lens"] for i in indices)
    print(f"  Lenses: {lenses_str}")

    candidates = []
    with ThreadPoolExecutor(max_workers=n_candidates) as executor:
        future_map: dict = {}
        for idx in indices:
            future = executor.submit(
                call_optimizer,
                current_prompt,
                metrics,
                composite,
                errors,
                history,
                phase,
                current_batch_cost,
                iteration,
                consecutive_rejects,
                idx,
            )
            future_map[future] = idx

        for future in as_completed(future_map):
            idx = future_map[future]
            lens = DIVERSITY_STRATEGIES[idx % n_strats]["lens"]
            try:
                new_prompt, summary = future.result()
                candidates.append(
                    {
                        "new_prompt": new_prompt,
                        "summary": summary,
                        "lens": lens,
                        "strategy_index": idx,
                    }
                )
            except Exception as exc:
                elapsed_info = f" ({exc})" if len(str(exc)) < 100 else ""
                print(f"  [{lens}] FAILED{elapsed_info}")

    print(f"  {len(candidates)}/{n_candidates} candidates succeeded")
    return candidates


# ---------------------------------------------------------------------------
# Version management
# ---------------------------------------------------------------------------


def save_version(
    prompt_text: str,
    version: str,
    metrics: dict[str, float],
    composite: float,
) -> Path:
    """Save a prompt version to disk with accompanying scores JSON.

    Args:
        prompt_text: Full text of the prompt.
        version: Version label (e.g., "v6.1_baseline" or "v6.2").
        metrics: Per-metric scores.
        composite: Composite score.

    Returns:
        Path to the saved prompt file.
    """
    VERSIONS_DIR.mkdir(exist_ok=True)

    prompt_path = VERSIONS_DIR / f"{version}.md"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    scores_path = VERSIONS_DIR / f"{version}_scores.json"
    scores_data = {
        "version": version,
        "composite": round(composite, 6),
        "metrics": {k: round(v, 6) for k, v in metrics.items()},
        "timestamp": datetime.now(UTC).isoformat(),
    }
    scores_path.write_text(json.dumps(scores_data, indent=2), encoding="utf-8")

    return prompt_path


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------


def run_full_evaluation(
    prompt: str,
    all_papers: list[dict],
    version_label: str,
    phase: str,
    rapid: bool = False,
) -> tuple[float, dict[str, float], dict[str, float], list[dict]]:
    """Run extraction on ALL papers and return (composite, metrics, costs, extractions).

    Args:
        prompt: Current extraction prompt text.
        all_papers: Full paper pool to evaluate against.
        version_label: Label used when saving the version snapshot.
        phase: Current optimization phase ("quality" or "cost"), printed in header.

    Returns:
        Tuple of (composite, metrics_dict, costs_dict, raw_extractions).
    """
    print(f"\n{'#' * 70}")
    print(f"FULL EVALUATION — {version_label}  [phase: {phase}]")
    print(f"{'#' * 70}")
    extractions = run_all_extractions(prompt, all_papers, timeout=600, rapid=rapid)
    composite, metrics = score_all(extractions, rapid=rapid)
    costs = sum_costs(extractions)
    # Print detailed results
    print(f"\nFull-eval composite: {composite:.4f}")
    for name, val in metrics.items():
        print(f"  {name:30s}: {val:.4f}")
    print(f"  Batch cost/paper: ${costs['avg_batch_per_paper']:.4f}")
    # Save version snapshot
    save_version(prompt, version_label, metrics, composite)
    return composite, metrics, costs, extractions


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the two-phase iterative prompt optimization loop."""
    parser = argparse.ArgumentParser(
        description="Iteratively optimize the KG extraction prompt using Sonnet (two-phase)."
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=200,
        help="Maximum iterations (default: 200 for overnight operation).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="Number of papers to sample per iteration (default: 3).",
    )
    parser.add_argument(
        "--no-rai14",
        action="store_true",
        default=False,
        help="Exclude the rai14 full-text paper from the test set.",
    )
    parser.add_argument(
        "--extra-papers",
        type=str,
        default=None,
        help="Path to JSON file with extra test papers [{id, title, full_text}].",
    )
    parser.add_argument(
        "--version-prefix",
        type=str,
        default="v7",
        help="Version label prefix for saved prompts (default: 'v7').",
    )
    parser.add_argument(
        "--skip-baseline",
        type=str,
        default=None,
        help="Path to existing baseline scores JSON — skip full baseline extraction.",
    )
    parser.add_argument(
        "--rapid",
        action="store_true",
        default=False,
        help="Rapid mode: aggressive truncation (Results+Methods+References only, 20K cap) and prefer short papers (<40K chars). ~3x faster iterations.",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=5,
        help="Number of parallel optimizer candidates per iteration (default: 5).",
    )
    args = parser.parse_args()

    include_rai14 = not args.no_rai14
    extra_corpus_path = (
        Path(args.extra_papers)
        if args.extra_papers
        else Path(SCRIPT_DIR / "optimize" / "extra_test_papers.json")
    )
    if not extra_corpus_path.exists():
        extra_corpus_path = None
    version_prefix = args.version_prefix
    sample_size = args.sample_size
    rapid = args.rapid
    n_candidates = args.candidates

    print("=" * 70)
    print("KG Extraction Prompt Optimizer  [Two-Phase, Overnight Mode]")
    print("=" * 70)
    print(f"  Max iterations : {args.max_iterations}")
    print(f"  Sample size    : {sample_size} papers/iteration")
    print(f"  Candidates     : {n_candidates} per iteration (tournament)")
    print(f"  Rapid mode     : {rapid}")
    print(f"  Include rai14  : {include_rai14}")
    print(f"  Extra papers   : {extra_corpus_path}")
    print(f"  Version prefix : {version_prefix}")
    print(f"  Phase 1 target : composite > {PHASE1_TARGET}")
    print(f"  Phase 2 floor  : quality > {PHASE2_QUALITY_FLOOR}")
    print()

    # --- Load ALL papers -------------------------------------------------------
    print("Loading all test papers...")
    all_papers = load_test_papers(
        micro_indices=list(range(100)),  # Filtered by corpus length and _MIN_TEXT_LENGTH
        include_rai14=include_rai14,
        extra_corpus_path=extra_corpus_path,
        max_text_length=80_000 if rapid else 0,
    )
    print(f"  Loaded {len(all_papers)} papers total:")
    for p in all_papers:
        print(f"    [{p['id']}] {p['title'][:60]}")
    print()

    # --- Load current prompt --------------------------------------------------
    current_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    print(f"Loaded prompt from: {PROMPT_PATH}")
    print(f"  Prompt length: {len(current_prompt)} chars")
    print()

    # --- Baseline ---------------------------------------------------------------
    if args.skip_baseline:
        # Load existing baseline scores and run a quick sample for error analysis
        baseline_scores_path = Path(args.skip_baseline)
        print(f"Loading baseline scores from: {baseline_scores_path}")
        baseline_data = json.loads(baseline_scores_path.read_text(encoding="utf-8"))
        baseline_metrics = baseline_data["metrics"]

        # Fill in any metrics missing from old scoring (e.g. evidence_depth)
        for name in METRIC_WEIGHTS:
            if name not in baseline_metrics:
                baseline_metrics[name] = 0.0
                print(f"  Note: metric '{name}' missing from saved scores — set to 0.0")

        # Recompute composite with rapid-aware weights if needed
        if rapid:
            active = {k: v for k, v in METRIC_WEIGHTS.items() if k not in RAPID_EXCLUDE}
            total_weight = sum(active.values())
            baseline_composite = (
                sum(active[n] * baseline_metrics.get(n, 0.0) for n in active) / total_weight
            )
            print(f"  Rapid mode: excluded {sorted(RAPID_EXCLUDE)} from composite")
        else:
            baseline_composite = baseline_data["composite"]

        baseline_batch_cost = 0.035  # Estimated from previous API run
        baseline_label = baseline_data.get("version", f"{version_prefix}.0_baseline")
        print(f"\nBaseline composite score: {baseline_composite:.4f} (from saved scores)")
        print()

        # Run a quick 3-paper sample for initial error analysis
        print("Running quick sample extraction for initial error analysis...")
        init_sample = random.sample(all_papers, min(sample_size, len(all_papers)))
        print(f"  Sample papers: {[p['id'] for p in init_sample]}")
        baseline_extractions = run_all_extractions(
            current_prompt, init_sample, timeout=600, rapid=rapid
        )
        init_costs = sum_costs(baseline_extractions)
        print(f"  Sample batch cost/paper: ${init_costs['avg_batch_per_paper']:.4f}")
        print()
    else:
        # Full baseline extraction on ALL papers
        print("Running baseline extraction on ALL papers...")
        baseline_label = f"{version_prefix}.0_baseline"
        baseline_composite, baseline_metrics, baseline_costs, baseline_extractions = (
            run_full_evaluation(
                current_prompt, all_papers, baseline_label, phase="quality", rapid=rapid
            )
        )
        print(f"\nBaseline composite score: {baseline_composite:.4f}")
        print()

        baseline_batch_cost = baseline_costs["avg_batch_per_paper"]
        print(f"  Cost this round:     ${baseline_costs['total_regular']:.3f} (regular API)")
        print(f"  Batch cost/paper:    ${baseline_batch_cost:.4f}")
        print(
            f"  Tokens (in/out):     {baseline_costs['total_input_tokens']:,} / {baseline_costs['total_output_tokens']:,}"
        )
        print()

    # --- Tracking variables ---------------------------------------------------
    best_prompt = current_prompt
    best_composite = baseline_composite
    best_metrics = baseline_metrics
    best_batch_cost = baseline_batch_cost

    # For error analysis we keep the last extractions
    last_sample_extractions: list[dict] = baseline_extractions

    history: list[dict] = []
    accepted_count = 0
    rejected_count = 0
    consecutive_rejects = 0
    cumulative_cost = 0.0 if args.skip_baseline else baseline_costs["total_regular"]

    phase: str = "quality"
    phase_transition_iteration: int | None = None
    full_evaluations: list[dict] = [
        {
            "iteration": 0,
            "label": baseline_label,
            "composite": round(baseline_composite, 6),
            "batch_cost_per_paper": round(baseline_batch_cost, 4),
        }
    ]

    # Track accepted count for full-eval checkpoints
    accepted_since_last_full_eval = 0

    # High-water mark: track the best composite seen across ALL iterations
    # (including rejected ones) so the best prompt wins at the end.
    hwm_composite = best_composite
    hwm_prompt = best_prompt
    hwm_metrics = dict(best_metrics)
    hwm_iteration = 0

    # --- Optimization loop ----------------------------------------------------
    for iteration in range(1, args.max_iterations + 1):
        print("=" * 70)
        print(f"Iteration {iteration} / {args.max_iterations}  [phase: {phase}]")
        print("=" * 70)

        # (a) Pick screening paper + validation papers (non-overlapping)
        random.shuffle(all_papers)
        screen_paper = all_papers[0]
        validation_papers = all_papers[1 : 1 + sample_size]
        print(f"  Screen paper: {screen_paper['id']}")
        print(f"  Validation papers: {[p['id'] for p in validation_papers]}")

        # (b) Analyze errors from last extractions (baseline on first iteration)
        errors = analyze_errors(last_sample_extractions)

        # (c) Check for convergence (no errors to fix)
        if not errors:
            print("No error patterns detected — prompt has converged.")
            break

        # (d) Print top 5 errors
        print(f"\nTop error patterns ({len(errors)} total):")
        for i, ep in enumerate(errors[:5], 1):
            print(f"  {i}. [{ep.category}] severity={ep.severity:.2f}, freq={ep.frequency}")
            print(f"     {ep.description[:100]}")
        print()

        # (e) Run optimizer tournament
        t0 = time.monotonic()
        candidates = run_optimizer_tournament(
            best_prompt,
            best_metrics,
            best_composite,
            errors,
            history,
            phase=phase,
            current_batch_cost=best_batch_cost,
            iteration=iteration,
            consecutive_rejects=consecutive_rejects,
            n_candidates=n_candidates,
        )
        tournament_elapsed = time.monotonic() - t0
        print(f"  Tournament completed in {tournament_elapsed:.1f}s")

        if not candidates:
            print("  All optimizer candidates failed — skipping iteration.")
            history.append(
                {
                    "iteration": iteration,
                    "outcome": "optimizer_error",
                    "error": "all tournament candidates failed",
                    "phase": phase,
                    "batch_cost_per_paper": round(best_batch_cost, 6),
                }
            )
            consecutive_rejects += 1
            rejected_count += 1
            if consecutive_rejects >= MAX_CONSECUTIVE_REJECTS:
                print(f"{MAX_CONSECUTIVE_REJECTS} consecutive failures — stopping.")
                break
            continue

        # (f) Validate candidates (check {PAPER_TEXT}, change ratio)
        valid_candidates = []
        for cand in candidates:
            prompt = cand["new_prompt"]
            if "{PAPER_TEXT}" not in prompt:
                prompt = prompt.rstrip() + "\n\n{PAPER_TEXT}\n"
                cand["new_prompt"] = prompt
            change_ratio = compute_change_ratio(best_prompt, prompt)
            cand["change_ratio"] = change_ratio
            if change_ratio > MAX_CHANGE_RATIO:
                print(f"  [{cand['lens']}] SKIP: change ratio {change_ratio:.2%} exceeds limit")
                continue
            valid_candidates.append(cand)

        if not valid_candidates:
            print("  All candidates exceeded change ratio — skipping iteration.")
            history.append(
                {
                    "iteration": iteration,
                    "outcome": "rejected_too_large",
                    "error": "all tournament candidates exceeded change ratio",
                    "phase": phase,
                    "batch_cost_per_paper": round(best_batch_cost, 6),
                }
            )
            consecutive_rejects += 1
            rejected_count += 1
            if consecutive_rejects >= MAX_CONSECUTIVE_REJECTS:
                print(f"{MAX_CONSECUTIVE_REJECTS} consecutive rejects — stopping.")
                break
            continue

        # (g) Screen: extract each valid candidate on 1 paper in parallel
        print(f"\nScreening {len(valid_candidates)} candidates on [{screen_paper['id']}]...")
        screen_results: list[dict] = []
        with ThreadPoolExecutor(max_workers=len(valid_candidates)) as executor:
            future_map: dict = {}
            for i, cand in enumerate(valid_candidates):
                future = executor.submit(
                    run_all_extractions,
                    cand["new_prompt"],
                    [screen_paper],
                    600,
                    None,
                    rapid,
                )
                future_map[future] = i

            for future in as_completed(future_map):
                idx = future_map[future]
                cand = valid_candidates[idx]
                extractions = future.result()
                s_composite, s_metrics = score_all(extractions, rapid=rapid)
                s_costs = sum_costs(extractions)
                screen_results.append(
                    {
                        "candidate_idx": idx,
                        "composite": s_composite,
                        "metrics": s_metrics,
                        "extractions": extractions,
                        "costs": s_costs,
                    }
                )
                print(
                    f"  [{cand['lens']}] screen composite: {s_composite:.4f}"
                    f" (change: {cand['change_ratio']:.1%})"
                )

        # Add screening costs to cumulative cost
        for sr in screen_results:
            cumulative_cost += sr["costs"].get("total_regular", 0)

        # (h) Pick best screened candidate
        screen_results.sort(key=lambda r: r["composite"], reverse=True)
        best_screen = screen_results[0]
        best_cand = valid_candidates[best_screen["candidate_idx"]]

        print(f"\n  Best candidate: [{best_cand['lens']}] composite {best_screen['composite']:.4f}")

        if best_screen["composite"] <= best_composite:
            print(
                f"  REJECT: best screen composite {best_screen['composite']:.4f}"
                f" <= baseline {best_composite:.4f}"
            )
            iter_label = f"{version_prefix}.{iteration}"
            save_version(
                best_cand["new_prompt"],
                iter_label,
                best_screen["metrics"],
                best_screen["composite"],
            )
            print(f"  Saved prompt as {iter_label}")

            history.append(
                {
                    "iteration": iteration,
                    "outcome": "rejected_no_improvement",
                    "old_composite": round(best_composite, 6),
                    "new_composite": round(best_screen["composite"], 6),
                    "optimizer_summary": best_cand["summary"],
                    "lens": best_cand["lens"],
                    "n_candidates": len(candidates),
                    "n_valid": len(valid_candidates),
                    "phase": phase,
                    "batch_cost_per_paper": round(
                        best_screen["costs"].get("avg_batch_per_paper", 0), 6
                    ),
                }
            )
            consecutive_rejects += 1
            rejected_count += 1

            # Update high-water mark
            if best_screen["composite"] > hwm_composite:
                hwm_composite = best_screen["composite"]
                hwm_prompt = best_cand["new_prompt"]
                hwm_metrics = dict(best_screen["metrics"])
                hwm_iteration = iteration

            if consecutive_rejects >= MAX_CONSECUTIVE_REJECTS:
                print(f"{MAX_CONSECUTIVE_REJECTS} consecutive rejects — stopping.")
                break
            continue

        # (i) Validation: extract best candidate on 3 different papers
        print(f"\n  Validating [{best_cand['lens']}] on {[p['id'] for p in validation_papers]}...")
        val_extractions = run_all_extractions(
            best_cand["new_prompt"], validation_papers, timeout=600, rapid=rapid
        )
        val_composite, val_metrics = score_all(val_extractions, rapid=rapid)
        val_costs = sum_costs(val_extractions)
        cumulative_cost += val_costs["total_regular"]
        new_batch_cost = val_costs["avg_batch_per_paper"]

        print(f"  Validation composite: {val_composite:.4f}")
        print(f"  Cost this validation: ${val_costs['total_regular']:.3f}")
        print(f"  Batch cost/paper:     ${new_batch_cost:.4f}")
        batch_delta = new_batch_cost - baseline_batch_cost
        if baseline_batch_cost > 0 and batch_delta / baseline_batch_cost > 0.50:
            print(
                f"  BATCH COST SPIKE: +{batch_delta / baseline_batch_cost:.0%} vs baseline"
                f" (${baseline_batch_cost:.4f} -> ${new_batch_cost:.4f})"
            )

        # Update high-water mark
        if val_composite > hwm_composite:
            hwm_composite = val_composite
            hwm_prompt = best_cand["new_prompt"]
            hwm_metrics = dict(val_metrics)
            hwm_iteration = iteration

        # (j) Print metric comparison (validation metrics vs baseline)
        print("\nMetric comparison (old vs new — validation):")
        print(f"  {'Metric':<30}  {'Old':>7}  {'New':>7}  {'Delta':>8}")
        print(f"  {'-' * 30}  {'-' * 7}  {'-' * 7}  {'-' * 8}")
        for name in METRIC_WEIGHTS:
            old_val = best_metrics.get(name, 0.0)
            new_val = val_metrics.get(name, 0.0)
            delta = new_val - old_val
            excluded = " (excl)" if rapid and name in RAPID_EXCLUDE else ""
            flag = " *" if abs(delta) >= 0.01 else ""
            print(f"  {name:<30}  {old_val:7.4f}  {new_val:7.4f}  {delta:+8.4f}{flag}{excluded}")
        composite_delta = val_composite - best_composite
        print(
            f"  {'COMPOSITE':<30}  {best_composite:7.4f}  {val_composite:7.4f}  {composite_delta:+8.4f}"
        )
        print()

        # Save version
        iter_label = f"{version_prefix}.{iteration}"
        save_version(best_cand["new_prompt"], iter_label, val_metrics, val_composite)
        print(f"  Saved prompt as {iter_label}")

        # (k) Phase-aware accept/reject logic (based on VALIDATION composite)
        accepted = False

        if phase == "quality":
            if val_composite > best_composite:
                accepted = True
            else:
                print(
                    f"  REJECT: validation composite did not improve"
                    f" ({val_composite:.4f} <= {best_composite:.4f})"
                )
                history.append(
                    {
                        "iteration": iteration,
                        "outcome": "rejected_no_improvement",
                        "old_composite": round(best_composite, 6),
                        "new_composite": round(val_composite, 6),
                        "screen_composite": round(best_screen["composite"], 6),
                        "optimizer_summary": best_cand["summary"],
                        "lens": best_cand["lens"],
                        "n_candidates": len(candidates),
                        "n_valid": len(valid_candidates),
                        "papers_used": [p["id"] for p in validation_papers],
                        "phase": phase,
                        "batch_cost_per_paper": round(new_batch_cost, 6),
                    }
                )
                consecutive_rejects += 1
                rejected_count += 1
                if consecutive_rejects >= MAX_CONSECUTIVE_REJECTS:
                    print(f"{MAX_CONSECUTIVE_REJECTS} consecutive rejects — stopping.")
                    break
                continue

        else:  # phase == "cost"
            if val_composite < PHASE2_QUALITY_FLOOR:
                print(
                    f"  REJECT [cost phase]: quality {val_composite:.4f}"
                    f" dropped below floor {PHASE2_QUALITY_FLOOR}"
                )
                history.append(
                    {
                        "iteration": iteration,
                        "outcome": "rejected_below_quality_floor",
                        "old_composite": round(best_composite, 6),
                        "new_composite": round(val_composite, 6),
                        "optimizer_summary": best_cand["summary"],
                        "lens": best_cand["lens"],
                        "phase": phase,
                        "batch_cost_per_paper": round(new_batch_cost, 6),
                    }
                )
                consecutive_rejects += 1
                rejected_count += 1
                if consecutive_rejects >= MAX_CONSECUTIVE_REJECTS:
                    print(f"{MAX_CONSECUTIVE_REJECTS} consecutive rejects — stopping.")
                    break
                continue

            if new_batch_cost < best_batch_cost:
                accepted = True
                print(
                    f"  Cost improved: ${best_batch_cost:.4f} -> ${new_batch_cost:.4f}"
                    f" ({(best_batch_cost - new_batch_cost) / best_batch_cost:.1%} reduction)"
                )
            elif val_composite > best_composite:
                accepted = True
                print("  Quality improved (cost did not worsen materially) — accepting.")
            else:
                print(
                    f"  REJECT [cost phase]: no cost reduction and no quality improvement"
                    f" (cost ${new_batch_cost:.4f}, composite {val_composite:.4f})"
                )
                history.append(
                    {
                        "iteration": iteration,
                        "outcome": "rejected_no_improvement",
                        "old_composite": round(best_composite, 6),
                        "new_composite": round(val_composite, 6),
                        "old_batch_cost": round(best_batch_cost, 6),
                        "new_batch_cost": round(new_batch_cost, 6),
                        "optimizer_summary": best_cand["summary"],
                        "lens": best_cand["lens"],
                        "phase": phase,
                        "batch_cost_per_paper": round(new_batch_cost, 6),
                    }
                )
                consecutive_rejects += 1
                rejected_count += 1
                if consecutive_rejects >= MAX_CONSECUTIVE_REJECTS:
                    print(f"{MAX_CONSECUTIVE_REJECTS} consecutive rejects — stopping.")
                    break
                continue

        # (l) Accept
        print(
            f"  ACCEPT: composite {best_composite:.4f} -> {val_composite:.4f} [{best_cand['lens']}]"
        )

        history.append(
            {
                "iteration": iteration,
                "outcome": "accepted",
                "version": iter_label,
                "old_composite": round(best_composite, 6),
                "new_composite": round(val_composite, 6),
                "screen_composite": round(best_screen["composite"], 6),
                "delta": round(composite_delta, 6),
                "optimizer_summary": best_cand["summary"],
                "lens": best_cand["lens"],
                "n_candidates": len(candidates),
                "n_valid": len(valid_candidates),
                "regressed_metrics": [],
                "papers_used": [p["id"] for p in validation_papers],
                "phase": phase,
                "batch_cost_per_paper": round(new_batch_cost, 6),
            }
        )

        best_prompt = best_cand["new_prompt"]
        best_composite = val_composite
        best_metrics = val_metrics
        best_batch_cost = new_batch_cost
        last_sample_extractions = val_extractions
        accepted_count += 1
        accepted_since_last_full_eval += 1
        consecutive_rejects = 0

        # (m) Full evaluation checkpoint every 5 accepted iterations
        if accepted_since_last_full_eval >= 5:
            full_eval_label = f"{version_prefix}.{iteration}_fulleval"
            full_eval_composite, full_eval_metrics, full_eval_costs, _ = run_full_evaluation(
                best_prompt, all_papers, full_eval_label, phase=phase, rapid=rapid
            )
            full_evaluations.append(
                {
                    "iteration": iteration,
                    "label": full_eval_label,
                    "composite": round(full_eval_composite, 6),
                    "batch_cost_per_paper": round(full_eval_costs["avg_batch_per_paper"], 4),
                }
            )
            accepted_since_last_full_eval = 0

            # Phase 2: enforce quality floor on full eval
            if phase == "cost" and full_eval_composite < PHASE2_QUALITY_FLOOR:
                print(
                    f"\nFull-eval quality {full_eval_composite:.4f} dropped below floor"
                    f" {PHASE2_QUALITY_FLOOR} — reverting to last good checkpoint."
                )
                # Revert to the last known-good state (we don't have it here, so stop cost phase)
                print("Stopping cost-reduction phase to preserve quality.")
                break

            # Update best from full eval (more reliable than sample)
            best_composite = full_eval_composite
            best_metrics = full_eval_metrics
            best_batch_cost = full_eval_costs["avg_batch_per_paper"]

            # Phase transition check (quality -> cost)
            if phase == "quality" and full_eval_composite > PHASE1_TARGET:
                print(f"\n{'*' * 70}")
                print(f"PHASE TRANSITION: Quality target {PHASE1_TARGET} reached!")
                print("Switching to Phase 2: Cost Reduction")
                print(f"{'*' * 70}\n")
                phase = "cost"
                phase_transition_iteration = iteration

        # (n) Check convergence
        if composite_delta < CONVERGENCE_THRESHOLD and phase == "quality":
            print(f"\nConverged: delta {composite_delta:.4f} < threshold {CONVERGENCE_THRESHOLD}.")
            break

    # --- Final full evaluation ------------------------------------------------
    # Use the high-water mark prompt (best composite seen across ALL iterations)
    print(f"\nHigh-water mark: composite {hwm_composite:.4f} at iteration {hwm_iteration}")
    print("Running final full evaluation using high-water mark prompt...")
    final_label = f"{version_prefix}_final"
    final_composite, final_metrics, final_costs, _ = run_full_evaluation(
        hwm_prompt, all_papers, final_label, phase=phase, rapid=rapid
    )
    full_evaluations.append(
        {
            "iteration": "final",
            "label": final_label,
            "composite": round(final_composite, 6),
            "batch_cost_per_paper": round(final_costs["avg_batch_per_paper"], 4),
            "hwm_iteration": hwm_iteration,
            "hwm_sample_composite": round(hwm_composite, 6),
        }
    )

    # Use final full eval as the authoritative best
    best_composite = final_composite

    # --- Final summary --------------------------------------------------------
    print()
    print("=" * 70)
    print("Optimization complete")
    print("=" * 70)
    print(f"  Iterations run : {len(history)}")
    print(f"  Accepted       : {accepted_count}")
    print(f"  Rejected       : {rejected_count}")
    print(f"  Baseline score : {baseline_composite:.4f}")
    print(f"  HWM sample     : {hwm_composite:.4f} (iter {hwm_iteration})")
    print(f"  Final score    : {final_composite:.4f}")
    improvement = final_composite - baseline_composite
    print(f"  Improvement    : {improvement:+.4f}")
    print(f"  Cumulative cost:   ${cumulative_cost:.3f} (regular API, this run)")
    print(f"  Baseline batch/paper: ${baseline_batch_cost:.4f}")
    print(f"  Final batch/paper:   ${final_costs['avg_batch_per_paper']:.4f}")
    if phase_transition_iteration is not None:
        print(f"  Phase transition at iteration: {phase_transition_iteration}")
    print()

    # Write best prompt back to PROMPT_PATH if improved — use HWM prompt
    if final_composite > baseline_composite:
        PROMPT_PATH.write_text(hwm_prompt, encoding="utf-8")
        print(f"Updated {PROMPT_PATH} with high-water mark prompt (iter {hwm_iteration}).")
    else:
        print("No improvement found — original prompt unchanged.")

    # Save optimization log
    LOG_PATH.parent.mkdir(exist_ok=True)
    log_entry = {
        "run_timestamp": datetime.now(UTC).isoformat(),
        "config": {
            "max_iterations": args.max_iterations,
            "sample_size": sample_size,
            "include_rai14": include_rai14,
            "convergence_threshold": CONVERGENCE_THRESHOLD,
            "quality_regression_threshold": QUALITY_REGRESSION_THRESHOLD,
            "max_change_ratio": MAX_CHANGE_RATIO,
            "max_consecutive_rejects": MAX_CONSECUTIVE_REJECTS,
            "phase1_target": PHASE1_TARGET,
            "phase2_quality_floor": PHASE2_QUALITY_FLOOR,
        },
        "papers": [{"id": p["id"], "title": p["title"]} for p in all_papers],
        "baseline_composite": round(baseline_composite, 6),
        "final_composite": round(final_composite, 6),
        "improvement": round(improvement, 6),
        "accepted": accepted_count,
        "rejected": rejected_count,
        "cumulative_cost_regular": round(cumulative_cost, 4),
        "baseline_batch_cost_per_paper": round(baseline_batch_cost, 4),
        "final_batch_cost_per_paper": round(final_costs["avg_batch_per_paper"], 4),
        "phase_transition_iteration": phase_transition_iteration,
        "full_evaluations": full_evaluations,
        "history": history,
    }

    # Append to existing log or create new
    if LOG_PATH.exists():
        try:
            existing = json.loads(LOG_PATH.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = [existing]
        except (json.JSONDecodeError, OSError):
            existing = []
        existing.append(log_entry)
        LOG_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    else:
        LOG_PATH.write_text(json.dumps([log_entry], indent=2), encoding="utf-8")

    print(f"Saved optimization log to {LOG_PATH}")


if __name__ == "__main__":
    main()
