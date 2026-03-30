#!/usr/bin/env python3
"""Run v6 KG extraction test on the Rai14 paper using Haiku via claude -p.

Compares v6 output against v5 baseline for predicate canonicalization,
evidence_strength alignment, causal_type coverage, and quantitative_context.
"""

import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPT_PATH = SCRIPT_DIR / "kg_extraction_prompt.md"
PAPER_PATH = SCRIPT_DIR / "rai14_fulltext.txt"
OUTPUT_PATH = SCRIPT_DIR / "extraction_test_haiku_v6.json"
V5_PATH = SCRIPT_DIR / "extraction_test_haiku_v5.json"

# --- Canonical predicate sets for validation ---
TIER1_CANONICAL = {
    "induces",
    "inhibits",
    "is_required_for",
    "is_sufficient_for",
    "regulates",
    "correlates_with",
    "interacts_with",
    "differentiates_into",
    "is_located_in",
    "is_marker_of",
    "is_component_of",
    "colocalizes_with",
}
TIER2_SPECIFIC = {
    "phosphorylates",
    "degrades",
    "stabilizes",
    "transports",
    "modifies",
    "converts",
    "maintains",
}
VALID_PREDICATES = TIER1_CANONICAL | TIER2_SPECIFIC
REMOVED_PREDICATES = {
    "activates",
    "promotes",
    "upregulates",
    "enhances",
    "suppresses",
    "blocks",
    "downregulates",
    "prevents",
    "reduces",
    "disrupts",
    "binds_to",
    "localizes_to",
    "is_expressed_in",
    "is_necessary_for",
    "enables",
    "mediates",
}
VALID_EVIDENCE_STRENGTHS = {
    "direct_experimental",
    "indirect_experimental",
    "observational",
    "computational",
    "review_citation",
}

# Load the v6 prompt
prompt_text = PROMPT_PATH.read_text()
marker = "{PAPER_TEXT}"
idx = prompt_text.find(marker)
system_prompt = prompt_text[:idx].rstrip() if idx != -1 else prompt_text

# Load the paper text
paper_text = PAPER_PATH.read_text()
print(f"Paper text: {len(paper_text)} chars")
print(f"System prompt: {len(system_prompt)} chars")

# Build the user prompt
user_prompt = (
    "Extract all falsifiable claims from the following paper as structured JSON "
    "according to the schema in your system prompt. Output ONLY valid JSON.\n\n"
    "---\n\n" + paper_text
)

# Call claude -p with haiku model
print("Running v6 extraction via claude -p (haiku)...")
start = time.time()

result = subprocess.run(
    [
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
    ],
    input=user_prompt,
    capture_output=True,
    text=True,
    timeout=300,
)

elapsed = time.time() - start
print(f"Extraction took {elapsed:.1f}s")

if result.returncode != 0:
    print(f"STDERR: {result.stderr[:2000]}")
    sys.exit(1)

raw_output = result.stdout.strip()
print(f"Raw output: {len(raw_output)} chars")

# Extract JSON from output
try:
    data = json.loads(raw_output)
except json.JSONDecodeError:
    fence_matches = list(re.finditer(r"```(?:json)?\s*\n?(.*?)```", raw_output, re.DOTALL))
    if fence_matches:
        best = max(fence_matches, key=lambda m: len(m.group(1)))
        data = json.loads(best.group(1).strip())
    else:
        first = raw_output.find("{")
        last = raw_output.rfind("}")
        if first == -1 or last <= first:
            print(f"No JSON found in output. First 500 chars:\n{raw_output[:500]}")
            sys.exit(1)
        data = json.loads(raw_output[first : last + 1])

# Save
with open(OUTPUT_PATH, "w") as f:
    json.dump(data, f, indent=2)

claims = data.get("claims", [])
evidence = data.get("evidence", [])

# ===================================================================
# V6 Extraction Stats
# ===================================================================
print("\n" + "=" * 60)
print("V6 EXTRACTION RESULTS")
print("=" * 60)
print(f"Claims: {len(claims)}")
print(f"Evidence units: {len(evidence)}")

# --- Predicate analysis ---
all_predicates = [c.get("predicate", "") for c in claims]
pred_counts = Counter(all_predicates)
canonical_used = [p for p in all_predicates if p in TIER1_CANONICAL]
specific_used = [p for p in all_predicates if p in TIER2_SPECIFIC]
removed_used = [p for p in all_predicates if p in REMOVED_PREDICATES]
unknown_used = [
    p for p in all_predicates if p not in VALID_PREDICATES and p not in REMOVED_PREDICATES
]

print("\nPredicate distribution:")
for pred, count in pred_counts.most_common():
    tier = (
        "T1"
        if pred in TIER1_CANONICAL
        else "T2"
        if pred in TIER2_SPECIFIC
        else "REMOVED"
        if pred in REMOVED_PREDICATES
        else "UNKNOWN"
    )
    print(f"  {pred}: {count} [{tier}]")

print(
    f"\nCanonical (Tier 1): {len(canonical_used)}/{len(all_predicates)} ({100 * len(canonical_used) / max(len(all_predicates), 1):.0f}%)"
)
print(
    f"Specific (Tier 2):  {len(specific_used)}/{len(all_predicates)} ({100 * len(specific_used) / max(len(all_predicates), 1):.0f}%)"
)
print(
    f"REMOVED (BAD):      {len(removed_used)}/{len(all_predicates)} ({100 * len(removed_used) / max(len(all_predicates), 1):.0f}%)"
)
print(f"Unknown (BAD):      {len(unknown_used)}/{len(all_predicates)}")
if removed_used:
    print(f"  Removed predicates used: {Counter(removed_used)}")
if unknown_used:
    print(f"  Unknown predicates used: {Counter(unknown_used)}")

# --- Evidence strength analysis ---
claim_strengths = [c.get("evidence_strength", "") for c in claims]
ev_strengths = [e.get("evidence_strength", "") for e in evidence]
invalid_claim_strengths = [s for s in claim_strengths if s not in VALID_EVIDENCE_STRENGTHS]
invalid_ev_strengths = [s for s in ev_strengths if s not in VALID_EVIDENCE_STRENGTHS]
print(f"\nEvidence strength (claims): {Counter(claim_strengths)}")
print(f"Evidence strength (evidence): {Counter(ev_strengths)}")
if invalid_claim_strengths:
    print(f"  INVALID claim strengths: {Counter(invalid_claim_strengths)}")
if invalid_ev_strengths:
    print(f"  INVALID evidence strengths: {Counter(invalid_ev_strengths)}")

# --- V6-specific field coverage ---
has_model_system = sum(1 for c in claims if c.get("model_system"))
has_organism = sum(1 for c in claims if c.get("organism"))
has_certainty = sum(1 for c in claims if c.get("certainty"))
has_section_source = sum(1 for c in claims if c.get("section_source"))
has_quant_ctx = sum(
    1
    for c in claims
    if c.get("quantitative_context")
    and c["quantitative_context"] is not None
    and any(v for v in c["quantitative_context"].values() if v)
)
mechanistic = [c for c in claims if c.get("claim_type") == "mechanistic_causal"]
has_causal_type = sum(1 for c in mechanistic if c.get("causal_type"))

print("\nField coverage:")
print(f"  model_system:        {has_model_system}/{len(claims)}")
print(f"  organism:            {has_organism}/{len(claims)}")
print(f"  certainty:           {has_certainty}/{len(claims)}")
print(f"  section_source:      {has_section_source}/{len(claims)}")
print(f"  quantitative_context: {has_quant_ctx}/{len(claims)}")
print(f"  causal_type (of mechanistic): {has_causal_type}/{len(mechanistic)}")

# --- Direction convention ---
neg_direction = [c for c in claims if c.get("direction") == "negative"]
print(f"\nDirection: negative claims: {len(neg_direction)}/{len(claims)}")
for c in neg_direction:
    print(f"  {c['claim_id']}: {c['predicate']} dir=negative — {c['natural_language'][:80]}")

# --- Claim type distribution ---
type_dist = Counter(c.get("claim_type", "unknown") for c in claims)
section_dist = Counter(c.get("section_source", "unknown") for c in claims)
print(f"\nClaim types: {dict(type_dist)}")
print(f"Section sources: {dict(section_dist)}")

# --- Citation contexts ---
citations = data.get("citation_contexts", [])
print(f"\nCitation contexts: {len(citations)}")
if citations:
    rel_dist = Counter(cc.get("relationship", "") for cc in citations)
    print(f"  Relationships: {dict(rel_dist)}")

# ===================================================================
# V5 vs V6 Comparison
# ===================================================================
if V5_PATH.exists():
    v5 = json.loads(V5_PATH.read_text())
    v5_claims = v5.get("claims", [])
    v5_evidence = v5.get("evidence", [])
    v5_predicates = [c.get("predicate", "") for c in v5_claims]
    v5_removed = [p for p in v5_predicates if p in REMOVED_PREDICATES]
    v5_canonical = [p for p in v5_predicates if p in TIER1_CANONICAL]
    v5_mechanistic = [c for c in v5_claims if c.get("claim_type") == "mechanistic_causal"]
    v5_has_causal = sum(1 for c in v5_mechanistic if c.get("causal_type"))
    v5_strengths = [c.get("evidence_strength", "") for c in v5_claims]
    v5_invalid_strengths = [s for s in v5_strengths if s not in VALID_EVIDENCE_STRENGTHS]
    v5_quant = sum(
        1
        for c in v5_claims
        if c.get("quantitative_context")
        and c["quantitative_context"] is not None
        and any(v for v in c["quantitative_context"].values() if v)
    )

    print("\n" + "=" * 60)
    print("V5 vs V6 COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<40} {'V5':>8} {'V6':>8} {'Delta':>8}")
    print("-" * 60)
    print(
        f"{'Total claims':<40} {len(v5_claims):>8} {len(claims):>8} {len(claims) - len(v5_claims):>+8}"
    )
    print(
        f"{'Total evidence units':<40} {len(v5_evidence):>8} {len(evidence):>8} {len(evidence) - len(v5_evidence):>+8}"
    )
    print(
        f"{'Canonical predicates (Tier 1)':<40} {len(v5_canonical):>8} {len(canonical_used):>8} {len(canonical_used) - len(v5_canonical):>+8}"
    )
    print(
        f"{'REMOVED predicates used (BAD)':<40} {len(v5_removed):>8} {len(removed_used):>8} {len(removed_used) - len(v5_removed):>+8}"
    )
    print(
        f"{'Invalid evidence_strength':<40} {len(v5_invalid_strengths):>8} {len(invalid_claim_strengths):>8} {len(invalid_claim_strengths) - len(v5_invalid_strengths):>+8}"
    )
    print(
        f"{'causal_type on mechanistic claims':<40} {v5_has_causal:>8} {has_causal_type:>8} {has_causal_type - v5_has_causal:>+8}"
    )
    print(
        f"{'quantitative_context populated':<40} {v5_quant:>8} {has_quant_ctx:>8} {has_quant_ctx - v5_quant:>+8}"
    )

    # Predicate shift detail
    v5_pred_dist = Counter(v5_predicates)
    print(f"\nV5 predicate distribution: {dict(v5_pred_dist)}")

print(f"\nSaved to: {OUTPUT_PATH}")
