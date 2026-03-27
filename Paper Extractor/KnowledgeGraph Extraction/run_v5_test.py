#!/usr/bin/env python3
"""Run v5 KG extraction test on the Rai14 paper using local Haiku via claude -p."""

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
OUTPUT_PATH = SCRIPT_DIR / "extraction_test_haiku_v5.json"

# Load the v5 prompt
prompt_text = PROMPT_PATH.read_text()
# Split at {PAPER_TEXT} placeholder
marker = "{PAPER_TEXT}"
idx = prompt_text.find(marker)
if idx != -1:
    system_prompt = prompt_text[:idx].rstrip()
else:
    system_prompt = prompt_text

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

# Write system prompt to a temp file to avoid CLI arg length limits
SYSTEM_PROMPT_TMP = SCRIPT_DIR / "_system_prompt_tmp.md"
SYSTEM_PROMPT_TMP.write_text(system_prompt)

# Call claude -p with haiku model, system prompt from file, user prompt via stdin
print("Running extraction via claude -p (haiku)...")
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
        "2",
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
    print(f"STDERR: {result.stderr[:1000]}")
    sys.exit(1)

raw_output = result.stdout.strip()
print(f"Raw output: {len(raw_output)} chars")

# Extract JSON from output
try:
    data = json.loads(raw_output)
except json.JSONDecodeError:
    # Try markdown fence
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw_output, re.DOTALL)
    if fence_match:
        data = json.loads(fence_match.group(1).strip())
    else:
        # Try first { to last }
        first = raw_output.find("{")
        last = raw_output.rfind("}")
        if first == -1 or last <= first:
            print(f"No JSON found in output. First 500 chars:\n{raw_output[:500]}")
            sys.exit(1)
        data = json.loads(raw_output[first : last + 1])

# Save
with open(OUTPUT_PATH, "w") as f:
    json.dump(data, f, indent=2)

# Quick stats
claims = data.get("claims", [])
evidence = data.get("evidence", [])
print("\n=== V5 Extraction Results ===")
print(f"Claims: {len(claims)}")
print(f"Evidence units: {len(evidence)}")

# Count v5-specific fields
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

print(f"Claims with model_system: {has_model_system}/{len(claims)}")
print(f"Claims with organism: {has_organism}/{len(claims)}")
print(f"Claims with certainty: {has_certainty}/{len(claims)}")
print(f"Claims with section_source: {has_section_source}/{len(claims)}")
print(f"Claims with quantitative_context: {has_quant_ctx}/{len(claims)}")

# Assertion type distribution
type_dist = Counter(c.get("claim_type", "unknown") for c in claims)
print(f"\nClaim types: {dict(type_dist)}")

section_dist = Counter(c.get("section_source", "unknown") for c in claims)
print(f"Section sources: {dict(section_dist)}")

print(f"\nSaved to: {OUTPUT_PATH}")
