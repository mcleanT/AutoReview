"""
prepare_extraction_batches.py

Reads v2 extractions and converts them to PaperExtraction-compatible format,
then creates augmentation batch files for downstream subagents.
"""

import json
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "output/arise/arise_rag_v2/snapshots/05_extraction.json"
OUTPUT_DIR = REPO_ROOT / "output/arise/arise_rag_v3"
BASE_OUTPUT = OUTPUT_DIR / "extractions_base.json"
BATCHES_DIR = OUTPUT_DIR / "augmentation_batches"

BATCH_SIZE = 50
MAX_METHODS_CHARS = 500
MAX_FINDINGS_CHARS = 500


def convert_extraction(paper_id: str, legacy: dict) -> dict:
    """Convert a single legacy extraction to PaperExtraction-compatible dict."""

    # --- key_findings ---
    raw_findings = legacy.get("key_findings") or []
    key_findings = []
    for f in raw_findings:
        key_findings.append(
            {
                "paper_id": paper_id,
                "claim": f.get("claim", ""),
                "evidence_strength": f.get("evidence_strength", ""),
                # rename quantitative_results → quantitative_result
                "quantitative_result": f.get("quantitative_results", ""),
                "context": None,
            }
        )

    # --- limitations: list[str] → single string ---
    raw_limitations = legacy.get("limitations") or []
    if isinstance(raw_limitations, list):
        limitations = "; ".join(raw_limitations)
    else:
        limitations = str(raw_limitations)

    # --- relationships: string → empty list (not structured enough to parse) ---
    relationships: list = []

    # --- domain_specific_fields ---
    domain_specific_fields = {
        "category": legacy.get("category"),
        "source_type": legacy.get("source_type"),
    }

    return {
        "paper_id": paper_id,
        "title": legacy.get("title", ""),
        "authors": legacy.get("authors", []),
        "year": legacy.get("year"),
        "doi": legacy.get("doi", ""),
        "key_findings": key_findings,
        "methods_summary": legacy.get("methods_summary", ""),
        "limitations": limitations,
        "relationships": relationships,
        "methodology_details": None,
        "domain_specific_fields": domain_specific_fields,
        "study_design": None,
        "quality_score": None,
        "sample_size": None,
    }


def make_batch_entry(paper_id: str, extraction: dict) -> dict:
    """Build the minimal representation needed for augmentation subagents."""

    # key_findings_text: join all claims as bullet points, truncated
    claims = [f["claim"] for f in extraction["key_findings"] if f.get("claim")]
    findings_text = "\n".join(f"- {c}" for c in claims)
    if len(findings_text) > MAX_FINDINGS_CHARS:
        findings_text = findings_text[:MAX_FINDINGS_CHARS] + "…"

    methods = (extraction.get("methods_summary") or "")[:MAX_METHODS_CHARS]

    return {
        "paper_id": paper_id,
        "title": extraction.get("title", ""),
        "methods_summary": methods,
        "key_findings_text": findings_text,
        "category": (extraction.get("domain_specific_fields") or {}).get("category"),
    }


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load legacy extractions
    # ------------------------------------------------------------------
    print(f"Reading: {INPUT_PATH}")
    with INPUT_PATH.open() as fh:
        raw = json.load(fh)

    legacy_extractions: dict = raw.get("extractions", {})
    print(f"Found {len(legacy_extractions)} legacy extractions")

    # ------------------------------------------------------------------
    # 2. Convert all extractions
    # ------------------------------------------------------------------
    converted: dict[str, dict] = {}
    for paper_id, legacy in legacy_extractions.items():
        converted[paper_id] = convert_extraction(paper_id, legacy)

    # ------------------------------------------------------------------
    # 3. Save combined base file
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with BASE_OUTPUT.open("w") as fh:
        json.dump(converted, fh, indent=2)
    print(f"Saved {len(converted)} converted extractions → {BASE_OUTPUT}")

    # ------------------------------------------------------------------
    # 4. Create augmentation batch files
    # ------------------------------------------------------------------
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)

    paper_ids = list(converted.keys())
    num_batches = math.ceil(len(paper_ids) / BATCH_SIZE)

    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = start + BATCH_SIZE
        batch_ids = paper_ids[start:end]

        batch = [make_batch_entry(pid, converted[pid]) for pid in batch_ids]

        batch_file = BATCHES_DIR / f"batch_{batch_idx + 1}.json"
        with batch_file.open("w") as fh:
            json.dump(batch, fh, indent=2)

    print(f"Created {num_batches} augmentation batches (≤{BATCH_SIZE} papers each) → {BATCHES_DIR}")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Total extractions converted : {len(converted)}")
    print(f"  Augmentation batches created: {num_batches}")
    print(f"  Base output                 : {BASE_OUTPUT}")
    print(f"  Batches directory           : {BATCHES_DIR}")


if __name__ == "__main__":
    main()
