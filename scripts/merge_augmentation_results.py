#!/usr/bin/env python3
"""Merge augmentation results into base extractions and update snapshots."""

import glob
import hashlib
import json
import os
from collections import Counter

BASE_DIR = "/Users/mst36/Desktop/Projects/Science/AutoReview/output/arise/arise_rag_v3"
BASE_EXTRACTIONS = os.path.join(BASE_DIR, "extractions_base.json")
AUG_RESULTS_GLOB = os.path.join(BASE_DIR, "augmentation_results", "batch_*.json")
OUTPUT_SNAPSHOT = os.path.join(BASE_DIR, "snapshots", "05_extraction.json")
LATEST_SNAPSHOT = os.path.join(BASE_DIR, "snapshots", "latest.json")


def _compute_checksum(data: dict) -> str:
    """SHA256 of JSON-serialised dict, excluding _checksum key."""
    payload = {k: v for k, v in data.items() if k != "_checksum"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_base_extractions() -> dict:
    with open(BASE_EXTRACTIONS) as f:
        return json.load(f)


def load_augmentation_results() -> list[dict]:
    """Load and concatenate all batch_*.json files, sorted numerically."""
    batch_files = sorted(
        glob.glob(AUG_RESULTS_GLOB),
        key=lambda p: int(os.path.basename(p).replace("batch_", "").replace(".json", "")),
    )
    records = []
    for path in batch_files:
        with open(path) as f:
            batch = json.load(f)
        records.extend(batch)
        print(f"  Loaded {os.path.basename(path)}: {len(batch)} records")
    return records


def merge(base: dict, aug_records: list[dict]) -> tuple[dict, dict]:
    """Merge augmentation fields into base extractions.

    Returns:
        merged: updated extractions dict
        stats: counts for reporting
    """
    merged = {k: dict(v) for k, v in base.items()}  # deep-copy at dict level

    matched = 0
    unmatched = []
    study_design_dist: Counter = Counter()

    for rec in aug_records:
        pid = rec["paper_id"]
        if pid not in merged:
            unmatched.append(pid)
            continue

        merged[pid]["study_design"] = rec.get("study_design")
        merged[pid]["quality_score"] = rec.get("quality_score")
        merged[pid]["sample_size"] = rec.get("sample_size")
        matched += 1

        sd = rec.get("study_design")
        if sd:
            study_design_dist[sd] += 1
        else:
            study_design_dist["<null>"] += 1

    stats = {
        "total_base": len(base),
        "total_aug_records": len(aug_records),
        "matched": matched,
        "unmatched": len(unmatched),
        "unmatched_ids": unmatched,
        "has_study_design": sum(1 for v in merged.values() if v.get("study_design")),
        "has_quality_score": sum(1 for v in merged.values() if v.get("quality_score") is not None),
        "has_sample_size": sum(1 for v in merged.values() if v.get("sample_size") is not None),
        "study_design_distribution": dict(study_design_dist.most_common()),
    }
    return merged, stats


def save_extraction_snapshot(merged: dict) -> None:
    os.makedirs(os.path.dirname(OUTPUT_SNAPSHOT), exist_ok=True)
    with open(OUTPUT_SNAPSHOT, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nSaved extraction snapshot: {OUTPUT_SNAPSHOT}")


def update_latest_snapshot(merged: dict) -> None:
    with open(LATEST_SNAPSHOT) as f:
        snapshot = json.load(f)

    snapshot["extractions"] = merged
    snapshot["current_phase"] = "extraction"
    # Bump schema version (keep same major, add timestamp awareness)
    snapshot["_schema_version"] = snapshot.get("_schema_version", 1)
    # Recompute checksum over the full updated payload
    snapshot["_checksum"] = _compute_checksum(snapshot)

    with open(LATEST_SNAPSHOT, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"Updated latest snapshot: {LATEST_SNAPSHOT}")
    print(f"  _schema_version: {snapshot['_schema_version']}")
    print(f"  _checksum: {snapshot['_checksum']}")
    print(f"  current_phase: {snapshot['current_phase']}")


def print_stats(stats: dict) -> None:
    print("\n=== Merge Statistics ===")
    print(f"Total base extractions  : {stats['total_base']}")
    print(f"Total augmentation recs : {stats['total_aug_records']}")
    print(f"Successfully merged     : {stats['matched']}")
    print(f"Unmatched paper IDs     : {stats['unmatched']}")
    if stats["unmatched_ids"]:
        for pid in stats["unmatched_ids"]:
            print(f"  - {pid}")
    print("\nFields populated after merge:")
    print(f"  study_design   : {stats['has_study_design']}/{stats['total_base']}")
    print(f"  quality_score  : {stats['has_quality_score']}/{stats['total_base']}")
    print(f"  sample_size    : {stats['has_sample_size']}/{stats['total_base']}")
    print("\nStudy design distribution:")
    for sd, count in sorted(stats["study_design_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {sd:<35} {count:>4}")


def main() -> None:
    print("Loading base extractions...")
    base = load_base_extractions()
    print(f"  {len(base)} papers loaded")

    print("\nLoading augmentation results...")
    aug_records = load_augmentation_results()
    print(f"  {len(aug_records)} total augmentation records")

    print("\nMerging...")
    merged, stats = merge(base, aug_records)

    save_extraction_snapshot(merged)
    update_latest_snapshot(merged)
    print_stats(stats)


if __name__ == "__main__":
    main()
