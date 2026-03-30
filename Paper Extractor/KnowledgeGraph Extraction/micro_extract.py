#!/usr/bin/env python3
"""Micro extraction — 10-paper direct API test using the v5 KG pipeline.

Calls the Anthropic Messages API directly (not batch) so results are immediate.
Uses the same prompt, schema, truncation, and coercion as batch_extract_kg.py.

Usage:
    ANTHROPIC_API_KEY=sk-... python micro_extract.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_AUTOREVIEW_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_AUTOREVIEW_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

import anthropic
from batch_extract_kg import (
    MAX_OUTPUT_TOKENS,
    MODEL,
    TRUNCATION_CONFIG,
    _coerce_kg_dict,
    _extract_json_str,
    _load_system_prompt,
    _paper_hash,
)
from kg_schema import KGExtraction

from autoreview.extraction.truncation import section_aware_truncate

OUTPUT_DIR = _SCRIPT_DIR / "gastruloid_run" / "micro_v5"


def run_micro(api_key: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sample_path = _SCRIPT_DIR / "gastruloid_run" / "micro_sample.json"
    papers = json.loads(sample_path.read_text())
    system_prompt = _load_system_prompt()

    client = anthropic.Anthropic(api_key=api_key)

    results = []
    total_input = 0
    total_output = 0

    for i, paper in enumerate(papers):
        doi = paper.get("doi", "")
        title = paper.get("title", "")[:60]
        phash = _paper_hash(doi, paper.get("title"))
        text = paper.get("full_text", "")
        text_len = len(text)

        if not text:
            print(f"  [{i + 1}/10] SKIP (no text): {title}")
            continue

        if len(text) > 5000:
            text = section_aware_truncate(text, 100_000, TRUNCATION_CONFIG)

        print(f"  [{i + 1}/10] Extracting: {title}... ({text_len:,} chars)", flush=True)
        t0 = time.time()

        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.0,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": "{"},
                ],
            ) as stream:
                response = stream.get_final_message()

            content = response.content[0].text if response.content else ""
            if content and not content.lstrip().startswith("{"):
                content = "{" + content

            elapsed = time.time() - t0
            in_tok = response.usage.input_tokens
            out_tok = response.usage.output_tokens
            total_input += in_tok
            total_output += out_tok

            json_str = _extract_json_str(content)
            raw_dict = json.loads(json_str)
            coerced = _coerce_kg_dict(raw_dict)
            validated = KGExtraction.model_validate(coerced)

            # Save extraction
            out_path = OUTPUT_DIR / f"{phash}.json"
            out_path.write_text(
                json.dumps(validated.model_dump(), indent=2, default=str, ensure_ascii=False)
            )

            n_claims = len(validated.claims)
            n_evidence = len(validated.evidence)
            n_citations = len(validated.citation_contexts)
            stop = response.stop_reason

            result = {
                "doi": doi,
                "title": paper.get("title", ""),
                "hash": phash,
                "status": "success",
                "claims": n_claims,
                "evidence": n_evidence,
                "citation_contexts": n_citations,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "output_chars": len(content),
                "elapsed_s": round(elapsed, 1),
                "stop_reason": stop,
                "text_chars": text_len,
            }
            results.append(result)

            print(
                f"          OK: {n_claims} claims, {n_evidence} evidence, {n_citations} cit_ctx | "
                f"{out_tok:,} out tokens | {elapsed:.1f}s | stop={stop}"
            )

        except Exception as e:
            elapsed = time.time() - t0
            print(f"          FAIL: {e!s:.120}")
            # Save raw for debugging
            if "content" in dir():
                (OUTPUT_DIR / f"{phash}_raw.txt").write_text(content)
            results.append(
                {
                    "doi": doi,
                    "title": paper.get("title", ""),
                    "hash": phash,
                    "status": "error",
                    "error": str(e)[:200],
                    "elapsed_s": round(elapsed, 1),
                    "text_chars": text_len,
                }
            )

    # Summary
    successes = [r for r in results if r["status"] == "success"]
    failures = [r for r in results if r["status"] != "success"]

    input_cost = total_input * 0.80 / 1_000_000
    output_cost = total_output * 4.00 / 1_000_000
    total_cost = input_cost + output_cost

    print(f"\n{'=' * 70}")
    print("  Micro v5 Extraction Complete")
    print(f"{'=' * 70}")
    print(f"  Succeeded: {len(successes)} / {len(results)}")
    print(f"  Failed:    {len(failures)}")
    if successes:
        claims = [r["claims"] for r in successes]
        evidence = [r["evidence"] for r in successes]
        out_toks = [r["output_tokens"] for r in successes]
        print(
            f"  Claims:    {sum(claims)} total (avg {sum(claims) / len(claims):.1f}/paper, "
            f"min={min(claims)}, max={max(claims)})"
        )
        print(f"  Evidence:  {sum(evidence)} total (avg {sum(evidence) / len(evidence):.1f}/paper)")
        print(f"  Output tokens: {total_output:,} (avg {total_output / len(successes):,.0f}/paper)")
        print(f"  Input tokens:  {total_input:,}")
        print(f"  Cost (non-batch): ${total_cost:.2f}")
        print(f"  Cost (batch est): ${total_cost / 2:.2f}")
        for r in successes:
            trunc = " TRUNCATED" if r["stop_reason"] == "max_tokens" else ""
            print(
                f"    {r['doi'][:40]:<42} {r['claims']:>3}c {r['evidence']:>3}e "
                f"{r['output_tokens']:>7,}tok {r['elapsed_s']:>5.1f}s{trunc}"
            )
    print(f"{'=' * 70}")

    # Save results
    log = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "papers": len(results),
        "successes": len(successes),
        "failures": len(failures),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "cost_non_batch_usd": round(total_cost, 4),
        "cost_batch_est_usd": round(total_cost / 2, 4),
        "results": results,
    }
    log_path = OUTPUT_DIR / "micro_run_log.json"
    log_path.write_text(json.dumps(log, indent=2, default=str))
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    import os

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY")
        sys.exit(1)
    run_micro(api_key)
