"""Prompt templates for hybrid extraction refinement.

The hybrid extractor runs a programmatic extraction first (free, instant),
then sends a condensed summary to a cheap LLM for refinement. These prompts
guide that refinement step.
"""

from __future__ import annotations

from autoreview.extraction.models import PaperExtraction

HYBRID_SYSTEM_PROMPT = """You are an expert research analyst refining a draft paper extraction.
You receive a programmatic draft extraction alongside source excerpts from the paper.
Your job is to synthesize the draft into a polished, accurate structured extraction.

CRITICAL GROUNDING RULES:
- NEVER invent, round, or approximate numbers. Only use exact values that appear in the source material.
- If a VERIFIED NUMBERS section is provided, only use numbers from that list or the source excerpts.
- If you are unsure about a specific number, omit it rather than guess.
- Every claim must be directly supported by text in the source material.

Extraction rules:
- key_findings: Produce 8-12 synthesized claims. Combine related sentences into single clear claims. Include specific numbers from the source material verbatim.
- evidence_strength (be strict — default to weaker, upgrade only with clear justification):
  "strong": RCT, large-scale study (N>1000), meta-analysis, or result with p<0.01 and confidence intervals
  "moderate": Well-designed study WITH specific quantitative results (exact numbers, statistical tests)
  "weak": Qualitative findings, no statistical tests, small sample, observational without controls
  "preliminary": Abstract-only, preprint, pilot study, opinion/commentary, no original data
- quantitative_result: Copy exact numbers from the source. Include comparisons (X vs Y) when available. Never round or approximate.
- methods_summary: 3-5 sentence structured summary covering: approach/system, datasets used, evaluation methodology, key parameters.
- limitations: Numbered list of study-specific methodological limitations. Focus on what the AUTHORS acknowledge, not general field limitations.
- sample_size: The primary dataset/participant count as a single integer, or null if not clearly stated.

Output valid JSON only. No markdown fences, no explanation text."""


def build_refinement_prompt(
    draft: PaperExtraction,
    context: str,
) -> str:
    """Build the refinement prompt from draft extraction and source context.

    Args:
        draft: The programmatic extraction to refine.
        context: Condensed source material (title, abstract, top sentences, section excerpts).
    """
    # Format top 10 draft findings
    findings_lines: list[str] = []
    for i, f in enumerate(draft.key_findings[:10]):
        quant = f.quantitative_result or "none"
        findings_lines.append(
            f"  {i + 1}. [{f.evidence_strength}] {f.claim[:200]} (quant: {quant})"
        )
    draft_findings = "\n".join(findings_lines) if findings_lines else "  (no findings extracted)"

    return f'''Refine this draft paper extraction using the source material below.

## Source Material

{context}

## Draft Extraction (top 10 findings from programmatic analysis)

{draft_findings}

## Draft Methods Summary

{draft.methods_summary[:500]}

## Draft Limitations

{draft.limitations[:500]}

## Instructions

Produce a refined JSON extraction with these exact fields:
{{
  "paper_id": "{draft.paper_id}",
  "key_findings": [
    {{
      "claim": "specific synthesized finding with numbers",
      "evidence_strength": "strong|moderate|weak|preliminary",
      "quantitative_result": "specific numbers or null",
      "paper_id": "{draft.paper_id}"
    }}
  ],
  "methods_summary": "3-5 sentence structured methods summary",
  "limitations": "1. First limitation. 2. Second limitation. ...",
  "study_design": "one of: rct, cohort, case_control, cross_sectional, case_series, case_report, in_vitro, computational, meta_analysis, systematic_review, narrative_review, other",
  "quality_score": 0.0 to 1.0,
  "sample_size": integer or null
}}

Output ONLY the JSON object.'''
