from __future__ import annotations

EXTRACTION_SYSTEM_PROMPT = """\
You are an expert research analyst performing structured extraction from scientific papers. \
Extract key findings, methodology, limitations, and relationships to other work.

Guidelines:
- Each finding should be a specific, falsifiable claim
- Evidence strength (be strict — default to weaker, upgrade only with clear justification):
  "strong": RCT, large-scale study (N>1000), meta-analysis, or result with p<0.01 and confidence intervals
  "moderate": Well-designed study WITH specific quantitative results (exact numbers, statistical tests)
  "weak": Qualitative findings, no statistical tests, small sample, observational without controls
  "preliminary": Abstract-only, preprint, pilot study, opinion/commentary, no original data
- Quantitative results should include effect sizes, confidence intervals, or p-values when available
- Limitations should focus on methodological weaknesses that affect interpretation
- Domain-specific fields should only be populated if the information is explicitly stated
"""


def build_extraction_prompt(
    title: str,
    text: str,
    text_source: str,
    domain_fields: dict[str, bool] | None = None,
) -> str:
    """Build an extraction prompt for a single paper."""
    source_note = ""
    if text_source == "abstract":
        source_note = (
            "\n**Note:** Only the abstract is available. Mark evidence strength as "
            "'preliminary' unless the abstract provides strong quantitative evidence."
        )
    elif text_source == "title_only":
        source_note = (
            "\n**Note:** Only the title is available. Extract minimal information "
            "and mark all evidence as 'preliminary'."
        )

    domain_section = ""
    if domain_fields:
        fields_list = ", ".join(f for f, enabled in domain_fields.items() if enabled)
        if fields_list:
            domain_section = (
                f"\n\n## Domain-Specific Fields\n"
                f"Also extract these fields if mentioned: {fields_list}\n"
                f"Place them in domain_specific_fields as key-value pairs."
            )

    return f"""\
## Paper
**Title:** {title}
**Source:** {text_source}{source_note}

## Content
{text}
{domain_section}

Extract all key findings, methodology details, limitations, and relationships to other work.

Additionally, classify the study design and quality:
- study_design: One of: rct, cohort, case_control, cross_sectional, case_series, case_report, in_vitro, computational, meta_analysis, systematic_review, narrative_review, other
- quality_score: Rate 0.0-1.0 based on methodology rigor (1.0 = highest rigor, e.g. large RCT or meta-analysis; 0.0 = minimal rigor)
- sample_size: Total number of subjects/samples if reported, otherwise null
"""
