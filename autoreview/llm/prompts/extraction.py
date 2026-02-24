from __future__ import annotations


EXTRACTION_SYSTEM_PROMPT = """\
You are an expert research analyst performing structured extraction from scientific papers. \
Extract key findings, methodology, limitations, and relationships to other work.

Guidelines:
- Each finding should be a specific, falsifiable claim
- Evidence strength: "strong" (RCT, large-scale study, meta-analysis), "moderate" (well-designed \
observational), "weak" (small sample, methodological limitations), "preliminary" (abstract-only, \
preprint, pilot)
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
"""
