"""Generate PDF reports for each Mycelium extraction run.

Reads three JSON extraction outputs, parses them with the Pydantic schema,
renders a structured Markdown report for each, and converts to PDF via weasyprint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import markdown
from weasyprint import HTML

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUTS = Path(__file__).parent
SCHEMA_DIR = OUTPUTS.parent / "src" / "mycelium"

RUNS = [
    {
        "json_path": OUTPUTS / "extraction_test_opus.json",
        "pdf_path": OUTPUTS / "extraction_report_opus_v1.pdf",
        "model_label": "claude-opus-4 (v1 schema)",
    },
    {
        "json_path": OUTPUTS / "extraction_test_sonnet_v2.json",
        "pdf_path": OUTPUTS / "extraction_report_sonnet_v2.pdf",
        "model_label": "claude-sonnet-4-5 (v2 schema)",
    },
    {
        "json_path": OUTPUTS / "extraction_test_haiku.json",
        "pdf_path": OUTPUTS / "extraction_report_haiku_v2.pdf",
        "model_label": "claude-haiku-4-5 (v2 schema)",
    },
]

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
body {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 11pt;
    line-height: 1.5;
    margin: 40px;
    color: #1a1a1a;
}
h1 {
    color: #0072B2;
    border-bottom: 3px solid #0072B2;
    padding-bottom: 8px;
    font-size: 20pt;
}
h2 {
    color: #333;
    border-bottom: 1px solid #ccc;
    padding-bottom: 4px;
    margin-top: 30px;
    font-size: 16pt;
}
h3 {
    color: #D55E00;
    margin-top: 20px;
    font-size: 13pt;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
}
th, td {
    border: 1px solid #ddd;
    padding: 6px 10px;
    text-align: left;
    font-size: 10pt;
}
th {
    background: #f5f5f5;
    font-weight: bold;
}
blockquote {
    border-left: 3px solid #0072B2;
    padding-left: 12px;
    color: #555;
    margin: 10px 0;
}
code {
    background: #f4f4f4;
    padding: 2px 4px;
    border-radius: 3px;
    font-size: 10pt;
}
hr {
    border: none;
    border-top: 1px solid #eee;
    margin: 20px 0;
}
.meta-block {
    background: #f9f9f9;
    border: 1px solid #ddd;
    padding: 12px 16px;
    border-radius: 4px;
    margin-bottom: 20px;
    font-size: 10pt;
}
@page {
    margin: 1in;
    @bottom-center {
        content: counter(page);
        font-size: 9pt;
        color: #999;
    }
}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def esc(val: object) -> str:
    """Return a safe string representation, replacing None/empty with em-dash."""
    if val is None:
        return "—"
    s = str(val).strip()
    return s if s else "—"


def esc_md(val: object) -> str:
    """Escape markdown special characters in a value string."""
    s = esc(val)
    # Escape pipe characters inside table cells
    return s.replace("|", "\\|")


def trunc(s: str, n: int = 120) -> str:
    return s[:n] + "…" if len(s) > n else s


def fmt_list(items: list) -> str:
    """Format a list of OntologyTerm-like dicts or strings as a comma-separated string."""
    if not items:
        return "—"
    parts = []
    for item in items:
        if isinstance(item, dict):
            name = item.get("term_name") or item.get("surface_form") or str(item)
            parts.append(name)
        else:
            parts.append(str(item))
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Markdown builders
# ---------------------------------------------------------------------------


def build_header(data: dict, model_label: str) -> str:
    prov = data.get("paper_provenance", {})
    meta = data.get("extraction_metadata", {})

    title = esc(prov.get("title"))
    authors = prov.get("authors", [])
    first_authors = [
        a.get("name", "") for a in authors if a.get("role") in ("first_author", "senior_author")
    ]
    author_str = "; ".join(first_authors) or "—"
    journal = esc(prov.get("journal"))
    pub_date = esc(prov.get("publication_date"))
    doi = esc(prov.get("doi"))
    peer = (
        "Yes"
        if prov.get("peer_reviewed")
        else ("No" if prov.get("peer_reviewed") is False else "—")
    )

    ext_model = esc(meta.get("extraction_model"))
    ext_version = esc(meta.get("extraction_version"))
    ext_ts = esc(meta.get("extraction_timestamp"))
    char_count = esc(meta.get("paper_char_count"))
    duration = meta.get("extraction_duration_seconds")
    duration_str = f"{duration:.1f} s" if isinstance(duration, (int, float)) else "—"

    has_citations = bool(data.get("citation_contexts"))
    schema_version = "v2 (CitationContext present)" if has_citations else "v1 (no CitationContext)"

    assertions = data.get("assertion_drafts", [])
    evidence = data.get("evidence_units", [])
    citations = data.get("citation_contexts", [])

    novel = sum(
        1
        for a in assertions
        if (a.get("epistemic_status") or {}).get("function") == "novel_finding"
    )
    background = sum(
        1 for a in assertions if (a.get("epistemic_status") or {}).get("function") == "background"
    )

    lines = [
        f"# Mycelium Extraction Report: {model_label}",
        "",
        '<div class="meta-block">',
        "",
        f"**Paper:** {title}",
        "",
        f"**Authors (first/senior):** {author_str}",
        "",
        f"**Journal:** {journal} | **Date:** {pub_date} | **Peer-reviewed:** {peer}",
        "",
        f"**DOI:** {doi}",
        "",
        f"**Extraction model:** {ext_model} | **Pipeline version:** {ext_version}",
        "",
        f"**Timestamp:** {ext_ts} | **Paper chars:** {char_count} | **Duration:** {duration_str}",
        "",
        f"**Schema version:** {schema_version}",
        "",
        "</div>",
        "",
        "---",
        "",
        "## Summary Statistics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Assertion drafts | {len(assertions)} |",
        f"| Evidence units | {len(evidence)} |",
        f"| Citation contexts | {len(citations)} |",
        f"| Novel findings | {novel} |",
        f"| Background claims | {background} |",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def build_assertions(data: dict) -> str:
    assertions = data.get("assertion_drafts", [])
    if not assertions:
        return "## Assertion Drafts\n\n*No assertion drafts extracted.*\n\n---\n\n"

    lines = ["## Assertion Drafts", ""]

    for i, a in enumerate(assertions, 1):
        draft_id = esc(a.get("draft_id", f"a_{i:03d}"))
        canonical = esc(a.get("canonical_form"))
        natural = esc(a.get("natural_language"))
        negatable = esc(a.get("negatable_form"))

        label = f"A-{i:03d}"
        lines.append(f"### {label}: {trunc(canonical, 90)}")
        lines.append("")
        lines.append(f"**Natural language:** {natural}")
        lines.append("")
        lines.append(f"**Negatable form:** {negatable}")
        lines.append("")

        # Core fields table
        ep = a.get("epistemic_status") or {}
        sub = a.get("subject_entity") or {}
        obj = a.get("object_entity") or {}
        lines += [
            "| Field | Value |",
            "|-------|-------|",
            f"| Draft ID | {esc_md(draft_id)} |",
            f"| Type | {esc_md(a.get('assertion_type'))} |",
            f"| Causal type | {esc_md(a.get('causal_type'))} |",
            f"| Direction | {esc_md(a.get('direction'))} |",
            f"| Predicate | {esc_md(a.get('predicate'))} |",
            f"| Subject | {esc_md(sub.get('surface_form'))} ({esc_md(sub.get('entity_type'))}) |",
            f"| Object | {esc_md(obj.get('surface_form') if obj else None)} ({esc_md(obj.get('entity_type') if obj else None)}) |",
            f"| Section | {esc_md(ep.get('section'))} |",
            f"| Function | {esc_md(ep.get('function'))} |",
            f"| Primary | {esc_md(ep.get('is_primary'))} |",
            "",
        ]

        # Scope
        scope = a.get("scope") or {}
        species_str = fmt_list(scope.get("species", []))
        cell_type_str = fmt_list(scope.get("cell_type", []))
        in_vitro = scope.get("in_vitro")
        in_vitro_str = "Yes" if in_vitro is True else ("No" if in_vitro is False else "—")
        condition_str = esc(scope.get("condition"))

        lines += [
            "**Scope:**",
            f"- Species: {species_str}",
            f"- Cell type: {cell_type_str}",
            f"- in_vitro: {in_vitro_str}",
            f"- Condition: {condition_str}",
            "",
        ]

        # Hedging
        hedge = a.get("hedging") or {}
        verbatim = esc(hedge.get("verbatim_hedge"))
        certainty = esc(hedge.get("certainty"))
        generalizability = esc(hedge.get("generalizability"))
        causality = esc(hedge.get("causality_hedge"))

        lines += [
            "**Hedging:**",
            f'- Verbatim: *"{verbatim}"*',
            f"- Certainty: {certainty} | Generalizability: {generalizability} | Causality: {causality}",
            "",
        ]

        # Evidence links
        ev_ids = a.get("evidence_unit_ids", [])
        ev_str = ", ".join(ev_ids) if ev_ids else "—"
        lines.append(f"**Evidence:** {ev_str}")
        lines.append("")

        # Provenance
        prov = a.get("provenance") or {}
        source = esc(prov.get("source_sentence"))
        lines.append(f'**Source:** *"{trunc(source, 300)}"*')
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def build_evidence(data: dict) -> str:
    units = data.get("evidence_units", [])
    if not units:
        return "## Evidence Units\n\n*No evidence units extracted.*\n\n---\n\n"

    lines = ["## Evidence Units", ""]

    for i, e in enumerate(units, 1):
        ev_id = esc(e.get("evidence_id", f"e_{i:03d}"))
        exp = e.get("experiment") or {}
        res = e.get("results") or {}
        tags = e.get("methodological_tags") or {}

        desc = esc(exp.get("description"))
        label = f"E-{i:03d}"
        lines.append(f"### {label}: {trunc(desc, 100)}")
        lines.append("")

        # Top-level table
        assertion_ids = e.get("assertion_draft_ids", [])
        assoc_str = ", ".join(assertion_ids) if assertion_ids else "—"
        lines += [
            "| Field | Value |",
            "|-------|-------|",
            f"| Evidence ID | {esc_md(ev_id)} |",
            f"| Direction | {esc_md(e.get('evidence_direction'))} |",
            f"| Strength | {esc_md(e.get('evidence_strength'))} |",
            f"| Linked assertions | {esc_md(assoc_str)} |",
            "",
        ]

        # Experiment
        pert_type = esc(exp.get("perturbation_type"))
        pert_target = esc(exp.get("perturbation_target"))
        pert_method = esc(exp.get("perturbation_method"))
        lines += [
            "**Experiment:**",
            f"- Model system: {esc(exp.get('model_system'))}",
            f"- Organism: {esc(exp.get('organism'))}",
            f"- Perturbation: {pert_type} → {pert_target} via {pert_method}",
            f"- Readout: {esc(exp.get('readout'))}",
            f"- Control: {esc(exp.get('control_description'))}",
            "",
        ]

        # Results
        assay_types = tags.get("assay_types", [])
        assay_str = ", ".join(assay_types) if assay_types else "—"
        lines += [
            "**Results:**",
            f"- Direction: {esc(res.get('result_direction'))}",
            f"- Effect: {esc(res.get('effect_description'))}",
            f"- Effect size: {esc(res.get('effect_size'))}",
            f"- p-value: {esc(res.get('p_value'))}",
            f"- Sample size: {esc(res.get('sample_size'))}",
            f"- Key figure: {esc(res.get('key_figure'))}",
            "",
            f"**Methods:** {assay_str}",
            "",
        ]

        # Limitations
        limitations = e.get("limitations_stated_by_authors", [])
        if limitations:
            lines.append("**Limitations (author-stated):**")
            for lim in limitations:
                lines.append(f"- {lim}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def build_citations(data: dict) -> str:
    citations = data.get("citation_contexts", [])
    if not citations:
        return ""

    lines = ["## Citation Contexts (v2)", ""]

    for i, c in enumerate(citations, 1):
        cit_id = esc(c.get("citation_id", f"c_{i:03d}"))
        label = f"C-{i:03d}"
        citing = esc(c.get("citing_sentence"))
        claimed = esc(c.get("cited_claim_paraphrase"))

        lines.append(f"### {label}")
        lines.append("")
        lines.append(f'**Citing sentence:** *"{trunc(citing, 400)}"*')
        lines.append("")
        lines.append(f"**Cited claim:** {claimed}")
        lines.append("")

        linked = c.get("linked_assertion_draft_ids", [])
        linked_str = ", ".join(linked) if linked else "—"
        lines += [
            "| Field | Value |",
            "|-------|-------|",
            f"| Citation ID | {esc_md(cit_id)} |",
            f"| Relationship | {esc_md(c.get('relationship'))} |",
            f"| Reference | {esc_md(c.get('cited_source_ref_key'))} |",
            f"| DOI | {esc_md(c.get('cited_source_doi'))} |",
            f"| Linked assertions | {esc_md(linked_str)} |",
            f"| Section | {esc_md(c.get('section'))} |",
            "",
            "---",
            "",
        ]

    return "\n".join(lines)


def build_funding_conflicts(data: dict) -> str:
    prov = data.get("paper_provenance", {})
    funding = prov.get("funding_sources", [])
    coi = esc(prov.get("conflicts_of_interest"))
    data_avail = esc(prov.get("data_availability"))

    lines = ["## Paper Provenance Details", ""]

    if funding:
        lines.append("**Funding sources:**")
        for f in funding:
            lines.append(f"- {f}")
        lines.append("")

    lines.append(f"**Conflicts of interest:** {coi}")
    lines.append("")
    lines.append(f"**Data availability:** {data_avail}")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_report(run: dict) -> tuple[str, int]:
    """Generate a PDF for one run. Returns (pdf_path, approx_page_count)."""
    json_path = run["json_path"]
    pdf_path = run["pdf_path"]
    model_label = run["model_label"]

    print(f"\n[{model_label}] Reading {json_path.name} …")
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)

    # Build markdown sections
    md_parts = [
        build_header(data, model_label),
        build_assertions(data),
        build_evidence(data),
        build_citations(data),
        build_funding_conflicts(data),
    ]
    full_md = "\n".join(md_parts)

    # Convert to HTML
    html_body = markdown.markdown(
        full_md,
        extensions=["tables", "fenced_code", "nl2br"],
    )
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    # Render PDF
    print(f"[{model_label}] Rendering PDF …")
    HTML(string=html_doc).write_pdf(str(pdf_path))

    size_kb = pdf_path.stat().st_size // 1024
    print(f"[{model_label}] Done → {pdf_path.name} ({size_kb} KB)")
    return str(pdf_path), size_kb


def main() -> None:
    results = []
    for run in RUNS:
        try:
            pdf_path, size_kb = generate_report(run)
            results.append((run["model_label"], pdf_path, size_kb, None))
        except Exception as exc:
            print(f"ERROR for {run['model_label']}: {exc}", file=sys.stderr)
            results.append((run["model_label"], run["pdf_path"], 0, str(exc)))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label, path, kb, err in results:
        if err:
            print(f"  FAILED  {label}: {err}")
        else:
            print(f"  OK      {label}: {Path(path).name} ({kb} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
