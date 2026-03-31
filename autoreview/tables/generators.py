"""Markdown table generation for review papers."""

from __future__ import annotations

from typing import Any

import structlog

from autoreview.models.visuals import TableMetadata, VisualInsertionAnchor

logger = structlog.get_logger()


def generate_retrieval_table() -> TableMetadata:
    """Generate retrieval paradigm comparison table (Table 1)."""
    md = (
        "| Method | Mechanism | Key Systems | Strengths | Limitations |\n"
        "|--------|-----------|-------------|-----------|-------------|\n"
        "| Dense bi-encoder | Semantic embedding similarity | DPR, M3-Embedding"
        " | Semantic matching, zero-shot transfer | Representation bottleneck, domain shift |\n"  # noqa: E501
        "| Sparse lexical | Term overlap (BM25, TF-IDF) | BM25, SPLADE"
        " | Exact match, no GPU, domain robust | No semantic understanding |\n"
        "| Hybrid fusion | Score combination of dense + sparse | RRF pipelines"
        " | Best of both, robust across domains | Complexity, tuning fusion weights |\n"  # noqa: E501
        "| Late interaction | Per-token similarity (MaxSim) | ColBERT, ColPali"
        " | Fine-grained matching, scalable | Larger index size |\n"
        "| Learned sparse | Neural term weighting | SPLADE, ANCE-PRF"
        " | Inverted index efficiency + semantics | Training data requirements |"
    )

    return TableMetadata(
        key="table1_retrieval",
        markdown=md,
        caption="Table 1. Comparison of retrieval paradigms for RAG systems.",
        anchor=VisualInsertionAnchor(section_id="sec_3_1", position="before"),
    )


def generate_domain_table(theme_counts: dict[str, int] | None = None) -> TableMetadata:
    """Generate domain application summary table (Table 2)."""
    counts = theme_counts or {}
    rows = [
        (
            "Biomedical/Clinical",
            counts.get("Biomedical", "~190"),
            "Patient safety, regulatory compliance, EHR heterogeneity",
            "BiomedRAG, Patho-AgenticRAG",
        ),
        (
            "Legal",
            counts.get("Legal", "~25"),
            "Citation networks, jurisdiction hierarchy",
            "COLIEE systems",
        ),
        (
            "Financial",
            counts.get("Financial", "~15"),
            "Temporal sensitivity, numerical reasoning",
            "SEC filing evaluators",
        ),
        (
            "Education",
            counts.get("Education", "~20"),
            "Pedagogical accuracy, curriculum alignment",
            "Educational QA systems",
        ),
        (
            "Enterprise",
            counts.get("Enterprise", "~30"),
            "Multilingual, metadata enrichment, privacy",
            "Enterprise RAG platforms",
        ),
        (
            "Multimodal",
            counts.get("Multimodal", "~104"),
            "Cross-modal retrieval, document understanding",
            "ColPali, M3DocRAG, EVisRAG",
        ),
    ]
    header = (
        "| Domain | Papers | Key Challenges | Representative Systems |\n"
        "|--------|--------|----------------|----------------------|"
    )
    body = "\n".join(f"| {d} | {p} | {c} | {s} |" for d, p, c, s in rows)
    md = f"{header}\n{body}"

    return TableMetadata(
        key="table2_domains",
        markdown=md,
        caption="Table 2. Summary of RAG applications across domains.",
        anchor=VisualInsertionAnchor(section_id="sec_10", position="before"),
    )


def generate_evaluation_table() -> TableMetadata:
    """Generate evaluation framework comparison table (Table 3)."""
    md = (
        "| Framework | Metrics | Scope | Automation | Key Limitation |\n"
        "|-----------|---------|-------|------------|---------------|\n"
        "| RAGAS | Faithfulness, relevance, context | Component-level"
        " | Fully automated (LLM) | Sensitive to judge model |\n"
        "| ARES | Confidence intervals on quality | End-to-end + component"
        " | Automated with PPI | Requires labeled calibration |\n"
        "| RAGBench | TRACe explainability scores | End-to-end | Automated | English-only |\n"
        "| ASTRID | Multi-dimension faithfulness | Faithfulness focus"
        " | Automated | Narrow scope |\n"
        "| Human eval | Task-specific rubrics | End-to-end | Manual"
        " | Expensive, low reliability |"
    )

    return TableMetadata(
        key="table3_evaluation",
        markdown=md,
        caption="Table 3. Comparison of RAG evaluation frameworks.",
        anchor=VisualInsertionAnchor(section_id="sec_8_1", position="before"),
    )


def generate_takeaways_table(outline: dict[str, Any], llm: Any | None = None) -> TableMetadata:
    """Generate key takeaways table (Table 4). Uses LLM if available, else outline descriptions."""
    sections = outline.get("sections", [])
    body_sections = [
        s
        for s in sections
        if s.get("id", "").startswith("sec_")
        and s["id"] not in ("sec_1", "sec_2", "sec_13", "sec_14")
    ]

    rows = []
    for s in body_sections:
        sid = s.get("id", "")
        title = s.get("title", "")
        desc = s.get("description", title)
        # Use first sentence of description as takeaway (no LLM needed for basic version)
        takeaway = desc.split(". ")[0] + "." if ". " in desc else desc
        num = sid.replace("sec_", "§")
        rows.append(f"| {num} {title} | {takeaway} |")

    header = "| Section | Key Takeaway |\n|---------|-------------|"
    md = f"{header}\n" + "\n".join(rows)

    return TableMetadata(
        key="table4_takeaways",
        markdown=md,
        caption="Table 4. Key findings across review sections.",
        anchor=VisualInsertionAnchor(section_id="sec_13", position="before"),
    )


def generate_all_tables(kb: Any, llm: Any | None = None) -> dict[str, TableMetadata]:
    """Generate all 4 tables and return metadata dict."""
    # Extract theme counts for domain table
    theme_counts: dict[str, int] = {}
    if kb.evidence_map and hasattr(kb.evidence_map, "themes"):
        for t in kb.evidence_map.themes:
            name = t.name if hasattr(t, "name") else str(t)
            count = len(t.paper_ids) if hasattr(t, "paper_ids") else 0
            theme_counts[name] = count

    if isinstance(kb.outline, dict):
        outline = kb.outline
    else:
        logger.warning("table_generation.outline_not_dict", type=type(kb.outline).__name__)
        outline = {}

    return {
        "table1_retrieval": generate_retrieval_table(),
        "table2_domains": generate_domain_table(theme_counts),
        "table3_evaluation": generate_evaluation_table(),
        "table4_takeaways": generate_takeaways_table(outline, llm),
    }
