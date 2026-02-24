from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import structlog

from autoreview.models.knowledge_base import KnowledgeBase
from autoreview.models.paper import CandidatePaper
from autoreview.output.bibliography import BibliographyFormatter

logger = structlog.get_logger()

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _resolve_citations(
    text: str,
    papers: list[CandidatePaper],
    style: str = "apa",
) -> tuple[str, list[str]]:
    """Replace [@paper_id] markers with numbered or author-year citations.

    Returns:
        Tuple of (resolved text, ordered list of cited paper IDs).
    """
    paper_map = {p.id: p for p in papers}
    cited_ids: list[str] = []
    citation_numbers: dict[str, int] = {}

    def replace_citation(match: re.Match) -> str:
        pid = match.group(1)
        if pid not in citation_numbers:
            citation_numbers[pid] = len(citation_numbers) + 1
            cited_ids.append(pid)

        paper = paper_map.get(pid)
        num = citation_numbers[pid]

        if style == "vancouver":
            return f"[{num}]"
        elif paper and paper.authors:
            first_author = paper.authors[0].split()[0]  # Last name
            year = paper.year or "n.d."
            return f"({first_author}, {year})"
        else:
            return f"[{num}]"

    resolved = re.sub(r"\[@([^\]]+)\]", replace_citation, text)
    return resolved, cited_ids


class OutputFormatter:
    """Formats the final review paper output."""

    def __init__(self, style: str = "apa") -> None:
        self.style = style

    def format_markdown(self, kb: KnowledgeBase) -> str:
        """Format the review as Markdown."""
        if not kb.full_draft:
            return ""

        all_papers = [sp.paper for sp in kb.screened_papers] + kb.candidate_papers
        resolved, cited_ids = _resolve_citations(kb.full_draft, all_papers, self.style)

        bib_formatter = BibliographyFormatter(style=self.style)
        unique_papers = {p.id: p for p in all_papers}
        cited_papers = [unique_papers[pid] for pid in cited_ids if pid in unique_papers]
        bibliography = bib_formatter.format_bibliography(cited_papers, cited_ids)

        return f"{resolved}\n\n---\n\n## References\n\n{bibliography}\n"

    def format_latex(self, kb: KnowledgeBase) -> tuple[str, str]:
        """Format the review as LaTeX.

        Returns:
            Tuple of (latex_content, bibtex_content).
        """
        if not kb.full_draft:
            return "", ""

        all_papers = [sp.paper for sp in kb.screened_papers] + kb.candidate_papers
        unique_papers = {p.id: p for p in all_papers}

        # Generate BibTeX entries
        bibtex_entries = []
        for pid, paper in unique_papers.items():
            key = pid[:12]
            authors = " and ".join(paper.authors) if paper.authors else "Unknown"
            title = _latex_escape(paper.title)
            journal = _latex_escape(paper.journal or "")
            year = paper.year or 0
            doi = paper.doi or ""
            bibtex_entries.append(
                f"@article{{{key},\n"
                f"  author = {{{authors}}},\n"
                f"  title = {{{title}}},\n"
                f"  journal = {{{journal}}},\n"
                f"  year = {{{year}}},\n"
                f"  doi = {{{doi}}},\n"
                f"}}"
            )

        bibtex = "\n\n".join(bibtex_entries)

        # Convert markdown to basic LaTeX
        latex = _markdown_to_latex(kb.full_draft)

        return latex, bibtex

    def save(self, kb: KnowledgeBase, output_dir: str, fmt: str = "markdown") -> list[str]:
        """Save formatted output to files.

        Returns:
            List of created file paths.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        created: list[str] = []

        if fmt in ("markdown", "all"):
            md_path = out / "review.md"
            md_path.write_text(self.format_markdown(kb))
            created.append(str(md_path))
            logger.info("output.saved", format="markdown", path=str(md_path))

        if fmt in ("latex", "all"):
            latex, bibtex = self.format_latex(kb)
            tex_path = out / "review.tex"
            bib_path = out / "references.bib"
            tex_path.write_text(latex)
            bib_path.write_text(bibtex)
            created.extend([str(tex_path), str(bib_path)])
            logger.info("output.saved", format="latex", path=str(tex_path))

        if fmt in ("docx", "all"):
            try:
                import pypandoc
                md_content = self.format_markdown(kb)
                docx_path = out / "review.docx"
                pypandoc.convert_text(md_content, "docx", format="md", outputfile=str(docx_path))
                created.append(str(docx_path))
                logger.info("output.saved", format="docx", path=str(docx_path))
            except ImportError:
                logger.warning("output.docx_skipped", reason="pypandoc not installed")
            except OSError:
                logger.warning("output.docx_skipped", reason="pandoc not found on PATH")

        return created


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters."""
    special = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
               "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    for char, replacement in special.items():
        text = text.replace(char, replacement)
    return text


def _markdown_to_latex(md: str) -> str:
    """Basic Markdown to LaTeX conversion."""
    lines = md.split("\n")
    result = [
        r"\documentclass{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{hyperref}",
        r"\begin{document}",
        "",
    ]

    for line in lines:
        if line.startswith("# "):
            result.append(f"\\title{{{_latex_escape(line[2:])}}}")
            result.append(r"\maketitle")
        elif line.startswith("## "):
            result.append(f"\\section{{{_latex_escape(line[3:])}}}")
        elif line.startswith("### "):
            result.append(f"\\subsection{{{_latex_escape(line[4:])}}}")
        elif line.startswith("#### "):
            result.append(f"\\subsubsection{{{_latex_escape(line[5:])}}}")
        else:
            result.append(line)

    result.append(r"\end{document}")
    return "\n".join(result)
