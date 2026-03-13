from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import jinja2
import structlog

from autoreview.models.knowledge_base import KnowledgeBase
from autoreview.models.paper import CandidatePaper
from autoreview.output.bibliography import BibliographyFormatter

logger = structlog.get_logger()

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
    keep_trailing_newline=True,
    undefined=jinja2.StrictUndefined,
)


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
        if paper is None:
            logger.warning("citation.unresolvable", paper_id=pid)
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


def _build_bibtex_key_map(papers: list[CandidatePaper]) -> dict[str, str]:
    """Build a mapping from paper ID to BibTeX key.

    Uses the first 12 characters of the paper ID as the BibTeX key,
    matching the convention used in BibTeX entry generation.

    Returns:
        Dict mapping full paper_id -> bibtex_key.
    """
    return {p.id: p.id[:12] for p in papers}


def _resolve_citations_latex(
    text: str,
    papers: list[CandidatePaper],
) -> tuple[str, list[str]]:
    r"""Replace [@paper_id] markers with \cite{key} commands for LaTeX.

    The BibTeX key is derived from the paper ID (first 12 chars), matching
    the keys generated in the .bib file.

    Returns:
        Tuple of (resolved text, ordered list of cited paper IDs).
    """
    key_map = _build_bibtex_key_map(papers)
    paper_map = {p.id: p for p in papers}
    cited_ids: list[str] = []
    seen: set[str] = set()

    def replace_citation(match: re.Match) -> str:
        pid = match.group(1)
        if pid not in seen:
            seen.add(pid)
            cited_ids.append(pid)

        if pid not in paper_map:
            logger.warning("citation.unresolvable", paper_id=pid)

        bib_key = key_map.get(pid, pid[:12])
        return f"\\cite{{{bib_key}}}"

    resolved = re.sub(r"\[@([^\]]+)\]", replace_citation, text)
    return resolved, cited_ids


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters."""
    special = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
               "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    for char, replacement in special.items():
        text = text.replace(char, replacement)
    return text


def _markdown_body_to_latex(md: str) -> str:
    """Convert Markdown body text to LaTeX sectioning commands.

    This converts only the body content (headings, paragraphs). It does NOT
    produce a full document -- the Jinja2 template handles the preamble,
    \\begin{document}, bibliography, and \\end{document}.
    """
    lines = md.split("\n")
    result: list[str] = []

    for line in lines:
        if line.startswith("# "):
            # Top-level heading becomes section (title is handled by template)
            result.append(f"\\section{{{_latex_escape(line[2:])}}}")
        elif line.startswith("## "):
            result.append(f"\\section{{{_latex_escape(line[3:])}}}")
        elif line.startswith("### "):
            result.append(f"\\subsection{{{_latex_escape(line[4:])}}}")
        elif line.startswith("#### "):
            result.append(f"\\subsubsection{{{_latex_escape(line[5:])}}}")
        else:
            result.append(line)

    return "\n".join(result)


def _markdown_to_latex(md: str) -> str:
    """Basic Markdown to LaTeX conversion (full document).

    Kept for backward compatibility. For new code, prefer using the Jinja2
    template with ``_markdown_body_to_latex`` for body conversion.
    """
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


class OutputFormatter:
    """Formats the final review paper output."""

    def __init__(self, style: str = "apa") -> None:
        self.style = style

    @staticmethod
    def _collect_papers(kb: KnowledgeBase) -> list:
        """Collect papers preferring screened (enriched) metadata over raw candidates."""
        screened_ids = {sp.paper.id for sp in kb.screened_papers}
        return [sp.paper for sp in kb.screened_papers] + [
            cp for cp in kb.candidate_papers if cp.id not in screened_ids
        ]

    def format_markdown(self, kb: KnowledgeBase) -> str:
        """Format the review as Markdown using the Jinja2 template."""
        if not kb.full_draft:
            return ""

        all_papers = self._collect_papers(kb)
        resolved, cited_ids = _resolve_citations(kb.full_draft, all_papers, self.style)

        bib_formatter = BibliographyFormatter(style=self.style)
        unique_papers = {p.id: p for p in all_papers}
        cited_papers = [unique_papers[pid] for pid in cited_ids if pid in unique_papers]
        bibliography = bib_formatter.format_bibliography(cited_papers, cited_ids)

        template = _jinja_env.get_template("review_paper.md.jinja2")
        return template.render(
            title=kb.topic,
            date=datetime.now(UTC).strftime("%Y-%m-%d"),
            domain=kb.domain,
            body=resolved,
            bibliography=bibliography,
        )

    def format_latex(self, kb: KnowledgeBase) -> tuple[str, str]:
        """Format the review as LaTeX with proper BibTeX citations.

        Returns:
            Tuple of (latex_content, bibtex_content).
        """
        if not kb.full_draft:
            return "", ""

        all_papers = self._collect_papers(kb)
        unique_papers = {p.id: p for p in all_papers}

        # Resolve [@paper_id] markers to \cite{key} commands
        resolved, cited_ids = _resolve_citations_latex(kb.full_draft, all_papers)

        # Convert markdown body to LaTeX sectioning (no preamble)
        latex_body = _markdown_body_to_latex(resolved)

        # Generate BibTeX entries for cited papers only
        bibtex_entries = []
        for pid in cited_ids:
            paper = unique_papers.get(pid)
            if not paper:
                continue
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

        # Render the LaTeX document using Jinja2 template
        template = _jinja_env.get_template("review_paper.tex.jinja2")
        latex = template.render(
            title=_latex_escape(kb.topic),
            date=datetime.now(UTC).strftime("%Y-%m-%d"),
            body=latex_body,
        )

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
