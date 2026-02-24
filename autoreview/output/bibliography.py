from __future__ import annotations

from autoreview.models.paper import CandidatePaper


class BibliographyFormatter:
    """Formats citations and bibliography entries."""

    def __init__(self, style: str = "apa") -> None:
        self.style = style

    def format_entry(self, paper: CandidatePaper) -> str:
        """Format a single bibliography entry."""
        if self.style == "vancouver":
            return self._format_vancouver(paper)
        elif self.style == "acs":
            return self._format_acs(paper)
        return self._format_apa(paper)

    def _format_apa(self, paper: CandidatePaper) -> str:
        authors = self._format_apa_authors(paper.authors)
        year = f" ({paper.year})." if paper.year else " (n.d.)."
        title = f" {paper.title}."
        journal = f" *{paper.journal}*." if paper.journal else ""
        doi = f" https://doi.org/{paper.doi}" if paper.doi else ""
        return f"{authors}{year}{title}{journal}{doi}"

    def _format_vancouver(self, paper: CandidatePaper) -> str:
        authors = ", ".join(paper.authors[:6])
        if len(paper.authors) > 6:
            authors += ", et al"
        year = f" {paper.year}." if paper.year else "."
        title = f" {paper.title}."
        journal = f" {paper.journal}." if paper.journal else ""
        doi = f" doi:{paper.doi}" if paper.doi else ""
        return f"{authors}.{year}{title}{journal}{doi}"

    def _format_acs(self, paper: CandidatePaper) -> str:
        authors = "; ".join(paper.authors[:6])
        if len(paper.authors) > 6:
            authors += "; et al."
        title = f" {paper.title}."
        journal = f" *{paper.journal}*" if paper.journal else ""
        year = f" **{paper.year}**." if paper.year else "."
        doi = f" DOI: {paper.doi}" if paper.doi else ""
        return f"{authors}.{title}{journal}{year}{doi}"

    def _format_apa_authors(self, authors: list[str]) -> str:
        if not authors:
            return "Unknown"
        if len(authors) == 1:
            return authors[0]
        if len(authors) == 2:
            return f"{authors[0]} & {authors[1]}"
        if len(authors) <= 20:
            return ", ".join(authors[:-1]) + f", & {authors[-1]}"
        # 20+ authors: first 19, ..., last
        return ", ".join(authors[:19]) + f", ... {authors[-1]}"

    def format_bibliography(
        self,
        papers: list[CandidatePaper],
        cited_ids: list[str] | None = None,
    ) -> str:
        """Format a full bibliography section."""
        if cited_ids:
            paper_map = {p.id: p for p in papers}
            ordered = [paper_map[pid] for pid in cited_ids if pid in paper_map]
        else:
            ordered = sorted(papers, key=lambda p: (p.authors[0] if p.authors else "", p.year or 0))

        entries = []
        for i, paper in enumerate(ordered, 1):
            entry = self.format_entry(paper)
            if self.style == "vancouver":
                entries.append(f"{i}. {entry}")
            else:
                entries.append(entry)

        return "\n\n".join(entries)
