---
name: format-output
description: Output formatting API — citation pipeline, templates, format conversion for AutoReview
---

# Output Formatting

## Quick Reference

| Output Format | Key Files | Requirements | CLI Flag |
|---|---|---|---|
| Markdown | `output/formatter.py`, `output/templates/review_paper.md.jinja2` | None | `--format markdown` (default) |
| LaTeX | `output/formatter.py`, `output/templates/review_paper.tex.jinja2` | None (built-in converter) | `--format latex` |
| DOCX | `output/formatter.py` | `pypandoc` + Pandoc on PATH | `--format docx` |
| All | All of the above | `pypandoc` + Pandoc on PATH | `--format all` |

**Output files produced:**

| Format | Files Created |
|---|---|
| `markdown` | `review.md` |
| `latex` | `review.tex`, `references.bib` |
| `docx` | `review.docx` |
| `all` | All of the above |

---

## Citation Pipeline

Citations flow through four stages from raw draft to final output:

### Stage 1: Marker Insertion (Section Writing)

Section writers embed `[@paper_id]` markers in the draft text during the `section_writing` pipeline node. The `paper_id` matches the `CandidatePaper.id` field assigned during search.

### Stage 2: Citation Validation (Assembly Node)

`CitationValidator` checks markers against known extractions at two points:

- **Per-section** (in `section_writing` node): `CitationValidator.validate_section()` checks each section's citations against its assigned `section_paper_ids` and the full `kb.extractions` dict.
- **Full draft** (in `assembly` node): `CitationValidator.validate_full_draft()` checks all citations across the assembled document.

Invalid citations (phantom references) become `CRITICAL` critique issues. Uncited assigned papers become `MAJOR` issues. Both feed into the critique-revision loop.

### Stage 3: Citation Resolution (Output Formatting)

`_resolve_citations()` in `formatter.py` replaces `[@paper_id]` markers with the final citation form:

- **Vancouver style** (`style="vancouver"`): `[@paper_id]` becomes `[1]`, `[2]`, etc. (numbered in order of first appearance)
- **APA style** (`style="apa"`): `[@paper_id]` becomes `(AuthorLastName, Year)` when author data is available, falls back to `[N]`
- **LaTeX**: `_resolve_citations_latex()` replaces `[@paper_id]` with `\cite{key}` where the BibTeX key is `paper_id[:12]`

The regex used: `r"\[@([^\]]+)\]"`

Both functions return `(resolved_text, ordered_cited_ids)` -- the ordered list of cited IDs drives bibliography generation.

### Stage 4: Bibliography Generation

`BibliographyFormatter.format_bibliography()` takes the ordered `cited_ids` list and produces the formatted reference list. Papers appear in citation order (first-appearance order from the text).

For LaTeX output, BibTeX entries are generated inline in `OutputFormatter.format_latex()` as `@article{key, ...}` blocks and written to `references.bib`.

---

## Key Classes and Methods

### `OutputFormatter` (`autoreview/output/formatter.py`)

```python
class OutputFormatter:
    def __init__(self, style: str = "apa") -> None
    # style: "apa" | "vancouver" | "acs" -- sets citation format

    @staticmethod
    def _collect_papers(kb: KnowledgeBase) -> list[CandidatePaper]
    # Merges screened papers and raw candidates, preferring screened metadata

    def format_markdown(self, kb: KnowledgeBase) -> str
    # Resolves citations, generates bibliography, renders review_paper.md.jinja2
    # Returns complete Markdown string with YAML frontmatter

    def format_latex(self, kb: KnowledgeBase) -> tuple[str, str]
    # Resolves citations to \cite{} commands, generates BibTeX entries,
    # converts Markdown body to LaTeX sectioning, renders review_paper.tex.jinja2
    # Returns (latex_content, bibtex_content)

    def save(self, kb: KnowledgeBase, output_dir: str, fmt: str = "markdown") -> list[str]
    # fmt: "markdown" | "latex" | "docx" | "all"
    # Writes files to output_dir, returns list of created file paths
    # DOCX uses pypandoc.convert_text() -- gracefully skips if pypandoc/pandoc missing
```

**Pipeline integration**: Called in `cli.py` after pipeline completion:
```python
formatter = OutputFormatter(style=config.writing.citation_format)
created = formatter.save(kb, output_dir, fmt=output_format)
```

The `config.writing.citation_format` comes from `WritingConfig.citation_format` (default: `"apa"`).

### `BibliographyFormatter` (`autoreview/output/bibliography.py`)

```python
class BibliographyFormatter:
    def __init__(self, style: str = "apa") -> None
    # style: "apa" | "vancouver" | "acs"

    def format_entry(self, paper: CandidatePaper) -> str
    # Dispatches to _format_apa, _format_vancouver, or _format_acs

    def format_bibliography(
        self,
        papers: list[CandidatePaper],
        cited_ids: list[str] | None = None,
    ) -> str
    # When cited_ids provided: orders papers by citation appearance order
    # When cited_ids=None: orders alphabetically by first author + year
    # Vancouver adds numbered prefixes ("1. ...", "2. ...")
    # Returns entries joined by double newlines
```

**Supported citation styles:**

| Style | In-text | Bibliography Entry Format |
|---|---|---|
| `apa` | `(Author, Year)` | `Authors (Year). Title. *Journal*. DOI` |
| `vancouver` | `[N]` | `N. Authors. Year. Title. Journal. doi:DOI` |
| `acs` | `[N]` (fallback) | `Authors. Title. *Journal* **Year**. DOI: DOI` |

**APA author formatting rules** (`_format_apa_authors`):
- 1 author: `Smith`
- 2 authors: `Smith & Jones`
- 3-20 authors: `Smith, Jones, ..., & Last`
- 21+ authors: first 19, `...`, last

**Vancouver**: Lists up to 6 authors, then `et al`.
**ACS**: Lists up to 6 authors separated by `;`, then `; et al.`

### `CitationValidator` (`autoreview/validation/citation_validator.py`)

```python
class CitationValidator:
    def validate_section(
        self,
        text: str,
        section_paper_ids: list[str],
        extractions: dict[str, PaperExtraction],
    ) -> CitationValidationReport
    # Checks: valid (in extractions), invalid (not in extractions), uncited (assigned but not cited)

    def validate_full_draft(
        self,
        full_draft: str,
        extractions: dict[str, PaperExtraction],
    ) -> CitationValidationReport
    # Same but for full document; uncited = any extracted paper never cited anywhere

    @staticmethod
    def to_critique_issues(report: CitationValidationReport) -> list[CritiqueIssue]
    # Severity mapping:
    #   invalid citations (phantom references) -> CRITICAL
    #   uncited assigned papers -> MAJOR
    #   suspicious attributions -> MAJOR
```

**`CitationValidationReport` model:**
```python
class CitationValidationReport(AutoReviewModel):
    section_id: str = ""
    valid_citations: list[str]       # paper IDs found in extractions
    invalid_citations: list[str]     # paper IDs NOT in extractions (phantom)
    uncited_papers: list[str]        # assigned papers never cited
    suspicious_attributions: list[SuspiciousAttribution]
    total_citation_markers: int      # total [@...] occurrences (including repeats)
    unique_citations: int            # distinct paper IDs cited
```

---

## Jinja2 Templates

Templates live in `autoreview/output/templates/`.

### `review_paper.md.jinja2`

```
---
title: "{{ title }}"
date: "{{ date }}"
domain: "{{ domain }}"
---

# {{ title }}

{{ body }}

---

## References

{{ bibliography }}
```

Variables: `title` (kb.topic), `date` (UTC date), `domain` (kb.domain), `body` (resolved draft), `bibliography` (formatted refs).

### `review_paper.tex.jinja2`

```latex
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{hyperref}
\usepackage[margin=1in]{geometry}
\usepackage{natbib}

\title{<title>}
\date{<date>}

\begin{document}
\maketitle

{{ body }}

\bibliographystyle{plainnat}
\bibliography{references}

\end{document}
```

Variables: `title` (LaTeX-escaped kb.topic), `date` (UTC date), `body` (LaTeX-converted body with `\cite{}` commands).

The LaTeX pipeline uses `natbib` with `plainnat` style. The companion `references.bib` file is generated alongside the `.tex` file.

---

## How to Add New Formats and Styles

### Adding a new output format

1. Add a new method to `OutputFormatter` (e.g., `format_rst(self, kb: KnowledgeBase) -> str`)
2. Add a corresponding branch in `OutputFormatter.save()` under the `fmt` check
3. Optionally add a Jinja2 template in `autoreview/output/templates/`

### Adding a new citation style

1. Add a `_format_<style>` method to `BibliographyFormatter` in `bibliography.py`
2. Add the dispatch case in `BibliographyFormatter.format_entry()`
3. If the in-text citation format differs, add a branch in `_resolve_citations()` in `formatter.py`

### Creating custom Jinja2 templates

Templates use `jinja2.StrictUndefined` (missing variables raise errors rather than silently rendering empty). The Jinja2 environment is configured with:
- `FileSystemLoader` pointed at `autoreview/output/templates/`
- `keep_trailing_newline=True`

To add a new template:
1. Create `autoreview/output/templates/your_template.jinja2`
2. Load it with `_jinja_env.get_template("your_template.jinja2")`
3. Call `.render()` with the required variables

---

## Configuration

Output formatting is controlled by `WritingConfig` in `autoreview/config/models.py`:

```python
class WritingConfig(BaseModel):
    style: str = "academic"              # writing style preset
    citation_format: str = "apa"         # "apa" | "vancouver" | "acs"
    writing_temperature: float = 0.3     # LLM temperature for writing
    analysis_temperature: float = 0.0    # LLM temperature for analysis
```

The `citation_format` field is passed to `OutputFormatter(style=...)` and `BibliographyFormatter(style=...)`.

Domain YAML overrides (e.g., `config/defaults/biomedical.yaml`):
```yaml
writing:
  citation_format: vancouver
```

CLI override: `--format` flag controls output file format (markdown/latex/docx/all), not citation style. Citation style comes from domain config.

---

## Common Issues

| Problem | Cause | Fix |
|---|---|---|
| Missing citations in output | `paper_id` in `[@...]` marker does not match any `CandidatePaper.id` | Check `CitationValidationReport.invalid_citations`; verify IDs match between search and extraction stages |
| Broken LaTeX output | Special characters not escaped | `_latex_escape()` handles `& % $ # _ { } ~ ^`; check for unescaped chars in paper titles/abstracts |
| DOCX generation skipped | `pypandoc` not installed or Pandoc not on PATH | `pip install pypandoc` and install Pandoc system binary |
| Duplicate references | Same paper ingested with different IDs from different sources | Check deduplication in search aggregator (`search/aggregator.py`); dedup is DOI-based |
| Bibliography order wrong | `cited_ids` list not passed to `format_bibliography()` | Ensure `_resolve_citations()` return value is threaded through; it orders by first appearance |
| Empty output file | `kb.full_draft` is empty/None | Pipeline did not complete assembly stage; check snapshots for last successful node |
| `StrictUndefined` error | Template variable missing from `.render()` call | Ensure all template variables (`title`, `date`, `domain`, `body`, `bibliography`) are passed |
| Citations show `[N]` instead of `(Author, Year)` | Paper has no author data or style is set to `vancouver` | Check `CandidatePaper.authors` list; verify `config.writing.citation_format` setting |
| LaTeX `\cite{}` keys don't match `.bib` | BibTeX key derivation mismatch | Both use `paper_id[:12]`; ensure consistency via `_build_bibtex_key_map()` |
| Uncited papers flagged as MAJOR issues | Papers assigned to section outline but never referenced in text | Section writer should cite all assigned papers; if intentional, remove from section assignment in outline |
