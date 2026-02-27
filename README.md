# AutoReview

Fully autonomous pipeline for generating publication-ready scientific review papers. Given a topic, AutoReview searches the literature, extracts structured evidence, synthesizes findings, writes a complete review, and self-critiques iteratively until quality thresholds are met — no human intervention required.

## Key Features

- **End-to-end automation** — from research question to formatted review paper
- **Multi-source search** — PubMed, Semantic Scholar, OpenAlex, and Perplexity Sonar
- **Structured extraction** — findings, methods, relationships, and limitations per paper
- **Evidence synthesis** — thematic clustering, contradiction detection, consensus identification, gap analysis
- **Three-level self-critique** — outline, per-section, and holistic review with configurable rubrics
- **Domain-agnostic** — ships with biomedical, CS/AI, and chemistry presets; add new domains via YAML
- **Crash recovery** — pipeline state saved after every stage; resume from any snapshot

## Architecture

```
[Query Expansion] → [Multi-Source Search] → [Screen & Deduplicate]
                                                     ↓
                          [Parallel Extraction (per paper)]
                                                     ↓
                    [Thematic Clustering + Contradiction Detection]
                                                     ↓
                              [Outline Generation] → [Outline Critique] ←→ [Revise]
                                                     ↓
                              [Gap-Aware Supplementary Search]
                                                     ↓
                         [Section Writing (with cross-section context)]
                                                     ↓
                              [Per-Section Critique] ←→ [Revision]
                                                     ↓
                    [Assemble Draft] → [Holistic Critique] ←→ [Revision]
                                                     ↓
                              [Final Polish] → [Formatted Output]
```

Each stage is an async DAG node with typed Pydantic inputs/outputs. Pipeline state is serialized to JSON after every node for crash recovery.

## Quick Start

### Install

```bash
# Clone and install with uv
git clone https://github.com/your-username/autoreview.git
cd autoreview
uv sync --all-extras
```

### Set Up API Keys

```bash
cp .env.example .env
# Edit .env with your API keys (at minimum: ANTHROPIC_API_KEY and ENTREZ_EMAIL)
```

### Generate a Review

```bash
autoreview run "the role of gut microbiome in neurodegenerative diseases"
```

## CLI Commands

### `autoreview run`

Run the full pipeline to generate a review paper.

```bash
autoreview run "your research topic" \
  --domain biomedical \
  --format markdown \
  --output-dir output/ \
  --model claude-sonnet-4-20250514 \
  --verbose
```

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--domain`, `-d` | Domain preset (`biomedical`, `cs_ai`, `chemistry`, `general`) | `general` |
| `--format`, `-f` | Output format (`markdown`, `latex`, `docx`) | `markdown` |
| `--output-dir`, `-o` | Output directory | `output` |
| `--model`, `-m` | Override LLM model | config default |
| `--provider`, `-p` | LLM provider (`claude`, `ollama`) | auto-detect |
| `--fresh` | Clear previous outputs before running | `false` |
| `--verbose`, `-v` | Enable verbose logging | `false` |

### `autoreview resume`

Resume a pipeline from a saved snapshot.

```bash
autoreview resume output/snapshots/extraction.json --start-from clustering
```

### `autoreview inspect`

Inspect a pipeline snapshot to see progress and statistics.

```bash
autoreview inspect output/snapshots/clustering.json
```

### `autoreview evaluate`

Evaluate a generated review against a published reference PDF.

```bash
autoreview evaluate output/review.md reference.pdf --output-dir output/evaluations
```

### `autoreview benchmark`

Benchmark LLM providers by running targeted pipeline stages.

```bash
autoreview benchmark "your topic" --models "claude-sonnet-4-20250514,qwen3.5:35b"
```

## Domain Configuration

Domains are configured via YAML files in `autoreview/config/defaults/`. No code changes needed to add a new domain.

Included presets: `biomedical`, `cs_ai`, `chemistry`

Each preset configures:
- Search databases and parameters
- Domain-specific extraction fields
- Critique rubric weights
- Writing style and citation format
- Required outline sections

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on adding new domains.

## MCP Server

AutoReview includes an MCP server that exposes literature search as tools for Claude Code:

```bash
# Run directly
python mcp_server.py
```

The `.mcp.json` file is pre-configured for use with Claude Code. Available tools: `search_pubmed`, `search_semantic_scholar`, `search_openalex`.

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.11+ |
| Async | `asyncio` |
| LLM SDK | `anthropic` (async), Ollama |
| Data models | Pydantic v2 |
| HTTP client | `httpx` |
| CLI | `typer` |
| Pipeline state | JSON snapshots |
| Output templating | `jinja2` |
| Format conversion | `pypandoc` |
| Testing | `pytest` + `pytest-asyncio` |
| Logging | `structlog` |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and PR guidelines.

## License

[MIT](LICENSE)
