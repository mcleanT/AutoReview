# Contributing to AutoReview

Thanks for your interest in contributing! This guide covers everything you need to get started.

## Development Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/autoreview.git
   cd autoreview
   ```

2. **Install dependencies** (requires [uv](https://docs.astral.sh/uv/))
   ```bash
   uv sync --all-extras
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Verify the install**
   ```bash
   autoreview --help
   ```

## Running Tests

```bash
# Run all unit tests
pytest --tb=line

# Skip tests that call real APIs
pytest -m "not integration" --tb=line

# Run a specific test file
pytest tests/test_extraction/test_extractor.py --tb=line
```

Tests use `pytest` with `pytest-asyncio` for async test support. LLM calls are mocked in unit tests — integration tests that hit real APIs are marked with `@pytest.mark.integration`.

## Code Style

- **Type hints** on all function signatures
- **Pydantic v2 models** for all data structures crossing module boundaries
- **Async by default** for any I/O-bound operation
- **Structured logging** with `structlog` — log LLM token usage per call
- **Prompts** are constructed programmatically in `autoreview/llm/prompts/`, not as inline strings

Linting and formatting:

```bash
ruff check .
ruff format .
mypy autoreview/
```

## Project Structure

```
autoreview/
├── analysis/       # Clustering, gap detection, evidence chains
├── config/         # Domain config loading + YAML defaults
├── critique/       # Self-critique system (outline, section, holistic)
├── evaluation/     # Evaluation against reference papers
├── extraction/     # Structured data extraction from papers
├── llm/            # LLM provider abstraction + prompts
├── models/         # Core Pydantic models (KnowledgeBase, Paper, etc.)
├── output/         # Formatting, bibliography, templates
├── pipeline/       # DAG runner, pipeline nodes, remediation
├── search/         # PubMed, Semantic Scholar, OpenAlex, Perplexity
├── validation/     # Citation validation
└── writing/        # Outline, section writing, narrative planning
```

## Adding a New Domain

Create a YAML file in `autoreview/config/defaults/` following the existing patterns (`biomedical.yaml`, `cs_ai.yaml`, `chemistry.yaml`). No code changes needed — the config loader picks up new YAML files automatically.

The YAML file configures:
- Which search databases to use
- Domain-specific extraction fields
- Critique rubric weights
- Writing style and citation format
- Required outline sections

## Adding a Pipeline Node

The pipeline is a DAG defined in `autoreview/pipeline/nodes.py`. Each node is an async callable with typed Pydantic inputs/outputs. To add a new node:

1. Define the node method in `PipelineNodes`
2. Register it in the DAG with its dependencies
3. Add corresponding prompts in `autoreview/llm/prompts/`
4. Write tests with mocked LLM responses

## Pull Requests

1. Create a feature branch from `main`
2. Make your changes with tests
3. Ensure `pytest --tb=line` passes
4. Ensure `ruff check .` passes
5. Open a PR with a clear description of what changed and why

## Reporting Issues

Open an issue on GitHub with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Relevant logs or error messages
