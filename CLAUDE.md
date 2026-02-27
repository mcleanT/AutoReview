# AutoReview

Fully autonomous pipeline for generating publication-ready scientific review papers. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design — pipeline DAG, data models, stage details, search strategy, critique system, and domain configuration.

---

## Development Conventions

### Code Style
- Type hints on all function signatures
- Pydantic models for all data structures crossing module boundaries
- Async by default for any I/O-bound operation
- Structured logging with `structlog` — log LLM token usage per call
- Prompts constructed programmatically in `llm/prompts/`, not inline strings

### Testing
- Unit tests for each module with mocked LLM responses
- Integration tests for the full pipeline with a small fixture corpus
- Use `pytest-asyncio` for async tests
- Test critique rubrics with known-good and known-bad drafts

### Error Handling
- Retry with exponential backoff on API rate limits (search sources + LLM)
- Graceful degradation: if a search source fails, continue with remaining sources
- Pipeline state snapshots enable restart from last successful node
