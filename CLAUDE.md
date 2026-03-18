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

### Review Depth
- Three depth levels: `low` (~4K words), `medium` (~8K, default), `deep` (~25K+)
- Depth config lives in `config/depth.py` — `DepthProfile`, `EvidenceWeightedAllocator`, `classify_section_type`
- Depth flows through: outline prompt → narrative prompt → section writing prompt
- `EvidenceWeightedAllocator` runs inside the outline node (no separate DAG node)
- Critique system is depth-unaware — same rubric regardless of depth

---

## Living Repository Protocol

Read `.living/INDEX.md` for a compact summary of project knowledge before starting work. Read full files only when the current task touches those areas:
- `.living/conventions.md` — project-specific conventions (read when writing code)
- `.living/decisions.md` — project decisions log (read when making architectural choices)
- `.living/learnings.md` — lessons learned (read when debugging or encountering known issues)

After significant actions:
- Log non-obvious decisions to `.living/decisions.md`
- Log unexpected findings or gotchas to `.living/learnings.md`
- Check `../.living/learnings.md` for cross-project insights from the Science portfolio
