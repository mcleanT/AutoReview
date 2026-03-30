# Last Session — 2026-03-29

## Focus
KG extraction prompt audit — deep comparison of v8.7 extraction against source paper

## Key Outcomes
- Ran fresh Haiku extraction of rai14 paper with v8.7 prompt: 10 claims, 47 evidence, 12 citation contexts
- Dispatched Opus for systematic paper-vs-extraction audit
- Discovered severe under-extraction: model plans ~52 claims but only writes 10
- 70% of evidence orphaned, all citation contexts reference non-existent claims
- Per-claim quality is decent (7/10 accuracy) but completeness is 3/10
- Root cause unclear: may be output token limit, prompt verbosity, or Haiku capability
- v8.7 prompt blocked for production use until truncation/completeness resolved

## Open Questions
- Is the under-extraction purely a max_tokens issue? CLI does not expose this flag
- Would v6 prompt still produce 34 claims through the same experiment_runner pipeline?
- Would Sonnet extraction solve both quality and completeness?

## State
- Branch: main
- No uncommitted code changes (audit report in /tmp/)
