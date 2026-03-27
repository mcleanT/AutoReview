---
name: kg-extract
description: >
  Extract knowledge graph claims from scientific papers using the v4 KG extraction pipeline.
  Use when the user says "extract claims", "KG extraction", "extract knowledge graph",
  "run extraction on [paper]", provides a DOI for extraction, or asks to "pull claims from"
  a paper. Handles single papers (local extraction in-chat) and corpus batches (API, requires
  confirmation). Applies the standard v4 prompt, schema validation, and predicate coercion.
---

# KG Claim Extraction

Extract structured, falsifiable scientific claims from papers into knowledge graph JSON.

## What this skill does

- **Single paper**: Reads paper text, applies the KG extraction prompt, validates output against the Pydantic schema, applies coercion, writes JSON
- **Corpus batch**: Configures and launches `batch_extract_kg.py` against the Anthropic Batches API (MUST confirm with user first)

## Files (relative to AutoReview root)

| File | Role |
|------|------|
| `Paper Extractor/KnowledgeGraph Extraction/kg_extraction_prompt.md` | Extraction prompt template (v4) |
| `Paper Extractor/KnowledgeGraph Extraction/kg_schema.py` | Pydantic schema (`KGExtraction`, `KGClaim`, `KGEvidence`, `KGEvidenceLink`) |
| `Paper Extractor/KnowledgeGraph Extraction/batch_extract_kg.py` | Batch API runner + coercion pipeline |
| `autoreview/knowledge_graph/ingest.py` | Graph ingestion (reads extractions into KG) |

---

## Detecting input type

```
User provides DOI, title, or URL   →  Single paper (fetch + local extract)
User provides file path (.pdf/.txt) →  Single paper (read + local extract)
User provides papers.json index     →  Ask: single paper or batch?
User provides directory path        →  Batch extraction (CONFIRM FIRST)
No input specified                  →  Ask what to extract
```

---

## Single Paper: Local Extraction

### Step 1 — Get paper text

**If DOI/title/URL**: Use the `fetch-papers` skill to download the PDF, then extract text:
```python
# For PDFs, use the pdf skill or pdftotext
# For papers.json entries, read the full_text field directly
```

**If file path**: Read directly. Supports `.pdf`, `.txt`, or `.json` (papers.json format with `full_text` field).

### Step 2 — Truncate paper text

Apply the same truncation rules as `batch_extract_kg.py`:

| Section | Limit |
|---------|-------|
| Results | Full (no cap) |
| Discussion | Full (no cap) |
| Methods | 5,000 chars |
| Introduction | 4,000 chars |
| Abstract | **DROP** (skip entirely) |
| Conclusion | **DROP** |
| References | **DROP** |

Total paper text cap: 100,000 chars.

### Step 3 — Read the prompt template

Read `Paper Extractor/KnowledgeGraph Extraction/kg_extraction_prompt.md`. The prompt ends with `{PAPER_TEXT}` — replace that marker with the truncated paper text.

**Do NOT modify the prompt.** The prompt is the standard and should be used as-is.

### Step 4 — Extract

You (the current model in this chat) ARE the extractor. Generate the full JSON extraction following the prompt's instructions exactly. Output the complete JSON object with all claims, evidence units, and metadata.

Key rules from the prompt (reminders, not overrides):
- Predicates: CLOSED SET of 34 valid predicates — do not invent new ones
- `claim_type` vs `section_source`: DIFFERENT FIELDS — don't confuse them
- `evidence_links`: Each link has `evidence_id` + `direction` (supports/refutes/mixed/not_applicable)
- `result_summary`: What the experiment CONCLUDED, not what was done
- `key_figure`: Every evidence unit must reference a specific figure panel

### Step 5 — Coerce and validate

Apply coercion rules from `batch_extract_kg.py`:

**Predicate coercion map** (invalid → valid):
```
lacks        → maintains (+ direction="negative")
forms        → induces
contains     → is_component_of
generates    → induces
expresses    → is_expressed_in
represses    → suppresses
models       → correlates_with
recapitulates → correlates_with
is_active    → is_expressed_in
is_active_in → is_expressed_in
develops     → induces
exhibits     → maintains
differs      → correlates_with
provides     → enables
controls     → regulates
```

**Post-processing rules:**
- `evidence_ids` (old format) → convert to `evidence_links` with `direction: "supports"`
- `claim_type == "absence"` → flip all evidence_links from `supports` to `refutes`
- Bare string evidence links → wrap in `{"evidence_id": ..., "direction": "supports"}`

**Schema validation**: Parse the JSON through `KGExtraction` from `kg_schema.py`. Fix any validation errors.

### Step 6 — Write output

Write the validated JSON to:
- If a corpus directory exists (e.g., `gastruloid_run/`): `{corpus}/extractions_kg/{paper_hash}.json`
- Otherwise: `/tmp/kg_extraction_{doi_slug}.json`

Report to user: claim count, evidence count, section_source distribution, any coercion actions taken.

---

## Corpus Batch: API Extraction

### MANDATORY: Confirm before launching

**You MUST ask the user for explicit confirmation before running batch extraction.** Present:

```
This will submit {N} papers to the Anthropic Batches API.
  Model: claude-haiku-4-5-20251001
  Estimated cost: ~${cost} (at $0.25/1M input + $1.25/1M output)
  Output directory: {path}/extractions_kg/

Proceed? (yes/no)
```

Cost estimate: ~$0.04/paper for Haiku with uncapped output.

### Running the batch

```bash
cd "Paper Extractor/KnowledgeGraph Extraction"
ANTHROPIC_API_KEY=... python batch_extract_kg.py
```

The script:
1. Reads `{corpus}/papers.json`
2. Skips papers already in `extractions_kg/`
3. Submits to Anthropic Message Batches API
4. Polls for completion
5. Applies coercion + schema validation on each result
6. Writes per-paper JSON + batch run log

### After batch completes

Report:
- Papers submitted / succeeded / failed
- Total claims extracted, mean per paper
- Section source distribution
- Any coercion actions (predicate remaps, evidence_links migrations)
- Cost from API response

---

## Output schema summary

```
KGExtraction
├── doi, title, journal, publication_date
├── claims: list[KGClaim]
│   ├── claim_id, natural_language
│   ├── subject: KGEntity {name, type, ontology_id}
│   ├── predicate (from closed vocabulary)
│   ├── object: KGEntity {name, type, ontology_id}
│   ├── direction (positive/negative)
│   ├── claim_type (mechanistic_causal/correlational/comparative/existence/absence/conditional/methodological)
│   ├── causal_type (necessary/sufficient/necessary_and_sufficient/contributory/modulatory/null)
│   ├── conditions: KGConditions {species, cell_type, tissue, treatment, developmental_stage, in_vitro}
│   ├── evidence_strength, certainty, section_source, source_doi
│   └── evidence_links: list[KGEvidenceLink] {evidence_id, direction}
├── evidence: list[KGEvidence]
│   ├── evidence_id, description, result_summary
│   ├── model_system, organism, perturbation, readout
│   ├── result_direction, effect_size, p_value, sample_size
│   ├── key_figure, approach, assay_types
└── extraction_model, extraction_timestamp
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Invalid predicate in output | Coercion map handles automatically; if new invalid predicate appears, add to `_PREDICATE_COERCION_MAP` in batch_extract_kg.py |
| `claim_type` has section_source value | Prompt v4 eliminated this; coercion maps it if it recurs |
| All evidence_links are "supports" | Post-processing flips absence claims; for other cases, the few-shot example in the prompt teaches refutes |
| Missing key_figure on evidence | Prompt requires it; if missing, scan paper text for figure references |
| Pydantic validation fails | Check for missing required fields (result_summary, evidence_links); coercion adds defaults |
| `evidence_ids` instead of `evidence_links` | Coercion auto-migrates to evidence_links format |

---

## What this skill does NOT do

- Modify the prompt or schema (those are the locked-in standard)
- Run batch extraction without explicit user confirmation
- Extract from abstracts (prompt skips them)
- Replace the AutoReview extraction pipeline (mycelium prompt + batch_extract.py remain for review paper generation)
