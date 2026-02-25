# Writing Quality: Flow, Segmentation, and Narrative Architecture

**Date**: 2026-02-25

---

## Context

The generated review papers have good content but poor readability. Specific problems:

1. **Predictable structure** — every section follows the same pattern regardless of the evidence shape (background → findings → limitations), because the outline mandates fixed headings and the writer is given no structural guidance beyond "synthesize."
2. **Segmentation at both levels** — paragraphs read as isolated findings blocks; sections feel like independent essays with no bridging.
3. **No narrative thread** — each section writer sees evidence and a heading, not the role it plays in a larger argument. The paper has coverage but no thesis.

Root causes in code:
- `section_writer.py`: temperature=0.3 produces formulaic, safe output
- `llm/prompts/writing.py`: writing prompt tells the LLM *what* to cover, not *how* to structure the argument
- `critique/section_critic.py` / `holistic_critic.py`: critique rubrics don't evaluate flow, structural variety, or paragraph cohesion — so revision loops don't fix style issues
- No pre-writing narrative planning stage

---

## Approach

Two-track implementation:

### Track 1: Quick Wins (prompt + temperature + critique)

Changes to existing code, no new stages.

**1a. Writing temperature** (`autoreview/writing/section_writer.py`):
- Changed writing call from `temperature=0.3` → `temperature=0.55`
- Revision and extraction remain at 0.3 (deterministic is correct there)

**1b. Section writing prompt** (`autoreview/llm/prompts/writing.py`):

Added three new directives to `SECTION_WRITING_SYSTEM_PROMPT`:

```
## Structural Approach
Choose the structure that best serves this section's evidence — do not default to
background → findings → limitations. Viable structures:
- Comparative: contrast two competing accounts or methodological traditions
- Problem-solution: establish what was unknown, then trace how the field addressed it
- Chronological development: trace how understanding of a concept evolved
- Argument-rebuttal: state the dominant view, then present the evidence that challenges it
Let the evidence shape the section architecture, not a template.

## Paragraph Discipline
Each paragraph advances exactly one claim. The opening sentence states the claim;
the rest develops it with evidence, qualification, or contrast. Avoid paragraphs
that are lists of findings dressed as prose.

## Transitions
Begin with a transition from the preceding section.
End with a sentence that creates forward momentum into the following section.
```

**1c. Adjacent context** (`autoreview/writing/section_writer.py`):
- Expanded from `preceding_text[-2000:]` → full preceding section text (no truncation)
- Following section hint stays at first 500 chars (structural preview, not full text)

**1d. Critique rubric additions** (`autoreview/llm/prompts/critique.py`):

`SECTION_CRITIQUE_SYSTEM_PROMPT` — added:
- `structural_variety` (0–1): Does the section use evidence-appropriate structure, or a default template?
- `paragraph_cohesion` (0–1): Does each paragraph advance a distinct claim?

`HOLISTIC_CRITIQUE_SYSTEM_PROMPT` — added:
- `prose_flow` (0–1): Does prose flow within paragraphs, across transitions, and between sections?

---

### Track 2: NarrativeArchitect Stage (new pipeline node)

**New models** — `autoreview/models/narrative.py`:

```python
class SectionNarrativeDirective(BaseModel):
    section_id: str
    narrative_role: str           # e.g. "sets up the central tension of the field"
    central_claim: str            # thesis statement for this section
    structural_suggestion: str    # e.g. "comparative: mechanistic vs. epidemiological evidence"
    key_insights: list[str]       # 3-5 analytical insights to foreground (not paper IDs)
    transition_from_prev: str     # hint for opening transition
    transition_to_next: str       # hint for closing sentence / setup for next section

class NarrativePlan(BaseModel):
    central_argument: str              # paper's one-sentence thesis
    narrative_arc: str                 # how argument builds from intro → conclusion
    section_directives: list[SectionNarrativeDirective]
```

**New class** — `autoreview/writing/narrative_architect.py`:

`NarrativeArchitect.plan()` takes outline, evidence_map, scope_document and returns a
`NarrativePlan` via a single `generate_structured()` call.

**New prompts** — `autoreview/llm/prompts/narrative.py`:

System prompt instructs the LLM to act as a scientific editor planning narrative structure,
not writing prose. Directives are framed as soft hints — writers deviate if evidence warrants.

User prompt provides:
- Scope document
- Flattened outline (section IDs, titles, descriptions)
- Evidence summary: themes, consensus claims, contradictions, identified gaps

**KnowledgeBase update** (`autoreview/models/knowledge_base.py`):
```python
narrative_plan: NarrativePlan | None = None
```

`PipelinePhase.NARRATIVE_PLANNING` added between `OUTLINE` and `GAP_SEARCH`.

**Pipeline node** (`autoreview/pipeline/nodes.py`):
- New `narrative_planning` node inserted between `outline` and `section_writing`
- Input: `KnowledgeBase` with validated outline + evidence map
- Output: `KnowledgeBase` with `narrative_plan` populated

**Section writer integration** (`autoreview/writing/section_writer.py`):
- `write_section()` accepts optional `directive: SectionNarrativeDirective | None`
- If present, appended to prompt as a clearly-labelled soft-hint block:
  ```
  ## Narrative Guidance (soft hints — deviate if evidence warrants)
  **Role in paper**: {directive.narrative_role}
  **Central claim for this section**: {directive.central_claim}
  **Structural suggestion**: {directive.structural_suggestion}
  **Key insights to foreground**: {bullet list}
  **Opening transition hint**: {directive.transition_from_prev}
  **Closing transition hint**: {directive.transition_to_next}
  ```
- `write_all_sections()` accepts optional `narrative_plan` and builds a directive lookup
  by `section_id`

---

## Files Changed

| File | Change |
|---|---|
| `autoreview/writing/section_writer.py` | Temperature 0.3→0.55, full adjacent context, directive integration |
| `autoreview/llm/prompts/writing.py` | Added structural flexibility, paragraph discipline, transition directives; `narrative_guidance` param |
| `autoreview/llm/prompts/critique.py` | Added `structural_variety`, `paragraph_cohesion`, `prose_flow` dimensions |
| `autoreview/models/knowledge_base.py` | Added `narrative_plan` field, `NARRATIVE_PLANNING` phase |
| `autoreview/pipeline/nodes.py` | Added `narrative_planning` node; passes `narrative_plan` to `write_all_sections` |
| **New** `autoreview/models/narrative.py` | `SectionNarrativeDirective`, `NarrativePlan` models |
| **New** `autoreview/writing/narrative_architect.py` | `NarrativeArchitect` class |
| **New** `autoreview/llm/prompts/narrative.py` | System + user prompts for narrative planning |

---

## Verification Checklist

1. **Unit test `NarrativeArchitect`** with a mocked LLM response — verify it produces a `NarrativePlan` with directives for all outline sections
2. **Unit test section writer** with a mock `SectionNarrativeDirective` — verify the directive appears correctly formatted in the assembled prompt
3. **Unit test critique rubrics** — verify `structural_variety` and `paragraph_cohesion` scores appear in `CritiqueReport.dimension_scores`
4. **Integration test** with a small fixture corpus — run the full pipeline and verify:
   - `kb.narrative_plan` is populated after the `narrative_planning` node
   - Section drafts reference their structural approach in the first paragraph
   - Holistic critique `dimension_scores` contains `prose_flow`
5. **Qualitative check**: Run on a known topic and manually verify:
   - Sections have varied opening structures
   - Paragraphs open with a claim rather than a citation
   - At least one section uses non-IMRAD structure
