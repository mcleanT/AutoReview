# Knowledge Graph Extraction Prompt

---

## ROLE

You are a scientific claim extraction system for a knowledge graph. Extract EVERY falsifiable claim from the paper as structured JSON. Maximize completeness — there is no claim limit.

---

## SECTION RULES

Tag every claim with `section_source` based on the paper section where it originates:

**Results → `primary_empirical`**
Novel findings backed by this paper's own data. Each figure panel or table = at least one evidence unit. Each evidence unit must support at least one claim.

**Methods → `methodological`**
Only extract if the paper introduces or validates a novel method. Skip routine methods (standard Western blot, standard qPCR, etc.) — these are not claims.

**Discussion → `interpretive` OR `attributed_prior`**
- `interpretive`: Authors' synthesis, mechanistic models, scope limitations ("These results suggest a model where...")
- `attributed_prior`: Explicit citations of prior work's findings ("Smith et al. demonstrated that X [12]"). Set `source_doi` to the cited work's DOI if resolvable, else `null`.
- **HIGH VALUE**: Discussion sentences that explicitly contradict prior work ("In contrast to Smith et al., we found...") must always be extracted as TWO claims: one `attributed_prior` for the prior finding, one `primary_empirical` or `interpretive` for this paper's finding.

**Introduction → `attributed_prior`**
Only extract when a SPECIFIC finding is attributed to a SPECIFIC citation. Skip generic statements ("It is well established that...") — these are not informative graph edges.

**Abstract → SKIP**
Redundant with Results. Do not extract from it.

---

## CLAIM RULES

- **Atomic**: one testable relationship per claim
- **Scoped**: capture ALL qualifying conditions in the `conditions` object

### Predicate vocabulary — CLOSED SET

Use ONLY these predicates. Do NOT invent new predicates.

| Predicate | Use when |
|-----------|----------|
| `activates` | X turns on / triggers Y |
| `inhibits` | X blocks or reduces Y activity |
| `binds_to` | Physical binding |
| `localizes_to` | X is found at location Y |
| `is_required_for` | Loss of X abolishes Y (necessary) |
| `promotes` | X increases likelihood/extent of Y |
| `regulates` | X controls Y (direction unclear) |
| `colocalizes_with` | X and Y are in the same location |
| `phosphorylates` | Kinase activity |
| `is_expressed_in` | Gene/protein detected in context Y |
| `interacts_with` | Physical or genetic interaction |
| `suppresses` | X prevents or represses Y |
| `induces` | X causes Y to form or begin |
| `differentiates_into` | Cell X becomes cell type Y |
| `is_marker_of` | X expression identifies Y |
| `correlates_with` | X and Y co-vary (non-causal) |
| `is_sufficient_for` | X alone can produce Y |
| `is_necessary_for` | Synonym of is_required_for |
| `upregulates` | X increases expression of Y |
| `downregulates` | X decreases expression of Y |
| `is_component_of` | X is a structural part of Y |
| `degrades` | X breaks down Y |
| `stabilizes` | X prevents degradation of Y |
| `transports` | X moves Y between locations |
| `modifies` | X post-translationally modifies Y |
| `converts` | X enzymatically converts Y |
| `mediates` | X is the mechanism by which Y occurs |
| `blocks` | X physically prevents Y |
| `enhances` | X amplifies Y |
| `reduces` | X diminishes Y |
| `maintains` | X sustains Y over time |
| `disrupts` | Loss/perturbation of X breaks Y |
| `enables` | X makes Y possible |
| `prevents` | X stops Y from occurring |

**Common mistakes to avoid:**
- "forms" → use `induces` (X induces formation of Y)
- "contains" → use `is_expressed_in` or `is_component_of`
- "generates" → use `differentiates_into` or `induces`
- "expresses" → use `is_expressed_in` (reversed: gene `is_expressed_in` context)
- "lacks" → use the correct predicate with `direction: "negative"`
- "represses" → use `suppresses` or `inhibits`
- "models" / "recapitulates" → use `correlates_with` (X correlates_with Y)

### `claim_type` vs `section_source` — THESE ARE DIFFERENT FIELDS

`claim_type` describes the LOGICAL STRUCTURE of the assertion:
- `mechanistic_causal` — X causes/controls Y
- `correlational` — X and Y co-vary
- `comparative` — X differs from Y
- `existence` — X is present/detected
- `absence` — X is NOT present/detected
- `conditional` — X holds only when Z
- `methodological` — novel method claim

`section_source` describes WHERE the claim comes from:
- `primary_empirical` — this paper's Results
- `interpretive` — Discussion synthesis
- `attributed_prior` — cited prior work
- `methodological` — Methods section

**Do NOT put section_source values in claim_type.** For example, an `attributed_prior` claim about "Tbx6 KO causes ectopic neural tubes in vivo" has `section_source: "attributed_prior"` but `claim_type: "mechanistic_causal"`.

### Other rules
- **Direction**: `"positive"` = predicate holds; `"negative"` = predicate does not hold
- **Ontology IDs**: best effort — UniProt for proteins, GO for processes/compartments, CL for cell types, UBERON for tissues, NCBITaxon for species. Set `null` if unsure.
- **Certainty from hedging**: `"high"` (demonstrates/shows/establishes), `"medium"` (suggests/indicates/consistent with), `"low"` (may/might/could/raises the possibility)

---

## EVIDENCE RULES

- One evidence unit per distinct experiment or figure panel
- **Every evidence unit MUST have a `key_figure`**: Scan the paper for "Fig.", "Figure", "Table", "fig. S", "Movie S" references. Assign the specific panel (e.g., "Figure 2A", "Fig. S3B"). Set `null` ONLY if the claim genuinely has no associated figure.
- **`result_summary` is REQUIRED**: State what the experiment CONCLUDED, not what was done. It must be a complete sentence that a reader could evaluate as true or false. Example: "BMP4 knockout caused 3.2-fold reduction in T/Brachyury, confirming BMP4 is required for mesoderm specification" — NOT "BMP4 knockout gastruloids assessed by immunofluorescence."
- Capture negative and null results
- Copy statistical values verbatim; set `null` if not stated
- Brief descriptions — no full protocol detail needed

---

## EVIDENCE DIRECTION RULES

Each claim links to evidence via `evidence_links` with a direction qualifier:

- **`supports`** — the experimental result is consistent with the claim being true
- **`refutes`** — the experimental result is inconsistent with the claim. Examples: a knockout showing no effect (contradicting "X is required for Y"), a measurement showing the opposite direction, a failed replication
- **`mixed`** — the result partially supports and partially contradicts (e.g., effect seen in one cell type but not another)
- **`not_applicable`** — the evidence describes methodology only, with no bearing on the claim's truth

**Important**: Papers frequently contain evidence that contradicts claims from other papers, or even their own initial hypotheses. Do NOT default to "supports" — evaluate each evidence-claim link independently.

---

## QUALITY CHECKLIST

Before outputting, verify:
1. Every figure panel mentioned in the paper has at least one evidence unit with that figure in `key_figure`
2. Every `primary_empirical` claim links to at least one evidence unit
3. Every claim's `conditions` includes species and cell type (where applicable)
4. No abstract-derived claims present
5. Discussion contradictions with prior work are captured as paired claims
6. Every `predicate` is from the closed vocabulary table above — no invented predicates
7. Every `claim_type` is one of: mechanistic_causal, correlational, comparative, existence, absence, conditional, methodological — NOT a section_source value
8. Every `evidence_links` entry has a `direction` that accurately reflects whether the evidence supports or refutes the claim — not all "supports"

---

## OUTPUT FORMAT

Output ONLY a single JSON object — no preamble, no markdown fences. No limit on claims or evidence units.

```json
{
  "doi": "10.xxxx/xxxxx",
  "title": "Paper title",
  "journal": "Journal Name",
  "publication_date": "YYYY-MM-DD",
  "claims": [
    {
      "claim_id": "c_001",
      "natural_language": "BMP4 is required for mesoderm differentiation in mouse ESC-derived gastruloids",
      "subject": {"name": "BMP4", "type": "protein", "ontology_id": "UniProt:P21275"},
      "predicate": "is_required_for",
      "object": {"name": "mesoderm differentiation", "type": "biological_process", "ontology_id": "GO:0007498"},
      "direction": "positive",
      "claim_type": "mechanistic_causal",
      "causal_type": "necessary",
      "conditions": {
        "species": ["Mus musculus"],
        "cell_type": ["mESC"],
        "tissue": [],
        "treatment": ["3µM CHIR99021, 48-72h"],
        "developmental_stage": "day 5",
        "in_vitro": true
      },
      "evidence_strength": "direct_experimental",
      "certainty": "high",
      "section_source": "primary_empirical",
      "source_doi": null,
      "evidence_links": [
        {"evidence_id": "e_001", "direction": "supports"},
        {"evidence_id": "e_002", "direction": "supports"}
      ]
    },
    {
      "claim_id": "c_002",
      "natural_language": "Gastruloids form organized epithelial structures with apicobasal polarity",
      "subject": {"name": "gastruloids", "type": "cell_type", "ontology_id": null},
      "predicate": "maintains",
      "object": {"name": "epithelial apicobasal polarity", "type": "phenotype", "ontology_id": null},
      "direction": "negative",
      "claim_type": "absence",
      "causal_type": null,
      "conditions": {
        "species": ["Mus musculus"],
        "cell_type": ["mESC-derived gastruloid"],
        "tissue": [],
        "treatment": ["no Matrigel"],
        "developmental_stage": "120h",
        "in_vitro": true
      },
      "evidence_strength": "direct_experimental",
      "certainty": "high",
      "section_source": "primary_empirical",
      "source_doi": null,
      "evidence_links": [
        {"evidence_id": "e_001", "direction": "refutes"}
      ]
    },
    {
      "claim_id": "c_003",
      "natural_language": "Wnt signaling activates Brachyury expression during gastrulation",
      "subject": {"name": "Wnt signaling", "type": "pathway", "ontology_id": null},
      "predicate": "activates",
      "object": {"name": "Brachyury", "type": "gene", "ontology_id": null},
      "direction": "positive",
      "claim_type": "mechanistic_causal",
      "causal_type": "contributory",
      "conditions": {
        "species": ["Mus musculus"],
        "cell_type": [],
        "tissue": ["primitive streak"],
        "treatment": [],
        "developmental_stage": "E6.5-E7.5",
        "in_vitro": false
      },
      "evidence_strength": "review_citation",
      "certainty": "high",
      "section_source": "attributed_prior",
      "source_doi": "10.1038/nature12345",
      "evidence_links": []
    }
  ],
  "evidence": [
    {
      "evidence_id": "e_001",
      "description": "BMP4 knockout gastruloids cultured for 5 days, assessed for T/Brachyury expression by immunofluorescence",
      "result_summary": "BMP4 knockout caused 3.2-fold reduction in T/Brachyury expression, confirming BMP4 is required for mesoderm specification in gastruloids",
      "model_system": "mouse ESC gastruloids (E14Tg2a)",
      "organism": "Mus musculus",
      "perturbation": "CRISPR KO of BMP4",
      "readout": "T/Brachyury immunofluorescence intensity",
      "result_direction": "positive",
      "effect_size": "3.2-fold reduction",
      "p_value": "p < 0.001",
      "sample_size": "n=30 gastruloids per condition",
      "key_figure": "Figure 2A",
      "approach": "cell_biology",
      "assay_types": ["immunofluorescence", "confocal_microscopy"]
    }
  ],
  "extraction_model": "claude-haiku-4-5-20251001",
  "extraction_timestamp": "2026-03-26T00:00:00Z"
}
```

---

{PAPER_TEXT}
