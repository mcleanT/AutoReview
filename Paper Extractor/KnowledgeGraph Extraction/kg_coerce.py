"""Shared coercion pipeline for KG extractions.

Pre-processes raw LLM JSON dicts to match the KGExtraction schema by
normalising enum values, migrating deprecated fields, and filling
required defaults.  Import ``coerce_kg_dict`` from this module wherever
KG extraction output needs to be coerced.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Enum coercion maps
# ---------------------------------------------------------------------------

_VALID_CLAIM_TYPES = {
    "mechanistic_causal",
    "correlational",
    "comparative",
    "existence",
    "absence",
    "conditional",
    "methodological",
}
_CLAIM_TYPE_MAP: dict[str, str] = {
    "causal": "mechanistic_causal",
    "observational": "correlational",
}

_VALID_EVIDENCE_STRENGTHS = {
    "direct_experimental",
    "indirect_experimental",
    "observational",
    "computational",
    "review_citation",
}
_EVIDENCE_STRENGTH_MAP: dict[str, str] = {
    "systematic_review_meta_analysis": "review_citation",
    "randomized_controlled_trial": "direct_experimental",
    "observational_controlled": "observational",
    "observational_uncontrolled": "observational",
    "computational_prediction": "computational",
    "case_report": "observational",
    "expert_opinion": "review_citation",
}

_VALID_ENTITY_TYPES = {
    "protein",
    "gene",
    "rna",
    "small_molecule",
    "pathway",
    "biological_process",
    "phenotype",
    "cellular_compartment",
    "organism",
    "cell_type",
    "disease",
    "tissue",
    "method",
    "other",
}

_VALID_APPROACHES = {
    "biochemical_assay",
    "cell_biology",
    "genetics",
    "omics",
    "imaging",
    "computational",
    "clinical",
    "animal_model",
    "in_vitro_model",
    "structural_biology",
    "pharmacology",
    "citation_reference",
}

_VALID_SECTION_SOURCES = {
    "primary_empirical",
    "interpretive",
    "attributed_prior",
    "methodological",
}
_SECTION_SOURCE_MAP: dict[str, str] = {
    "results": "primary_empirical",
    "novel_finding": "primary_empirical",
    "discussion": "interpretive",
    "interpretation": "interpretive",
    "hypothesis": "interpretive",
    "background": "attributed_prior",
    "introduction": "attributed_prior",
    "methods": "methodological",
    "methodological_note": "methodological",
}

_VALID_RESULT_DIRECTIONS = {"positive", "negative", "null", "not_reported"}

_VALID_EVIDENCE_DIRECTIONS = {"supports", "refutes", "mixed", "not_applicable"}

_VALID_CERTAINTIES = {"high", "medium", "low"}

_VALID_DIRECTIONS = {"positive", "negative"}

_VALID_PREDICATES = {
    "activates",
    "inhibits",
    "binds_to",
    "localizes_to",
    "is_required_for",
    "promotes",
    "regulates",
    "colocalizes_with",
    "phosphorylates",
    "is_expressed_in",
    "interacts_with",
    "suppresses",
    "induces",
    "differentiates_into",
    "is_marker_of",
    "correlates_with",
    "is_sufficient_for",
    "is_necessary_for",
    "upregulates",
    "downregulates",
    "is_component_of",
    "degrades",
    "stabilizes",
    "transports",
    "modifies",
    "converts",
    "mediates",
    "blocks",
    "enhances",
    "reduces",
    "maintains",
    "disrupts",
    "enables",
    "prevents",
}

# Maps invalid predicates Haiku commonly generates to valid ones.
# Entries with a tuple value (predicate, direction) also override direction.
# Entries with a plain string value only remap the predicate.
_PREDICATE_COERCION_MAP: dict[str, str | tuple[str, str]] = {
    # "lacks" → X does not maintain Y; direction flipped to negative
    "lacks": ("maintains", "negative"),
    "forms": "induces",
    "contains": "is_component_of",
    "generates": "induces",
    "expresses": "is_expressed_in",
    "represses": "suppresses",
    "models": "correlates_with",
    "recapitulates": "correlates_with",
    "is_active": "is_expressed_in",
    "is_active_in": "is_expressed_in",
    # v4 regressions
    "develops": "induces",
    "exhibits": "maintains",
    "differs": "correlates_with",
    "provides": "enables",
    "controls": "regulates",
}

_VALID_CAUSAL_TYPES = {
    "necessary",
    "sufficient",
    "necessary_and_sufficient",
    "contributory",
    "modulatory",
}


# ---------------------------------------------------------------------------
# Coercion function
# ---------------------------------------------------------------------------


def coerce_kg_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Pre-process LLM JSON to match KGExtraction schema."""
    # Ensure top-level required strings
    for key in ("doi", "title", "journal", "publication_date"):
        if key not in d or d[key] is None:
            d[key] = ""

    # Ensure lists
    d.setdefault("claims", [])
    d.setdefault("evidence", [])

    # Coerce evidence units
    for ev in d.get("evidence", []):
        if not isinstance(ev, dict):
            continue
        ev.setdefault("evidence_id", "e_000")
        ev.setdefault("description", "")
        ev.setdefault("result_summary", "")
        ev.setdefault("model_system", "")
        ev.setdefault("organism", "")
        ev.setdefault("readout", "")
        ev.setdefault("assay_types", [])
        # Fix None values in required string fields (setdefault only covers absent keys)
        for req_str in ("description", "result_summary", "model_system", "organism", "readout"):
            if ev.get(req_str) is None:
                ev[req_str] = ""

        # Coerce result_direction
        rd = ev.get("result_direction", "not_reported")
        if rd not in _VALID_RESULT_DIRECTIONS:
            ev["result_direction"] = "not_reported"

        # Coerce approach
        approach = ev.get("approach", "cell_biology")
        if approach not in _VALID_APPROACHES:
            ev["approach"] = "cell_biology"  # safe default

        # Coerce v5 citation-stub evidence fields
        for str_field in ("evidence_strength", "citing_sentence", "source_doi"):
            val = ev.get(str_field)
            if val is not None and not isinstance(val, str):
                ev[str_field] = str(val)

    # Coerce claims
    for claim in d.get("claims", []):
        if not isinstance(claim, dict):
            continue
        claim.setdefault("claim_id", "c_000")
        claim.setdefault("natural_language", "")
        claim.setdefault("predicate", "unknown")

        # evidence_ids → evidence_links migration
        if "evidence_ids" in claim and "evidence_links" not in claim:
            claim["evidence_links"] = [
                {"evidence_id": eid, "direction": "supports"} for eid in claim.pop("evidence_ids")
            ]
        elif "evidence_ids" in claim and "evidence_links" in claim:
            claim.pop("evidence_ids")  # prefer evidence_links if both present

        claim.setdefault("evidence_links", [])

        # Coerce each evidence_link object
        coerced_links = []
        for link in claim["evidence_links"]:
            if isinstance(link, str):
                # bare string → object
                link = {"evidence_id": link, "direction": "supports"}
            if link.get("direction") not in _VALID_EVIDENCE_DIRECTIONS:
                link["direction"] = "supports"  # safe default
            coerced_links.append(link)
        claim["evidence_links"] = coerced_links

        # Post-processing: auto-set "refutes" for absence claims
        # If a claim is typed as "absence", the evidence demonstrates that
        # something does NOT hold — it refutes the positive form.
        # Note: direction="negative" alone is NOT sufficient — a negative
        # correlation (anti-correlation) is a real finding that evidence
        # supports, not refutes.
        claim_ct = claim.get("claim_type", "existence")
        if claim_ct == "absence":
            for link in claim["evidence_links"]:
                if link.get("direction") == "supports":
                    link["direction"] = "refutes"

        # Ensure required fields exist (truncation repair can create incomplete claims)
        claim.setdefault("direction", "positive")
        claim.setdefault("claim_type", "existence")
        claim.setdefault("evidence_strength", "direct_experimental")
        claim.setdefault("certainty", "medium")
        claim.setdefault("section_source", "primary_empirical")

        # Coerce direction
        direction = claim.get("direction", "positive")
        if direction not in _VALID_DIRECTIONS:
            claim["direction"] = "positive"

        # Coerce predicate — map invalid values to nearest valid predicate
        predicate = claim.get("predicate", "maintains")
        if predicate not in _VALID_PREDICATES:
            coercion = _PREDICATE_COERCION_MAP.get(predicate)
            if isinstance(coercion, tuple):
                claim["predicate"], claim["direction"] = coercion
            elif isinstance(coercion, str):
                claim["predicate"] = coercion
            # Unknown invalid predicate: leave as-is (schema will reject if truly invalid)

        # Coerce claim_type
        ct = claim.get("claim_type", "existence")
        if ct not in _VALID_CLAIM_TYPES:
            claim["claim_type"] = _CLAIM_TYPE_MAP.get(ct, "existence")

        # Coerce causal_type
        ctype = claim.get("causal_type")
        if ctype is not None and ctype not in _VALID_CAUSAL_TYPES:
            claim["causal_type"] = None

        # Coerce evidence_strength
        es = claim.get("evidence_strength", "direct_experimental")
        if es not in _VALID_EVIDENCE_STRENGTHS:
            claim["evidence_strength"] = _EVIDENCE_STRENGTH_MAP.get(es, "direct_experimental")

        # Coerce certainty
        cert = claim.get("certainty", "medium")
        if cert not in _VALID_CERTAINTIES:
            claim["certainty"] = "medium"

        # Coerce section_source
        ss = claim.get("section_source", "primary_empirical")
        if ss not in _VALID_SECTION_SOURCES:
            claim["section_source"] = _SECTION_SOURCE_MAP.get(ss, "primary_empirical")

        # Coerce entities
        for entity_key in ("subject", "object"):
            ent = claim.get(entity_key)
            if ent is None:
                claim[entity_key] = {"name": "unknown", "type": "other", "ontology_id": None}
            elif isinstance(ent, dict):
                ent.setdefault("name", "unknown")
                ent.setdefault("type", "other")
                etype = ent.get("type", "other")
                if etype not in _VALID_ENTITY_TYPES:
                    ent["type"] = "other"

        # Coerce conditions
        conds = claim.get("conditions")
        if conds is None:
            claim["conditions"] = {}
        elif isinstance(conds, dict):
            # Ensure list fields
            for list_key in ("species", "cell_type", "tissue", "treatment"):
                val = conds.get(list_key)
                if val is None:
                    conds[list_key] = []
                elif isinstance(val, str):
                    conds[list_key] = [val]
            # developmental_stage must be str|None, not list
            ds = conds.get("developmental_stage")
            if ds is not None and not isinstance(ds, str):
                if isinstance(ds, list) and ds:
                    conds["developmental_stage"] = str(ds[0])
                else:
                    conds["developmental_stage"] = None

        # Coerce v5 claim-level context fields
        for str_field in ("model_system", "organism"):
            val = claim.get(str_field)
            if val is not None and not isinstance(val, str):
                claim[str_field] = str(val)
        qc = claim.get("quantitative_context")
        if qc is not None and not isinstance(qc, dict):
            claim["quantitative_context"] = None

    return d
