from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def sample_paper_provenance() -> dict:
    return {
        "doi": "10.1038/s41586-020-0001",
        "pmid": "12345678",
        "title": "Wnt signaling in gastruloid development",
        "authors": [
            {
                "name": "Smith, Alice",
                "orcid": None,
                "affiliations": ["Lab A, MIT"],
                "role": "first_author",
            },
            {
                "name": "Jones, Bob",
                "orcid": None,
                "affiliations": ["Lab A, MIT"],
                "role": "last_author",
            },
        ],
        "journal": "Nature",
        "publication_date": "2023-01-15",
        "peer_reviewed": True,
        "preprint_doi": None,
        "funding_sources": ["NIH R01"],
        "conflicts_of_interest": [],
        "data_availability": "GEO: GSE12345",
    }


@pytest.fixture
def sample_assertion_draft() -> dict:
    return {
        "draft_id": "a_001",
        "natural_language": "Wnt signaling is required for mesoderm formation in gastruloids",
        "canonical_form": "Wnt signaling -> is_required_for -> mesoderm formation",
        "negatable_form": "Wnt signaling is not required for mesoderm formation",
        "subject_entity": {
            "surface_form": "Wnt signaling",
            "canonical_name": "Wnt signaling pathway",
            "ontology_id": "GO:0016055",
            "ontology_source": "GO",
            "entity_type": "pathway",
            "aliases": ["Wnt pathway", "canonical Wnt"],
        },
        "object_entity": {
            "surface_form": "mesoderm formation",
            "canonical_name": "mesoderm formation",
            "ontology_id": "GO:0001707",
            "ontology_source": "GO",
            "entity_type": "biological_process",
            "aliases": ["mesoderm development"],
        },
        "predicate": "is_required_for",
        "direction": "positive",
        "assertion_type": "mechanistic_causal",
        "causal_type": "necessary",
        "scope": "gastruloid_specific",
        "conditions": [],
        "hedging": None,
        "epistemic_status": "established",
        "evidence_unit_ids": ["e_001"],
        "parent_assertion_ids": [],
        "provenance": {"extraction_method": "llm", "confidence": 0.9},
    }


@pytest.fixture
def sample_evidence_unit() -> dict:
    return {
        "evidence_id": "e_001",
        "assertion_draft_ids": ["a_001"],
        "evidence_direction": "supports",
        "evidence_strength": "direct_experimental",
        "experiment": {
            "description": "CHIR99021 (Wnt agonist) treatment of gastruloids",
            "model_system": "Mouse ESC-derived gastruloids",
            "organism": "Mus musculus",
            "perturbation_type": "chemical",
            "readout": "T/Brachyury immunofluorescence",
            "control_description": "Untreated gastruloids",
        },
        "results": {
            "result_direction": "positive",
            "effect_description": "CHIR treatment induced robust mesoderm marker expression",
            "effect_size": "3.5-fold increase",
            "sample_size": "n=50 gastruloids per condition",
            "key_figure": "Fig. 2A",
        },
        "methodological_tags": {
            "approach_category": "in_vitro_model",
            "assay_types": ["immunofluorescence"],
            "blinding_reported": None,
            "randomization_reported": None,
        },
        "limitations_stated_by_authors": [],
        "source_section": "results",
        "source_text_span": "Treatment with CHIR99021...",
    }


@pytest.fixture
def sample_citation_context() -> dict:
    return {
        "citation_id": "c_001",
        "citing_sentence": (
            "Previous work demonstrated that Wnt signaling is essential for"
            " mesoderm specification (Smith et al., 2020)."
        ),
        "cited_source_doi": "10.1016/j.cell.2020.0001",
        "cited_source_pmid": "87654321",
        "cited_claim_paraphrase": "Wnt is essential for mesoderm",
        "relationship": "supports",
        "linked_assertion_draft_ids": ["a_001"],
        "section": "introduction",
    }


@pytest.fixture
def sample_extraction_json(
    sample_paper_provenance: dict,
    sample_assertion_draft: dict,
    sample_evidence_unit: dict,
    sample_citation_context: dict,
) -> dict:
    return {
        "paper_provenance": sample_paper_provenance,
        "assertion_drafts": [sample_assertion_draft],
        "evidence_units": [sample_evidence_unit],
        "citation_contexts": [sample_citation_context],
        "extraction_metadata": {"model": "haiku", "timestamp": "2025-01-01T00:00:00Z"},
    }


@pytest.fixture
def sample_extraction_dir(
    tmp_path: Path,
    sample_extraction_json: dict,
) -> Path:
    """Write 3 extraction JSONs to a temp dir with slight variations."""
    for i, paper_hash in enumerate(["aaa111", "bbb222", "ccc333"]):
        data = json.loads(json.dumps(sample_extraction_json))  # deep copy
        data["paper_provenance"]["doi"] = f"10.1038/paper-{i}"
        data["paper_provenance"]["title"] = f"Paper {i} on gastruloids"
        data["assertion_drafts"][0]["draft_id"] = f"a_{i:03d}"
        data["evidence_units"][0]["evidence_id"] = f"e_{i:03d}"
        data["evidence_units"][0]["assertion_draft_ids"] = [f"a_{i:03d}"]
        data["assertion_drafts"][0]["evidence_unit_ids"] = [f"e_{i:03d}"]

        # Paper 2: same entities, different predicate synonym -> should merge after normalization
        if i == 1:
            data["assertion_drafts"][0]["predicate"] = "is_necessary_for"

        # Paper 3: different entities entirely
        if i == 2:
            data["assertion_drafts"][0]["subject_entity"]["canonical_name"] = (
                "BMP signaling pathway"
            )
            data["assertion_drafts"][0]["subject_entity"]["ontology_id"] = "GO:0030509"
            data["assertion_drafts"][0]["object_entity"]["canonical_name"] = (
                "dorsal-ventral axis specification"
            )
            data["assertion_drafts"][0]["object_entity"]["ontology_id"] = "GO:0009953"
            data["assertion_drafts"][0]["predicate"] = "induces"

        (tmp_path / f"{paper_hash}.json").write_text(json.dumps(data, indent=2))

    # Also write a non-JSON file that should be skipped
    (tmp_path / "debug_raw.txt").write_text("raw extraction text")

    return tmp_path


@pytest.fixture
def sample_v5_claim() -> dict:
    """A v5-format claim with conditions, certainty, section_source."""
    return {
        "claim_id": "c_001",
        "natural_language": "BMP4 induces mesoderm differentiation in mouse ESC gastruloids",
        "subject": {
            "name": "BMP4",
            "type": "protein",
            "ontology_id": "UniProt:P21275",
        },
        "object": {
            "name": "mesoderm differentiation",
            "type": "biological_process",
            "ontology_id": "GO:0007498",
        },
        "predicate": "induces",
        "direction": "positive",
        "claim_type": "mechanistic_causal",
        "causal_type": "sufficient",
        "conditions": {
            "species": ["Mus musculus"],
            "cell_type": ["mESC"],
            "tissue": [],
            "treatment": ["10 ng/mL BMP4"],
            "developmental_stage": "day 5",
            "in_vitro": True,
        },
        "evidence_strength": "direct_experimental",
        "certainty": "high",
        "section_source": "primary_empirical",
        "source_doi": None,
        "model_system": "mouse ESC gastruloids",
        "organism": "Mus musculus",
        "quantitative_context": {
            "concentration": "10 ng/mL BMP4",
            "timepoint": "48h",
            "dose": None,
        },
        "evidence_links": [{"evidence_id": "e_001", "direction": "supports"}],
    }


@pytest.fixture
def sample_v5_evidence() -> dict:
    """A v5-format evidence unit."""
    return {
        "evidence_id": "e_001",
        "description": "BMP4 treatment of mouse ESC gastruloids at day 3",
        "result_summary": "BMP4 at 10 ng/mL induced robust T/Brachyury expression by day 5",
        "model_system": "mouse ESC gastruloids",
        "organism": "Mus musculus",
        "perturbation": None,
        "readout": "T/Brachyury immunofluorescence",
        "result_direction": "positive",
        "effect_size": "3.5-fold increase",
        "p_value": "p < 0.001",
        "sample_size": "n=50",
        "key_figure": "Figure 2A",
        "approach": "cell_biology",
        "assay_types": ["immunofluorescence"],
        "evidence_strength": "direct_experimental",
    }


@pytest.fixture
def sample_v5_extraction_dir(
    tmp_path: Path,
    sample_v5_claim: dict,
    sample_v5_evidence: dict,
) -> Path:
    """Write v5-format extraction JSONs with varying conditions for v2 graph testing.

    Creates 3 papers:
    - Paper 1: BMP4 induces mesoderm (mouse, in vitro, gastruloids)
    - Paper 2: BMP4 induces mesoderm (mouse, in vitro, gastruloids) — same context, should merge
    - Paper 3: BMP4 induces mesoderm (human, in vitro, iPSC organoids) — different context, separate
    """
    paper_configs = [
        ("mouse_p1", "Mus musculus", "mouse ESC gastruloids", ["mESC"]),
        ("mouse_p2", "Mus musculus", "mouse ESC gastruloids", ["E14Tg2a"]),
        ("human_p3", "Homo sapiens", "human iPSC organoids", ["iPSC"]),
    ]
    for i, (paper_hash, organism, model_sys, cell_types) in enumerate(paper_configs):
        claim = json.loads(json.dumps(sample_v5_claim))
        evidence = json.loads(json.dumps(sample_v5_evidence))

        claim["claim_id"] = f"c_{i + 1:03d}"
        claim["organism"] = organism
        claim["model_system"] = model_sys
        claim["conditions"]["species"] = [organism]
        claim["conditions"]["cell_type"] = cell_types
        claim["evidence_links"] = [{"evidence_id": f"e_{i + 1:03d}", "direction": "supports"}]

        evidence["evidence_id"] = f"e_{i + 1:03d}"
        evidence["organism"] = organism
        evidence["model_system"] = model_sys

        data = {
            "doi": f"10.1234/paper-{i}",
            "title": f"Paper {i} on BMP4",
            "journal": "Nature",
            "publication_date": f"2023-0{i + 1}-15",
            "claims": [claim],
            "evidence": [evidence],
            "citation_contexts": [],
        }
        (tmp_path / f"{paper_hash}.json").write_text(json.dumps(data, indent=2))

    return tmp_path
