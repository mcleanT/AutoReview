"""
Builder script for Mycelium v2 extraction of:
  Lobos Patorniti et al. (2023) — "Rai14 is a novel interactor of Invariant chain
  that regulates macropinocytosis"
  Front. Immunol. 14:1182180. doi: 10.3389/fimmu.2023.1182180

v2 schema rules:
  - NO background assertion drafts (is_primary=False / function="background")
  - ONLY novel findings as AssertionDrafts (all is_primary=True)
  - Background claims from prior work → CitationContext objects
  - Skip generic bulk citations; only capture specific-finding citations
"""

import pathlib
import sys

# ---------------------------------------------------------------------------
# Ensure project src is importable
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mycelium.extraction_schema import (
    AssertionDraft,
    AssertionTypeEnum,
    Author,
    CausalityHedgeEnum,
    CausalTypeEnum,
    CitationContext,
    CitationRelationshipEnum,
    Condition,
    ConditionTypeEnum,
    DirectionEnum,
    Entity,
    EntityTypeEnum,
    EpistemicStatus,
    EvidenceDirectionEnum,
    EvidenceStrengthEnum,
    EvidenceUnit,
    Experiment,
    ExtractionMetadata,
    ExtractionResult,
    FunctionEnum,
    HedgeLevelEnum,
    Hedging,
    MethodologicalTags,
    OntologyTerm,
    PaperProvenance,
    Provenance,
    ResultDirectionEnum,
    Results,
    Scope,
    SectionEnum,
)


# ---------------------------------------------------------------------------
# Helper: human/MelJuSo scope
# ---------------------------------------------------------------------------
def human_meljuso_scope(in_vitro: bool = True) -> Scope:
    return Scope(
        species=[
            OntologyTerm(
                term_id="NCBITaxon:9606",
                term_name="Homo sapiens",
                ontology="NCBI Taxonomy",
                surface_form="human",
            )
        ],
        tissue=[],
        cell_type=[
            OntologyTerm(
                term_id="CL:0001087",
                term_name="melanoma cell",
                ontology="CL",
                surface_form="MelJuSo cells",
            )
        ],
        disease=[
            OntologyTerm(
                term_id="MONDO:0005105",
                term_name="melanoma",
                ontology="MONDO",
                surface_form="melanoma",
            )
        ],
        condition=None,
        developmental_stage=None,
        in_vitro=in_vitro,
    )


def mouse_bmdc_scope() -> Scope:
    return Scope(
        species=[
            OntologyTerm(
                term_id="NCBITaxon:10090",
                term_name="Mus musculus",
                ontology="NCBI Taxonomy",
                surface_form="C57BL/6 mice",
            )
        ],
        tissue=[
            OntologyTerm(
                term_id="UBERON:0002371",
                term_name="bone marrow",
                ontology="UBERON",
                surface_form="bone marrow",
            )
        ],
        cell_type=[
            OntologyTerm(
                term_id="CL:0000451",
                term_name="dendritic cell",
                ontology="CL",
                surface_form="bone marrow-derived dendritic cells (BMDCs)",
            )
        ],
        disease=[],
        condition=None,
        developmental_stage="8-16 weeks",
        in_vitro=True,
    )


def dual_scope() -> Scope:
    """Scope covering both MelJuSo and BMDC contexts."""
    return Scope(
        species=[
            OntologyTerm(
                term_id="NCBITaxon:9606",
                term_name="Homo sapiens",
                ontology="NCBI Taxonomy",
                surface_form="human",
            ),
            OntologyTerm(
                term_id="NCBITaxon:10090",
                term_name="Mus musculus",
                ontology="NCBI Taxonomy",
                surface_form="mouse",
            ),
        ],
        tissue=[],
        cell_type=[
            OntologyTerm(
                term_id="CL:0001087",
                term_name="melanoma cell",
                ontology="CL",
                surface_form="MelJuSo cells",
            ),
            OntologyTerm(
                term_id="CL:0000451",
                term_name="dendritic cell",
                ontology="CL",
                surface_form="BMDCs",
            ),
        ],
        disease=[
            OntologyTerm(
                term_id="MONDO:0005105",
                term_name="melanoma",
                ontology="MONDO",
                surface_form="melanoma",
            )
        ],
        condition=None,
        developmental_stage=None,
        in_vitro=True,
    )


# ---------------------------------------------------------------------------
# Shared entities
# ---------------------------------------------------------------------------
RAI14 = Entity(
    surface_form="Rai14",
    canonical_name="RAI14",
    ontology_id="UniProt:Q9UHD9",
    ontology_source="UniProt",
    entity_type=EntityTypeEnum.PROTEIN,
    aliases=["Retinoic Acid-Induced 14", "NORPEG", "ankycorbin"],
)

II_CD74 = Entity(
    surface_form="Invariant chain (Ii)",
    canonical_name="CD74",
    ontology_id="UniProt:P04233",
    ontology_source="UniProt",
    entity_type=EntityTypeEnum.PROTEIN,
    aliases=["Ii", "CD74", "Invariant chain", "Ii p33"],
)

MYOSIN_II = Entity(
    surface_form="myosin II",
    canonical_name="MYH9",
    ontology_id="UniProt:P35579",
    ontology_source="UniProt",
    entity_type=EntityTypeEnum.PROTEIN,
    aliases=["non-muscle myosin IIA", "NMIIA", "myosin II heavy chain"],
)

MHC_II = Entity(
    surface_form="MHC II",
    canonical_name="HLA-DR",
    ontology_id="UniProt:P01903",
    ontology_source="UniProt",
    entity_type=EntityTypeEnum.PROTEIN,
    aliases=["HLA-DR", "HLA-DR1 b", "MHC class II"],
)

MACROPINOCYTOSIS = Entity(
    surface_form="macropinocytosis",
    canonical_name="macropinocytosis",
    ontology_id="GO:0044351",
    ontology_source="GO",
    entity_type=EntityTypeEnum.OTHER,
    aliases=["macropinosome formation", "macropinocytic activity"],
)

CELL_MIGRATION = Entity(
    surface_form="cell migration",
    canonical_name="cell migration",
    ontology_id="GO:0016477",
    ontology_source="GO",
    entity_type=EntityTypeEnum.OTHER,
    aliases=["cell motility", "DC migration"],
)

MEMBRANE_RUFFLES = Entity(
    surface_form="membrane ruffles",
    canonical_name="membrane ruffle",
    ontology_id="GO:0001726",
    ontology_source="GO",
    entity_type=EntityTypeEnum.OTHER,
    aliases=["plasma membrane ruffles", "membrane ruffle"],
)

PAK = Entity(
    surface_form="PAK",
    canonical_name="PAK1",
    ontology_id="UniProt:Q13153",
    ontology_source="UniProt",
    entity_type=EntityTypeEnum.PROTEIN,
    aliases=["p21-activated kinase", "PAK1/2/3"],
)

MACROPINOSOME_CLOSURE = Entity(
    surface_form="macropinosome closure",
    canonical_name="macropinosome closure",
    ontology_id="GO:0099615",
    ontology_source="GO",
    entity_type=EntityTypeEnum.OTHER,
    aliases=["macropinosome sealing"],
)

# ---------------------------------------------------------------------------
# EVIDENCE UNITS
# ---------------------------------------------------------------------------
evidence_units = [
    # e_001: Yeast two-hybrid identifies Rai14 as Ii interactor
    EvidenceUnit(
        evidence_id="e_001",
        assertion_draft_ids=["a_001"],
        evidence_direction=EvidenceDirectionEnum.SUPPORTS,
        evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
        experiment=Experiment(
            description="Yeast two-hybrid screen using full-length Ii p33 as bait against a human placenta cDNA library (performed by Hybrigenics Services, Paris, France). Rai14 was identified as a positive hit (Supplementary Table 1).",
            model_system="Yeast two-hybrid in vitro screen, human placenta cDNA library",
            organism="Homo sapiens",
            organism_strain=None,
            perturbation_type="none",
            perturbation_target=None,
            perturbation_method=None,
            readout="Yeast reporter gene activation (growth on selective media)",
            control_description="Negative controls inherent to yeast two-hybrid system",
        ),
        results=Results(
            result_direction=ResultDirectionEnum.POSITIVE,
            effect_description="Rai14 identified as one of the strongest positive candidates for Ii interaction",
            effect_size=None,
            statistical_test=None,
            p_value=None,
            confidence_interval=None,
            sample_size=None,
            key_figure="Supplementary Table 1",
        ),
        methodological_tags=MethodologicalTags(
            approach_category="biochemical_assay",
            assay_types=["yeast_two_hybrid"],
            blinding_reported=None,
            randomization_reported=None,
        ),
        limitations_stated_by_authors=[
            "Yeast two-hybrid may produce false positives; interaction requires independent validation"
        ],
        source_section="results",
        source_text_span="Rai14 was identified as a positive hit (Supplementary Table 1).",
    ),
    # e_002: Co-IP — Ii pulls down Rai14 (anti-Ii direction)
    EvidenceUnit(
        evidence_id="e_002",
        assertion_draft_ids=["a_001"],
        evidence_direction=EvidenceDirectionEnum.SUPPORTS,
        evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
        experiment=Experiment(
            description="Co-immunoprecipitation from MelJuSo cell lysates using anti-Ii antibody (MB741, BD Biosciences cat. 555538, IP 1:200). Immunoprecipitated samples and whole cell lysates analyzed by Western blot with antibodies against Ii and Rai14 (Abcam ab137118).",
            model_system="MelJuSo human melanoma cell line",
            organism="Homo sapiens",
            organism_strain=None,
            perturbation_type="none",
            perturbation_target=None,
            perturbation_method=None,
            readout="Co-immunoprecipitation band detection by Western blot",
            control_description="Mouse IgG2aK isotype control (BD Biosciences cat. 555571)",
        ),
        results=Results(
            result_direction=ResultDirectionEnum.POSITIVE,
            effect_description="Rai14 was specifically pulled down by endogenous Ii and not by IgG2aK isotype control",
            effect_size=None,
            statistical_test=None,
            p_value=None,
            confidence_interval=None,
            sample_size=None,
            key_figure="Figure 1A",
        ),
        methodological_tags=MethodologicalTags(
            approach_category="biochemical_assay",
            assay_types=["co-immunoprecipitation", "Western_blot"],
            blinding_reported=None,
            randomization_reported=None,
        ),
        limitations_stated_by_authors=["Co-IP does not distinguish direct from indirect binding"],
        source_section="results",
        source_text_span="Rai14 was successfully pulled down by the endogenous Ii and not by a mouse IgG2a k isotype control, indicating that Rai14 specifically binds to Ii (Figure 1A).",
    ),
    # e_003: Co-IP — Rai14 pulls down Ii (anti-Rai14 direction)
    EvidenceUnit(
        evidence_id="e_003",
        assertion_draft_ids=["a_001"],
        evidence_direction=EvidenceDirectionEnum.SUPPORTS,
        evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
        experiment=Experiment(
            description="Co-immunoprecipitation from MelJuSo cell lysates using anti-Rai14 antibody (Abcam ab137118, IP 1:200). Western blot analysis with antibodies against Ii and Rai14.",
            model_system="MelJuSo human melanoma cell line",
            organism="Homo sapiens",
            organism_strain=None,
            perturbation_type="none",
            perturbation_target=None,
            perturbation_method=None,
            readout="Co-immunoprecipitation of endogenous Ii by anti-Rai14, detected by Western blot",
            control_description="IgG isotype control (BD Biosciences cat. 550875)",
        ),
        results=Results(
            result_direction=ResultDirectionEnum.POSITIVE,
            effect_description="Endogenous Ii was co-immunoprecipitated by Rai14 and not by IgG isotype control",
            effect_size=None,
            statistical_test=None,
            p_value=None,
            confidence_interval=None,
            sample_size=None,
            key_figure="Figure 1B",
        ),
        methodological_tags=MethodologicalTags(
            approach_category="biochemical_assay",
            assay_types=["co-immunoprecipitation", "Western_blot"],
            blinding_reported=None,
            randomization_reported=None,
        ),
        limitations_stated_by_authors=[],
        source_section="results",
        source_text_span="Also, in this case, endogenous Ii was co-immunoprecipitated by Rai14 and not by the IgG isotype control (Figure 1B).",
    ),
    # e_004: Live imaging — Rai14 and Ii co-localize at membrane ruffles and macropinosomes
    EvidenceUnit(
        evidence_id="e_004",
        assertion_draft_ids=["a_002", "a_003"],
        evidence_direction=EvidenceDirectionEnum.SUPPORTS,
        evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
        experiment=Experiment(
            description="Live cell imaging of MelJuSo cells co-transfected with Ii p33 and Rai14-GFP. Fluorescently labeled anti-Ii antibody (M-B741 conjugated with Alexa Fluor 555) added to medium and imaged on spinning disk confocal at 30-second intervals. Manders' colocalization coefficients quantified using ImageJ.",
            model_system="MelJuSo human melanoma cell line",
            organism="Homo sapiens",
            organism_strain=None,
            perturbation_type="genetic_gain_of_function",
            perturbation_target="RAI14",
            perturbation_method="transient transfection with Rai14-GFP plasmid (pCMV6-AC-Rai14-GFP, OriGene)",
            readout="Colocalization by Manders' coefficient; live imaging of membrane ruffle dynamics",
            control_description="No perturbation control; Rai14-GFP only (Supplementary Figure 1A)",
        ),
        results=Results(
            result_direction=ResultDirectionEnum.POSITIVE,
            effect_description="Rai14-GFP and Ii colocalized on membrane ruffles, nascent vesicles, and forming macropinosomes. Rai14 colocalized with Ii at early endocytic stages until an Ii-positive vesicle pinched off. At steady state, Rai14-GFP was detected on domains of Ii-positive endosomes.",
            effect_size=None,
            statistical_test="Manders' colocalization coefficient",
            p_value=None,
            confidence_interval=None,
            sample_size=None,
            key_figure="Figures 1C, D, E; Video 2",
        ),
        methodological_tags=MethodologicalTags(
            approach_category="imaging",
            assay_types=["live_cell_imaging", "confocal_microscopy", "colocalization_analysis"],
            blinding_reported=None,
            randomization_reported=None,
        ),
        limitations_stated_by_authors=[],
        source_section="results",
        source_text_span="Rai14-GFP and MHC II colocalized on membrane ruffles, and these ruffles lead to the formation of macropinosomes.",
    ),
    # e_005: GFP-TRAP co-IP — Rai14 in MHC II complex with Ii
    EvidenceUnit(
        evidence_id="e_005",
        assertion_draft_ids=["a_004"],
        evidence_direction=EvidenceDirectionEnum.SUPPORTS,
        evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
        experiment=Experiment(
            description="GFP-TRAP immunoprecipitation (Chromotek GFP-Trap_MA) from lysates of MelJuSo cells stably expressing HLA-DR1 b-GFP. Western blot analysis for endogenous Ii and endogenous Rai14. Control: MelJuSo cells expressing GFP alone.",
            model_system="MelJuSo cells stably expressing HLA-DR1 b-GFP",
            organism="Homo sapiens",
            organism_strain=None,
            perturbation_type="genetic_gain_of_function",
            perturbation_target="HLA-DRB1",
            perturbation_method="stable expression of HLA-DR1 b-GFP",
            readout="Co-immunoprecipitation of endogenous Ii and Rai14 with HLA-DR1 b-GFP by Western blot",
            control_description="MelJuSo cells expressing GFP alone",
        ),
        results=Results(
            result_direction=ResultDirectionEnum.POSITIVE,
            effect_description="Cells expressing HLA-DR1 b-GFP (but not GFP alone) co-immunoprecipitated both endogenous Ii and Rai14, confirming Rai14 interacts with Ii in a complex with MHC II",
            effect_size=None,
            statistical_test=None,
            p_value=None,
            confidence_interval=None,
            sample_size=None,
            key_figure="Figure 2A",
        ),
        methodological_tags=MethodologicalTags(
            approach_category="biochemical_assay",
            assay_types=["GFP-TRAP_immunoprecipitation", "Western_blot"],
            blinding_reported=None,
            randomization_reported=None,
        ),
        limitations_stated_by_authors=[],
        source_section="results",
        source_text_span="Cells expressing HLA-DR1 b-GFP (but not cells expressing GFP) were able to co-immunoprecipitate both endogenous Ii and Rai14 (Figure 2A), confirming that Rai14 interacts with Ii in a complex with MHC II.",
    ),
    # e_006: Confocal imaging — MHC II plasma membrane distribution upon Rai14 siRNA
    EvidenceUnit(
        evidence_id="e_006",
        assertion_draft_ids=["a_005"],
        evidence_direction=EvidenceDirectionEnum.SUPPORTS,
        evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
        experiment=Experiment(
            description="MelJuSo cells stably expressing HLA-DR1 b-GFP transfected with siRNA against Rai14 (Rai14 siRNA#1, Eurofinss MWG Operon) or siRNA control. Cells imaged by confocal laser scanning microscopy (Zeiss LSM880, 63x objective). Plasma membrane GFP fraction quantified using FIJI (IntDen2/IntDen1×100).",
            model_system="MelJuSo HLA-DR1 b-GFP stable cell line",
            organism="Homo sapiens",
            organism_strain=None,
            perturbation_type="genetic_loss_of_function",
            perturbation_target="RAI14",
            perturbation_method="siRNA (Rai14 siRNA#1, Eurofinss MWG Operon)",
            readout="Percentage of HLA-DR1 b-GFP at plasma membrane vs total",
            control_description="siRNA negative control (sense 5'-ACUUCGAGCGUGCAUGGCUTT-3')",
        ),
        results=Results(
            result_direction=ResultDirectionEnum.POSITIVE,
            effect_description="HLA-DR1 b-GFP at the plasma membrane increased 1.5-fold in Rai14 knockdown cells compared to control",
            effect_size="1.5-fold increase",
            statistical_test="two-tailed unpaired Student's t-test",
            p_value="P < 0.01",
            confidence_interval=None,
            sample_size="n=3 independent experiments",
            key_figure="Figures 2B-E",
        ),
        methodological_tags=MethodologicalTags(
            approach_category="imaging",
            assay_types=["confocal_microscopy", "immunofluorescence", "image_quantification"],
            blinding_reported=None,
            randomization_reported=None,
        ),
        limitations_stated_by_authors=[],
        source_section="results",
        source_text_span="the presence of HLA-DR1 b-GFP at the plasma membrane increased 1.5-fold in cells knocked down for Rai14 (Figures 2B–E).",
    ),
    # e_007: Flow cytometry — surface MHC II and Ii levels upon Rai14 siRNA
    EvidenceUnit(
        evidence_id="e_007",
        assertion_draft_ids=["a_005"],
        evidence_direction=EvidenceDirectionEnum.SUPPORTS,
        evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
        experiment=Experiment(
            description="FACS analysis of surface and total levels of HLA-DR (L243 Alexa-647) and Ii (MB741 Alexa-647) in MelJuSo cells transfected with siRai14 vs siRNA control. Surface staining in intact cells; total levels in saponin-permeabilized cells. Analyzed on BD Fortessa; data analyzed with FlowJo.",
            model_system="MelJuSo human melanoma cell line",
            organism="Homo sapiens",
            organism_strain=None,
            perturbation_type="genetic_loss_of_function",
            perturbation_target="RAI14",
            perturbation_method="siRNA",
            readout="Mean fluorescence intensity of surface and total MHC II and Ii by flow cytometry",
            control_description="siRNA negative control; IgG2aK isotype control",
        ),
        results=Results(
            result_direction=ResultDirectionEnum.POSITIVE,
            effect_description="Surface levels of both endogenous MHC II and Ii increased 1.5-fold in Rai14-silenced cells; total protein levels were unaffected",
            effect_size="1.5-fold increase in surface levels; total levels not significant",
            statistical_test="two-tailed unpaired Student's t-test",
            p_value="P < 0.01 (surface); ns (total)",
            confidence_interval=None,
            sample_size="n=3 independent experiments",
            key_figure="Figure 2F; Supplementary Figure 2A",
        ),
        methodological_tags=MethodologicalTags(
            approach_category="cell_biology",
            assay_types=["flow_cytometry", "FACS"],
            blinding_reported=None,
            randomization_reported=None,
        ),
        limitations_stated_by_authors=[],
        source_section="results",
        source_text_span="Flow cytometry analysis indeed confirmed that the surface levels of both endogenous MHC II and Ii increased 1.5-fold in cells silenced for Rai14 compared to control cells, while the total levels remain unaffected (Figure 2F; Supplementary Figure 2A).",
    ),
    # e_008: MHC II internalization assay (antibody uptake, two siRNAs)
    EvidenceUnit(
        evidence_id="e_008",
        assertion_draft_ids=["a_005"],
        evidence_direction=EvidenceDirectionEnum.SUPPORTS,
        evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
        experiment=Experiment(
            description="MelJuSo cells transfected with siRai14#1, siRai14#2, or siRNA control incubated with anti-MHC II antibody conjugated to Alexa Fluor 647 for 45 minutes, then fixed at 0, 30, and 60 minutes. MHC II internalization quantified as percentage of area occupied by MHC II-positive endosomes over total cell area using FIJI particle analysis.",
            model_system="MelJuSo human melanoma cell line",
            organism="Homo sapiens",
            organism_strain=None,
            perturbation_type="genetic_loss_of_function",
            perturbation_target="RAI14",
            perturbation_method="siRNA (two independent siRNAs: siRai14#1 and siRai14#2, Eurofinss)",
            readout="Percentage of MHC II-positive endosomal area relative to total cell area at 0, 30, 60 min",
            control_description="siRNA negative control",
        ),
        results=Results(
            result_direction=ResultDirectionEnum.POSITIVE,
            effect_description="Rai14 knockdown cells internalized ~40% less MHC II into endosomes than control cells at 60 minutes post-antibody addition",
            effect_size="~40% decrease in MHC II in endosomes at t=60 min",
            statistical_test="two-tailed unpaired Student's t-test",
            p_value="*P < 0.05, **P < 0.01 for t = 60 min",
            confidence_interval=None,
            sample_size="n=3 independent experiments; n>45 cells per condition",
            key_figure="Figures 2G-I; Supplementary Figure 2B",
        ),
        methodological_tags=MethodologicalTags(
            approach_category="cell_biology",
            assay_types=[
                "antibody_uptake_assay",
                "immunofluorescence",
                "confocal_microscopy",
                "image_quantification",
            ],
            blinding_reported=None,
            randomization_reported=None,
        ),
        limitations_stated_by_authors=[],
        source_section="results",
        source_text_span="Rai14 knocked-down cells internalized less MHC II in endosomes than the control cells, with a decrease of almost 40% of MHC II in endosomes 60 minutes after the uptake (Figures 2G–I).",
    ),
    # e_009: Live imaging — Rai14-GFP colocalizes with MHC II on ruffles forming macropinosomes
    EvidenceUnit(
        evidence_id="e_009",
        assertion_draft_ids=["a_002", "a_006"],
        evidence_direction=EvidenceDirectionEnum.SUPPORTS,
        evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
        experiment=Experiment(
            description="Time-lapse video microscopy of MelJuSo cells transfected with Rai14-GFP. Anti-HLA-DR antibody (L243, BD Biosciences) conjugated with Alexa 647 added to culture medium before imaging. Colocalization analyzed using JACoP plugin (ImageJ) with Manders' correlation coefficient. Imaged on spinning disk confocal at 1-second intervals.",
            model_system="MelJuSo human melanoma cell line",
            organism="Homo sapiens",
            organism_strain=None,
            perturbation_type="genetic_gain_of_function",
            perturbation_target="RAI14",
            perturbation_method="transient transfection with Rai14-GFP",
            readout="Colocalization of Rai14-GFP and MHC II on membrane ruffles and macropinosomes",
            control_description="None (observational)",
        ),
        results=Results(
            result_direction=ResultDirectionEnum.POSITIVE,
            effect_description="Rai14-GFP and MHC II colocalized on membrane ruffles that led to the formation of macropinosomes",
            effect_size=None,
            statistical_test="Manders' correlation coefficient",
            p_value=None,
            confidence_interval=None,
            sample_size="n=4 independent experiments, n=25 cells",
            key_figure="Figure 3A; Video 3",
        ),
        methodological_tags=MethodologicalTags(
            approach_category="imaging",
            assay_types=["live_cell_imaging", "spinning_disk_confocal", "colocalization_analysis"],
            blinding_reported=None,
            randomization_reported=None,
        ),
        limitations_stated_by_authors=[],
        source_section="results",
        source_text_span="Rai14-GFP and MHC II colocalized on membrane ruffles, and these ruffles lead to the formation of macropinosomes (Figure 3A; Video 3).",
    ),
    # e_010: Dextran macropinocytosis assay — MelJuSo siRai14 (with rescue)
    EvidenceUnit(
        evidence_id="e_010",
        assertion_draft_ids=["a_006", "a_007"],
        evidence_direction=EvidenceDirectionEnum.SUPPORTS,
        evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
        experiment=Experiment(
            description="MelJuSo HLA-DR1 b-GFP cells transfected with siRai14 or siRNA control, incubated with 70 kDa Dextran Alexa Fluor 555 (100 µg/ml) for 30 minutes. Macropinocytic index (MI = dextran-positive area / cell area × 100) and macropinosome area quantified using FIJI particle analysis. Rescue experiment: siRai14 cells re-transfected with Rai14-GFP.",
            model_system="MelJuSo HLA-DR1 b-GFP stable cell line",
            organism="Homo sapiens",
            organism_strain=None,
            perturbation_type="genetic_loss_of_function",
            perturbation_target="RAI14",
            perturbation_method="siRNA + rescue with Rai14-GFP re-expression",
            readout="Macropinocytic index; macropinosome area",
            control_description="siRNA negative control",
        ),
        results=Results(
            result_direction=ResultDirectionEnum.POSITIVE,
            effect_description="Macropinocytic index decreased by half in Rai14-silenced cells; macropinosome area decreased. Reintroduction of Rai14 rescued the macropinocytosis defect.",
            effect_size="~50% decrease in macropinocytic index; rescue confirmed specificity",
            statistical_test="two-tailed unpaired Student's t-test",
            p_value="*P < 0.05 (MI); **P < 0.01; *P < 0.05 (area)",
            confidence_interval=None,
            sample_size="n=3 independent experiments; n=60 cells per condition",
            key_figure="Figures 3B-D",
        ),
        methodological_tags=MethodologicalTags(
            approach_category="cell_biology",
            assay_types=["dextran_uptake_assay", "confocal_microscopy", "image_quantification"],
            blinding_reported=None,
            randomization_reported=None,
        ),
        limitations_stated_by_authors=[],
        source_section="results",
        source_text_span="the macropinocytic index...decreased by half in cells silenced for Rai14, and that this was a consequence of smaller macropinosome area (Figures 3B–D). The reintroduction of Rai14 in cells silenced for this protein rescued the macropinocytosis defect, further validating the specificity of Rai14 siRNA.",
    ),
    # e_011: PtdIns(4,5)P2 biosensor assay — macropinosome closure
    EvidenceUnit(
        evidence_id="e_011",
        assertion_draft_ids=["a_008"],
        evidence_direction=EvidenceDirectionEnum.SUPPORTS,
        evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
        experiment=Experiment(
            description="MelJuSo cells transfected with PH-PLCd-GFP (PtdIns(4,5)P2 biosensor, gift from Tamas Balla) and siRNA control or siRai14. Anti-HLA-DR L243 Alexa Fluor 647 added to medium. Time-lapse imaging on SoRa Spinning Disk microscope (UPLSAPO 60x/0.30 Silicon Oil, 30-second intervals). Percentage of cells retaining PtdIns(4,5)P2 at nascent macropinosome membrane quantified.",
            model_system="MelJuSo human melanoma cell line",
            organism="Homo sapiens",
            organism_strain=None,
            perturbation_type="genetic_loss_of_function",
            perturbation_target="RAI14",
            perturbation_method="siRNA",
            readout="Percentage of cells retaining PtdIns(4,5)P2 at macropinosome membrane during closure",
            control_description="siRNA negative control; PH-PLCd-GFP alone",
        ),
        results=Results(
            result_direction=ResultDirectionEnum.POSITIVE,
            effect_description="Percentage of cells retaining PtdIns(4,5)P2 at membrane of nascent macropinosomes dramatically increased in Rai14-silenced cells compared to control, indicating defective macropinosome closure",
            effect_size=None,
            statistical_test="two-tailed unpaired Student's t-test",
            p_value="**P < 0.01",
            confidence_interval=None,
            sample_size="n=3 independent experiments; n≥32 cells per condition",
            key_figure="Figures 4A-C; Videos 4, 5",
        ),
        methodological_tags=MethodologicalTags(
            approach_category="imaging",
            assay_types=["live_cell_imaging", "biosensor_assay", "spinning_disk_confocal"],
            blinding_reported=None,
            randomization_reported=None,
        ),
        limitations_stated_by_authors=[],
        source_section="results",
        source_text_span="in cells silenced for Rai14, the percentage of cells retaining PtdIns(4,5)P2 at the membrane of nascent macropinosomes dramatically increased (Figures 4B, C; Video 5), suggesting that Rai14 is required for macropinosome closure.",
    ),
    # e_012: Dextran macropinocytosis assay — BMDCs, open dish
    EvidenceUnit(
        evidence_id="e_012",
        assertion_draft_ids=["a_006"],
        evidence_direction=EvidenceDirectionEnum.SUPPORTS,
        evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
        experiment=Experiment(
            description="BMDCs from 8-16 week old C57BL/6 mice transfected with FlexiTube siRai14 (Qiagen cat. SI01396318) or control siRNA by nucleofection (Amaxa Nucleofector II, program Y-001). Cells incubated with 70 kDa Dextran Alexa Fluor 555 (100 µg/ml) for 15 minutes. Macropinocytic index and macropinosome area quantified from confocal images.",
            model_system="Bone marrow-derived dendritic cells (BMDCs), C57BL/6 mice",
            organism="Mus musculus",
            organism_strain="C57BL/6",
            perturbation_type="genetic_loss_of_function",
            perturbation_target="Rai14",
            perturbation_method="siRNA (FlexiTube Rai14 siRNA, Qiagen SI01396318)",
            readout="Macropinocytic index; macropinosome area",
            control_description="siRNA negative control (Qiagen sense 5'-UUCUCCGAACGUGUCACGUTT-3')",
        ),
        results=Results(
            result_direction=ResultDirectionEnum.POSITIVE,
            effect_description="30% decrease in macropinocytic index and ~30% decrease in macropinosome area in Rai14-silenced BMDCs",
            effect_size="~30% decrease in macropinocytic index; ~30% decrease in macropinosome area",
            statistical_test="two-tailed paired Student's t-test",
            p_value="*P < 0.05",
            confidence_interval=None,
            sample_size="n=3 independent experiments; n=60 cells per condition",
            key_figure="Figures 5A-C",
        ),
        methodological_tags=MethodologicalTags(
            approach_category="cell_biology",
            assay_types=["dextran_uptake_assay", "confocal_microscopy", "image_quantification"],
            blinding_reported=None,
            randomization_reported=None,
        ),
        limitations_stated_by_authors=[],
        source_section="results",
        source_text_span="In primary bone marrow-derived DCs (BMDCs) silenced for Rai14, there was a 30% decrease in the macropinocytic index (Figures 5A, B). Similarly, the macropinosome area decreased by approximately 30% in cells knocked down for Rai14 (Figure 5C).",
    ),
    # e_013: Macropinocytosis in BMDCs in microchannels
    EvidenceUnit(
        evidence_id="e_013",
        assertion_draft_ids=["a_006"],
        evidence_direction=EvidenceDirectionEnum.SUPPORTS,
        evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
        experiment=Experiment(
            description="BMDCs loaded into 5×8 µm micro-fabricated PDMS channels (4D Cell), treated with siRai14 or control siRNA. After 16 h, channels filled with 10 kDa Alexa Fluor 555-conjugated dextran (100 µg/ml) for 50 min. Cells imaged on SoRa Spinning Disk microscope every minute for 20 min. Percentage of cells containing macropinosomes quantified.",
            model_system="Bone marrow-derived dendritic cells (BMDCs), C57BL/6 mice, confined in microchannels",
            organism="Mus musculus",
            organism_strain="C57BL/6",
            perturbation_type="genetic_loss_of_function",
            perturbation_target="Rai14",
            perturbation_method="siRNA",
            readout="Percentage of cells with internalized dextran (macropinosomes) in microchannels",
            control_description="siRNA negative control",
        ),
        results=Results(
            result_direction=ResultDirectionEnum.POSITIVE,
            effect_description="Almost 40% decrease in the number of cells containing macropinosomes upon Rai14 depletion in microchannels",
            effect_size="~40% decrease",
            statistical_test="two-tailed paired Student's t-test",
            p_value="*P < 0.05",
            confidence_interval=None,
            sample_size="n=3 independent experiments; n≥40 cells per condition",
            key_figure="Figures 5D-E",
        ),
        methodological_tags=MethodologicalTags(
            approach_category="cell_biology",
            assay_types=["microchannel_assay", "dextran_uptake_assay", "spinning_disk_confocal"],
            blinding_reported=None,
            randomization_reported=None,
        ),
        limitations_stated_by_authors=[],
        source_section="results",
        source_text_span="Analysis of BMDCs in microchannels revealed an almost 40% decrease in the number of cells containing macropinosomes upon Rai14 depletion (Figures 5D, E).",
    ),
    # e_014: PAK phosphorylation assay — BMDCs siRai14
    EvidenceUnit(
        evidence_id="e_014",
        assertion_draft_ids=["a_009"],
        evidence_direction=EvidenceDirectionEnum.SUPPORTS,
        evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
        experiment=Experiment(
            description="BMDCs transfected with siRai14 or siRNA control. Cell lysates analyzed by Western blot using antibodies against Rai14, phosphorylated PAK1 (Ser144)/PAK2 (Ser141) (Cell Signaling Technology 2606S, 1:200), and total PAK1/2/3 (Cell Signaling Technology 2604S, 1:500). GAPDH (Chemicon MAB374) as loading control. Band intensity quantified with Image Lab (Bio-Rad).",
            model_system="Bone marrow-derived dendritic cells (BMDCs), C57BL/6 mice",
            organism="Mus musculus",
            organism_strain="C57BL/6",
            perturbation_type="genetic_loss_of_function",
            perturbation_target="Rai14",
            perturbation_method="siRNA",
            readout="Ratio of phospho-PAK to total PAK by Western blot densitometry",
            control_description="siRNA negative control",
        ),
        results=Results(
            result_direction=ResultDirectionEnum.POSITIVE,
            effect_description="PAK phosphorylation decreased by almost 50% in BMDCs silenced for Rai14 compared to control",
            effect_size="~50% decrease in phospho-PAK levels",
            statistical_test="two-tailed paired Student's t-test",
            p_value="*P < 0.05",
            confidence_interval=None,
            sample_size="n=4 independent experiments; n=5 for Rai14 level quantification",
            key_figure="Figures 5F-H",
        ),
        methodological_tags=MethodologicalTags(
            approach_category="biochemical_assay",
            assay_types=["Western_blot", "densitometry"],
            blinding_reported=None,
            randomization_reported=None,
        ),
        limitations_stated_by_authors=[],
        source_section="results",
        source_text_span="quantification of the levels of PAK phosphorylation revealed a decrease of almost 50% in BMDCs silenced for Rai14 compared to control cells (Figures 5F–H).",
    ),
    # e_015: BMDC migration speed in microchannels
    EvidenceUnit(
        evidence_id="e_015",
        assertion_draft_ids=["a_010"],
        evidence_direction=EvidenceDirectionEnum.SUPPORTS,
        evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
        experiment=Experiment(
            description="BMDCs transfected with siRNA control or siRai14, loaded into 5×8 µm micro-fabricated fibronectin-coated channels. Imaged for 20 h on Nikon TiE epiﬂuorescence video-microscope (10X objective, 1 image/min). Kymograph extraction and velocity analysis performed using ImageJ. Speed fluctuations calculated as SD/mean instantaneous speed.",
            model_system="Bone marrow-derived dendritic cells (BMDCs), C57BL/6 mice, confined in microchannels",
            organism="Mus musculus",
            organism_strain="C57BL/6",
            perturbation_type="genetic_loss_of_function",
            perturbation_target="Rai14",
            perturbation_method="siRNA",
            readout="Mean cell speed (µm/min); speed fluctuations (SD/mean instantaneous speed)",
            control_description="siRNA negative control",
        ),
        results=Results(
            result_direction=ResultDirectionEnum.POSITIVE,
            effect_description="BMDCs silenced for Rai14 moved faster (mean speed 7.1 µm/min) than control BMDCs (mean speed 5.4 µm/min), and showed fewer speed fluctuations (changed direction less frequently)",
            effect_size="7.1 µm/min vs 5.4 µm/min (Rai14 KD vs control)",
            statistical_test="two-tailed paired Student's t-test",
            p_value="**P<0.01 (speed); *P<0.01 (fluctuations)",
            confidence_interval=None,
            sample_size="n=4 independent experiments; n>150 cells per condition",
            key_figure="Figures 6A-C",
        ),
        methodological_tags=MethodologicalTags(
            approach_category="cell_biology",
            assay_types=["microchannel_migration_assay", "kymograph_analysis", "live_cell_imaging"],
            blinding_reported=None,
            randomization_reported=None,
        ),
        limitations_stated_by_authors=[],
        source_section="results",
        source_text_span="BMDCs silenced for Rai14 move faster (mean speed 7.1 µm/min) than control BMDCs (mean speed 5.4 µm/min) (Figures 6A, B). In addition, upon depletion of Rai14, BMDCs show fewer local speed variations compared to control cells, indicating that the cells knocked down for Rai14 change direction less frequently (Figures 6A–C).",
    ),
    # e_016: Myosin II density map — front localization in microchannels
    EvidenceUnit(
        evidence_id="e_016",
        assertion_draft_ids=["a_011"],
        evidence_direction=EvidenceDirectionEnum.SUPPORTS,
        evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
        experiment=Experiment(
            description="10^6 BMDCs loaded in 5×8 µm fibronectin-coated microchannels, allowed to migrate overnight, then fixed with 3% PFA and immunostained with anti-non-muscle myosin IIA (Abcam ab24762, 1:2000). Imaged on Olympus FluoView FV1000 confocal (60× PlanApo NA 1.35). Density maps generated in ImageJ (cells cropped to average size, background subtracted, normalized, projected). Front/back ratio quantified in rectangles corresponding to 20% front and 20% back of cell.",
            model_system="Bone marrow-derived dendritic cells (BMDCs), C57BL/6 mice, confined in microchannels",
            organism="Mus musculus",
            organism_strain="C57BL/6",
            perturbation_type="genetic_loss_of_function",
            perturbation_target="Rai14",
            perturbation_method="siRNA",
            readout="Myosin II fluorescence density map front-to-back ratio",
            control_description="siRNA negative control",
        ),
        results=Results(
            result_direction=ResultDirectionEnum.POSITIVE,
            effect_description="Rai14 silencing decreased the amount of myosin II at the front of BMDCs migrating in microfabricated channels, as revealed by density maps of mean myosin II distribution",
            effect_size=None,
            statistical_test="two-tailed unpaired Student's t-test",
            p_value="*P<0.05",
            confidence_interval=None,
            sample_size="n=3 independent experiments; n>40 cells per condition",
            key_figure="Supplementary Figure 3",
        ),
        methodological_tags=MethodologicalTags(
            approach_category="imaging",
            assay_types=["immunofluorescence", "confocal_microscopy", "density_map_analysis"],
            blinding_reported=None,
            randomization_reported=None,
        ),
        limitations_stated_by_authors=[],
        source_section="results",
        source_text_span="Silencing of Rai14 indeed decreased the amount of myosin II at the front of BMDCs migrating into microfabricated channels, as revealed by density maps of the mean myosin II distribution (Supplementary Figure 3).",
    ),
    # e_017: Rai14-Ii colocalization in BMDCs by immunofluorescence
    EvidenceUnit(
        evidence_id="e_017",
        assertion_draft_ids=["a_003"],
        evidence_direction=EvidenceDirectionEnum.SUPPORTS,
        evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
        experiment=Experiment(
            description="BMDCs fixed and immunostained with antibodies against Rai14 (Abcam ab137118, IF 1:40) and Ii (anti-mouse CD74 In-1, BD Biosciences 555317, IF 1:50). Nuclei stained with DAPI. Confocal imaging on Olympus FluoView FV1000 IX81 (60× PlanApo NA 1.35). Normalized fluorescence intensity profiles along line scan quantified using ImageJ.",
            model_system="Bone marrow-derived dendritic cells (BMDCs), C57BL/6 mice",
            organism="Mus musculus",
            organism_strain="C57BL/6",
            perturbation_type="none",
            perturbation_target=None,
            perturbation_method=None,
            readout="Colocalization by fluorescence intensity profile; Rai14 and Ii on macropinosome-like vesicles",
            control_description="No secondary antibody negative control (standard protocol)",
        ),
        results=Results(
            result_direction=ResultDirectionEnum.POSITIVE,
            effect_description="Rai14 and Ii colocalized on large macropinosome-like vesicles in BMDCs",
            effect_size=None,
            statistical_test="Fluorescence intensity profile analysis",
            p_value=None,
            confidence_interval=None,
            sample_size="n=2 independent experiments; n=78 vesicles from 37 cells",
            key_figure="Figures 6D, E",
        ),
        methodological_tags=MethodologicalTags(
            approach_category="imaging",
            assay_types=["immunofluorescence", "confocal_microscopy", "line_scan_analysis"],
            blinding_reported=None,
            randomization_reported=None,
        ),
        limitations_stated_by_authors=[],
        source_section="results",
        source_text_span="Our results show that, in line with our hypothesis, the two proteins colocalize on large, macropinosome-like vesicles (Figures 6D, E).",
    ),
    # e_018: Co-IP Rai14-myosin II in DC lysates
    EvidenceUnit(
        evidence_id="e_018",
        assertion_draft_ids=["a_012"],
        evidence_direction=EvidenceDirectionEnum.SUPPORTS,
        evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
        experiment=Experiment(
            description="DC lysates subjected to co-immunoprecipitation with anti-Rai14 antibody (Abcam ab137118) or IgG isotype control. Total lysate and immunoprecipitates analyzed by Western blot with antibodies against myosin II (Abcam ab24762) and Rai14.",
            model_system="Primary dendritic cells",
            organism="Mus musculus",
            organism_strain="C57BL/6",
            perturbation_type="none",
            perturbation_target=None,
            perturbation_method=None,
            readout="Co-immunoprecipitation of myosin II with Rai14 by Western blot",
            control_description="IgG isotype control",
        ),
        results=Results(
            result_direction=ResultDirectionEnum.POSITIVE,
            effect_description="Rai14 co-immunoprecipitated myosin II from DC lysates but not in IgG control, confirming Rai14-myosin II physical interaction",
            effect_size=None,
            statistical_test=None,
            p_value=None,
            confidence_interval=None,
            sample_size=None,
            key_figure="Figure 6F",
        ),
        methodological_tags=MethodologicalTags(
            approach_category="biochemical_assay",
            assay_types=["co-immunoprecipitation", "Western_blot"],
            blinding_reported=None,
            randomization_reported=None,
        ),
        limitations_stated_by_authors=["Co-IP does not distinguish direct from indirect binding"],
        source_section="results",
        source_text_span="Co-immunoprecipitation experiments revealed that Rai14 is indeed able to bind myosin II (Figure 6F).",
    ),
    # e_019: GST-pulldown — GST-myosin II heavy chain tail pulls down Rai14 and Ii; fails when Rai14 depleted
    EvidenceUnit(
        evidence_id="e_019",
        assertion_draft_ids=["a_012", "a_013"],
        evidence_direction=EvidenceDirectionEnum.SUPPORTS,
        evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
        experiment=Experiment(
            description="GST-tagged myosin II heavy chain tail (amino acids 1795-1960, from pGEX2T-myosin II heavy chain tail) expressed in E. coli BL21(DE3), purified using Glutathione Sepharose 4B (Cytiva). 10 µg purified GST or GST-myosin II heavy chain tail incubated with 200 µl lysates from MelJuSo cells transfected with siRNA control or siRai14. Affinity chromatography, then Western blot for GST, Rai14, and Ii (antibodies as listed). Coomassie Blue staining for input verification.",
            model_system="MelJuSo human melanoma cell line (in vitro GST-pulldown)",
            organism="Homo sapiens",
            organism_strain=None,
            perturbation_type="genetic_loss_of_function",
            perturbation_target="RAI14",
            perturbation_method="siRNA (siRai14 condition vs control)",
            readout="Pulldown of Rai14 and Ii by GST-myosin II heavy chain tail from cell lysates",
            control_description="GST alone; siRNA control lysate",
        ),
        results=Results(
            result_direction=ResultDirectionEnum.POSITIVE,
            effect_description="Purified GST-myosin II heavy chain tail specifically precipitates both Ii and Rai14 from total MelJuSo cell extracts. Critically, GST-myosin II tail is unable to pull down Ii from cells silenced for Rai14, indicating that Rai14 is required to bridge Ii to myosin II.",
            effect_size=None,
            statistical_test=None,
            p_value=None,
            confidence_interval=None,
            sample_size=None,
            key_figure="Figure 6G",
        ),
        methodological_tags=MethodologicalTags(
            approach_category="biochemical_assay",
            assay_types=["GST_pulldown", "SDS-PAGE", "Western_blot", "affinity_chromatography"],
            blinding_reported=None,
            randomization_reported=None,
        ),
        limitations_stated_by_authors=[],
        source_section="results",
        source_text_span="while purified GST-tagged myosin II heavy chain tail specifically precipitates both Ii and Rai14 from total MelJuSo cell extracts, it is unable to pull down Ii from cells silenced for Rai14 (Figure 6G), suggesting that Rai14 bridges Ii to myosin II.",
    ),
]

# ---------------------------------------------------------------------------
# ASSERTION DRAFTS (all is_primary=True, no background)
# ---------------------------------------------------------------------------
assertion_drafts = [
    # a_001: Rai14 physically interacts with Ii (novel protein-protein interaction)
    AssertionDraft(
        draft_id="a_001",
        natural_language="Rai14 physically interacts with Invariant chain (Ii/CD74) as demonstrated by co-immunoprecipitation in human cells and confirmed by yeast two-hybrid.",
        canonical_form="RAI14 — physically_interacts_with — CD74/Ii",
        negatable_form="RAI14 does NOT physically interact with CD74/Ii",
        subject_entity=RAI14,
        object_entity=II_CD74,
        predicate="physically_interacts_with",
        direction=DirectionEnum.POSITIVE,
        assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
        causal_type=None,
        scope=human_meljuso_scope(),
        conditions=[],
        hedging=Hedging(
            verbatim_hedge="confirm the interaction",
            certainty=HedgeLevelEnum.HIGH,
            generalizability=HedgeLevelEnum.MEDIUM,
            causality_hedge=CausalityHedgeEnum.CORRELATIONAL,
        ),
        epistemic_status=EpistemicStatus(
            section=SectionEnum.RESULTS,
            function=FunctionEnum.NOVEL_FINDING,
            is_primary=True,
            cited_source=None,
        ),
        evidence_unit_ids=["e_001", "e_002", "e_003"],
        parent_assertion_ids=[],
        provenance=Provenance(
            source_sentence="Taken together, these results confirm the interaction between Rai14 and Ii identified in the yeast two-hybrid screen.",
            section_name="Results — Rai14 is a novel interactor of Invariant chain",
            char_offset_start=None,
            char_offset_end=None,
        ),
    ),
    # a_002: Rai14 localizes to membrane ruffles and nascent macropinosomes
    AssertionDraft(
        draft_id="a_002",
        natural_language="Rai14 localizes to membrane ruffles and nascent macropinosomes in antigen-presenting cells.",
        canonical_form="RAI14 — localizes_to — membrane ruffles and nascent macropinosomes",
        negatable_form="RAI14 does NOT localize to membrane ruffles or nascent macropinosomes",
        subject_entity=RAI14,
        object_entity=MEMBRANE_RUFFLES,
        predicate="localizes_to",
        direction=DirectionEnum.POSITIVE,
        assertion_type=AssertionTypeEnum.EXISTENCE,
        causal_type=None,
        scope=dual_scope(),
        conditions=[],
        hedging=Hedging(
            verbatim_hedge="Rai14-GFP was mainly localized at the plasma membrane and membrane ruffles",
            certainty=HedgeLevelEnum.HIGH,
            generalizability=HedgeLevelEnum.MEDIUM,
            causality_hedge=CausalityHedgeEnum.CORRELATIONAL,
        ),
        epistemic_status=EpistemicStatus(
            section=SectionEnum.RESULTS,
            function=FunctionEnum.NOVEL_FINDING,
            is_primary=True,
            cited_source=None,
        ),
        evidence_unit_ids=["e_004", "e_009"],
        parent_assertion_ids=[],
        provenance=Provenance(
            source_sentence="We show that Rai14 localizes to membrane ruffles and nascent macropinosomes.",
            section_name="Results — Rai14 is a novel interactor of Invariant chain",
            char_offset_start=None,
            char_offset_end=None,
        ),
    ),
    # a_003: Rai14 and Ii colocalize on membrane ruffles and macropinosome-like vesicles
    AssertionDraft(
        draft_id="a_003",
        natural_language="Rai14 and Ii colocalize on membrane ruffles, nascent vesicles, and macropinosome-like structures in antigen-presenting cells.",
        canonical_form="RAI14 — colocalizes_with — CD74/Ii [at membrane ruffles and macropinosomes]",
        negatable_form="RAI14 does NOT colocalize with CD74/Ii at membrane ruffles or macropinosomes",
        subject_entity=RAI14,
        object_entity=II_CD74,
        predicate="colocalizes_with",
        direction=DirectionEnum.POSITIVE,
        assertion_type=AssertionTypeEnum.CORRELATIONAL,
        causal_type=None,
        scope=dual_scope(),
        conditions=[],
        hedging=Hedging(
            verbatim_hedge="we detected Ii together with Rai14 on membrane ruffles, nascent vesicles, and forming macropinosomes",
            certainty=HedgeLevelEnum.HIGH,
            generalizability=HedgeLevelEnum.MEDIUM,
            causality_hedge=CausalityHedgeEnum.CORRELATIONAL,
        ),
        epistemic_status=EpistemicStatus(
            section=SectionEnum.RESULTS,
            function=FunctionEnum.NOVEL_FINDING,
            is_primary=True,
            cited_source=None,
        ),
        evidence_unit_ids=["e_004", "e_017"],
        parent_assertion_ids=["a_001"],
        provenance=Provenance(
            source_sentence="we detected Ii together with Rai14 on membrane ruffles, nascent vesicles, and forming macropinosomes.",
            section_name="Results — Rai14 is a novel interactor of Invariant chain",
            char_offset_start=None,
            char_offset_end=None,
        ),
    ),
    # a_004: Rai14 exists in a complex with MHC II and Ii
    AssertionDraft(
        draft_id="a_004",
        natural_language="Rai14 is present in a protein complex containing both MHC II and Ii in human antigen-presenting cells.",
        canonical_form="RAI14 — co-complex_with — MHC II–Ii complex",
        negatable_form="RAI14 is NOT present in a complex with MHC II and Ii",
        subject_entity=RAI14,
        object_entity=Entity(
            surface_form="MHC II-Ii complex",
            canonical_name="MHC II-Ii complex",
            ontology_id=None,
            ontology_source=None,
            entity_type=EntityTypeEnum.OTHER,
            aliases=["HLA-DR-CD74 complex"],
        ),
        predicate="co-complex_with",
        direction=DirectionEnum.POSITIVE,
        assertion_type=AssertionTypeEnum.EXISTENCE,
        causal_type=None,
        scope=human_meljuso_scope(),
        conditions=[],
        hedging=Hedging(
            verbatim_hedge="confirming that Rai14 interacts with Ii in a complex with MHC II",
            certainty=HedgeLevelEnum.HIGH,
            generalizability=HedgeLevelEnum.MEDIUM,
            causality_hedge=CausalityHedgeEnum.CORRELATIONAL,
        ),
        epistemic_status=EpistemicStatus(
            section=SectionEnum.RESULTS,
            function=FunctionEnum.NOVEL_FINDING,
            is_primary=True,
            cited_source=None,
        ),
        evidence_unit_ids=["e_005"],
        parent_assertion_ids=["a_001"],
        provenance=Provenance(
            source_sentence="Cells expressing HLA-DR1 b-GFP (but not cells expressing GFP) were able to co-immunoprecipitate both endogenous Ii and Rai14 (Figure 2A), confirming that Rai14 interacts with Ii in a complex with MHC II.",
            section_name="Results — Rai14 depletion retains MHC II at the plasma membrane",
            char_offset_start=None,
            char_offset_end=None,
        ),
    ),
    # a_005: Rai14 is required for MHC II internalization
    AssertionDraft(
        draft_id="a_005",
        natural_language="Rai14 is required for efficient internalization of MHC II from the plasma membrane; Rai14 depletion increases surface MHC II and reduces endosomal MHC II uptake.",
        canonical_form="RAI14 — is_required_for — MHC II internalization",
        negatable_form="RAI14 is NOT required for MHC II internalization; Rai14 depletion does not affect surface MHC II levels or internalization rate",
        subject_entity=RAI14,
        object_entity=Entity(
            surface_form="MHC II internalization",
            canonical_name="MHC class II endocytosis",
            ontology_id="GO:0032586",
            ontology_source="GO",
            entity_type=EntityTypeEnum.OTHER,
            aliases=["MHC II uptake", "HLA-DR internalization"],
        ),
        predicate="is_required_for",
        direction=DirectionEnum.POSITIVE,
        assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
        causal_type=CausalTypeEnum.NECESSARY,
        scope=human_meljuso_scope(),
        conditions=[],
        hedging=Hedging(
            verbatim_hedge="these results indicate that Rai14 is required for MHC II internalization",
            certainty=HedgeLevelEnum.HIGH,
            generalizability=HedgeLevelEnum.MEDIUM,
            causality_hedge=CausalityHedgeEnum.CAUSAL,
        ),
        epistemic_status=EpistemicStatus(
            section=SectionEnum.RESULTS,
            function=FunctionEnum.NOVEL_FINDING,
            is_primary=True,
            cited_source=None,
        ),
        evidence_unit_ids=["e_006", "e_007", "e_008"],
        parent_assertion_ids=[],
        provenance=Provenance(
            source_sentence="Altogether, these results indicate that Rai14 is required for MHC II internalization.",
            section_name="Results — Rai14 depletion retains MHC II at the plasma membrane",
            char_offset_start=None,
            char_offset_end=None,
        ),
    ),
    # a_006: Rai14 is required for macropinocytosis (positive regulator)
    AssertionDraft(
        draft_id="a_006",
        natural_language="Rai14 is a positive regulator of macropinocytosis; its depletion reduces macropinocytic index and macropinosome size in both human melanoma cells and murine dendritic cells.",
        canonical_form="RAI14 — positively_regulates — macropinocytosis",
        negatable_form="RAI14 does NOT positively regulate macropinocytosis; its depletion does not reduce macropinocytic activity",
        subject_entity=RAI14,
        object_entity=MACROPINOCYTOSIS,
        predicate="positively_regulates",
        direction=DirectionEnum.POSITIVE,
        assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
        causal_type=CausalTypeEnum.CONTRIBUTORY,
        scope=dual_scope(),
        conditions=[],
        hedging=Hedging(
            verbatim_hedge="these results indicate that Rai14 is required for macropinocytosis",
            certainty=HedgeLevelEnum.HIGH,
            generalizability=HedgeLevelEnum.MEDIUM,
            causality_hedge=CausalityHedgeEnum.CAUSAL,
        ),
        epistemic_status=EpistemicStatus(
            section=SectionEnum.RESULTS,
            function=FunctionEnum.NOVEL_FINDING,
            is_primary=True,
            cited_source=None,
        ),
        evidence_unit_ids=["e_010", "e_012", "e_013"],
        parent_assertion_ids=[],
        provenance=Provenance(
            source_sentence="Altogether, these results indicate that Rai14 is required for macropinocytosis.",
            section_name="Results — Silencing of Rai14 inhibits macropinocytosis",
            char_offset_start=None,
            char_offset_end=None,
        ),
    ),
    # a_007: Rai14 localization at membrane ruffles drives MHC II macropinocytic uptake
    AssertionDraft(
        draft_id="a_007",
        natural_language="MHC II is internalized via macropinocytosis from Rai14-positive membrane ruffles that close into macropinosomes.",
        canonical_form="RAI14-positive membrane ruffles — are_entry_sites_for — MHC II macropinocytic uptake",
        negatable_form="MHC II is NOT internalized from Rai14-positive membrane ruffles via macropinocytosis",
        subject_entity=Entity(
            surface_form="Rai14-positive membrane ruffles",
            canonical_name="membrane ruffle",
            ontology_id="GO:0001726",
            ontology_source="GO",
            entity_type=EntityTypeEnum.OTHER,
            aliases=["Rai14-MHC II positive ruffles"],
        ),
        object_entity=Entity(
            surface_form="MHC II macropinocytic uptake",
            canonical_name="macropinocytic MHC II internalization",
            ontology_id=None,
            ontology_source=None,
            entity_type=EntityTypeEnum.OTHER,
            aliases=["MHC II macropinocytosis"],
        ),
        predicate="are_sites_of",
        direction=DirectionEnum.POSITIVE,
        assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
        causal_type=CausalTypeEnum.CONTRIBUTORY,
        scope=human_meljuso_scope(),
        conditions=[],
        hedging=Hedging(
            verbatim_hedge="Rai14-GFP and MHC II colocalized on membrane ruffles, and these ruffles lead to the formation of macropinosomes",
            certainty=HedgeLevelEnum.HIGH,
            generalizability=HedgeLevelEnum.MEDIUM,
            causality_hedge=CausalityHedgeEnum.UNCLEAR,
        ),
        epistemic_status=EpistemicStatus(
            section=SectionEnum.RESULTS,
            function=FunctionEnum.NOVEL_FINDING,
            is_primary=True,
            cited_source=None,
        ),
        evidence_unit_ids=["e_009", "e_010"],
        parent_assertion_ids=["a_002", "a_006"],
        provenance=Provenance(
            source_sentence="Rai14-GFP and MHC II colocalized on membrane ruffles, and these ruffles lead to the formation of macropinosomes (Figure 3A; Video 3).",
            section_name="Results — Silencing of Rai14 inhibits macropinocytosis",
            char_offset_start=None,
            char_offset_end=None,
        ),
    ),
    # a_008: Rai14 is required for macropinosome closure (PtdIns(4,5)P2 depletion)
    AssertionDraft(
        draft_id="a_008",
        natural_language="Rai14 is required for macropinosome closure; Rai14 depletion causes PtdIns(4,5)P2 to persist abnormally at the membrane of nascent macropinosomes.",
        canonical_form="RAI14 — is_required_for — macropinosome closure [PtdIns(4,5)P2 depletion]",
        negatable_form="RAI14 is NOT required for macropinosome closure; its depletion does not affect PtdIns(4,5)P2 dynamics at macropinosome membrane",
        subject_entity=RAI14,
        object_entity=MACROPINOSOME_CLOSURE,
        predicate="is_required_for",
        direction=DirectionEnum.POSITIVE,
        assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
        causal_type=CausalTypeEnum.NECESSARY,
        scope=human_meljuso_scope(),
        conditions=[
            Condition(
                condition_type=ConditionTypeEnum.BIOLOGICAL_CONTEXT,
                value="macropinosome formation from membrane ruffles",
            )
        ],
        hedging=Hedging(
            verbatim_hedge="suggesting that Rai14 is required for macropinosome closure",
            certainty=HedgeLevelEnum.MEDIUM,
            generalizability=HedgeLevelEnum.MEDIUM,
            causality_hedge=CausalityHedgeEnum.CAUSAL,
        ),
        epistemic_status=EpistemicStatus(
            section=SectionEnum.RESULTS,
            function=FunctionEnum.NOVEL_FINDING,
            is_primary=True,
            cited_source=None,
        ),
        evidence_unit_ids=["e_011"],
        parent_assertion_ids=["a_006"],
        provenance=Provenance(
            source_sentence="in cells silenced for Rai14, the percentage of cells retaining PtdIns(4,5)P2 at the membrane of nascent macropinosomes dramatically increased (Figures 4B, C; Video 5), suggesting that Rai14 is required for macropinosome closure.",
            section_name="Results — Silencing of Rai14 inhibits macropinocytosis",
            char_offset_start=None,
            char_offset_end=None,
        ),
    ),
    # a_009: Rai14 is required for PAK phosphorylation / activation
    AssertionDraft(
        draft_id="a_009",
        natural_language="Rai14 is required for PAK activation (phosphorylation) in dendritic cells; Rai14 depletion reduces PAK phosphorylation by ~50%.",
        canonical_form="RAI14 — positively_regulates — PAK phosphorylation [in BMDCs]",
        negatable_form="RAI14 does NOT regulate PAK phosphorylation; its depletion does not reduce PAK activation",
        subject_entity=RAI14,
        object_entity=PAK,
        predicate="positively_regulates_phosphorylation_of",
        direction=DirectionEnum.POSITIVE,
        assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
        causal_type=CausalTypeEnum.CONTRIBUTORY,
        scope=mouse_bmdc_scope(),
        conditions=[],
        hedging=Hedging(
            verbatim_hedge="quantification of the levels of PAK phosphorylation revealed a decrease of almost 50% in BMDCs silenced for Rai14",
            certainty=HedgeLevelEnum.HIGH,
            generalizability=HedgeLevelEnum.MEDIUM,
            causality_hedge=CausalityHedgeEnum.CAUSAL,
        ),
        epistemic_status=EpistemicStatus(
            section=SectionEnum.RESULTS,
            function=FunctionEnum.NOVEL_FINDING,
            is_primary=True,
            cited_source=None,
        ),
        evidence_unit_ids=["e_014"],
        parent_assertion_ids=["a_006"],
        provenance=Provenance(
            source_sentence="quantification of the levels of PAK phosphorylation revealed a decrease of almost 50% in BMDCs silenced for Rai14 compared to control cells (Figures 5F–H).",
            section_name="Results — Silencing of Rai14 inhibits macropinocytosis",
            char_offset_start=None,
            char_offset_end=None,
        ),
    ),
    # a_010: Rai14 negatively regulates dendritic cell migration speed
    AssertionDraft(
        draft_id="a_010",
        natural_language="Rai14 negatively regulates dendritic cell migration speed; Rai14-depleted BMDCs migrate faster and exhibit fewer directional changes than wild-type cells.",
        canonical_form="RAI14 — negatively_regulates — BMDC migration speed",
        negatable_form="RAI14 does NOT negatively regulate BMDC migration speed; its depletion does not increase cell speed",
        subject_entity=RAI14,
        object_entity=CELL_MIGRATION,
        predicate="negatively_regulates",
        direction=DirectionEnum.NEGATIVE,
        assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
        causal_type=CausalTypeEnum.CONTRIBUTORY,
        scope=mouse_bmdc_scope(),
        conditions=[
            Condition(
                condition_type=ConditionTypeEnum.BIOLOGICAL_CONTEXT,
                value="confined in 5×8 µm microfabricated channels",
            )
        ],
        hedging=Hedging(
            verbatim_hedge="BMDCs silenced for Rai14 move faster (mean speed 7.1 µm/min) than control BMDCs (mean speed 5.4 µm/min)",
            certainty=HedgeLevelEnum.HIGH,
            generalizability=HedgeLevelEnum.MEDIUM,
            causality_hedge=CausalityHedgeEnum.CAUSAL,
        ),
        epistemic_status=EpistemicStatus(
            section=SectionEnum.RESULTS,
            function=FunctionEnum.NOVEL_FINDING,
            is_primary=True,
            cited_source=None,
        ),
        evidence_unit_ids=["e_015"],
        parent_assertion_ids=[],
        provenance=Provenance(
            source_sentence="BMDCs silenced for Rai14 move faster (mean speed 7.1 µm/min) than control BMDCs (mean speed 5.4 µm/min) (Figures 6A, B). In addition, upon depletion of Rai14, BMDCs show fewer local speed variations compared to control cells, indicating that the cells knocked down for Rai14 change direction less frequently (Figures 6A–C).",
            section_name="Results — Rai14 negatively regulates BMDC migration",
            char_offset_start=None,
            char_offset_end=None,
        ),
    ),
    # a_011: Rai14 depletion displaces myosin II from the leading edge of migrating DCs
    AssertionDraft(
        draft_id="a_011",
        natural_language="Rai14 depletion reduces myosin II recruitment to the leading edge of migrating BMDCs in confined environments.",
        canonical_form="RAI14 — is_required_for — myosin II recruitment to cell front [in migrating BMDCs]",
        negatable_form="RAI14 depletion does NOT affect myosin II localization at the leading edge of migrating BMDCs",
        subject_entity=RAI14,
        object_entity=Entity(
            surface_form="myosin II at cell front",
            canonical_name="MYH9 front localization",
            ontology_id=None,
            ontology_source=None,
            entity_type=EntityTypeEnum.OTHER,
            aliases=["myosin II leading edge recruitment"],
        ),
        predicate="is_required_for",
        direction=DirectionEnum.POSITIVE,
        assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
        causal_type=CausalTypeEnum.CONTRIBUTORY,
        scope=mouse_bmdc_scope(),
        conditions=[
            Condition(
                condition_type=ConditionTypeEnum.BIOLOGICAL_CONTEXT,
                value="migrating in 5×8 µm microfabricated channels",
            )
        ],
        hedging=Hedging(
            verbatim_hedge="Silencing of Rai14 indeed decreased the amount of myosin II at the front of BMDCs migrating into microfabricated channels",
            certainty=HedgeLevelEnum.HIGH,
            generalizability=HedgeLevelEnum.MEDIUM,
            causality_hedge=CausalityHedgeEnum.CAUSAL,
        ),
        epistemic_status=EpistemicStatus(
            section=SectionEnum.RESULTS,
            function=FunctionEnum.NOVEL_FINDING,
            is_primary=True,
            cited_source=None,
        ),
        evidence_unit_ids=["e_016"],
        parent_assertion_ids=["a_010"],
        provenance=Provenance(
            source_sentence="Silencing of Rai14 indeed decreased the amount of myosin II at the front of BMDCs migrating into microfabricated channels, as revealed by density maps of the mean myosin II distribution (Supplementary Figure 3).",
            section_name="Results — Rai14 negatively regulates BMDC migration",
            char_offset_start=None,
            char_offset_end=None,
        ),
    ),
    # a_012: Rai14 physically interacts with myosin II
    AssertionDraft(
        draft_id="a_012",
        natural_language="Rai14 physically interacts with myosin II (non-muscle myosin IIA) as demonstrated by co-immunoprecipitation and GST-pulldown.",
        canonical_form="RAI14 — physically_interacts_with — myosin II (non-muscle myosin IIA/MYH9)",
        negatable_form="RAI14 does NOT physically interact with myosin II (non-muscle myosin IIA)",
        subject_entity=RAI14,
        object_entity=MYOSIN_II,
        predicate="physically_interacts_with",
        direction=DirectionEnum.POSITIVE,
        assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
        causal_type=None,
        scope=dual_scope(),
        conditions=[],
        hedging=Hedging(
            verbatim_hedge="Co-immunoprecipitation experiments revealed that Rai14 is indeed able to bind myosin II",
            certainty=HedgeLevelEnum.HIGH,
            generalizability=HedgeLevelEnum.MEDIUM,
            causality_hedge=CausalityHedgeEnum.CORRELATIONAL,
        ),
        epistemic_status=EpistemicStatus(
            section=SectionEnum.RESULTS,
            function=FunctionEnum.NOVEL_FINDING,
            is_primary=True,
            cited_source=None,
        ),
        evidence_unit_ids=["e_018", "e_019"],
        parent_assertion_ids=[],
        provenance=Provenance(
            source_sentence="Co-immunoprecipitation experiments revealed that Rai14 is indeed able to bind myosin II (Figure 6F).",
            section_name="Results — Rai14 negatively regulates BMDC migration",
            char_offset_start=None,
            char_offset_end=None,
        ),
    ),
    # a_013: Rai14 bridges Ii to myosin II (scaffolding function)
    AssertionDraft(
        draft_id="a_013",
        natural_language="Rai14 acts as a molecular bridge between Invariant chain (Ii) and myosin II; its depletion abolishes the Ii–myosin II interaction.",
        canonical_form="RAI14 — bridges — CD74/Ii to myosin II [enabling Ii-myosin II interaction]",
        negatable_form="RAI14 does NOT bridge Ii to myosin II; its depletion does not affect the Ii-myosin II interaction",
        subject_entity=RAI14,
        object_entity=Entity(
            surface_form="Ii-myosin II interaction",
            canonical_name="CD74-MYH9 interaction",
            ontology_id=None,
            ontology_source=None,
            entity_type=EntityTypeEnum.OTHER,
            aliases=["Ii-myosin II binding", "CD74-myosin IIA complex"],
        ),
        predicate="is_required_for",
        direction=DirectionEnum.POSITIVE,
        assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
        causal_type=CausalTypeEnum.NECESSARY,
        scope=human_meljuso_scope(),
        conditions=[],
        hedging=Hedging(
            verbatim_hedge="suggesting that Rai14 bridges Ii to myosin II",
            certainty=HedgeLevelEnum.MEDIUM,
            generalizability=HedgeLevelEnum.MEDIUM,
            causality_hedge=CausalityHedgeEnum.CAUSAL,
        ),
        epistemic_status=EpistemicStatus(
            section=SectionEnum.RESULTS,
            function=FunctionEnum.NOVEL_FINDING,
            is_primary=True,
            cited_source=None,
        ),
        evidence_unit_ids=["e_019"],
        parent_assertion_ids=["a_001", "a_012"],
        provenance=Provenance(
            source_sentence="while purified GST-tagged myosin II heavy chain tail specifically precipitates both Ii and Rai14 from total MelJuSo cell extracts, it is unable to pull down Ii from cells silenced for Rai14 (Figure 6G), suggesting that Rai14 bridges Ii to myosin II.",
            section_name="Results — Rai14 negatively regulates BMDC migration",
            char_offset_start=None,
            char_offset_end=None,
        ),
    ),
    # a_014: Rai14 is a positive regulator of macropinocytosis and negative regulator of migration (functional antagonism)
    AssertionDraft(
        draft_id="a_014",
        natural_language="Rai14 coordinates the antagonistic relationship between macropinocytosis and cell migration in antigen-presenting cells, acting as a positive regulator of macropinocytosis and a negative regulator of migration.",
        canonical_form="RAI14 — coordinates_antagonism_between — macropinocytosis and cell migration [in APCs]",
        negatable_form="RAI14 does NOT coordinate the antagonism between macropinocytosis and cell migration in APCs",
        subject_entity=RAI14,
        object_entity=Entity(
            surface_form="macropinocytosis-migration antagonism",
            canonical_name="macropinocytosis-migration antagonism",
            ontology_id=None,
            ontology_source=None,
            entity_type=EntityTypeEnum.OTHER,
            aliases=["antigen uptake vs migration tradeoff"],
        ),
        predicate="coordinates",
        direction=DirectionEnum.POSITIVE,
        assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
        causal_type=CausalTypeEnum.CONTRIBUTORY,
        scope=mouse_bmdc_scope(),
        conditions=[],
        hedging=Hedging(
            verbatim_hedge="Rai14 is a positive regulator of macropinocytosis and a negative regulator of cell migration, two antagonistic processes in antigen-presenting cells",
            certainty=HedgeLevelEnum.MEDIUM,
            generalizability=HedgeLevelEnum.MEDIUM,
            causality_hedge=CausalityHedgeEnum.CAUSAL,
        ),
        epistemic_status=EpistemicStatus(
            section=SectionEnum.ABSTRACT,
            function=FunctionEnum.INTERPRETATION,
            is_primary=True,
            cited_source=None,
        ),
        evidence_unit_ids=["e_010", "e_012", "e_013", "e_015"],
        parent_assertion_ids=["a_006", "a_010"],
        provenance=Provenance(
            source_sentence="we demonstrated that, similar to Ii, Rai14 is a positive regulator of macropinocytosis and a negative regulator of cell migration, two antagonistic processes in antigen-presenting cells.",
            section_name="Abstract",
            char_offset_start=None,
            char_offset_end=None,
        ),
    ),
    # a_015: Hypothesis — Rai14 as scaffold links Ii to actin/myosin network
    AssertionDraft(
        draft_id="a_015",
        natural_language="Rai14 may act as a scaffold protein linking Invariant chain (Ii) to the actomyosin network, thereby coordinating macropinocytosis, intracellular membrane traffic, and cell migration in antigen-presenting cells.",
        canonical_form="RAI14 — scaffold_linking — CD74/Ii to actomyosin network",
        negatable_form="RAI14 does NOT act as a scaffold linking Ii to the actomyosin network",
        subject_entity=RAI14,
        object_entity=Entity(
            surface_form="actomyosin network",
            canonical_name="actomyosin cytoskeleton",
            ontology_id="GO:0031941",
            ontology_source="GO",
            entity_type=EntityTypeEnum.OTHER,
            aliases=["actin-myosin II network", "cortical actomyosin"],
        ),
        predicate="scaffolds_onto",
        direction=DirectionEnum.POSITIVE,
        assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
        causal_type=CausalTypeEnum.CONTRIBUTORY,
        scope=dual_scope(),
        conditions=[],
        hedging=Hedging(
            verbatim_hedge="Rai14 is likely a scaffold protein that links Ii with myosin II and the actomyosin network",
            certainty=HedgeLevelEnum.LOW,
            generalizability=HedgeLevelEnum.MEDIUM,
            causality_hedge=CausalityHedgeEnum.UNCLEAR,
        ),
        epistemic_status=EpistemicStatus(
            section=SectionEnum.DISCUSSION,
            function=FunctionEnum.HYPOTHESIS,
            is_primary=True,
            cited_source=None,
        ),
        evidence_unit_ids=["e_019"],
        parent_assertion_ids=["a_013"],
        provenance=Provenance(
            source_sentence="Rai14 is likely a scaffold protein that links Ii with myosin II and the actomyosin network.",
            section_name="Discussion",
            char_offset_start=None,
            char_offset_end=None,
        ),
    ),
    # a_016: Hypothesis — Rai14 may link membrane shaping to actin in macropinocytosis
    AssertionDraft(
        draft_id="a_016",
        natural_language="Rai14 may act as a link between plasma membrane curvature/shaping and the cortical actin cytoskeleton during macropinocytosis.",
        canonical_form="RAI14 — links — membrane curvature/shaping to cortical actin [during macropinocytosis]",
        negatable_form="RAI14 does NOT link membrane curvature to the actin cytoskeleton during macropinocytosis",
        subject_entity=RAI14,
        object_entity=Entity(
            surface_form="membrane curvature and cortical actin link",
            canonical_name="membrane-actin interface",
            ontology_id=None,
            ontology_source=None,
            entity_type=EntityTypeEnum.OTHER,
            aliases=["membrane-cytoskeleton coupling"],
        ),
        predicate="links",
        direction=DirectionEnum.POSITIVE,
        assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
        causal_type=CausalTypeEnum.CONTRIBUTORY,
        scope=dual_scope(),
        conditions=[
            Condition(condition_type=ConditionTypeEnum.BIOLOGICAL_CONTEXT, value="macropinocytosis")
        ],
        hedging=Hedging(
            verbatim_hedge="it is therefore possible that Rai14 acts as a link between membrane shaping and actin in macropinocytosis",
            certainty=HedgeLevelEnum.LOW,
            generalizability=HedgeLevelEnum.MEDIUM,
            causality_hedge=CausalityHedgeEnum.UNCLEAR,
        ),
        epistemic_status=EpistemicStatus(
            section=SectionEnum.DISCUSSION,
            function=FunctionEnum.HYPOTHESIS,
            is_primary=True,
            cited_source=None,
        ),
        evidence_unit_ids=["e_011"],
        parent_assertion_ids=["a_008"],
        provenance=Provenance(
            source_sentence="it is therefore possible that Rai14 acts as a link between membrane shaping and actin in macropinocytosis.",
            section_name="Discussion",
            char_offset_start=None,
            char_offset_end=None,
        ),
    ),
    # a_017: Limitation — study uses overexpressed Rai14-GFP, not endogenous tagging
    AssertionDraft(
        draft_id="a_017",
        natural_language="Localization studies used overexpressed Rai14-GFP rather than endogenously tagged protein, which may not fully recapitulate native Rai14 distribution.",
        canonical_form="RAI14-GFP overexpression — may_not_reflect — endogenous RAI14 localization",
        negatable_form="RAI14-GFP overexpression faithfully recapitulates endogenous RAI14 localization",
        subject_entity=Entity(
            surface_form="Rai14-GFP overexpression",
            canonical_name="RAI14-GFP transgene",
            ontology_id=None,
            ontology_source=None,
            entity_type=EntityTypeEnum.OTHER,
            aliases=["pCMV6-AC-Rai14-GFP"],
        ),
        object_entity=Entity(
            surface_form="endogenous RAI14 localization",
            canonical_name="RAI14 subcellular localization",
            ontology_id=None,
            ontology_source=None,
            entity_type=EntityTypeEnum.OTHER,
            aliases=[],
        ),
        predicate="may_not_recapitulate",
        direction=DirectionEnum.NEGATIVE,
        assertion_type=AssertionTypeEnum.METHODOLOGICAL,
        causal_type=None,
        scope=human_meljuso_scope(),
        conditions=[],
        hedging=Hedging(
            verbatim_hedge=None,
            certainty=HedgeLevelEnum.LOW,
            generalizability=HedgeLevelEnum.LOW,
            causality_hedge=CausalityHedgeEnum.UNCLEAR,
        ),
        epistemic_status=EpistemicStatus(
            section=SectionEnum.METHODS,
            function=FunctionEnum.LIMITATION,
            is_primary=True,
            cited_source=None,
        ),
        evidence_unit_ids=[],
        parent_assertion_ids=["a_002"],
        provenance=Provenance(
            source_sentence="pCMV6-AC-Rai14-GFP was purchased from OriGene Technologies, Inc.",
            section_name="Materials and Methods — Antibodies, constructs, and reagents",
            char_offset_start=None,
            char_offset_end=None,
        ),
    ),
]

# ---------------------------------------------------------------------------
# CITATION CONTEXTS
# ---------------------------------------------------------------------------
citation_contexts = [
    # c_001: ref [7] — Faure-Andre et al. 2008 — Ii regulates DC migration via CD74
    CitationContext(
        citation_id="c_001",
        citing_sentence="Furthermore, by interacting with the actin motor myosin II, Ii regulates the macropinocytic and migratory ability of DCs in an antagonistic manner (7, 8).",
        cited_source_doi="10.1126/science.1159894",
        cited_source_pmid="19008446",
        cited_source_ref_key="(7)",
        cited_claim_paraphrase="CD74/Ii regulates dendritic cell migration; DCs lacking Ii display faster migration with fewer speed fluctuations than wild-type cells.",
        relationship=CitationRelationshipEnum.CONTEXTUALIZES,
        linked_assertion_draft_ids=["a_010", "a_014"],
        section=SectionEnum.INTRODUCTION,
    ),
    # c_002: ref [8] — Chabaud et al. 2015 — macropinocytosis-migration antagonism via myosin II
    CitationContext(
        citation_id="c_002",
        citing_sentence="The antagonism between fast cell migration and antigen uptake depends on the localization of myosin II mediated by Ii (8). When myosin II is recruited to the cell front by Ii, the macropinocytic activity increases, reducing the migration speed. On the other hand, myosin II localization at the cell rear is necessary for fast migration (7, 8).",
        cited_source_doi="10.1038/ncomms8526",
        cited_source_pmid="26130004",
        cited_source_ref_key="(8)",
        cited_claim_paraphrase="Cell migration and antigen capture (macropinocytosis) are antagonistic processes coupled by myosin II in dendritic cells. Ii recruits myosin II to the cell front to promote macropinocytosis and reduce migration; myosin II at the cell rear drives fast migration.",
        relationship=CitationRelationshipEnum.CONTEXTUALIZES,
        linked_assertion_draft_ids=["a_010", "a_011", "a_013", "a_014"],
        section=SectionEnum.INTRODUCTION,
    ),
    # c_003: ref [9] — Wolf et al. 2019 — N-Ank protein family membrane shaping
    CitationContext(
        citation_id="c_003",
        citing_sentence="Rai14 is a good example of an N-Ank protein as it binds to membranes via both hydrophobic and electrostatic interactions. Indeed, it has been shown that Rai14 has a direct role in shaping membranes (9).",
        cited_source_doi="10.1038/s41556-019-0381-7",
        cited_source_pmid="31481792",
        cited_source_ref_key="(9)",
        cited_claim_paraphrase="Ankyrin repeat-containing N-Ank proteins shape cellular membranes; they contain ankyrin repeats and an N-terminal amphipathic helix that senses membrane curvature and modulates membrane topology via hydrophobic/electrostatic interactions. Rai14 directly participates in membrane shaping.",
        relationship=CitationRelationshipEnum.CONTEXTUALIZES,
        linked_assertion_draft_ids=["a_016"],
        section=SectionEnum.INTRODUCTION,
    ),
    # c_004: refs [10, 11] — Peng/Mandai 2000; Qian et al. 2013a — Rai14 actin association
    CitationContext(
        citation_id="c_004",
        citing_sentence="The reported intracellular localization of Rai14 suggests that it also associates with the cortical actin cytoskeleton, F-actin stress fibers, and cell-cell adhesions sites (10–12).",
        cited_source_doi="10.1046/j.1365-2443.2000.00381.x",
        cited_source_pmid="11168582",
        cited_source_ref_key="(10-12)",
        cited_claim_paraphrase="Rai14 (ankycorbin) associates with the cortical actin cytoskeleton, F-actin stress fibers, and cell-cell adhesion sites; it is a cytoskeleton-associated protein linked to actin function and organization.",
        relationship=CitationRelationshipEnum.EXTENDS,
        linked_assertion_draft_ids=["a_002", "a_015"],
        section=SectionEnum.INTRODUCTION,
    ),
    # c_005: ref [2] — Landsverk et al. 2011 — Ii delays endosomal maturation
    CitationContext(
        citation_id="c_005",
        citing_sentence="Ii mediates endosome fusion and delays endosome maturation (2, 6).",
        cited_source_doi="10.1038/icb.2010.143",
        cited_source_pmid="21042330",
        cited_source_ref_key="(2)",
        cited_claim_paraphrase="Invariant chain increases the half-life of MHC II by delaying endosomal maturation.",
        relationship=CitationRelationshipEnum.CONTEXTUALIZES,
        linked_assertion_draft_ids=[],
        section=SectionEnum.INTRODUCTION,
    ),
    # c_006: ref [3] — Neefjes et al. 2011 Nat Rev Immunol — Ii chaperoning MHC II
    CitationContext(
        citation_id="c_006",
        citing_sentence="Ii interacts with the peptide-binding groove of MHC II to prevent the premature binding of endogenous peptides and chaperones new MHC II molecules to endosomes for the loading of antigenic peptides (2, 3).",
        cited_source_doi="10.1038/nri3084",
        cited_source_pmid="22076556",
        cited_source_ref_key="(3)",
        cited_claim_paraphrase="Towards a systems understanding of MHC class I and MHC class II antigen presentation; Ii prevents premature peptide binding and chaperones MHC II to endosomes.",
        relationship=CitationRelationshipEnum.CONTEXTUALIZES,
        linked_assertion_draft_ids=[],
        section=SectionEnum.INTRODUCTION,
    ),
    # c_007: ref [24] — Eby et al. 1998 / ref [25] — Dharmawardhane 2000 — PAK activates actin remodeling and macropinocytosis
    CitationContext(
        citation_id="c_007",
        citing_sentence="PAK is a serine/threonine kinase that, upon activation, is auto-phosphorylated, promoting actin cytoskeleton remodeling and macropinocytosis (24, 25).",
        cited_source_doi="10.1091/mbc.11.10.3341",
        cited_source_pmid="11029042",
        cited_source_ref_key="(24, 25)",
        cited_claim_paraphrase="PAK1 (p21-activated kinase) regulates macropinocytosis; its activation through auto-phosphorylation promotes actin cytoskeleton remodeling and macropinosome formation.",
        relationship=CitationRelationshipEnum.CONTEXTUALIZES,
        linked_assertion_draft_ids=["a_009"],
        section=SectionEnum.RESULTS,
    ),
    # c_008: ref [34] — Delorme-Walker et al. 2011 — PAK1 regulates myosin IIA distribution
    CitationContext(
        citation_id="c_008",
        citing_sentence="In addition, Rai14 silencing inhibits PAK activation, which, in agreement with previous reports, also contributes to myosin II displacement from the cell's leading edge (34).",
        cited_source_doi="10.1083/jcb.201010059",
        cited_source_pmid="21670215",
        cited_source_ref_key="(34)",
        cited_claim_paraphrase="PAK1 regulates focal adhesion strength, myosin IIA distribution, and actin dynamics to optimize cell migration; PAK1 activity controls myosin IIA localization at the cell front.",
        relationship=CitationRelationshipEnum.SUPPORTS,
        linked_assertion_draft_ids=["a_009", "a_011"],
        section=SectionEnum.DISCUSSION,
    ),
    # c_009: ref [36] — Kitamata et al. 2019 — ANKHD1 (another N-Ank protein) in early endosome fission
    CitationContext(
        citation_id="c_009",
        citing_sentence="Interestingly, another member of the N-Ank protein family, ANKHD1, induces fission on early endosomes (36). As we also observed Rai14 at the sites of vesicle scission from membrane tubules or enriched on endosomal domains (Figures 1C, D), it is tempting to speculate that Rai14 could also be involved in membrane fission.",
        cited_source_doi="10.1016/j.isci.2019.06.020",
        cited_source_pmid="31325784",
        cited_source_ref_key="(36)",
        cited_claim_paraphrase="ANKHD1, an N-Ank family protein, induces fission on early endosomes; this N-Ank protein shapes membranes during endosomal fission.",
        relationship=CitationRelationshipEnum.CONTEXTUALIZES,
        linked_assertion_draft_ids=["a_002"],
        section=SectionEnum.DISCUSSION,
    ),
    # c_010: ref [18–20] — PtdIns(4,5)P2 depletion at macropinosome formation
    CitationContext(
        citation_id="c_010",
        citing_sentence="Macropinosome formation involves loss of PtdIns(4,5)P2, which is present at the plasma membrane, from the macropinosome membrane as it internalized (18–20).",
        cited_source_doi="10.1242/jcs.252411",
        cited_source_pmid="34374776",
        cited_source_ref_key="(18-20)",
        cited_claim_paraphrase="PtdIns(4,5)P2 is depleted from the macropinosome membrane during macropinosome formation/closure; this PtdIns(4,5)P2 loss is a hallmark of macropinosome sealing.",
        relationship=CitationRelationshipEnum.CONTEXTUALIZES,
        linked_assertion_draft_ids=["a_008"],
        section=SectionEnum.RESULTS,
    ),
    # c_011: ref [6] — Margiotta et al. 2020 — Ii regulates endosomal fusion via SNARE Vti1b
    CitationContext(
        citation_id="c_011",
        citing_sentence="Invariant chain regulates endosomal fusion and maturation through an interaction with the SNARE Vti1b (6).",
        cited_source_doi="10.1242/jcs.244624",
        cited_source_pmid="32546540",
        cited_source_ref_key="(6)",
        cited_claim_paraphrase="Invariant chain regulates endosomal fusion and maturation through an interaction with the SNARE protein Vti1b.",
        relationship=CitationRelationshipEnum.CONTEXTUALIZES,
        linked_assertion_draft_ids=[],
        section=SectionEnum.INTRODUCTION,
    ),
    # c_012: refs [10, 11, 13, 14] — Rai14 is cytoskeleton-associated protein
    CitationContext(
        citation_id="c_012",
        citing_sentence="Furthermore, it has been proposed that Rai14 is a cytoskeleton-associated protein linked to actin function and organization (10, 11, 13, 14).",
        cited_source_doi="10.1371/journal.pone.0060656",
        cited_source_pmid="23560100",
        cited_source_ref_key="(10, 11, 13, 14)",
        cited_claim_paraphrase="Rai14 is a cytoskeleton-associated protein linked to actin function and organization, involved in regulating F-actin dynamics; it associates with the ectoplasmic specialization in the rat testis.",
        relationship=CitationRelationshipEnum.EXTENDS,
        linked_assertion_draft_ids=["a_015"],
        section=SectionEnum.INTRODUCTION,
    ),
]

# ---------------------------------------------------------------------------
# PAPER PROVENANCE
# ---------------------------------------------------------------------------
paper_provenance = PaperProvenance(
    doi="10.3389/fimmu.2023.1182180",
    pmid=None,
    title="Rai14 is a novel interactor of Invariant chain that regulates macropinocytosis",
    authors=[
        Author(
            name="Lobos Patorniti, Natacha",
            orcid=None,
            affiliations=["Department of Biosciences, University of Oslo, Oslo, Norway"],
            role="first_author",
        ),
        Author(
            name="Zulkefli, Khalisah Liyana",
            orcid=None,
            affiliations=["Department of Biosciences, University of Oslo, Oslo, Norway"],
            role="co_author",
        ),
        Author(
            name="McAdam, Martin E.",
            orcid=None,
            affiliations=[
                "Department of Biosciences, University of Oslo, Oslo, Norway",
                "Pharmaq part of Zoetis, Oslo, Norway",
            ],
            role="co_author",
        ),
        Author(
            name="Vargas, Pablo",
            orcid=None,
            affiliations=["Inserm U1151, Institut Necker Enfants Malades, Paris, France"],
            role="co_author",
        ),
        Author(
            name="Bakke, Oddmund",
            orcid=None,
            affiliations=["Department of Biosciences, University of Oslo, Oslo, Norway"],
            role="co_author",
        ),
        Author(
            name="Progida, Cinzia",
            orcid=None,
            affiliations=["Department of Biosciences, University of Oslo, Oslo, Norway"],
            role="senior_author",
        ),
    ],
    journal="Frontiers in Immunology",
    publication_date="2023-07-21",
    peer_reviewed=True,
    preprint_doi=None,
    funding_sources=[
        "Norwegian Research Council grant 287560 (to CP)",
        "Anders Jahre Foundation",
        "Den Grevelige Hjelmstjerne-Rosencroneske Foundation",
    ],
    conflicts_of_interest="The authors declare that the research was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest.",
    data_availability="The original contributions presented in the study are included in the article/Supplementary Material. Further inquiries can be directed to the corresponding author.",
)

# ---------------------------------------------------------------------------
# EXTRACTION METADATA
# ---------------------------------------------------------------------------
extraction_metadata = ExtractionMetadata(
    extraction_model="claude-sonnet-4-6",
    extraction_version="0.2.0",
    extraction_timestamp="2026-03-24T00:00:00Z",
    paper_char_count=None,
    extraction_duration_seconds=None,
)

# ---------------------------------------------------------------------------
# ASSEMBLE ExtractionResult
# ---------------------------------------------------------------------------
result = ExtractionResult(
    paper_provenance=paper_provenance,
    evidence_units=evidence_units,
    assertion_drafts=assertion_drafts,
    citation_contexts=citation_contexts,
    extraction_metadata=extraction_metadata,
)

# ---------------------------------------------------------------------------
# VALIDATE (Pydantic v2 — model_validate re-parses from JSON to confirm round-trip)
# ---------------------------------------------------------------------------
raw_json = result.model_dump_json(indent=2)
validated = ExtractionResult.model_validate_json(raw_json)

# ---------------------------------------------------------------------------
# WRITE OUTPUT
# ---------------------------------------------------------------------------
output_path = pathlib.Path(__file__).parent / "extraction_test_sonnet_v2.json"
output_path.write_text(raw_json, encoding="utf-8")
print(f"Wrote {output_path}")

# ---------------------------------------------------------------------------
# SUMMARY REPORT
# ---------------------------------------------------------------------------
print("\n=== EXTRACTION SUMMARY ===")
print(f"Assertion drafts total:     {len(validated.assertion_drafts)}")
is_primary_count = sum(1 for a in validated.assertion_drafts if a.epistemic_status.is_primary)
non_primary_count = len(validated.assertion_drafts) - is_primary_count
print(f"  is_primary=True:          {is_primary_count}")
print(f"  is_primary=False:         {non_primary_count}  (SHOULD BE 0 for v2)")

background_count = sum(
    1 for a in validated.assertion_drafts if a.epistemic_status.function == FunctionEnum.BACKGROUND
)
print(f"  function='background':    {background_count}  (SHOULD BE 0 for v2)")

print(f"\nEvidence units total:       {len(validated.evidence_units)}")
print(f"Citation contexts total:    {len(validated.citation_contexts)}")

# Relationship distribution
from collections import Counter

rel_dist = Counter(str(c.relationship) for c in validated.citation_contexts)
print("\nCitation context relationship distribution:")
for rel, count in sorted(rel_dist.items(), key=lambda x: -x[1]):
    print(f"  {rel:25s}: {count}")

# Function distribution
func_dist = Counter(str(a.epistemic_status.function) for a in validated.assertion_drafts)
print("\nAssertion draft function distribution:")
for func, count in sorted(func_dist.items(), key=lambda x: -x[1]):
    print(f"  {func:30s}: {count}")

print("\n=== VALIDATION PASSED ===")
