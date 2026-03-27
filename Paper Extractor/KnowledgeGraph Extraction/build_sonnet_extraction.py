"""Build the Rai14-Ii paper extraction result programmatically.

Paper: Lobos Patorniti et al. (2023) "Rai14 is a novel interactor of Invariant chain
that regulates macropinocytosis." Frontiers in Immunology.
DOI: 10.3389/fimmu.2023.1182180

Extraction model: claude-sonnet-4-6
"""

import sys

sys.path.insert(0, "/Users/mst36/Desktop/Projects/Science/Autonomous Science/src")

from mycelium.extraction_schema import (
    AssertionDraft,
    AssertionTypeEnum,
    Author,
    CausalityHedgeEnum,
    CausalTypeEnum,
    Condition,
    ConditionTypeEnum,
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
# Paper Provenance
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
            affiliations=["Department of Biosciences, University of Oslo, Oslo, Norway"],
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
        "Norwegian Research Council grant 287560",
        "Anders Jahre Foundation",
        "Den Grevelige Hjelmstjerne-Rosencroneske Foundation",
    ],
    conflicts_of_interest="The authors declare that the research was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest.",
    data_availability="The original contributions presented in the study are included in the article/Supplementary Material.",
)

# ---------------------------------------------------------------------------
# Shared scope helpers
# ---------------------------------------------------------------------------

human_species = OntologyTerm(
    term_id="NCBITaxon:9606",
    term_name="Homo sapiens",
    ontology="NCBI Taxonomy",
    surface_form="human",
)
mouse_species = OntologyTerm(
    term_id="NCBITaxon:10090",
    term_name="Mus musculus",
    ontology="NCBI Taxonomy",
    surface_form="mouse",
)
meljuso_ct = OntologyTerm(
    term_id="CL:0001087", term_name="melanoma cell", ontology="CL", surface_form="MelJuSo cells"
)
dc_ct = OntologyTerm(
    term_id="CL:0000451",
    term_name="dendritic cell",
    ontology="CL",
    surface_form="bone marrow-derived dendritic cells (BMDCs)",
)
melanoma_disease = OntologyTerm(
    term_id="MONDO:0005105", term_name="melanoma", ontology="MONDO", surface_form="melanoma"
)

rai14_entity = Entity(
    surface_form="Rai14",
    canonical_name="RAI14",
    ontology_id="UniProt:Q9UHD9",
    ontology_source="UniProt",
    entity_type=EntityTypeEnum.PROTEIN,
    aliases=["retinoic acid-induced 14", "NORPEG", "ankycorbin"],
)

ii_entity = Entity(
    surface_form="Ii",
    canonical_name="CD74",
    ontology_id="UniProt:P04233",
    ontology_source="UniProt",
    entity_type=EntityTypeEnum.PROTEIN,
    aliases=["Invariant chain", "CD74", "HLA class II histocompatibility antigen gamma chain"],
)

myosin2_entity = Entity(
    surface_form="myosin II",
    canonical_name="MYH9",
    ontology_id="UniProt:P35579",
    ontology_source="UniProt",
    entity_type=EntityTypeEnum.PROTEIN,
    aliases=["non-muscle myosin IIA", "NMIIA", "myosin heavy chain 9"],
)

mhcii_entity = Entity(
    surface_form="MHC II",
    canonical_name="HLA-DR",
    ontology_id="UniProt:P01903",
    ontology_source="UniProt",
    entity_type=EntityTypeEnum.PROTEIN,
    aliases=["HLA-DR", "HLA-DR1", "Major Histocompatibility Complex Class II"],
)

macropinocytosis_entity = Entity(
    surface_form="macropinocytosis",
    canonical_name="macropinocytosis",
    ontology_id="GO:0044351",
    ontology_source="GO",
    entity_type=EntityTypeEnum.OTHER,
    aliases=["macropinosome formation", "macropinocytic activity"],
)

migration_entity = Entity(
    surface_form="cell migration",
    canonical_name="cell migration",
    ontology_id="GO:0016477",
    ontology_source="GO",
    entity_type=EntityTypeEnum.OTHER,
    aliases=["cell motility", "DC migration"],
)

membrane_ruffle_entity = Entity(
    surface_form="membrane ruffles",
    canonical_name="membrane ruffle",
    ontology_id="GO:0001726",
    ontology_source="GO",
    entity_type=EntityTypeEnum.OTHER,
    aliases=["membrane ruffles", "lamellipodia"],
)

pak_entity = Entity(
    surface_form="PAK",
    canonical_name="PAK1",
    ontology_id="UniProt:Q13153",
    ontology_source="UniProt",
    entity_type=EntityTypeEnum.PROTEIN,
    aliases=["p21-activated kinase", "PAK1", "PAK2", "PAK3"],
)

pip2_entity = Entity(
    surface_form="PtdIns(4,5)P2",
    canonical_name="phosphatidylinositol 4,5-bisphosphate",
    ontology_id="CHEBI:18348",
    ontology_source="ChEBI",
    entity_type=EntityTypeEnum.CHEMICAL,
    aliases=["PI(4,5)P2", "PIP2"],
)

# ---------------------------------------------------------------------------
# Evidence Units
# ---------------------------------------------------------------------------

evidence_units = []

# e_001: Y2H screen - Rai14 identified as Ii interactor
e_001 = EvidenceUnit(
    evidence_id="e_001",
    assertion_draft_ids=["a_001"],
    evidence_direction=EvidenceDirectionEnum.SUPPORTS,
    evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
    experiment=Experiment(
        description="Yeast two-hybrid screen using full-length human Ii p33 as bait against a human placenta cDNA library (performed by Hybrigenics Services, Paris, France). Rai14 was identified as a positive hit.",
        model_system="Yeast two-hybrid system (Saccharomyces cerevisiae)",
        organism="Homo sapiens",
        organism_strain=None,
        perturbation_type="none",
        perturbation_target=None,
        perturbation_method=None,
        readout="Yeast two-hybrid reporter activation (positive hit identification)",
        control_description="Negative hits from the screen; Supplementary Table 1 lists all hits",
    ),
    results=Results(
        result_direction=ResultDirectionEnum.POSITIVE,
        effect_description="Rai14 identified as one of the strongest positive candidates in Y2H screen using Ii p33 as bait against human placenta cDNA library",
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
        "Y2H may detect indirect interactions or false positives; confirmed by co-IP"
    ],
    source_section="Results",
    source_text_span="Rai14 is a novel interactor of Invariant chain section",
)
evidence_units.append(e_001)

# e_002: Co-IP Ii pulls down Rai14 (Figure 1A)
e_002 = EvidenceUnit(
    evidence_id="e_002",
    assertion_draft_ids=["a_001"],
    evidence_direction=EvidenceDirectionEnum.SUPPORTS,
    evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
    experiment=Experiment(
        description="Co-immunoprecipitation from MelJuSo cell lysates using anti-Ii antibody (MB741, BD Biosciences cat. 555538) or isotype control (anti-IgG2aK, BD Biosciences cat. 555571). Immunoprecipitates and whole cell lysates analyzed by Western blot with anti-Ii and anti-Rai14 (ab137118, Abcam) antibodies.",
        model_system="MelJuSo human melanoma cell line",
        organism="Homo sapiens",
        organism_strain=None,
        perturbation_type="none",
        perturbation_target=None,
        perturbation_method=None,
        readout="Co-immunoprecipitation band by Western blot (Rai14 detection in Ii IP)",
        control_description="Mouse IgG2a kappa isotype control immunoprecipitation",
    ),
    results=Results(
        result_direction=ResultDirectionEnum.POSITIVE,
        effect_description="Rai14 was specifically pulled down by endogenous Ii but not by IgG2ak isotype control, confirming Rai14-Ii interaction",
        effect_size=None,
        statistical_test=None,
        p_value=None,
        confidence_interval=None,
        sample_size=None,
        key_figure="Figure 1A",
    ),
    methodological_tags=MethodologicalTags(
        approach_category="biochemical_assay",
        assay_types=["co-immunoprecipitation", "western_blot"],
        blinding_reported=None,
        randomization_reported=None,
    ),
    limitations_stated_by_authors=["Co-IP does not distinguish direct from indirect interaction"],
    source_section="Results",
    source_text_span="Figure 1A",
)
evidence_units.append(e_002)

# e_003: Reverse Co-IP Rai14 pulls down Ii (Figure 1B)
e_003 = EvidenceUnit(
    evidence_id="e_003",
    assertion_draft_ids=["a_001"],
    evidence_direction=EvidenceDirectionEnum.SUPPORTS,
    evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
    experiment=Experiment(
        description="Reverse co-immunoprecipitation from MelJuSo cell lysates using anti-Rai14 antibody (ab137118, Abcam, IP 1:200) or IgG isotype control (BD Biosciences cat. 550875). Immunoprecipitates analyzed by Western blot for Ii and Rai14.",
        model_system="MelJuSo human melanoma cell line",
        organism="Homo sapiens",
        organism_strain=None,
        perturbation_type="none",
        perturbation_target=None,
        perturbation_method=None,
        readout="Co-immunoprecipitation band by Western blot (Ii detection in Rai14 IP)",
        control_description="IgG isotype control immunoprecipitation",
    ),
    results=Results(
        result_direction=ResultDirectionEnum.POSITIVE,
        effect_description="Endogenous Ii was co-immunoprecipitated by Rai14 antibody but not by IgG isotype control, confirming bidirectional interaction",
        effect_size=None,
        statistical_test=None,
        p_value=None,
        confidence_interval=None,
        sample_size=None,
        key_figure="Figure 1B",
    ),
    methodological_tags=MethodologicalTags(
        approach_category="biochemical_assay",
        assay_types=["co-immunoprecipitation", "western_blot"],
        blinding_reported=None,
        randomization_reported=None,
    ),
    limitations_stated_by_authors=[],
    source_section="Results",
    source_text_span="Figure 1B",
)
evidence_units.append(e_003)

# e_004: Live-cell imaging Rai14-GFP + Ii co-localization on membrane ruffles (Figure 1C,D)
e_004 = EvidenceUnit(
    evidence_id="e_004",
    assertion_draft_ids=["a_002", "a_003"],
    evidence_direction=EvidenceDirectionEnum.SUPPORTS,
    evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
    experiment=Experiment(
        description="Live cell imaging of MelJuSo cells co-transfected with Ii p33 and Rai14-GFP. Alexa Fluor 555-conjugated anti-Ii antibody (M-B741) added before imaging. Mander's colocalization coefficients quantified using ImageJ.",
        model_system="MelJuSo human melanoma cell line",
        organism="Homo sapiens",
        organism_strain=None,
        perturbation_type="protein_overexpression",
        perturbation_target="RAI14, CD74",
        perturbation_method="Transient transfection with Rai14-GFP and Ii p33 constructs using Lipofectamine 2000",
        readout="Colocalization of Rai14-GFP and Ii-Alexa555 on membrane ruffles and nascent vesicles by live confocal imaging; Mander's colocalization coefficient",
        control_description="Cells without Rai14-GFP overexpression",
    ),
    results=Results(
        result_direction=ResultDirectionEnum.POSITIVE,
        effect_description="Rai14-GFP and Ii colocalized on membrane ruffles, nascent vesicles, and forming macropinosomes; Rai14 localized to plasma membrane and membrane ruffles; Rai14 and Ii colocalized at early stages of endocytosis until an Ii-positive vesicle pinched off",
        effect_size=None,
        statistical_test="Mander's colocalization coefficient",
        p_value=None,
        confidence_interval=None,
        sample_size=None,
        key_figure="Figures 1C, D; Video 2",
    ),
    methodological_tags=MethodologicalTags(
        approach_category="imaging",
        assay_types=["live_cell_imaging", "confocal_microscopy", "colocalization_analysis"],
        blinding_reported=None,
        randomization_reported=None,
    ),
    limitations_stated_by_authors=[
        "Colocalization does not prove direct interaction; protein overexpression may not reflect endogenous levels"
    ],
    source_section="Results",
    source_text_span="Figures 1C, D; Video 2",
)
evidence_units.append(e_004)

# e_005: Rai14 on Ii-positive endosomes at steady state (Figure 1E)
e_005 = EvidenceUnit(
    evidence_id="e_005",
    assertion_draft_ids=["a_003"],
    evidence_direction=EvidenceDirectionEnum.SUPPORTS,
    evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
    experiment=Experiment(
        description="MelJuSo cells co-transfected with Ii p33, mCherry-Ii p33, and Rai14-GFP. Steady-state imaging to assess Rai14 localization on Ii-positive endosomes.",
        model_system="MelJuSo human melanoma cell line",
        organism="Homo sapiens",
        organism_strain=None,
        perturbation_type="protein_overexpression",
        perturbation_target="RAI14, CD74",
        perturbation_method="Transient transfection",
        readout="Fluorescence intensity profile of Rai14-GFP on mCherry-Ii-positive endosomes",
        control_description=None,
    ),
    results=Results(
        result_direction=ResultDirectionEnum.POSITIVE,
        effect_description="At steady state, Rai14-GFP detected in domains on Ii-positive endosomes; normalized fluorescence intensity profiles showed colocalization on endosomal membranes",
        effect_size=None,
        statistical_test=None,
        p_value=None,
        confidence_interval=None,
        sample_size=None,
        key_figure="Figure 1E",
    ),
    methodological_tags=MethodologicalTags(
        approach_category="imaging",
        assay_types=["confocal_microscopy", "fluorescence_intensity_profiling"],
        blinding_reported=None,
        randomization_reported=None,
    ),
    limitations_stated_by_authors=["Overexpression system"],
    source_section="Results",
    source_text_span="Figure 1E",
)
evidence_units.append(e_005)

# e_006: GFP-TRAP MHC II co-IP of Rai14 (Figure 2A)
e_006 = EvidenceUnit(
    evidence_id="e_006",
    assertion_draft_ids=["a_004"],
    evidence_direction=EvidenceDirectionEnum.SUPPORTS,
    evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
    experiment=Experiment(
        description="GFP-TRAP immunoprecipitation from lysates of MelJuSo cells stably expressing HLA-DR1 beta-GFP or GFP alone (control). Immunoprecipitates analyzed by Western blot for Ii and Rai14.",
        model_system="MelJuSo cells stably expressing HLA-DR1 beta-GFP",
        organism="Homo sapiens",
        organism_strain=None,
        perturbation_type="protein_overexpression",
        perturbation_target="HLA-DR1",
        perturbation_method="Stable transfection with HLA-DR1 beta-GFP",
        readout="Co-immunoprecipitation of endogenous Ii and Rai14 with HLA-DR1 beta-GFP by Western blot",
        control_description="MelJuSo cells expressing GFP alone subjected to GFP-TRAP pulldown",
    ),
    results=Results(
        result_direction=ResultDirectionEnum.POSITIVE,
        effect_description="Cells expressing HLA-DR1 beta-GFP (but not GFP alone) co-immunoprecipitated both endogenous Ii and Rai14, confirming Rai14 is present in a complex with Ii and MHC II",
        effect_size=None,
        statistical_test=None,
        p_value=None,
        confidence_interval=None,
        sample_size=None,
        key_figure="Figure 2A",
    ),
    methodological_tags=MethodologicalTags(
        approach_category="biochemical_assay",
        assay_types=["GFP_trap_immunoprecipitation", "western_blot"],
        blinding_reported=None,
        randomization_reported=None,
    ),
    limitations_stated_by_authors=[],
    source_section="Results",
    source_text_span="Figure 2A",
)
evidence_units.append(e_006)

# e_007: Rai14 KD increases MHC II at plasma membrane (Figures 2B-E)
e_007 = EvidenceUnit(
    evidence_id="e_007",
    assertion_draft_ids=["a_005", "a_006"],
    evidence_direction=EvidenceDirectionEnum.SUPPORTS,
    evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
    experiment=Experiment(
        description="MelJuSo cells stably expressing HLA-DR1 beta-GFP transfected with siRNA against Rai14 or control siRNA. Confocal imaging to assess HLA-DR1 beta-GFP distribution between plasma membrane and endosomes. Quantification of plasma membrane percentage of total GFP signal using FIJI.",
        model_system="HLA-DR1 beta-GFP MelJuSo cells",
        organism="Homo sapiens",
        organism_strain=None,
        perturbation_type="genetic_loss_of_function",
        perturbation_target="RAI14",
        perturbation_method="siRNA transfection (Lipofectamine RNAiMAX, 72 hours)",
        readout="Percentage of HLA-DR1 beta-GFP at plasma membrane vs total (confocal imaging + FIJI quantification)",
        control_description="Control siRNA transfection",
    ),
    results=Results(
        result_direction=ResultDirectionEnum.POSITIVE,
        effect_description="HLA-DR1 beta-GFP presence at plasma membrane increased 1.5-fold in Rai14 knockdown cells compared to control",
        effect_size="1.5-fold increase",
        statistical_test="two-tailed unpaired Student's t-test",
        p_value="P < 0.01",
        confidence_interval=None,
        sample_size="three independent experiments",
        key_figure="Figures 2B-E",
    ),
    methodological_tags=MethodologicalTags(
        approach_category="cell_biology",
        assay_types=["siRNA_knockdown", "confocal_microscopy", "image_quantification"],
        blinding_reported=None,
        randomization_reported=None,
    ),
    limitations_stated_by_authors=[],
    source_section="Results",
    source_text_span="Figures 2B-E",
)
evidence_units.append(e_007)

# e_008: Flow cytometry surface MHC II and Ii increased upon Rai14 KD (Figure 2F)
e_008 = EvidenceUnit(
    evidence_id="e_008",
    assertion_draft_ids=["a_005", "a_006"],
    evidence_direction=EvidenceDirectionEnum.SUPPORTS,
    evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
    experiment=Experiment(
        description="Flow cytometry (BD Fortessa) analysis of surface and total levels of endogenous HLA-DR and Ii in MelJuSo cells transfected with siRai14 vs. control siRNA. Surface staining in intact cells; total staining in saponin-permeabilized cells.",
        model_system="MelJuSo human melanoma cell line",
        organism="Homo sapiens",
        organism_strain=None,
        perturbation_type="genetic_loss_of_function",
        perturbation_target="RAI14",
        perturbation_method="siRNA transfection",
        readout="Mean fluorescence intensity of surface and total HLA-DR and Ii by flow cytometry",
        control_description="Control siRNA-treated cells; IgG2aK isotype control",
    ),
    results=Results(
        result_direction=ResultDirectionEnum.POSITIVE,
        effect_description="Surface levels of both endogenous MHC II and Ii increased 1.5-fold in Rai14-silenced cells vs. control; total protein levels were unaffected (not significant)",
        effect_size="1.5-fold increase",
        statistical_test="two-tailed unpaired Student's t-test",
        p_value="**P < 0.01; ns, not significant",
        confidence_interval=None,
        sample_size="three independent experiments",
        key_figure="Figure 2F; Supplementary Figure 2A",
    ),
    methodological_tags=MethodologicalTags(
        approach_category="cell_biology",
        assay_types=["flow_cytometry", "siRNA_knockdown"],
        blinding_reported=None,
        randomization_reported=None,
    ),
    limitations_stated_by_authors=[],
    source_section="Results",
    source_text_span="Figure 2F",
)
evidence_units.append(e_008)

# e_009: MHC II internalization assay - Rai14 KD decreases endosomal MHC II (Figures 2G-I)
e_009 = EvidenceUnit(
    evidence_id="e_009",
    assertion_draft_ids=["a_006"],
    evidence_direction=EvidenceDirectionEnum.SUPPORTS,
    evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
    experiment=Experiment(
        description="MelJuSo cells transfected with siRai14#1, siRai14#2, or siRNA control incubated with anti-MHC II antibody (L243) conjugated to Alexa Fluor 647 for 45 minutes, then fixed at 0, 30, and 60 min post-treatment. Quantification of MHC II-positive endosomal area as percentage of cell area using ImageJ particle analysis.",
        model_system="MelJuSo human melanoma cell line",
        organism="Homo sapiens",
        organism_strain=None,
        perturbation_type="genetic_loss_of_function",
        perturbation_target="RAI14",
        perturbation_method="siRNA transfection using two independent siRNA sequences (siRai14#1 and siRai14#2)",
        readout="Percentage area of MHC II-positive endosomes over total cell area at 0, 30, 60 min",
        control_description="Control siRNA transfection",
    ),
    results=Results(
        result_direction=ResultDirectionEnum.POSITIVE,
        effect_description="Rai14 knockdown cells internalized less MHC II; almost 40% decrease of MHC II in endosomes at 60 minutes post-uptake compared to controls; two independent siRNAs gave consistent results",
        effect_size="~40% decrease",
        statistical_test="two-tailed unpaired Student's t-test",
        p_value="*P < 0.05, **P < 0.01 at t = 60 min",
        confidence_interval=None,
        sample_size="three independent experiments; n > 45 cells per condition",
        key_figure="Figures 2G-I; Supplementary Figure 2B",
    ),
    methodological_tags=MethodologicalTags(
        approach_category="cell_biology",
        assay_types=[
            "antibody_internalization_assay",
            "confocal_microscopy",
            "image_quantification",
            "siRNA_knockdown",
        ],
        blinding_reported=None,
        randomization_reported=None,
    ),
    limitations_stated_by_authors=[],
    source_section="Results",
    source_text_span="Figures 2G-I",
)
evidence_units.append(e_009)

# e_010: Live-cell imaging Rai14-GFP + MHC II colocalize on ruffles and macropinosomes (Figure 3A)
e_010 = EvidenceUnit(
    evidence_id="e_010",
    assertion_draft_ids=["a_007", "a_008"],
    evidence_direction=EvidenceDirectionEnum.SUPPORTS,
    evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
    experiment=Experiment(
        description="Time-lapse live cell imaging of MelJuSo cells transfected with Rai14-GFP. Alexa 647-labeled anti-HLA-DR (L243) antibody added immediately before imaging. Colocalization quantified using JACoP plugin (ImageJ) with Manders' correlation coefficient.",
        model_system="MelJuSo human melanoma cell line",
        organism="Homo sapiens",
        organism_strain=None,
        perturbation_type="protein_overexpression",
        perturbation_target="RAI14",
        perturbation_method="Transient transfection with pCMV6-AC-Rai14-GFP",
        readout="Colocalization of Rai14-GFP and MHC II on membrane ruffles and macropinosomes (Manders' coefficient)",
        control_description=None,
    ),
    results=Results(
        result_direction=ResultDirectionEnum.POSITIVE,
        effect_description="Rai14-GFP and MHC II colocalized on membrane ruffles; these ruffles closed into macropinosomes containing both Rai14 and MHC II",
        effect_size=None,
        statistical_test="Manders' correlation coefficient (JACoP plugin)",
        p_value=None,
        confidence_interval=None,
        sample_size="four independent experiments; n = 25",
        key_figure="Figure 3A; Video 3",
    ),
    methodological_tags=MethodologicalTags(
        approach_category="imaging",
        assay_types=["live_cell_imaging", "spinning_disk_microscopy", "colocalization_analysis"],
        blinding_reported=None,
        randomization_reported=None,
    ),
    limitations_stated_by_authors=["Overexpression may not reflect endogenous Rai14 localization"],
    source_section="Results",
    source_text_span="Figure 3A; Video 3",
)
evidence_units.append(e_010)

# e_011: Rai14 KD halves macropinocytic index in MelJuSo cells (Figures 3B-D)
e_011 = EvidenceUnit(
    evidence_id="e_011",
    assertion_draft_ids=["a_009"],
    evidence_direction=EvidenceDirectionEnum.SUPPORTS,
    evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
    experiment=Experiment(
        description="HLA-DR1 beta-GFP MelJuSo cells transfected with siRai14 or control siRNA, incubated with 70 kDa dextran-Alexa Fluor 555 for 30 min. Macropinocytic index (% cell area occupied by dextran-positive macropinosomes) and macropinosome area quantified from z-stack maximum intensity projections using ImageJ. Rescue experiment: siRai14 cells re-transfected with Rai14-GFP.",
        model_system="HLA-DR1 beta-GFP MelJuSo human melanoma cell line",
        organism="Homo sapiens",
        organism_strain=None,
        perturbation_type="genetic_loss_of_function",
        perturbation_target="RAI14",
        perturbation_method="siRNA transfection; rescue by Rai14-GFP re-expression",
        readout="Macropinocytic index (% cell area occupied by dextran macropinosomes); macropinosome area",
        control_description="siRNA control; rescue with Rai14-GFP re-expression as specificity control",
    ),
    results=Results(
        result_direction=ResultDirectionEnum.POSITIVE,
        effect_description="Macropinocytic index decreased by half (50%) in Rai14-silenced MelJuSo cells; macropinosome area also decreased; Rai14-GFP re-introduction rescued the defect, validating siRNA specificity",
        effect_size="~50% decrease in macropinocytic index",
        statistical_test="two-tailed unpaired Student's t-test",
        p_value="*P < 0.05; **P < 0.01",
        confidence_interval=None,
        sample_size="three independent experiments; n = 60 cells per condition",
        key_figure="Figures 3B-D",
    ),
    methodological_tags=MethodologicalTags(
        approach_category="cell_biology",
        assay_types=[
            "dextran_uptake_assay",
            "siRNA_knockdown",
            "confocal_microscopy",
            "image_quantification",
        ],
        blinding_reported=None,
        randomization_reported=None,
    ),
    limitations_stated_by_authors=[],
    source_section="Results",
    source_text_span="Figures 3B-D",
)
evidence_units.append(e_011)

# e_012: PtdIns(4,5)P2 retention at macropinocytic cup in Rai14 KD (Figures 4A-C)
e_012 = EvidenceUnit(
    evidence_id="e_012",
    assertion_draft_ids=["a_010"],
    evidence_direction=EvidenceDirectionEnum.SUPPORTS,
    evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
    experiment=Experiment(
        description="Time-lapse video microscopy of MelJuSo cells transfected with PH-PLCdelta-GFP (PtdIns(4,5)P2 biosensor) and control siRNA or siRai14. Alexa 647-labeled L243 (anti-HLA-DR) added before imaging on SoRa Spinning Disk microscope (UPLSAPO 60x/0.30 Silicon Oil). Images every 30 seconds. Quantification: percentage of cells retaining PtdIns(4,5)P2 at macropinosome membrane.",
        model_system="MelJuSo human melanoma cell line",
        organism="Homo sapiens",
        organism_strain=None,
        perturbation_type="genetic_loss_of_function",
        perturbation_target="RAI14",
        perturbation_method="siRNA transfection",
        readout="Percentage of cells retaining PtdIns(4,5)P2 at membrane of forming macropinosomes (PH-PLCdelta-GFP biosensor)",
        control_description="siRNA control transfection",
    ),
    results=Results(
        result_direction=ResultDirectionEnum.POSITIVE,
        effect_description="In Rai14-silenced cells, the percentage of cells retaining PtdIns(4,5)P2 at the membrane of nascent macropinosomes dramatically increased compared to controls where PtdIns(4,5)P2 was quickly depleted upon macropinosome formation",
        effect_size=None,
        statistical_test="two-tailed unpaired Student's t-test",
        p_value="**P < 0.01",
        confidence_interval=None,
        sample_size="three independent experiments; n ≥ 32 cells per condition",
        key_figure="Figures 4A-C; Videos 4, 5",
    ),
    methodological_tags=MethodologicalTags(
        approach_category="cell_biology",
        assay_types=[
            "live_cell_imaging",
            "biosensor_imaging",
            "siRNA_knockdown",
            "spinning_disk_microscopy",
        ],
        blinding_reported=None,
        randomization_reported=None,
    ),
    limitations_stated_by_authors=[],
    source_section="Results",
    source_text_span="Figures 4A-C; Videos 4, 5",
)
evidence_units.append(e_012)

# e_013: Rai14 KD decreases macropinocytic index in BMDCs (Figures 5A-C)
e_013 = EvidenceUnit(
    evidence_id="e_013",
    assertion_draft_ids=["a_009", "a_011"],
    evidence_direction=EvidenceDirectionEnum.SUPPORTS,
    evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
    experiment=Experiment(
        description="Primary bone marrow-derived DCs (BMDCs) from C57BL/6 mice transfected with FlexiTube Rai14 siRNA (Qiagen, cat. SI01396318) or control siRNA using Amaxa Nucleofector II electroporator. BMDCs incubated with 70 kDa dextran-Alexa Fluor 555 for 15 min. Macropinocytic index and macropinosome area quantified using ImageJ.",
        model_system="Bone marrow-derived dendritic cells (BMDCs) from C57BL/6 mice (8-16 week old)",
        organism="Mus musculus",
        organism_strain="C57BL/6",
        perturbation_type="genetic_loss_of_function",
        perturbation_target="Rai14",
        perturbation_method="siRNA transfection by electroporation (Amaxa Nucleofector II, program Y-001)",
        readout="Macropinocytic index; macropinosome area",
        control_description="Control siRNA transfection",
    ),
    results=Results(
        result_direction=ResultDirectionEnum.POSITIVE,
        effect_description="30% decrease in macropinocytic index and approximately 30% decrease in macropinosome area in BMDCs silenced for Rai14",
        effect_size="~30% decrease in macropinocytic index and macropinosome area",
        statistical_test="two-tailed paired Student's t-test",
        p_value="*P < 0.05",
        confidence_interval=None,
        sample_size="three independent experiments; n = 60 cells per condition",
        key_figure="Figures 5A-C",
    ),
    methodological_tags=MethodologicalTags(
        approach_category="cell_biology",
        assay_types=[
            "dextran_uptake_assay",
            "siRNA_knockdown",
            "confocal_microscopy",
            "image_quantification",
        ],
        blinding_reported=None,
        randomization_reported=None,
    ),
    limitations_stated_by_authors=[
        "Primary cells from mice, not human; may not fully replicate human DC biology"
    ],
    source_section="Results",
    source_text_span="Figures 5A-C",
)
evidence_units.append(e_013)

# e_014: Rai14 KD decreases macropinocytosis in BMDCs in microchannels (Figures 5D-E)
e_014 = EvidenceUnit(
    evidence_id="e_014",
    assertion_draft_ids=["a_009", "a_011"],
    evidence_direction=EvidenceDirectionEnum.SUPPORTS,
    evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
    experiment=Experiment(
        description="BMDCs transfected with siRai14 or control siRNA loaded into 5x8 µm microfabricated PDMS microchannels coated with fibronectin (20 µg/ml). After 16 h, channels filled with 100 µg/ml 10 kDa Alexa Fluor 555-conjugated dextran for 50 min and imaged on SoRa Spinning Disk microscope. Percentage of cells with internalized dextran quantified.",
        model_system="BMDCs from C57BL/6 mice in microfabricated channels (5x8 µm, 4D Cell)",
        organism="Mus musculus",
        organism_strain="C57BL/6",
        perturbation_type="genetic_loss_of_function",
        perturbation_target="Rai14",
        perturbation_method="siRNA transfection by electroporation",
        readout="Percentage of cells with internalized dextran (in microchannels)",
        control_description="Control siRNA transfection",
    ),
    results=Results(
        result_direction=ResultDirectionEnum.POSITIVE,
        effect_description="Almost 40% decrease in the number of cells containing macropinosomes upon Rai14 depletion in microchannels",
        effect_size="~40% decrease",
        statistical_test="two-tailed paired Student's t-test",
        p_value="*P < 0.05",
        confidence_interval=None,
        sample_size="three independent experiments; n ≥ 40 cells per condition",
        key_figure="Figures 5D-E",
    ),
    methodological_tags=MethodologicalTags(
        approach_category="cell_biology",
        assay_types=[
            "dextran_uptake_assay",
            "microchannel_assay",
            "siRNA_knockdown",
            "spinning_disk_microscopy",
        ],
        blinding_reported=None,
        randomization_reported=None,
    ),
    limitations_stated_by_authors=[],
    source_section="Results",
    source_text_span="Figures 5D-E",
)
evidence_units.append(e_014)

# e_015: Rai14 KD decreases PAK phosphorylation in BMDCs (Figures 5F-H)
e_015 = EvidenceUnit(
    evidence_id="e_015",
    assertion_draft_ids=["a_012"],
    evidence_direction=EvidenceDirectionEnum.SUPPORTS,
    evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
    experiment=Experiment(
        description="BMDCs transfected with siRai14 or control siRNA lysed and analyzed by Western blot for Rai14, phosphorylated PAK1 (Ser144)/PAK2 (Ser141) (Cell Signaling Technology, cat. 2606S), total PAK1/2/3 (Cell Signaling Technology, cat. 2604S), and GAPDH loading control. Band intensity quantified by densitometry (Image Lab Software, Bio-Rad).",
        model_system="BMDCs from C57BL/6 mice",
        organism="Mus musculus",
        organism_strain="C57BL/6",
        perturbation_type="genetic_loss_of_function",
        perturbation_target="Rai14",
        perturbation_method="siRNA transfection",
        readout="Levels of phosphorylated PAK (pPAK1 Ser144/pPAK2 Ser141) relative to total PAK and tubulin by Western blot densitometry",
        control_description="Control siRNA transfection",
    ),
    results=Results(
        result_direction=ResultDirectionEnum.POSITIVE,
        effect_description="Almost 50% decrease in PAK phosphorylation levels in Rai14-silenced BMDCs compared to control cells",
        effect_size="~50% decrease",
        statistical_test="two-tailed paired Student's t-test",
        p_value="*P < 0.05",
        confidence_interval=None,
        sample_size="four independent experiments",
        key_figure="Figures 5F-H",
    ),
    methodological_tags=MethodologicalTags(
        approach_category="biochemical_assay",
        assay_types=["western_blot", "siRNA_knockdown", "phosphorylation_assay"],
        blinding_reported=None,
        randomization_reported=None,
    ),
    limitations_stated_by_authors=[],
    source_section="Results",
    source_text_span="Figures 5F-H",
)
evidence_units.append(e_015)

# e_016: Rai14 KD increases BMDC migration speed and decreases speed fluctuations (Figures 6A-C)
e_016 = EvidenceUnit(
    evidence_id="e_016",
    assertion_draft_ids=["a_013"],
    evidence_direction=EvidenceDirectionEnum.SUPPORTS,
    evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
    experiment=Experiment(
        description="BMDCs transfected with siRai14 or control siRNA loaded into 5x8 µm microfabricated channels coated with fibronectin. Imaged for 20 h on epiﬂuorescence Olympus TiE microscope with 10x objective, 1 image/min. Kymograph extraction and velocity analysis performed using ImageJ.",
        model_system="BMDCs from C57BL/6 mice in microfabricated channels (5x8 µm)",
        organism="Mus musculus",
        organism_strain="C57BL/6",
        perturbation_type="genetic_loss_of_function",
        perturbation_target="Rai14",
        perturbation_method="siRNA transfection",
        readout="Mean cell speed (µm/min); speed fluctuations (s.d./mean instantaneous speed) by kymograph analysis",
        control_description="Control siRNA transfection",
    ),
    results=Results(
        result_direction=ResultDirectionEnum.POSITIVE,
        effect_description="BMDCs silenced for Rai14 moved faster (mean speed 7.1 µm/min) than control BMDCs (mean speed 5.4 µm/min); Rai14-depleted cells showed fewer speed fluctuations, indicating less frequent direction changes",
        effect_size="7.1 µm/min vs 5.4 µm/min (control)",
        statistical_test="two-tailed paired Student's t-test",
        p_value="**P < 0.01; *P < 0.01",
        confidence_interval=None,
        sample_size="four independent experiments; n > 150 cells per condition",
        key_figure="Figures 6A-C",
    ),
    methodological_tags=MethodologicalTags(
        approach_category="cell_biology",
        assay_types=[
            "microchannel_migration_assay",
            "kymograph_analysis",
            "live_cell_imaging",
            "siRNA_knockdown",
        ],
        blinding_reported=None,
        randomization_reported=None,
    ),
    limitations_stated_by_authors=[],
    source_section="Results",
    source_text_span="Figures 6A-C",
)
evidence_units.append(e_016)

# e_017: Rai14 and Ii colocalize on macropinosome-like vesicles in BMDCs (Figures 6D-E)
e_017 = EvidenceUnit(
    evidence_id="e_017",
    assertion_draft_ids=["a_003", "a_014"],
    evidence_direction=EvidenceDirectionEnum.SUPPORTS,
    evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
    experiment=Experiment(
        description="Immunofluorescence staining of fixed BMDCs with anti-Rai14 (ab137118, Abcam, IF 1:40) and anti-mouse CD74 (In-1, BD Biosciences, IF 1:50) antibodies. DAPI nuclear staining. Confocal imaging on Olympus FluoView 1000 IX81, 60x objective. Normalized fluorescence intensity profiles along vesicles quantified.",
        model_system="BMDCs from C57BL/6 mice",
        organism="Mus musculus",
        organism_strain="C57BL/6",
        perturbation_type="none",
        perturbation_target=None,
        perturbation_method=None,
        readout="Colocalization of endogenous Rai14 and Ii on macropinosome-like vesicles by immunofluorescence; normalized fluorescence intensity profiles",
        control_description=None,
    ),
    results=Results(
        result_direction=ResultDirectionEnum.POSITIVE,
        effect_description="Endogenous Rai14 and Ii colocalized on large macropinosome-like vesicles in BMDCs; intensity profiles showed overlapping signals",
        effect_size=None,
        statistical_test=None,
        p_value=None,
        confidence_interval=None,
        sample_size="two independent experiments; n = 78 vesicles from 37 cells",
        key_figure="Figures 6D-E",
    ),
    methodological_tags=MethodologicalTags(
        approach_category="imaging",
        assay_types=[
            "immunofluorescence",
            "confocal_microscopy",
            "fluorescence_intensity_profiling",
        ],
        blinding_reported=None,
        randomization_reported=None,
    ),
    limitations_stated_by_authors=[],
    source_section="Results",
    source_text_span="Figures 6D-E",
)
evidence_units.append(e_017)

# e_018: Co-IP Rai14 pulls down myosin II in DCs (Figure 6F)
e_018 = EvidenceUnit(
    evidence_id="e_018",
    assertion_draft_ids=["a_015"],
    evidence_direction=EvidenceDirectionEnum.SUPPORTS,
    evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
    experiment=Experiment(
        description="Co-immunoprecipitation from DC lysates using anti-Rai14 antibody (ab137118, Abcam) or IgG isotype control. Total lysate and immunoprecipitates analyzed by Western blot with anti-myosin IIA (ab24762, Abcam) and anti-Rai14 antibodies.",
        model_system="Dendritic cells",
        organism="Mus musculus",
        organism_strain="C57BL/6",
        perturbation_type="none",
        perturbation_target=None,
        perturbation_method=None,
        readout="Co-immunoprecipitation of myosin II with anti-Rai14 antibody by Western blot",
        control_description="IgG isotype control immunoprecipitation",
    ),
    results=Results(
        result_direction=ResultDirectionEnum.POSITIVE,
        effect_description="Rai14 antibody co-immunoprecipitated myosin IIA from DC lysates, confirming Rai14-myosin II interaction in primary immune cells",
        effect_size=None,
        statistical_test=None,
        p_value=None,
        confidence_interval=None,
        sample_size=None,
        key_figure="Figure 6F",
    ),
    methodological_tags=MethodologicalTags(
        approach_category="biochemical_assay",
        assay_types=["co-immunoprecipitation", "western_blot"],
        blinding_reported=None,
        randomization_reported=None,
    ),
    limitations_stated_by_authors=["Co-IP does not distinguish direct from indirect interaction"],
    source_section="Results",
    source_text_span="Figure 6F",
)
evidence_units.append(e_018)

# e_019: GST-myosin II tail pulldown of Rai14 and Ii; Rai14 KD abolishes Ii pulldown (Figure 6G)
e_019 = EvidenceUnit(
    evidence_id="e_019",
    assertion_draft_ids=["a_015", "a_016"],
    evidence_direction=EvidenceDirectionEnum.SUPPORTS,
    evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
    experiment=Experiment(
        description="GST-pulldown experiment: purified GST or GST-myosin II heavy chain tail (amino acids 1795-1960; pGEX2T construct) expressed in E. coli BL21(DE3), purified by Glutathione Sepharose 4B affinity chromatography. 10 µg purified GST-proteins incubated with 200 µl precleared MelJuSo lysates from cells transfected with siRNA control or siRai14. Affinity chromatography followed by Western blot for GST, Rai14, and Ii.",
        model_system="GST pulldown from MelJuSo cell lysates (siRNA control or siRai14)",
        organism="Homo sapiens",
        organism_strain=None,
        perturbation_type="genetic_loss_of_function",
        perturbation_target="RAI14",
        perturbation_method="siRNA transfection (compared to control siRNA)",
        readout="Co-pulldown of Rai14 and Ii with GST-myosin II heavy chain tail by Western blot",
        control_description="GST alone negative control; siRNA control lysate",
    ),
    results=Results(
        result_direction=ResultDirectionEnum.POSITIVE,
        effect_description="GST-myosin II heavy chain tail (but not GST alone) specifically precipitated both Ii and Rai14 from control MelJuSo lysates; GST-myosin II tail was unable to pull down Ii from Rai14-silenced cell lysates, suggesting Rai14 bridges Ii to myosin II",
        effect_size=None,
        statistical_test=None,
        p_value=None,
        confidence_interval=None,
        sample_size=None,
        key_figure="Figure 6G",
    ),
    methodological_tags=MethodologicalTags(
        approach_category="biochemical_assay",
        assay_types=["GST_pulldown", "affinity_chromatography", "western_blot", "siRNA_knockdown"],
        blinding_reported=None,
        randomization_reported=None,
    ),
    limitations_stated_by_authors=[
        "Pulldown uses only the myosin II tail domain (aa 1795-1960), not full-length myosin II"
    ],
    source_section="Results",
    source_text_span="Figure 6G",
)
evidence_units.append(e_019)

# e_020: Rai14 KD decreases myosin II at front of migrating BMDCs (Supplementary Figure 3)
e_020 = EvidenceUnit(
    evidence_id="e_020",
    assertion_draft_ids=["a_017"],
    evidence_direction=EvidenceDirectionEnum.SUPPORTS,
    evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
    experiment=Experiment(
        description="BMDCs transfected with siRai14 or control siRNA loaded into 5x8 µm microchannels, fixed after 16 h, and immunostained with anti-myosin II antibody. Cells imaged on confocal microscope (Olympus FluoView FV1000, 60x). Density maps of mean myosin II distribution generated using ImageJ. Front (20%) and back (20%) myosin II intensity quantified.",
        model_system="BMDCs from C57BL/6 mice in microfabricated channels",
        organism="Mus musculus",
        organism_strain="C57BL/6",
        perturbation_type="genetic_loss_of_function",
        perturbation_target="Rai14",
        perturbation_method="siRNA transfection",
        readout="Myosin II density maps; front-to-back ratio of myosin II intensity",
        control_description="Control siRNA transfection",
    ),
    results=Results(
        result_direction=ResultDirectionEnum.POSITIVE,
        effect_description="Rai14 silencing decreased the amount of myosin II at the front of BMDCs migrating in microchannels, as revealed by density maps; reduced front-to-back myosin II ratio",
        effect_size=None,
        statistical_test="two-tailed unpaired Student's t-test",
        p_value="*P < 0.05",
        confidence_interval=None,
        sample_size="three independent experiments; n > 40 cells per condition",
        key_figure="Supplementary Figure 3",
    ),
    methodological_tags=MethodologicalTags(
        approach_category="imaging",
        assay_types=[
            "immunofluorescence",
            "confocal_microscopy",
            "density_map_analysis",
            "siRNA_knockdown",
        ],
        blinding_reported=None,
        randomization_reported=None,
    ),
    limitations_stated_by_authors=[],
    source_section="Results",
    source_text_span="Supplementary Figure 3",
)
evidence_units.append(e_020)

# e_021: Rai14-GFP localizes to membrane ruffles (Supplementary Figure 1A)
e_021 = EvidenceUnit(
    evidence_id="e_021",
    assertion_draft_ids=["a_002"],
    evidence_direction=EvidenceDirectionEnum.SUPPORTS,
    evidence_strength=EvidenceStrengthEnum.DIRECT_EXPERIMENTAL,
    experiment=Experiment(
        description="Time-lapse video microscopy of MelJuSo cells transfected with Rai14-GFP alone. Live imaging to assess endogenous localization pattern of Rai14.",
        model_system="MelJuSo human melanoma cell line",
        organism="Homo sapiens",
        organism_strain=None,
        perturbation_type="protein_overexpression",
        perturbation_target="RAI14",
        perturbation_method="Transient transfection with pCMV6-AC-Rai14-GFP",
        readout="Subcellular localization of Rai14-GFP by live cell imaging",
        control_description=None,
    ),
    results=Results(
        result_direction=ResultDirectionEnum.POSITIVE,
        effect_description="Rai14-GFP was mainly localized at the plasma membrane and membrane ruffles, which became internalized into vesicles",
        effect_size=None,
        statistical_test=None,
        p_value=None,
        confidence_interval=None,
        sample_size=None,
        key_figure="Supplementary Figure 1A; Video 1",
    ),
    methodological_tags=MethodologicalTags(
        approach_category="imaging",
        assay_types=["live_cell_imaging", "spinning_disk_microscopy"],
        blinding_reported=None,
        randomization_reported=None,
    ),
    limitations_stated_by_authors=[
        "Overexpression construct; does not represent endogenous Rai14 levels"
    ],
    source_section="Results",
    source_text_span="Supplementary Figure 1A; Video 1",
)
evidence_units.append(e_021)

# ---------------------------------------------------------------------------
# Assertion Drafts
# ---------------------------------------------------------------------------

assertion_drafts = []

# -- BACKGROUND CLAIMS FROM INTRODUCTION --

# a_BG_001: MHC II is expressed on professional APCs
a_bg_001 = AssertionDraft(
    draft_id="a_bg_001",
    natural_language="Professional antigen-presenting cells such as dendritic cells and B-lymphocytes constitutively express MHC II molecules consisting of alpha-beta heterodimers.",
    canonical_form="Professional APCs (dendritic cells, B-lymphocytes) — constitutively_express — MHC II (alpha-beta heterodimers)",
    negatable_form="Professional APCs such as dendritic cells and B-lymphocytes do NOT constitutively express MHC II molecules.",
    subject_entity=Entity(
        surface_form="professional antigen-presenting cells",
        canonical_name="antigen-presenting cell",
        ontology_id="CL:0000145",
        ontology_source="CL",
        entity_type=EntityTypeEnum.CELL_TYPE,
        aliases=["APCs", "dendritic cells", "B-lymphocytes"],
    ),
    object_entity=mhcii_entity,
    predicate="constitutively_expresses",
    direction=None,
    assertion_type=AssertionTypeEnum.EXISTENCE,
    causal_type=None,
    scope=Scope(
        species=[human_species],
        tissue=[],
        cell_type=[dc_ct],
        disease=[],
        condition=None,
        developmental_stage=None,
        in_vitro=None,
    ),
    conditions=[],
    hedging=Hedging(
        verbatim_hedge="constitutively express",
        certainty=HedgeLevelEnum.HIGH,
        generalizability=HedgeLevelEnum.HIGH,
        causality_hedge=CausalityHedgeEnum.CAUSAL,
    ),
    epistemic_status=EpistemicStatus(
        section=SectionEnum.INTRODUCTION,
        function=FunctionEnum.BACKGROUND,
        is_primary=False,
        cited_source=None,
    ),
    evidence_unit_ids=[],
    parent_assertion_ids=[],
    provenance=Provenance(
        source_sentence="Professional antigen-presenting cells (APCs) such as dendritic cells (DCs) and B-lymphocytes constitutively express Major Histocompatibility Class II molecules (MHC II), which consist of ab heterodimers associated with self or antigenic peptides situated in their peptide-binding grooves.",
        section_name="Introduction",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_bg_001)

# a_BG_002: Ii acts as a chaperone for MHC II assembly and transport
a_bg_002 = AssertionDraft(
    draft_id="a_bg_002",
    natural_language="Invariant chain (Ii, CD74) acts as a chaperone that prevents premature peptide binding to MHC II and directs new MHC II complexes to endosomes for antigen loading.",
    canonical_form="Invariant chain (CD74) — acts_as_chaperone_for — MHC II (transport to antigen-loading endosomes)",
    negatable_form="Invariant chain does NOT act as a chaperone for MHC II or direct MHC II to endosomes.",
    subject_entity=ii_entity,
    object_entity=mhcii_entity,
    predicate="chaperones",
    direction=None,
    assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
    causal_type=CausalTypeEnum.NECESSARY,
    scope=Scope(
        species=[human_species],
        tissue=[],
        cell_type=[dc_ct],
        disease=[],
        condition=None,
        developmental_stage=None,
        in_vitro=None,
    ),
    conditions=[],
    hedging=Hedging(
        verbatim_hedge="provides a scaffold for the assembly of MHC II heterodimers",
        certainty=HedgeLevelEnum.HIGH,
        generalizability=HedgeLevelEnum.HIGH,
        causality_hedge=CausalityHedgeEnum.CAUSAL,
    ),
    epistemic_status=EpistemicStatus(
        section=SectionEnum.INTRODUCTION,
        function=FunctionEnum.BACKGROUND,
        is_primary=False,
        cited_source=None,
    ),
    evidence_unit_ids=[],
    parent_assertion_ids=[],
    provenance=Provenance(
        source_sentence="Invariant chain (Ii, CD74) is a type II transmembrane protein that self-associates into trimers and provides a scaffold for the assembly of MHC II heterodimers. Ii interacts with the peptide-binding groove of MHC II to prevent the premature binding of endogenous peptides and chaperones new MHC II molecules to endosomes for the loading of antigenic peptides.",
        section_name="Introduction",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_bg_002)

# a_BG_003: Ii leucine-based sorting motifs recruit AP-1 and AP-2
a_bg_003 = AssertionDraft(
    draft_id="a_bg_003",
    natural_language="Leucine-based sorting motifs in the N-terminal cytosolic tail of Ii recruit adaptor proteins AP-1 and AP-2 to mediate transport of new MHC II-Ii complexes from the trans-Golgi network to antigen-loading compartments.",
    canonical_form="Ii (leucine-based sorting motifs) — recruits — adaptor proteins AP-1 and AP-2 [mediating MHC II-Ii transport to antigen-loading compartments]",
    negatable_form="Leucine-based sorting motifs in Ii do NOT recruit adaptor proteins AP-1/AP-2 to mediate MHC II-Ii transport.",
    subject_entity=ii_entity,
    object_entity=Entity(
        surface_form="adaptor proteins-1 and -2",
        canonical_name="AP-1/AP-2",
        ontology_id=None,
        ontology_source=None,
        entity_type=EntityTypeEnum.PROTEIN,
        aliases=["AP-1", "AP-2", "adaptor protein 1", "adaptor protein 2"],
    ),
    predicate="recruits",
    direction=None,
    assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
    causal_type=CausalTypeEnum.NECESSARY,
    scope=Scope(
        species=[human_species],
        tissue=[],
        cell_type=[],
        disease=[],
        condition=None,
        developmental_stage=None,
        in_vitro=None,
    ),
    conditions=[],
    hedging=Hedging(
        verbatim_hedge="recruits adaptor proteins-1 and -2, which mediate the transport",
        certainty=HedgeLevelEnum.HIGH,
        generalizability=HedgeLevelEnum.MEDIUM,
        causality_hedge=CausalityHedgeEnum.CAUSAL,
    ),
    epistemic_status=EpistemicStatus(
        section=SectionEnum.INTRODUCTION,
        function=FunctionEnum.BACKGROUND,
        is_primary=False,
        cited_source=None,
    ),
    evidence_unit_ids=[],
    parent_assertion_ids=[],
    provenance=Provenance(
        source_sentence="Due to the presence of two leucine-based sorting motifs in its N-terminal cytosolic tail, Ii recruits adaptor proteins-1 and -2, which mediate the transport of new MHC II-Ii complexes to antigen-loading compartments from the trans-Golgi network, mainly via the cell surface.",
        section_name="Introduction",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_bg_003)

# a_BG_004: Ii interacts with myosin II to regulate macropinocytosis and migration antagonistically
a_bg_004 = AssertionDraft(
    draft_id="a_bg_004",
    natural_language="Invariant chain interacts with the actin motor myosin II to regulate macropinocytic and migratory activity of dendritic cells in an antagonistic manner.",
    canonical_form="Invariant chain — interacts_with — myosin II [regulating macropinocytosis and migration antagonistically in DCs]",
    negatable_form="Invariant chain does NOT interact with myosin II to regulate macropinocytosis and migration in dendritic cells.",
    subject_entity=ii_entity,
    object_entity=myosin2_entity,
    predicate="interacts_with",
    direction=None,
    assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
    causal_type=CausalTypeEnum.MODULATORY,
    scope=Scope(
        species=[mouse_species],
        tissue=[],
        cell_type=[dc_ct],
        disease=[],
        condition=None,
        developmental_stage=None,
        in_vitro=False,
    ),
    conditions=[],
    hedging=Hedging(
        verbatim_hedge="by interacting with the actin motor myosin II, Ii regulates the macropinocytic and migratory ability of DCs in an antagonistic manner",
        certainty=HedgeLevelEnum.HIGH,
        generalizability=HedgeLevelEnum.MEDIUM,
        causality_hedge=CausalityHedgeEnum.CAUSAL,
    ),
    epistemic_status=EpistemicStatus(
        section=SectionEnum.INTRODUCTION,
        function=FunctionEnum.BACKGROUND,
        is_primary=False,
        cited_source=None,
    ),
    evidence_unit_ids=[],
    parent_assertion_ids=[],
    provenance=Provenance(
        source_sentence="Furthermore, by interacting with the actin motor myosin II, Ii regulates the macropinocytic and migratory ability of DCs in an antagonistic manner (7, 8).",
        section_name="Introduction",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_bg_004)

# a_BG_005: Myosin II at DC front promotes macropinocytosis; at rear promotes migration
a_bg_005 = AssertionDraft(
    draft_id="a_bg_005",
    natural_language="Myosin II localization at the cell front in dendritic cells increases macropinocytic activity and reduces migration speed, while myosin II at the cell rear is necessary for fast migration.",
    canonical_form="Myosin II localization (cell front) — promotes — macropinocytosis [while reducing migration speed in DCs]",
    negatable_form="Myosin II localization at the DC cell front does NOT increase macropinocytosis or reduce migration speed.",
    subject_entity=myosin2_entity,
    object_entity=macropinocytosis_entity,
    predicate="promotes",
    direction=None,
    assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
    causal_type=CausalTypeEnum.SUFFICIENT,
    scope=Scope(
        species=[mouse_species],
        tissue=[],
        cell_type=[dc_ct],
        disease=[],
        condition="myosin II recruited to cell front by Ii",
        developmental_stage=None,
        in_vitro=False,
    ),
    conditions=[
        Condition(
            condition_type=ConditionTypeEnum.BIOLOGICAL_CONTEXT,
            value="myosin II recruited to the cell front by Ii",
        )
    ],
    hedging=Hedging(
        verbatim_hedge="When myosin II is recruited to the cell front by Ii, the macropinocytic activity increases, reducing the migration speed",
        certainty=HedgeLevelEnum.HIGH,
        generalizability=HedgeLevelEnum.MEDIUM,
        causality_hedge=CausalityHedgeEnum.CAUSAL,
    ),
    epistemic_status=EpistemicStatus(
        section=SectionEnum.INTRODUCTION,
        function=FunctionEnum.BACKGROUND,
        is_primary=False,
        cited_source=None,
    ),
    evidence_unit_ids=[],
    parent_assertion_ids=["a_bg_004"],
    provenance=Provenance(
        source_sentence="When myosin II is recruited to the cell front by Ii, the macropinocytic activity increases, reducing the migration speed. On the other hand, myosin II localization at the cell rear is necessary for fast migration.",
        section_name="Introduction",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_bg_005)

# a_BG_006: Rai14 is an N-Ank protein with membrane-binding and shaping activity
a_bg_006 = AssertionDraft(
    draft_id="a_bg_006",
    natural_language="Rai14 (ankycorbin/NORPEG) is a member of the N-Ank superfamily that binds to membranes via both hydrophobic and electrostatic interactions and has a direct role in shaping membranes.",
    canonical_form="Rai14 (N-Ank family member) — binds_and_shapes — membranes [via amphipathic helix and ankyrin repeats]",
    negatable_form="Rai14 does NOT bind or shape membranes via its N-Ank domain.",
    subject_entity=rai14_entity,
    object_entity=Entity(
        surface_form="membranes",
        canonical_name="plasma membrane",
        ontology_id="GO:0005886",
        ontology_source="GO",
        entity_type=EntityTypeEnum.OTHER,
        aliases=["membrane", "lipid bilayer"],
    ),
    predicate="binds_and_shapes",
    direction=None,
    assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
    causal_type=CausalTypeEnum.CONTRIBUTORY,
    scope=Scope(
        species=[human_species],
        tissue=[],
        cell_type=[],
        disease=[],
        condition=None,
        developmental_stage=None,
        in_vitro=None,
    ),
    conditions=[],
    hedging=Hedging(
        verbatim_hedge="it has been shown that Rai14 has a direct role in shaping membranes",
        certainty=HedgeLevelEnum.MEDIUM,
        generalizability=HedgeLevelEnum.MEDIUM,
        causality_hedge=CausalityHedgeEnum.CAUSAL,
    ),
    epistemic_status=EpistemicStatus(
        section=SectionEnum.INTRODUCTION,
        function=FunctionEnum.BACKGROUND,
        is_primary=False,
        cited_source=None,
    ),
    evidence_unit_ids=[],
    parent_assertion_ids=[],
    provenance=Provenance(
        source_sentence="Rai14 is a good example of an N-Ank protein as it binds to membranes via both hydrophobic and electrostatic interactions. Indeed, it has been shown that Rai14 has a direct role in shaping membranes (9).",
        section_name="Introduction",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_bg_006)

# a_BG_007: Rai14 associates with cortical actin, F-actin stress fibers, and cell-cell adhesion sites
a_bg_007 = AssertionDraft(
    draft_id="a_bg_007",
    natural_language="Rai14 associates with the cortical actin cytoskeleton, F-actin stress fibers, and cell-cell adhesion sites, and has been proposed as a cytoskeleton-associated protein linked to actin function and organization.",
    canonical_form="Rai14 — associates_with — cortical actin cytoskeleton and F-actin stress fibers",
    negatable_form="Rai14 does NOT associate with the cortical actin cytoskeleton or F-actin stress fibers.",
    subject_entity=rai14_entity,
    object_entity=Entity(
        surface_form="cortical actin cytoskeleton",
        canonical_name="actin cytoskeleton",
        ontology_id="GO:0015629",
        ontology_source="GO",
        entity_type=EntityTypeEnum.OTHER,
        aliases=["F-actin", "actin stress fibers", "cortical actin"],
    ),
    predicate="associates_with",
    direction=None,
    assertion_type=AssertionTypeEnum.CORRELATIONAL,
    causal_type=None,
    scope=Scope(
        species=[human_species],
        tissue=[],
        cell_type=[],
        disease=[],
        condition=None,
        developmental_stage=None,
        in_vitro=True,
    ),
    conditions=[],
    hedging=Hedging(
        verbatim_hedge="The reported intracellular localization of Rai14 suggests that it also associates with the cortical actin cytoskeleton",
        certainty=HedgeLevelEnum.MEDIUM,
        generalizability=HedgeLevelEnum.MEDIUM,
        causality_hedge=CausalityHedgeEnum.CORRELATIONAL,
    ),
    epistemic_status=EpistemicStatus(
        section=SectionEnum.INTRODUCTION,
        function=FunctionEnum.BACKGROUND,
        is_primary=False,
        cited_source=None,
    ),
    evidence_unit_ids=[],
    parent_assertion_ids=[],
    provenance=Provenance(
        source_sentence="The reported intracellular localization of Rai14 suggests that it also associates with the cortical actin cytoskeleton, F-actin stress ﬁbers, and cell-cell adhesions sites (10–12).",
        section_name="Introduction",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_bg_007)

# -- NOVEL FINDING CLAIMS --

# a_001: Rai14 interacts with Ii (novel finding, confirmed by Y2H + co-IP)
a_001 = AssertionDraft(
    draft_id="a_001",
    natural_language="Rai14 physically interacts with Invariant chain (Ii/CD74), as demonstrated by yeast two-hybrid screening and confirmed by reciprocal co-immunoprecipitation in human melanoma cells.",
    canonical_form="Rai14 — interacts_with — Invariant chain (CD74) [in MelJuSo human melanoma cells]",
    negatable_form="Rai14 does NOT interact with Invariant chain (CD74).",
    subject_entity=rai14_entity,
    object_entity=ii_entity,
    predicate="interacts_with",
    direction=None,
    assertion_type=AssertionTypeEnum.CORRELATIONAL,
    causal_type=None,
    scope=Scope(
        species=[human_species],
        tissue=[],
        cell_type=[meljuso_ct],
        disease=[melanoma_disease],
        condition=None,
        developmental_stage=None,
        in_vitro=True,
    ),
    conditions=[],
    hedging=Hedging(
        verbatim_hedge="confirm the interaction between Rai14 and Ii identified in the yeast two-hybrid screen",
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
)
assertion_drafts.append(a_001)

# a_002: Rai14 localizes to membrane ruffles and nascent macropinosomes
a_002 = AssertionDraft(
    draft_id="a_002",
    natural_language="Rai14 localizes to the plasma membrane and membrane ruffles, and is present on nascent macropinosomes during macropinosome formation.",
    canonical_form="Rai14 — localizes_to — membrane ruffles and nascent macropinosomes [in human melanoma cells]",
    negatable_form="Rai14 does NOT localize to membrane ruffles or nascent macropinosomes.",
    subject_entity=rai14_entity,
    object_entity=membrane_ruffle_entity,
    predicate="localizes_to",
    direction=None,
    assertion_type=AssertionTypeEnum.EXISTENCE,
    causal_type=None,
    scope=Scope(
        species=[human_species],
        tissue=[],
        cell_type=[meljuso_ct],
        disease=[melanoma_disease],
        condition=None,
        developmental_stage=None,
        in_vitro=True,
    ),
    conditions=[],
    hedging=Hedging(
        verbatim_hedge="Rai14-GFP was mainly localized at the plasma membrane and membrane ruffles, which also became internalized into vesicles",
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
    evidence_unit_ids=["e_004", "e_021"],
    parent_assertion_ids=[],
    provenance=Provenance(
        source_sentence="Rai14-GFP was mainly localized at the plasma membrane and membrane ruffles, which also became internalized into vesicles.",
        section_name="Results — Rai14 is a novel interactor of Invariant chain",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_002)

# a_003: Rai14 and Ii colocalize at membrane ruffles and on endosomes
a_003 = AssertionDraft(
    draft_id="a_003",
    natural_language="Rai14 and Ii colocalize on membrane ruffles, nascent vesicles, and Ii-positive endosomes in antigen-presenting cells.",
    canonical_form="Rai14 — colocalizes_with — Invariant chain [on membrane ruffles, nascent vesicles, and endosomes]",
    negatable_form="Rai14 and Ii do NOT colocalize on membrane ruffles, nascent vesicles, or endosomes.",
    subject_entity=rai14_entity,
    object_entity=ii_entity,
    predicate="colocalizes_with",
    direction=None,
    assertion_type=AssertionTypeEnum.CORRELATIONAL,
    causal_type=None,
    scope=Scope(
        species=[human_species, mouse_species],
        tissue=[],
        cell_type=[meljuso_ct, dc_ct],
        disease=[melanoma_disease],
        condition=None,
        developmental_stage=None,
        in_vitro=True,
    ),
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
    evidence_unit_ids=["e_004", "e_005", "e_017"],
    parent_assertion_ids=["a_001"],
    provenance=Provenance(
        source_sentence="Upon addition of the fluorescently labeled antibody targeting Ii, we detected Ii together with Rai14 on membrane ruffles, nascent vesicles, and forming macropinosomes.",
        section_name="Results — Rai14 is a novel interactor of Invariant chain",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_003)

# a_004: Rai14 is present in a complex with Ii and MHC II
a_004 = AssertionDraft(
    draft_id="a_004",
    natural_language="Rai14 is present in a trimeric complex with Invariant chain (Ii) and MHC II (HLA-DR1) in human melanoma cells.",
    canonical_form="Rai14 — forms_complex_with — Invariant chain and MHC II (HLA-DR1)",
    negatable_form="Rai14 is NOT present in a complex with Invariant chain and MHC II.",
    subject_entity=rai14_entity,
    object_entity=mhcii_entity,
    predicate="forms_complex_with",
    direction=None,
    assertion_type=AssertionTypeEnum.CORRELATIONAL,
    causal_type=None,
    scope=Scope(
        species=[human_species],
        tissue=[],
        cell_type=[meljuso_ct],
        disease=[melanoma_disease],
        condition=None,
        developmental_stage=None,
        in_vitro=True,
    ),
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
    evidence_unit_ids=["e_006"],
    parent_assertion_ids=["a_001"],
    provenance=Provenance(
        source_sentence="Cells expressing HLA-DR1 b-GFP (but not cells expressing GFP) were able to co-immunoprecipitate both endogenous Ii and Rai14, confirming that Rai14 interacts with Ii in a complex with MHC II.",
        section_name="Results — Rai14 depletion retains MHC II at the plasma membrane",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_004)

# a_005: Rai14 depletion increases surface MHC II and Ii levels
a_005 = AssertionDraft(
    draft_id="a_005",
    natural_language="Rai14 knockdown increases the surface levels of both endogenous MHC II and Ii by approximately 1.5-fold in human melanoma cells without affecting total protein levels.",
    canonical_form="Rai14 knockdown — increases — surface MHC II and Ii levels [1.5-fold; without affecting total protein]",
    negatable_form="Rai14 knockdown does NOT increase the surface levels of MHC II or Ii.",
    subject_entity=rai14_entity,
    object_entity=mhcii_entity,
    predicate="negatively_regulates_surface_expression",
    direction=None,
    assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
    causal_type=CausalTypeEnum.CONTRIBUTORY,
    scope=Scope(
        species=[human_species],
        tissue=[],
        cell_type=[meljuso_ct],
        disease=[melanoma_disease],
        condition=None,
        developmental_stage=None,
        in_vitro=True,
    ),
    conditions=[
        Condition(
            condition_type=ConditionTypeEnum.BIOLOGICAL_CONTEXT, value="siRNA knockdown of Rai14"
        )
    ],
    hedging=Hedging(
        verbatim_hedge="the surface levels of both endogenous MHC II and Ii increased 1.5-fold in cells silenced for Rai14 compared to control cells, while the total levels remain unaffected",
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
    evidence_unit_ids=["e_007", "e_008"],
    parent_assertion_ids=[],
    provenance=Provenance(
        source_sentence="Flow cytometry analysis indeed confirmed that the surface levels of both endogenous MHC II and Ii increased 1.5-fold in cells silenced for Rai14 compared to control cells, while the total levels remain unaffected.",
        section_name="Results — Rai14 depletion retains MHC II at the plasma membrane",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_005)

# a_006: Rai14 is required for MHC II internalization
a_006 = AssertionDraft(
    draft_id="a_006",
    natural_language="Rai14 is required for efficient MHC II internalization; its knockdown decreases MHC II in endosomes by approximately 40% at 60 minutes post-uptake.",
    canonical_form="Rai14 — is_required_for — MHC II internalization [in human APCs]",
    negatable_form="Rai14 is NOT required for MHC II internalization.",
    subject_entity=rai14_entity,
    object_entity=mhcii_entity,
    predicate="is_required_for",
    direction=None,
    assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
    causal_type=CausalTypeEnum.NECESSARY,
    scope=Scope(
        species=[human_species],
        tissue=[],
        cell_type=[meljuso_ct],
        disease=[melanoma_disease],
        condition=None,
        developmental_stage=None,
        in_vitro=True,
    ),
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
    evidence_unit_ids=["e_007", "e_008", "e_009"],
    parent_assertion_ids=["a_005"],
    provenance=Provenance(
        source_sentence="Altogether, these results indicate that Rai14 is required for MHC II internalization.",
        section_name="Results — Rai14 depletion retains MHC II at the plasma membrane",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_006)

# a_007: Rai14 and MHC II colocalize on membrane ruffles that form macropinosomes
a_007 = AssertionDraft(
    draft_id="a_007",
    natural_language="Rai14 and MHC II colocalize on membrane ruffles that subsequently close into macropinosomes in human melanoma cells.",
    canonical_form="Rai14 — colocalizes_with — MHC II [on membrane ruffles that form macropinosomes]",
    negatable_form="Rai14 and MHC II do NOT colocalize on membrane ruffles that form macropinosomes.",
    subject_entity=rai14_entity,
    object_entity=mhcii_entity,
    predicate="colocalizes_with",
    direction=None,
    assertion_type=AssertionTypeEnum.CORRELATIONAL,
    causal_type=None,
    scope=Scope(
        species=[human_species],
        tissue=[],
        cell_type=[meljuso_ct],
        disease=[melanoma_disease],
        condition=None,
        developmental_stage=None,
        in_vitro=True,
    ),
    conditions=[],
    hedging=Hedging(
        verbatim_hedge="Rai14-GFP and MHC II colocalized on membrane ruffles, and these ruffles lead to the formation of macropinosomes",
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
    evidence_unit_ids=["e_010"],
    parent_assertion_ids=["a_002"],
    provenance=Provenance(
        source_sentence="Intriguingly, Rai14-GFP and MHC II colocalized on membrane ruffles, and these ruffles lead to the formation of macropinosomes.",
        section_name="Results — Silencing of Rai14 inhibits macropinocytosis",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_007)

# a_008: Rai14 is present on membrane ruffles during macropinosome formation
a_008 = AssertionDraft(
    draft_id="a_008",
    natural_language="Rai14 is present on membrane ruffles during macropinosome formation in antigen-presenting cells.",
    canonical_form="Rai14 — present_at — membrane ruffles during macropinosome formation",
    negatable_form="Rai14 is NOT present on membrane ruffles during macropinosome formation.",
    subject_entity=rai14_entity,
    object_entity=macropinocytosis_entity,
    predicate="participates_in",
    direction=None,
    assertion_type=AssertionTypeEnum.EXISTENCE,
    causal_type=None,
    scope=Scope(
        species=[human_species],
        tissue=[],
        cell_type=[meljuso_ct],
        disease=[melanoma_disease],
        condition=None,
        developmental_stage=None,
        in_vitro=True,
    ),
    conditions=[],
    hedging=Hedging(
        verbatim_hedge="Rai14 localizes to membrane ruffles, where it forms macropinosomes",
        certainty=HedgeLevelEnum.HIGH,
        generalizability=HedgeLevelEnum.MEDIUM,
        causality_hedge=CausalityHedgeEnum.CORRELATIONAL,
    ),
    epistemic_status=EpistemicStatus(
        section=SectionEnum.ABSTRACT,
        function=FunctionEnum.NOVEL_FINDING,
        is_primary=True,
        cited_source=None,
    ),
    evidence_unit_ids=["e_010", "e_021"],
    parent_assertion_ids=["a_002"],
    provenance=Provenance(
        source_sentence="In line with this, we found that Rai14 localizes to membrane ruffles, where it forms macropinosomes.",
        section_name="Abstract",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_008)

# a_009: Rai14 is a positive regulator of macropinocytosis
a_009 = AssertionDraft(
    draft_id="a_009",
    natural_language="Rai14 positively regulates macropinocytosis; its depletion reduces the macropinocytic index and macropinosome area in both human melanoma cells and primary mouse dendritic cells.",
    canonical_form="Rai14 — positively_regulates — macropinocytosis [in human melanoma cells and mouse BMDCs]",
    negatable_form="Rai14 does NOT positively regulate macropinocytosis.",
    subject_entity=rai14_entity,
    object_entity=macropinocytosis_entity,
    predicate="positively_regulates",
    direction=None,
    assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
    causal_type=CausalTypeEnum.CONTRIBUTORY,
    scope=Scope(
        species=[human_species, mouse_species],
        tissue=[],
        cell_type=[meljuso_ct, dc_ct],
        disease=[melanoma_disease],
        condition=None,
        developmental_stage=None,
        in_vitro=True,
    ),
    conditions=[],
    hedging=Hedging(
        verbatim_hedge="we demonstrated that, similar to Ii, Rai14 is a positive regulator of macropinocytosis",
        certainty=HedgeLevelEnum.HIGH,
        generalizability=HedgeLevelEnum.MEDIUM,
        causality_hedge=CausalityHedgeEnum.CAUSAL,
    ),
    epistemic_status=EpistemicStatus(
        section=SectionEnum.ABSTRACT,
        function=FunctionEnum.NOVEL_FINDING,
        is_primary=True,
        cited_source=None,
    ),
    evidence_unit_ids=["e_011", "e_013", "e_014"],
    parent_assertion_ids=[],
    provenance=Provenance(
        source_sentence="we demonstrated that, similar to Ii, Rai14 is a positive regulator of macropinocytosis and a negative regulator of cell migration, two antagonistic processes in antigen-presenting cells.",
        section_name="Abstract",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_009)

# a_010: Rai14 is required for macropinosome closure (PtdIns(4,5)P2 depletion from macropinosome membrane)
a_010 = AssertionDraft(
    draft_id="a_010",
    natural_language="Rai14 is required for macropinosome closure, as Rai14 depletion causes retention of PtdIns(4,5)P2 at the membrane of nascent macropinosomes.",
    canonical_form="Rai14 — is_required_for — macropinosome closure [measured by PtdIns(4,5)P2 depletion from macropinosome membrane]",
    negatable_form="Rai14 is NOT required for macropinosome closure.",
    subject_entity=rai14_entity,
    object_entity=Entity(
        surface_form="macropinosome closure",
        canonical_name="macropinosome closure",
        ontology_id="GO:0044351",
        ontology_source="GO",
        entity_type=EntityTypeEnum.OTHER,
        aliases=["macropinosome sealing", "cup closure"],
    ),
    predicate="is_required_for",
    direction=None,
    assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
    causal_type=CausalTypeEnum.NECESSARY,
    scope=Scope(
        species=[human_species],
        tissue=[],
        cell_type=[meljuso_ct],
        disease=[melanoma_disease],
        condition=None,
        developmental_stage=None,
        in_vitro=True,
    ),
    conditions=[],
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
    evidence_unit_ids=["e_012"],
    parent_assertion_ids=["a_009"],
    provenance=Provenance(
        source_sentence="in cells silenced for Rai14, the percentage of cells retaining PtdIns(4,5)P2 at the membrane of nascent macropinosomes dramatically increased (Figures 4B, C; Video 5), suggesting that Rai14 is required for macropinosome closure.",
        section_name="Results — Silencing of Rai14 inhibits macropinocytosis",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_010)

# a_011: Rai14 is required for macropinocytosis in primary BMDCs
a_011 = AssertionDraft(
    draft_id="a_011",
    natural_language="Rai14 is required for macropinocytosis in primary mouse bone marrow-derived dendritic cells, as its knockdown reduces both the macropinocytic index and macropinosome area by approximately 30%.",
    canonical_form="Rai14 — is_required_for — macropinocytosis [in primary mouse BMDCs]",
    negatable_form="Rai14 is NOT required for macropinocytosis in primary BMDCs.",
    subject_entity=rai14_entity,
    object_entity=macropinocytosis_entity,
    predicate="is_required_for",
    direction=None,
    assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
    causal_type=CausalTypeEnum.NECESSARY,
    scope=Scope(
        species=[mouse_species],
        tissue=[],
        cell_type=[dc_ct],
        disease=[],
        condition=None,
        developmental_stage=None,
        in_vitro=True,
    ),
    conditions=[],
    hedging=Hedging(
        verbatim_hedge="Altogether, these results indicate that Rai14 is required for macropinocytosis",
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
    evidence_unit_ids=["e_013", "e_014"],
    parent_assertion_ids=["a_009"],
    provenance=Provenance(
        source_sentence="Altogether, these results indicate that Rai14 is required for macropinocytosis.",
        section_name="Results — Silencing of Rai14 inhibits macropinocytosis",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_011)

# a_012: Rai14 is required for PAK activation in BMDCs
a_012 = AssertionDraft(
    draft_id="a_012",
    natural_language="Rai14 is required for PAK (p21-activated kinase) activation in primary mouse bone marrow-derived dendritic cells; Rai14 knockdown reduces PAK phosphorylation by approximately 50%.",
    canonical_form="Rai14 — is_required_for — PAK activation [in mouse BMDCs]",
    negatable_form="Rai14 is NOT required for PAK activation in mouse BMDCs.",
    subject_entity=rai14_entity,
    object_entity=pak_entity,
    predicate="promotes_activation_of",
    direction=None,
    assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
    causal_type=CausalTypeEnum.CONTRIBUTORY,
    scope=Scope(
        species=[mouse_species],
        tissue=[],
        cell_type=[dc_ct],
        disease=[],
        condition=None,
        developmental_stage=None,
        in_vitro=True,
    ),
    conditions=[],
    hedging=Hedging(
        verbatim_hedge="quantification of the levels of PAK phosphorylation revealed a decrease of almost 50% in BMDCs silenced for Rai14 compared to control cells",
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
    parent_assertion_ids=["a_009"],
    provenance=Provenance(
        source_sentence="Interestingly, quantification of the levels of PAK phosphorylation revealed a decrease of almost 50% in BMDCs silenced for Rai14 compared to control cells.",
        section_name="Results — Silencing of Rai14 inhibits macropinocytosis",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_012)

# a_013: Rai14 negatively regulates BMDC migration speed and direction changes
a_013 = AssertionDraft(
    draft_id="a_013",
    natural_language="Rai14 negatively regulates dendritic cell migration speed and direction changes; Rai14 depletion increases BMDC mean speed from 5.4 to 7.1 µm/min and reduces speed fluctuations.",
    canonical_form="Rai14 — negatively_regulates — cell migration speed [in mouse BMDCs in microchannels]",
    negatable_form="Rai14 does NOT negatively regulate cell migration speed in BMDCs.",
    subject_entity=rai14_entity,
    object_entity=migration_entity,
    predicate="negatively_regulates",
    direction=None,
    assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
    causal_type=CausalTypeEnum.CONTRIBUTORY,
    scope=Scope(
        species=[mouse_species],
        tissue=[],
        cell_type=[dc_ct],
        disease=[],
        condition="confined in microchannels (5x8 µm)",
        developmental_stage=None,
        in_vitro=True,
    ),
    conditions=[
        Condition(
            condition_type=ConditionTypeEnum.BIOLOGICAL_CONTEXT,
            value="BMDCs confined in microfabricated 5x8 µm channels",
        )
    ],
    hedging=Hedging(
        verbatim_hedge="we demonstrated that, similar to Ii, Rai14 is a positive regulator of macropinocytosis and a negative regulator of cell migration",
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
    parent_assertion_ids=[],
    provenance=Provenance(
        source_sentence="The analysis revealed that BMDCs silenced for Rai14 move faster (mean speed 7.1 µm/min) than control BMDCs (mean speed 5.4 µm/min).",
        section_name="Results — Rai14 negatively regulates BMDC migration",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_013)

# a_014: Endogenous Rai14 and Ii colocalize on macropinosome-like vesicles in primary BMDCs
a_014 = AssertionDraft(
    draft_id="a_014",
    natural_language="Endogenous Rai14 and endogenous Ii colocalize on large macropinosome-like vesicles in primary mouse dendritic cells.",
    canonical_form="Rai14 (endogenous) — colocalizes_with — Ii (endogenous) [on macropinosome-like vesicles in primary BMDCs]",
    negatable_form="Endogenous Rai14 and Ii do NOT colocalize on macropinosome-like vesicles in primary BMDCs.",
    subject_entity=rai14_entity,
    object_entity=ii_entity,
    predicate="colocalizes_with",
    direction=None,
    assertion_type=AssertionTypeEnum.CORRELATIONAL,
    causal_type=None,
    scope=Scope(
        species=[mouse_species],
        tissue=[],
        cell_type=[dc_ct],
        disease=[],
        condition=None,
        developmental_stage=None,
        in_vitro=True,
    ),
    conditions=[],
    hedging=Hedging(
        verbatim_hedge="in line with our hypothesis, the two proteins colocalize on large, macropinosome-like vesicles",
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
    evidence_unit_ids=["e_017"],
    parent_assertion_ids=["a_001", "a_003"],
    provenance=Provenance(
        source_sentence="Our results show that, in line with our hypothesis, the two proteins colocalize on large, macropinosome-like vesicles.",
        section_name="Results — Rai14 negatively regulates BMDC migration",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_014)

# a_015: Rai14 interacts with myosin II
a_015 = AssertionDraft(
    draft_id="a_015",
    natural_language="Rai14 physically interacts with myosin II in dendritic cells and can be co-precipitated with the myosin II heavy chain tail domain from human melanoma cell extracts.",
    canonical_form="Rai14 — interacts_with — myosin II [in primary DCs and human melanoma cells]",
    negatable_form="Rai14 does NOT interact with myosin II.",
    subject_entity=rai14_entity,
    object_entity=myosin2_entity,
    predicate="interacts_with",
    direction=None,
    assertion_type=AssertionTypeEnum.CORRELATIONAL,
    causal_type=None,
    scope=Scope(
        species=[human_species, mouse_species],
        tissue=[],
        cell_type=[meljuso_ct, dc_ct],
        disease=[melanoma_disease],
        condition=None,
        developmental_stage=None,
        in_vitro=True,
    ),
    conditions=[],
    hedging=Hedging(
        verbatim_hedge="Here, we show that Rai14 also binds to myosin II",
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
    evidence_unit_ids=["e_018", "e_019"],
    parent_assertion_ids=[],
    provenance=Provenance(
        source_sentence="Here, we show that Rai14 also binds to myosin II, suggesting that Ii, myosin II, and Rai14 work together to coordinate macropinocytosis and cell motility.",
        section_name="Abstract",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_015)

# a_016: Rai14 bridges Ii to myosin II (Rai14 is required for Ii-myosin II interaction)
a_016 = AssertionDraft(
    draft_id="a_016",
    natural_language="Rai14 bridges Invariant chain to myosin II; depletion of Rai14 prevents the interaction between Ii and the myosin II heavy chain tail.",
    canonical_form="Rai14 — bridges — Invariant chain to myosin II [Rai14 depletion abolishes Ii-myosin II co-pulldown]",
    negatable_form="Rai14 does NOT bridge Invariant chain to myosin II.",
    subject_entity=rai14_entity,
    object_entity=myosin2_entity,
    predicate="bridges_interaction_between",
    direction=None,
    assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
    causal_type=CausalTypeEnum.NECESSARY,
    scope=Scope(
        species=[human_species],
        tissue=[],
        cell_type=[meljuso_ct],
        disease=[melanoma_disease],
        condition=None,
        developmental_stage=None,
        in_vitro=True,
    ),
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
    parent_assertion_ids=["a_001", "a_015"],
    provenance=Provenance(
        source_sentence="purified GST-tagged myosin II heavy chain tail specifically precipitates both Ii and Rai14 from total MelJuSo cell extracts, it is unable to pull down Ii from cells silenced for Rai14, suggesting that Rai14 bridges Ii to myosin II.",
        section_name="Results — Rai14 negatively regulates BMDC migration",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_016)

# a_017: Rai14 is required for myosin II recruitment to the front of migrating DCs
a_017 = AssertionDraft(
    draft_id="a_017",
    natural_language="Rai14 is required for myosin II recruitment to the leading edge of migrating dendritic cells; Rai14 depletion decreases myosin II accumulation at the cell front.",
    canonical_form="Rai14 — is_required_for — myosin II recruitment to leading edge [in migrating BMDCs in microchannels]",
    negatable_form="Rai14 is NOT required for myosin II recruitment to the leading edge of migrating DCs.",
    subject_entity=rai14_entity,
    object_entity=myosin2_entity,
    predicate="promotes_localization_of",
    direction=None,
    assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
    causal_type=CausalTypeEnum.CONTRIBUTORY,
    scope=Scope(
        species=[mouse_species],
        tissue=[],
        cell_type=[dc_ct],
        disease=[],
        condition="migrating in 5x8 µm microchannels",
        developmental_stage=None,
        in_vitro=True,
    ),
    conditions=[
        Condition(
            condition_type=ConditionTypeEnum.BIOLOGICAL_CONTEXT,
            value="BMDCs migrating in microfabricated 5x8 µm channels",
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
    evidence_unit_ids=["e_020"],
    parent_assertion_ids=["a_013", "a_015"],
    provenance=Provenance(
        source_sentence="Silencing of Rai14 indeed decreased the amount of myosin II at the front of BMDCs migrating into microfabricated channels, as revealed by density maps of the mean myosin II distribution.",
        section_name="Results — Rai14 negatively regulates BMDC migration",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_017)

# -- INTERPRETATION / DISCUSSION CLAIMS --

# a_018: Rai14 as scaffold protein linking Ii to myosin II and actomyosin network (interpretation)
a_018 = AssertionDraft(
    draft_id="a_018",
    natural_language="Rai14 may act as a scaffold protein that links Invariant chain to myosin II and the actomyosin network to coordinate macropinocytosis and cell migration in antigen-presenting cells.",
    canonical_form="Rai14 — acts_as_scaffold_linking — Invariant chain to myosin II and actomyosin network",
    negatable_form="Rai14 does NOT act as a scaffold protein linking Ii to myosin II and the actomyosin network.",
    subject_entity=rai14_entity,
    object_entity=myosin2_entity,
    predicate="acts_as_scaffold_linking",
    direction=None,
    assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
    causal_type=CausalTypeEnum.MODULATORY,
    scope=Scope(
        species=[human_species, mouse_species],
        tissue=[],
        cell_type=[dc_ct, meljuso_ct],
        disease=[],
        condition=None,
        developmental_stage=None,
        in_vitro=True,
    ),
    conditions=[],
    hedging=Hedging(
        verbatim_hedge="Rai14 is likely a scaffold protein that links Ii with myosin II and the actomyosin network",
        certainty=HedgeLevelEnum.LOW,
        generalizability=HedgeLevelEnum.MEDIUM,
        causality_hedge=CausalityHedgeEnum.UNCLEAR,
    ),
    epistemic_status=EpistemicStatus(
        section=SectionEnum.DISCUSSION,
        function=FunctionEnum.INTERPRETATION,
        is_primary=True,
        cited_source=None,
    ),
    evidence_unit_ids=["e_002", "e_003", "e_018", "e_019"],
    parent_assertion_ids=["a_001", "a_015", "a_016"],
    provenance=Provenance(
        source_sentence="Therefore, Rai14 is likely a scaffold protein that links Ii with myosin II and the actomyosin network.",
        section_name="Discussion",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_018)

# a_019: Rai14 depletion phenocopies Ii and myosin II loss (interpretation)
a_019 = AssertionDraft(
    draft_id="a_019",
    natural_language="Rai14 depletion phenocopies loss of Invariant chain or myosin II in dendritic cells, producing smaller macropinosomes, faster migration, and fewer speed fluctuations.",
    canonical_form="Rai14 depletion — phenocopies — Ii or myosin II loss [producing similar defects in macropinocytosis and migration in BMDCs]",
    negatable_form="Rai14 depletion does NOT phenocopy loss of Ii or myosin II in dendritic cells.",
    subject_entity=rai14_entity,
    object_entity=ii_entity,
    predicate="phenocopies_loss_of",
    direction=None,
    assertion_type=AssertionTypeEnum.COMPARATIVE,
    causal_type=None,
    scope=Scope(
        species=[mouse_species],
        tissue=[],
        cell_type=[dc_ct],
        disease=[],
        condition=None,
        developmental_stage=None,
        in_vitro=True,
    ),
    conditions=[],
    hedging=Hedging(
        verbatim_hedge="these results are similar to those reported for BMDCs lacking Ii or myosin II",
        certainty=HedgeLevelEnum.HIGH,
        generalizability=HedgeLevelEnum.MEDIUM,
        causality_hedge=CausalityHedgeEnum.CORRELATIONAL,
    ),
    epistemic_status=EpistemicStatus(
        section=SectionEnum.DISCUSSION,
        function=FunctionEnum.INTERPRETATION,
        is_primary=True,
        cited_source=None,
    ),
    evidence_unit_ids=["e_013", "e_016"],
    parent_assertion_ids=["a_009", "a_013"],
    provenance=Provenance(
        source_sentence="these results are similar to those reported for BMDCs lacking Ii or myosin II. Indeed, similarly to BMDCs silenced for Rai14, myosin II or Ii knockout BMDCs display smaller macropinosomes.",
        section_name="Discussion",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_019)

# a_020: Rai14 may be involved in membrane fission (hypothesis/speculation)
a_020 = AssertionDraft(
    draft_id="a_020",
    natural_language="Rai14 may be involved in membrane fission on endosomes, analogous to the N-Ank family member ANKHD1 which induces fission on early endosomes.",
    canonical_form="Rai14 (N-Ank family member) — may_participate_in — membrane fission [on endosomes]",
    negatable_form="Rai14 does NOT participate in membrane fission on endosomes.",
    subject_entity=rai14_entity,
    object_entity=Entity(
        surface_form="endosomal membrane fission",
        canonical_name="membrane fission",
        ontology_id="GO:0090148",
        ontology_source="GO",
        entity_type=EntityTypeEnum.OTHER,
        aliases=["membrane scission", "vesicle fission"],
    ),
    predicate="may_participate_in",
    direction=None,
    assertion_type=AssertionTypeEnum.EXISTENCE,
    causal_type=None,
    scope=Scope(
        species=[human_species],
        tissue=[],
        cell_type=[meljuso_ct],
        disease=[],
        condition=None,
        developmental_stage=None,
        in_vitro=True,
    ),
    conditions=[],
    hedging=Hedging(
        verbatim_hedge="it is tempting to speculate that Rai14 could also be involved in membrane fission",
        certainty=HedgeLevelEnum.LOW,
        generalizability=HedgeLevelEnum.LOW,
        causality_hedge=CausalityHedgeEnum.UNCLEAR,
    ),
    epistemic_status=EpistemicStatus(
        section=SectionEnum.DISCUSSION,
        function=FunctionEnum.HYPOTHESIS,
        is_primary=True,
        cited_source=None,
    ),
    evidence_unit_ids=["e_004", "e_005"],
    parent_assertion_ids=[],
    provenance=Provenance(
        source_sentence="As we also observed Rai14 at the sites of vesicle scission from membrane tubules or enriched on endosomal domains (Figures 1C, D), it is tempting to speculate that Rai14 could also be involved in membrane fission.",
        section_name="Discussion",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_020)

# a_021: Rai14 links membrane shaping and actin in macropinocytosis (interpretation)
a_021 = AssertionDraft(
    draft_id="a_021",
    natural_language="Rai14 may act as a link between membrane shaping and actin organization in macropinocytosis, consistent with its localization to membrane ruffles and closing macropinosomes and its membership in the N-Ank protein family.",
    canonical_form="Rai14 — links — membrane shaping to actin organization [during macropinocytosis]",
    negatable_form="Rai14 does NOT link membrane shaping to actin organization during macropinocytosis.",
    subject_entity=rai14_entity,
    object_entity=Entity(
        surface_form="actin organization",
        canonical_name="actin cytoskeleton organization",
        ontology_id="GO:0030036",
        ontology_source="GO",
        entity_type=EntityTypeEnum.OTHER,
        aliases=["actin remodeling", "actin polymerization"],
    ),
    predicate="links",
    direction=None,
    assertion_type=AssertionTypeEnum.EXISTENCE,
    causal_type=None,
    scope=Scope(
        species=[human_species],
        tissue=[],
        cell_type=[meljuso_ct],
        disease=[],
        condition=None,
        developmental_stage=None,
        in_vitro=True,
    ),
    conditions=[],
    hedging=Hedging(
        verbatim_hedge="it is therefore possible that Rai14 acts as a link between membrane shaping and actin in macropinocytosis",
        certainty=HedgeLevelEnum.LOW,
        generalizability=HedgeLevelEnum.LOW,
        causality_hedge=CausalityHedgeEnum.UNCLEAR,
    ),
    epistemic_status=EpistemicStatus(
        section=SectionEnum.DISCUSSION,
        function=FunctionEnum.INTERPRETATION,
        is_primary=True,
        cited_source=None,
    ),
    evidence_unit_ids=["e_021", "e_010"],
    parent_assertion_ids=["a_009"],
    provenance=Provenance(
        source_sentence="It is therefore possible that Rai14 acts as a link between membrane shaping and actin in macropinocytosis.",
        section_name="Discussion",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_021)

# a_022: Ii, Rai14, and myosin II work together to coordinate antigen uptake and migration
a_022 = AssertionDraft(
    draft_id="a_022",
    natural_language="Rai14, Invariant chain, and myosin II work together as a functional complex to coordinate macropinocytosis, intracellular membrane traffic, and cell migration in antigen-presenting cells.",
    canonical_form="Rai14 — works_in_complex_with — Invariant chain and myosin II [coordinating macropinocytosis and migration in APCs]",
    negatable_form="Rai14, Ii, and myosin II do NOT function as a complex to coordinate macropinocytosis and migration in APCs.",
    subject_entity=rai14_entity,
    object_entity=ii_entity,
    predicate="works_in_complex_with",
    direction=None,
    assertion_type=AssertionTypeEnum.MECHANISTIC_CAUSAL,
    causal_type=CausalTypeEnum.MODULATORY,
    scope=Scope(
        species=[human_species, mouse_species],
        tissue=[],
        cell_type=[dc_ct, meljuso_ct],
        disease=[],
        condition=None,
        developmental_stage=None,
        in_vitro=True,
    ),
    conditions=[],
    hedging=Hedging(
        verbatim_hedge="Altogether, our results indicate that Rai14, Ii, and myosin II work together to coordinate antigen uptake, intracellular membrane traffic, and, hence, cell migration in antigen-presenting cells",
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
    evidence_unit_ids=["e_002", "e_003", "e_016", "e_018", "e_019"],
    parent_assertion_ids=["a_001", "a_015", "a_016"],
    provenance=Provenance(
        source_sentence="Altogether, our results indicate that Rai14, Ii, and myosin II work together to coordinate antigen uptake, intracellular membrane traffic, and, hence, cell migration in antigen-presenting cells.",
        section_name="Results — Rai14 negatively regulates BMDC migration",
        char_offset_start=None,
        char_offset_end=None,
    ),
)
assertion_drafts.append(a_022)

# ---------------------------------------------------------------------------
# Extraction Metadata
# ---------------------------------------------------------------------------

extraction_metadata = ExtractionMetadata(
    extraction_model="claude-sonnet-4-6",
    extraction_version="0.1.0",
    extraction_timestamp="2026-03-24T00:00:00Z",
    paper_char_count=None,
    extraction_duration_seconds=None,
)

# ---------------------------------------------------------------------------
# Build ExtractionResult
# ---------------------------------------------------------------------------

result = ExtractionResult(
    paper_provenance=paper_provenance,
    evidence_units=evidence_units,
    assertion_drafts=assertion_drafts,
    extraction_metadata=extraction_metadata,
)

# ---------------------------------------------------------------------------
# Write output JSON
# ---------------------------------------------------------------------------

output_path = (
    "/Users/mst36/Desktop/Projects/Science/Autonomous Science/outputs/extraction_test_sonnet.json"
)

json_str = result.model_dump_json(indent=2)
with open(output_path, "w", encoding="utf-8") as fh:
    fh.write(json_str)

print(f"JSON written to: {output_path}")

# ---------------------------------------------------------------------------
# Validation summary
# ---------------------------------------------------------------------------

total_eu = len(result.evidence_units)
total_ad = len(result.assertion_drafts)

novel = [a for a in result.assertion_drafts if a.epistemic_status.is_primary]
background = [a for a in result.assertion_drafts if not a.epistemic_status.is_primary]

functions = {}
for a in result.assertion_drafts:
    fn = str(a.epistemic_status.function)
    functions[fn] = functions.get(fn, 0) + 1

print("\n=== VALIDATION SUMMARY ===")
print(f"Total evidence units:    {total_eu}")
print(f"Total assertion drafts:  {total_ad}")
print(f"  Novel (is_primary=True):      {len(novel)}")
print(f"  Background (is_primary=False): {len(background)}")
print("\nBy epistemic function:")
for fn, count in sorted(functions.items()):
    print(f"  {fn}: {count}")

# Check all evidence_unit_ids referenced in assertions exist
eu_ids = {e.evidence_id for e in result.evidence_units}
ad_ids = {a.draft_id for a in result.assertion_drafts}

dangling_eu_refs = []
for a in result.assertion_drafts:
    for eid in a.evidence_unit_ids:
        if eid not in eu_ids:
            dangling_eu_refs.append((a.draft_id, eid))

dangling_ad_refs = []
for e in result.evidence_units:
    for aid in e.assertion_draft_ids:
        if aid not in ad_ids:
            dangling_ad_refs.append((e.evidence_id, aid))

if dangling_eu_refs:
    print(f"\nWARNING: {len(dangling_eu_refs)} dangling evidence_unit_id refs: {dangling_eu_refs}")
else:
    print("\nAll evidence_unit_id references are valid.")

if dangling_ad_refs:
    print(f"WARNING: {len(dangling_ad_refs)} dangling assertion_draft_id refs: {dangling_ad_refs}")
else:
    print("All assertion_draft_id references are valid.")

print(f"\nJSON file size: {len(json_str):,} bytes")
print("Extraction complete.")
