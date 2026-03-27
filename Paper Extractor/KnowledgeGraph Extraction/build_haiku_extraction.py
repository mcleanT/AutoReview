#!/usr/bin/env python3
"""
Build Mycelium extraction for Rai14 paper (Lobos Patorniti et al., 2023)
programmatically using Pydantic models.

This avoids output token limits by constructing the entire ExtractionResult
in Python, then serializing to JSON.
"""

import json

# Add src to path
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mycelium.extraction_schema import (
    AssertionDraft,
    Author,
    CitationContext,
    Condition,
    Entity,
    EpistemicStatus,
    EvidenceUnit,
    Experiment,
    ExtractionMetadata,
    ExtractionResult,
    Hedging,
    MethodologicalTags,
    OntologyTerm,
    PaperProvenance,
    Provenance,
    Results,
    Scope,
)


def build_extraction():
    """Build the complete extraction result."""

    # =========================================================================
    # PAPER PROVENANCE
    # =========================================================================

    paper_provenance = PaperProvenance(
        doi="10.3389/fimmu.2023.1182180",
        pmid="37476277",
        title="Rai14 is a novel interactor of Invariant chain that regulates macropinocytosis",
        authors=[
            Author(
                name="Lobos Patorniti, Natacha",
                orcid=None,
                affiliations=["Department of Biosciences, University of Oslo, Oslo, Norway"],
                role="first_author",
            ),
            Author(
                name="Zulkeﬂi, Khalisah Liyana",
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
        funding_sources=[],
        conflicts_of_interest=None,
        data_availability=None,
    )

    # =========================================================================
    # EVIDENCE UNITS
    # =========================================================================

    evidence_units = [
        # e_001: Ii-Rai14 interaction by co-IP (endogenous)
        EvidenceUnit(
            evidence_id="e_001",
            assertion_draft_ids=["a_001"],
            evidence_direction="supports",
            evidence_strength="direct_experimental",
            experiment=Experiment(
                description="Co-immunoprecipitation of endogenous Rai14 with Invariant chain in MelJuSo cells. Anti-Ii antibody (MB741) used for pulldown; Western blot detection with anti-Rai14 and anti-Ii antibodies.",
                model_system="MelJuSo human melanoma cell line",
                organism="Homo sapiens",
                organism_strain=None,
                perturbation_type="none",
                perturbation_target=None,
                perturbation_method=None,
                readout="Co-immunoprecipitation band intensity by Western blot",
                control_description="IgG2a isotype control immunoprecipitation",
            ),
            results=Results(
                result_direction="positive",
                effect_description="Rai14 specifically co-immunoprecipitated with endogenous Ii",
                effect_size=None,
                statistical_test=None,
                p_value=None,
                confidence_interval=None,
                sample_size="n=3 independent experiments",
                key_figure="Figure 1A",
            ),
            methodological_tags=MethodologicalTags(
                approach_category="biochemical_assay",
                assay_types=["co_immunoprecipitation", "Western_blot"],
                blinding_reported=None,
                randomization_reported=None,
            ),
            limitations_stated_by_authors=[
                "Co-IP does not distinguish direct from indirect binding"
            ],
            source_section="results",
            source_text_span=None,
        ),
        # e_002: Ii-Rai14 interaction reverse co-IP
        EvidenceUnit(
            evidence_id="e_002",
            assertion_draft_ids=["a_001"],
            evidence_direction="supports",
            evidence_strength="direct_experimental",
            experiment=Experiment(
                description="Reverse co-immunoprecipitation: endogenous Ii co-pulled with anti-Rai14 antibody in MelJuSo cells, detected by Western blot",
                model_system="MelJuSo human melanoma cell line",
                organism="Homo sapiens",
                organism_strain=None,
                perturbation_type="none",
                perturbation_target=None,
                perturbation_method=None,
                readout="Co-immunoprecipitation band intensity by Western blot",
                control_description="IgG isotype control immunoprecipitation",
            ),
            results=Results(
                result_direction="positive",
                effect_description="Endogenous Ii specifically co-immunoprecipitated with anti-Rai14 pulldown",
                effect_size=None,
                statistical_test=None,
                p_value=None,
                confidence_interval=None,
                sample_size="n=3 independent experiments",
                key_figure="Figure 1B",
            ),
            methodological_tags=MethodologicalTags(
                approach_category="biochemical_assay",
                assay_types=["co_immunoprecipitation", "Western_blot"],
                blinding_reported=None,
                randomization_reported=None,
            ),
            limitations_stated_by_authors=[],
            source_section="results",
            source_text_span=None,
        ),
        # e_003: Rai14 at membrane ruffles and nascent macropinosomes (live imaging)
        EvidenceUnit(
            evidence_id="e_003",
            assertion_draft_ids=["a_002"],
            evidence_direction="supports",
            evidence_strength="observational_uncontrolled",
            experiment=Experiment(
                description="Live cell imaging of MelJuSo cells co-transfected with Ii p33 and Rai14-GFP; Ii labeled with Alexa Fluor 555-conjugated antibody (M-B741). Temporal colocalization monitored until vesicle pinching.",
                model_system="MelJuSo human melanoma cell line",
                organism="Homo sapiens",
                organism_strain=None,
                perturbation_type="none",
                perturbation_target=None,
                perturbation_method=None,
                readout="Colocalization of Rai14-GFP and Ii on membrane ruffles and forming macropinosomes (Manders' colocalization coefficient)",
                control_description="Cells with GFP-only control",
            ),
            results=Results(
                result_direction="positive",
                effect_description="Rai14 and Ii colocalize on membrane ruffles, nascent vesicles, and forming macropinosomes; colocalization persists until vesicle pinching",
                effect_size=None,
                statistical_test=None,
                p_value=None,
                confidence_interval=None,
                sample_size=None,
                key_figure="Figure 1C, D; Video 2",
            ),
            methodological_tags=MethodologicalTags(
                approach_category="imaging",
                assay_types=["live_cell_imaging", "immunofluorescence"],
                blinding_reported=None,
                randomization_reported=None,
            ),
            limitations_stated_by_authors=[],
            source_section="results",
            source_text_span=None,
        ),
        # e_004: Rai14-MHC II complex interaction
        EvidenceUnit(
            evidence_id="e_004",
            assertion_draft_ids=["a_003"],
            evidence_direction="supports",
            evidence_strength="direct_experimental",
            experiment=Experiment(
                description="GFP-TRAP immunoprecipitation using lysates from MelJuSo cells stably expressing HLA-DR1b-GFP (MHC II). Detection of co-precipitated Ii and Rai14 by Western blot.",
                model_system="MelJuSo human melanoma cell line stably expressing HLA-DR1b-GFP",
                organism="Homo sapiens",
                organism_strain=None,
                perturbation_type="none",
                perturbation_target=None,
                perturbation_method=None,
                readout="Co-immunoprecipitation of Ii and Rai14 with HLA-DR1b-GFP by Western blot",
                control_description="Cells expressing GFP alone; GFP-TRAP with GFP-only lysates",
            ),
            results=Results(
                result_direction="positive",
                effect_description="Both Ii and Rai14 co-immunoprecipitate with HLA-DR1b-GFP, indicating Rai14 interaction with Ii-MHC II complex",
                effect_size=None,
                statistical_test=None,
                p_value=None,
                confidence_interval=None,
                sample_size=None,
                key_figure="Figure 2A",
            ),
            methodological_tags=MethodologicalTags(
                approach_category="biochemical_assay",
                assay_types=["GFP_TRAP", "Western_blot"],
                blinding_reported=None,
                randomization_reported=None,
            ),
            limitations_stated_by_authors=[],
            source_section="results",
            source_text_span=None,
        ),
        # e_005: Rai14 depletion increases MHC II at plasma membrane
        EvidenceUnit(
            evidence_id="e_005",
            assertion_draft_ids=["a_004"],
            evidence_direction="supports",
            evidence_strength="direct_experimental",
            experiment=Experiment(
                description="siRNA knockdown of Rai14 in MelJuSo cells stably expressing HLA-DR1b-GFP. Quantification of HLA-DR1b-GFP plasma membrane localization by confocal imaging (percentage of total).",
                model_system="MelJuSo human melanoma cell line stably expressing HLA-DR1b-GFP",
                organism="Homo sapiens",
                organism_strain=None,
                perturbation_type="genetic_loss_of_function",
                perturbation_target="RAI14",
                perturbation_method="siRNA",
                readout="Percentage of HLA-DR1b-GFP at plasma membrane vs. intracellular compartments",
                control_description="siRNA control (non-targeting)",
            ),
            results=Results(
                result_direction="positive",
                effect_description="HLA-DR1b-GFP plasma membrane localization increased 1.5-fold upon Rai14 knockdown",
                effect_size="1.5-fold increase",
                statistical_test="two-tailed unpaired Student's t-test",
                p_value="p < 0.01",
                confidence_interval=None,
                sample_size="n=3 independent experiments; n > 45 cells per condition",
                key_figure="Figure 2B–E",
            ),
            methodological_tags=MethodologicalTags(
                approach_category="cell_biology",
                assay_types=["confocal_microscopy", "siRNA_knockdown"],
                blinding_reported=None,
                randomization_reported=None,
            ),
            limitations_stated_by_authors=[],
            source_section="results",
            source_text_span=None,
        ),
        # e_006: Rai14 knockdown impairs MHC II internalization kinetics
        EvidenceUnit(
            evidence_id="e_006",
            assertion_draft_ids=["a_005"],
            evidence_direction="supports",
            evidence_strength="direct_experimental",
            experiment=Experiment(
                description="MHC II uptake assay: MelJuSo cells with Rai14 knockdown incubated with anti-MHC II antibody (conjugated to Alexa Fluor 647) for 45 min, fixed at 0, 30, and 60 min time points. Quantification of MHC II-positive endosomes.",
                model_system="MelJuSo human melanoma cell line",
                organism="Homo sapiens",
                organism_strain=None,
                perturbation_type="genetic_loss_of_function",
                perturbation_target="RAI14",
                perturbation_method="siRNA (two independent siRNAs)",
                readout="Percentage of cell area occupied by MHC II-positive endosomes over time",
                control_description="siRNA control (non-targeting)",
            ),
            results=Results(
                result_direction="positive",
                effect_description="Rai14 knockdown cells showed ~40% decrease in MHC II in endosomes 60 minutes after uptake",
                effect_size="~40% decrease",
                statistical_test="two-tailed unpaired Student's t-test",
                p_value="p < 0.05 (t=60min)",
                confidence_interval=None,
                sample_size="n=3 independent experiments; n > 45 cells per condition",
                key_figure="Figure 2G–I",
            ),
            methodological_tags=MethodologicalTags(
                approach_category="cell_biology",
                assay_types=["live_cell_imaging", "siRNA_knockdown"],
                blinding_reported=None,
                randomization_reported=None,
            ),
            limitations_stated_by_authors=[],
            source_section="results",
            source_text_span=None,
        ),
        # e_007: Rai14 colocalizes with MHC II at membrane ruffles and macropinosomes
        EvidenceUnit(
            evidence_id="e_007",
            assertion_draft_ids=["a_002"],
            evidence_direction="supports",
            evidence_strength="observational_uncontrolled",
            experiment=Experiment(
                description="Live cell imaging of MelJuSo cells transfected with Rai14-GFP; anti-HLA-DR antibody (L243, Alexa 647) added during imaging. Temporal colocalization quantified by Manders' coefficient.",
                model_system="MelJuSo human melanoma cell line",
                organism="Homo sapiens",
                organism_strain=None,
                perturbation_type="none",
                perturbation_target=None,
                perturbation_method=None,
                readout="Colocalization of Rai14-GFP and HLA-DR on membrane ruffles and macropinosomes",
                control_description=None,
            ),
            results=Results(
                result_direction="positive",
                effect_description="Rai14 and MHC II colocalize on membrane ruffles that close into macropinosomes",
                effect_size=None,
                statistical_test=None,
                p_value=None,
                confidence_interval=None,
                sample_size="n=4 independent experiments (n=25)",
                key_figure="Figure 3A; Video 3",
            ),
            methodological_tags=MethodologicalTags(
                approach_category="imaging",
                assay_types=["live_cell_imaging", "immunofluorescence"],
                blinding_reported=None,
                randomization_reported=None,
            ),
            limitations_stated_by_authors=[],
            source_section="results",
            source_text_span=None,
        ),
        # e_008: Rai14 knockdown reduces macropinocytic index in MelJuSo
        EvidenceUnit(
            evidence_id="e_008",
            assertion_draft_ids=["a_006"],
            evidence_direction="supports",
            evidence_strength="direct_experimental",
            experiment=Experiment(
                description="Dextran uptake assay in HLA-DR1b-GFP MelJuSo cells: 70 kDa Alexa Fluor 555-conjugated dextran for 30 min. Live imaging quantifies macropinocytic index (% cell area with dextran) and macropinosome size. Rai14 rescue by re-introduction of Rai14-GFP.",
                model_system="MelJuSo human melanoma cell line stably expressing HLA-DR1b-GFP",
                organism="Homo sapiens",
                organism_strain=None,
                perturbation_type="genetic_loss_of_function",
                perturbation_target="RAI14",
                perturbation_method="siRNA",
                readout="Macropinocytic index (% cell area occupied by dextran-positive vesicles); macropinosome area (um²)",
                control_description="siRNA control; Rai14 rescue with Rai14-GFP re-introduction",
            ),
            results=Results(
                result_direction="positive",
                effect_description="Macropinocytic index decreased by ~50%, driven primarily by reduced macropinosome area; rescue restored phenotype",
                effect_size="~50% decrease in macropinocytic index",
                statistical_test="two-tailed unpaired Student's t-test",
                p_value="p < 0.05",
                confidence_interval=None,
                sample_size="n=3 independent experiments (n=60 cells per condition)",
                key_figure="Figure 3B–D",
            ),
            methodological_tags=MethodologicalTags(
                approach_category="cell_biology",
                assay_types=["live_cell_imaging", "siRNA_knockdown", "rescue_experiment"],
                blinding_reported=None,
                randomization_reported=None,
            ),
            limitations_stated_by_authors=[],
            source_section="results",
            source_text_span=None,
        ),
        # e_009: Rai14 required for PtdIns(4,5)P2 depletion during macropinosome closure
        EvidenceUnit(
            evidence_id="e_009",
            assertion_draft_ids=["a_007"],
            evidence_direction="supports",
            evidence_strength="direct_experimental",
            experiment=Experiment(
                description="Live cell imaging of MelJuSo cells transfected with PH-PLCδ-GFP (biosensor for PtdIns(4,5)P2) ± Rai14 knockdown. Anti-HLA-DR antibody (L243, Alexa 647) added during imaging. Quantification of % cells retaining PtdIns(4,5)P2 at macropinocytic cup membrane.",
                model_system="MelJuSo human melanoma cell line",
                organism="Homo sapiens",
                organism_strain=None,
                perturbation_type="genetic_loss_of_function",
                perturbation_target="RAI14",
                perturbation_method="siRNA",
                readout="% cells retaining PtdIns(4,5)P2 at membrane of nascent macropinosomes",
                control_description="siRNA control",
            ),
            results=Results(
                result_direction="positive",
                effect_description="Rai14 knockdown dramatically increased % cells retaining PtdIns(4,5)P2 at macropinocytic cup, indicating defect in macropinosome closure",
                effect_size=None,
                statistical_test="two-tailed unpaired Student's t-test",
                p_value="p < 0.01",
                confidence_interval=None,
                sample_size="n=3 independent experiments (n ≥32 cells per condition)",
                key_figure="Figure 4A–C; Videos 4–5",
            ),
            methodological_tags=MethodologicalTags(
                approach_category="imaging",
                assay_types=["live_cell_imaging", "lipid_biosensor"],
                blinding_reported=None,
                randomization_reported=None,
            ),
            limitations_stated_by_authors=[],
            source_section="results",
            source_text_span=None,
        ),
        # e_010: Rai14 knockdown reduces macropinocytosis in BMDCs
        EvidenceUnit(
            evidence_id="e_010",
            assertion_draft_ids=["a_006"],
            evidence_direction="supports",
            evidence_strength="direct_experimental",
            experiment=Experiment(
                description="Dextran uptake assay in BMDCs: 70 kDa Alexa Fluor 555-conjugated dextran for 15 min. Quantification of macropinocytic index and macropinosome area by live imaging.",
                model_system="Bone marrow-derived dendritic cells (BMDCs) from mice",
                organism="Mus musculus",
                organism_strain=None,
                perturbation_type="genetic_loss_of_function",
                perturbation_target="RAI14",
                perturbation_method="siRNA",
                readout="Macropinocytic index and macropinosome area",
                control_description="siRNA control",
            ),
            results=Results(
                result_direction="positive",
                effect_description="Macropinocytic index decreased by 30% in Rai14 knockdown BMDCs; macropinosome area decreased ~30%",
                effect_size="30% decrease in macropinocytic index; 30% decrease in macropinosome area",
                statistical_test="two-tailed paired Student's t-test",
                p_value="p < 0.05",
                confidence_interval=None,
                sample_size="n=3 independent experiments (n=60 cells per condition)",
                key_figure="Figure 5A–C",
            ),
            methodological_tags=MethodologicalTags(
                approach_category="cell_biology",
                assay_types=["live_cell_imaging", "siRNA_knockdown"],
                blinding_reported=None,
                randomization_reported=None,
            ),
            limitations_stated_by_authors=[],
            source_section="results",
            source_text_span=None,
        ),
        # e_011: Rai14 knockdown reduces macropinosome-containing BMDCs in microchannels
        EvidenceUnit(
            evidence_id="e_011",
            assertion_draft_ids=["a_006"],
            evidence_direction="supports",
            evidence_strength="direct_experimental",
            experiment=Experiment(
                description="BMDCs in 5×8 µm microfabricated channels. After 16h confinement, channels filled with 10 kDa Alexa Fluor 555-conjugated dextran; cells imaged 50 min later. Quantification of % cells with internalized dextran.",
                model_system="Bone marrow-derived dendritic cells (BMDCs) in microfabricated channels",
                organism="Mus musculus",
                organism_strain=None,
                perturbation_type="genetic_loss_of_function",
                perturbation_target="RAI14",
                perturbation_method="siRNA",
                readout="% cells containing dextran-positive macropinosomes",
                control_description="siRNA control",
            ),
            results=Results(
                result_direction="positive",
                effect_description="Rai14 knockdown decreased % cells with macropinosomes by ~40%",
                effect_size="~40% decrease",
                statistical_test="two-tailed paired Student's t-test",
                p_value="p < 0.05",
                confidence_interval=None,
                sample_size="n=3 independent experiments (n ≥40 cells per condition)",
                key_figure="Figure 5D–E",
            ),
            methodological_tags=MethodologicalTags(
                approach_category="cell_biology",
                assay_types=["live_cell_imaging", "siRNA_knockdown", "microchannel_assay"],
                blinding_reported=None,
                randomization_reported=None,
            ),
            limitations_stated_by_authors=[],
            source_section="results",
            source_text_span=None,
        ),
        # e_012: Rai14 knockdown reduces PAK phosphorylation in BMDCs
        EvidenceUnit(
            evidence_id="e_012",
            assertion_draft_ids=["a_008"],
            evidence_direction="supports",
            evidence_strength="direct_experimental",
            experiment=Experiment(
                description="Western blot analysis of lysates from BMDCs ± Rai14 knockdown using antibodies against phosphorylated PAK (p-PAK) and total PAK. Quantification of p-PAK/PAK ratio normalized to tubulin.",
                model_system="Bone marrow-derived dendritic cells (BMDCs)",
                organism="Mus musculus",
                organism_strain=None,
                perturbation_type="genetic_loss_of_function",
                perturbation_target="RAI14",
                perturbation_method="siRNA",
                readout="p-PAK (Ser199/204) and total PAK protein levels by Western blot; p-PAK/PAK ratio",
                control_description="siRNA control",
            ),
            results=Results(
                result_direction="positive",
                effect_description="p-PAK levels decreased by ~50% upon Rai14 knockdown",
                effect_size="~50% decrease in p-PAK/PAK ratio",
                statistical_test="two-tailed paired Student's t-test",
                p_value="p < 0.05",
                confidence_interval=None,
                sample_size="n=4 independent experiments",
                key_figure="Figure 5F–H",
            ),
            methodological_tags=MethodologicalTags(
                approach_category="biochemical_assay",
                assay_types=["Western_blot"],
                blinding_reported=None,
                randomization_reported=None,
            ),
            limitations_stated_by_authors=[],
            source_section="results",
            source_text_span=None,
        ),
        # e_013: Rai14 knockdown increases BMDC migration speed
        EvidenceUnit(
            evidence_id="e_013",
            assertion_draft_ids=["a_009"],
            evidence_direction="supports",
            evidence_strength="direct_experimental",
            experiment=Experiment(
                description="BMDCs in 5×8 µm microfabricated channels imaged for 20h using epiflourescence microscopy (10X objective; 1 phase image/min). Kymographs generated; cell speed and speed fluctuations quantified.",
                model_system="Bone marrow-derived dendritic cells (BMDCs) in microfabricated channels",
                organism="Mus musculus",
                organism_strain=None,
                perturbation_type="genetic_loss_of_function",
                perturbation_target="RAI14",
                perturbation_method="siRNA",
                readout="Cell speed (µm/min); speed fluctuations (s.d./mean instantaneous speed)",
                control_description="siRNA control",
            ),
            results=Results(
                result_direction="positive",
                effect_description="Rai14 knockdown BMDCs migrated faster (7.1 µm/min) vs control (5.4 µm/min); reduced speed fluctuations (fewer direction changes)",
                effect_size="7.1 vs 5.4 µm/min; p < 0.01",
                statistical_test="two-tailed paired Student's t-test",
                p_value="p < 0.01",
                confidence_interval=None,
                sample_size="n=4 independent experiments (n > 150 cells per condition)",
                key_figure="Figure 6A–C",
            ),
            methodological_tags=MethodologicalTags(
                approach_category="cell_biology",
                assay_types=["live_cell_imaging", "siRNA_knockdown", "microchannel_assay"],
                blinding_reported=None,
                randomization_reported=None,
            ),
            limitations_stated_by_authors=[],
            source_section="results",
            source_text_span=None,
        ),
        # e_014: Rai14 and Ii colocalize on macropinosomal vesicles in BMDCs
        EvidenceUnit(
            evidence_id="e_014",
            assertion_draft_ids=["a_002"],
            evidence_direction="supports",
            evidence_strength="observational_uncontrolled",
            experiment=Experiment(
                description="Immunofluorescence of fixed BMDCs stained with anti-Rai14 (green), anti-Ii (red), and DAPI (nuclei). Colocalization quantified by normalized fluorescence intensity profiles.",
                model_system="Bone marrow-derived dendritic cells (BMDCs)",
                organism="Mus musculus",
                organism_strain=None,
                perturbation_type="none",
                perturbation_target=None,
                perturbation_method=None,
                readout="Colocalization of Rai14 and Ii on large macropinosome-like vesicles (normalized fluorescence intensity profiles)",
                control_description=None,
            ),
            results=Results(
                result_direction="positive",
                effect_description="Rai14 and Ii colocalize on large macropinosome-like vesicles in primary BMDCs",
                effect_size=None,
                statistical_test=None,
                p_value=None,
                confidence_interval=None,
                sample_size="n=2 independent experiments; n=78 vesicles from 37 cells",
                key_figure="Figure 6D–E",
            ),
            methodological_tags=MethodologicalTags(
                approach_category="imaging",
                assay_types=["immunofluorescence"],
                blinding_reported=None,
                randomization_reported=None,
            ),
            limitations_stated_by_authors=[],
            source_section="results",
            source_text_span=None,
        ),
        # e_015: Rai14-myosin II binding by co-IP
        EvidenceUnit(
            evidence_id="e_015",
            assertion_draft_ids=["a_010"],
            evidence_direction="supports",
            evidence_strength="direct_experimental",
            experiment=Experiment(
                description="Co-immunoprecipitation of endogenous myosin II with Rai14 in BMDC lysates using anti-Rai14 antibody. Western blot detection with anti-myosin II and anti-Rai14 antibodies.",
                model_system="Bone marrow-derived dendritic cells (BMDCs)",
                organism="Mus musculus",
                organism_strain=None,
                perturbation_type="none",
                perturbation_target=None,
                perturbation_method=None,
                readout="Co-immunoprecipitation band intensity by Western blot",
                control_description="IgG isotype control immunoprecipitation",
            ),
            results=Results(
                result_direction="positive",
                effect_description="Myosin II specifically co-immunoprecipitated with endogenous Rai14",
                effect_size=None,
                statistical_test=None,
                p_value=None,
                confidence_interval=None,
                sample_size=None,
                key_figure="Figure 6F",
            ),
            methodological_tags=MethodologicalTags(
                approach_category="biochemical_assay",
                assay_types=["co_immunoprecipitation", "Western_blot"],
                blinding_reported=None,
                randomization_reported=None,
            ),
            limitations_stated_by_authors=[],
            source_section="results",
            source_text_span=None,
        ),
        # e_016: GST-myosin II tail pulldown of Rai14 and Ii; blocking with Rai14 depletion
        EvidenceUnit(
            evidence_id="e_016",
            assertion_draft_ids=["a_010", "a_011"],
            evidence_direction="supports",
            evidence_strength="direct_experimental",
            experiment=Experiment(
                description="GST-myosin II heavy chain tail (purified from bacteria) incubated with lysates from MelJuSo cells ± Rai14 knockdown. Affinity chromatography followed by Western blot detection of pulled-down Rai14 and Ii.",
                model_system="MelJuSo human melanoma cell line lysates ± Rai14 knockdown",
                organism="Homo sapiens",
                organism_strain=None,
                perturbation_type="genetic_loss_of_function",
                perturbation_target="RAI14",
                perturbation_method="siRNA",
                readout="Affinity-precipitated Rai14 and Ii detected by Western blot",
                control_description="GST alone (negative control); siRNA control lysates; Rai14 knockdown lysates",
            ),
            results=Results(
                result_direction="positive",
                effect_description="GST-myosin II tail pulled down both Rai14 and Ii from control lysates; Rai14 depletion prevented Ii pulldown, suggesting Rai14 bridges Ii to myosin II",
                effect_size=None,
                statistical_test=None,
                p_value=None,
                confidence_interval=None,
                sample_size=None,
                key_figure="Figure 6G",
            ),
            methodological_tags=MethodologicalTags(
                approach_category="biochemical_assay",
                assay_types=["GST_pulldown", "affinity_chromatography", "Western_blot"],
                blinding_reported=None,
                randomization_reported=None,
            ),
            limitations_stated_by_authors=[],
            source_section="results",
            source_text_span=None,
        ),
    ]

    # =========================================================================
    # ASSERTION DRAFTS (Novel findings only)
    # =========================================================================

    assertion_drafts = [
        # a_001: Rai14-Ii interaction
        AssertionDraft(
            draft_id="a_001",
            natural_language="Rai14 specifically binds to Invariant chain in human melanoma cells",
            canonical_form="RAI14 — directly_binds — Invariant_Chain (Ii/CD74)",
            negatable_form="Rai14 does NOT specifically bind to Invariant chain",
            subject_entity=Entity(
                surface_form="Rai14",
                canonical_name="RAI14",
                ontology_id="UniProt:Q9UHD9",
                ontology_source="UniProt",
                entity_type="protein",
                aliases=["retinoic acid induced 14", "ankycorbin", "NORPEG"],
            ),
            object_entity=Entity(
                surface_form="Invariant chain",
                canonical_name="CD74",
                ontology_id="UniProt:P04397",
                ontology_source="UniProt",
                entity_type="protein",
                aliases=["Ii", "CD74", "HLA-DR associated invariant chain"],
            ),
            predicate="directly_binds",
            direction="positive",
            assertion_type="mechanistic_causal",
            causal_type=None,
            scope=Scope(
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
                disease=[],
                condition=None,
                developmental_stage=None,
                in_vitro=True,
            ),
            conditions=[],
            hedging=Hedging(
                verbatim_hedge="Rai14 was detected as a putative interaction partner, and the interaction was confirmed by co-immunoprecipitation",
                certainty="high",
                generalizability="medium",
                causality_hedge="correlational",
            ),
            epistemic_status=EpistemicStatus(
                section="results",
                function="novel_finding",
                is_primary=True,
                cited_source=None,
            ),
            evidence_unit_ids=["e_001", "e_002"],
            parent_assertion_ids=[],
            provenance=Provenance(
                source_sentence="Retinoic acid-induced 14 (Rai14) was detected as a putative interaction partner, and the interaction was confirmed by co-immunoprecipitation.",
                section_name="Results: Rai14 is a novel interactor of Invariant chain",
                char_offset_start=None,
                char_offset_end=None,
            ),
        ),
        # a_002: Rai14 localizes to membrane ruffles and macropinosomes
        AssertionDraft(
            draft_id="a_002",
            natural_language="Rai14 localizes to actin-rich membrane ruffles and nascent macropinosomes in antigen-presenting cells",
            canonical_form="RAI14 — localizes_to — membrane_ruffles_and_macropinosomes",
            negatable_form="Rai14 does NOT localize to membrane ruffles and nascent macropinosomes",
            subject_entity=Entity(
                surface_form="Rai14",
                canonical_name="RAI14",
                ontology_id="UniProt:Q9UHD9",
                ontology_source="UniProt",
                entity_type="protein",
                aliases=["retinoic acid induced 14", "ankycorbin"],
            ),
            object_entity=Entity(
                surface_form="membrane ruffles",
                canonical_name="membrane ruffle",
                ontology_id="GO:0032591",
                ontology_source="GO",
                entity_type="other",
                aliases=["membrane ruffles", "actin-rich membrane protrusions"],
            ),
            predicate="localizes_to",
            direction="positive",
            assertion_type="existence",
            causal_type=None,
            scope=Scope(
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
                disease=[],
                condition=None,
                developmental_stage=None,
                in_vitro=True,
            ),
            conditions=[],
            hedging=Hedging(
                verbatim_hedge="we found that Rai14 localizes to membrane ruffles, where it forms macropinosomes",
                certainty="high",
                generalizability="medium",
                causality_hedge="correlational",
            ),
            epistemic_status=EpistemicStatus(
                section="results",
                function="novel_finding",
                is_primary=True,
                cited_source=None,
            ),
            evidence_unit_ids=["e_003", "e_007", "e_014"],
            parent_assertion_ids=[],
            provenance=Provenance(
                source_sentence="In line with this, we found that Rai14 localizes to membrane ruffles, where it forms macropinosomes.",
                section_name="Results",
                char_offset_start=None,
                char_offset_end=None,
            ),
        ),
        # a_003: Rai14 is part of Ii-MHC II complex
        AssertionDraft(
            draft_id="a_003",
            natural_language="Rai14 interacts with the Invariant chain-MHC II complex in human cells",
            canonical_form="RAI14 — associates_with — Ii_MHC_II_complex",
            negatable_form="Rai14 does NOT interact with the Invariant chain-MHC II complex",
            subject_entity=Entity(
                surface_form="Rai14",
                canonical_name="RAI14",
                ontology_id="UniProt:Q9UHD9",
                ontology_source="UniProt",
                entity_type="protein",
                aliases=["retinoic acid induced 14"],
            ),
            object_entity=Entity(
                surface_form="MHC II-Ii complex",
                canonical_name="MHC II",
                ontology_id="UniProt:P13760",
                ontology_source="UniProt",
                entity_type="protein",
                aliases=["HLA-DR", "Major Histocompatibility Complex Class II"],
            ),
            predicate="associates_with",
            direction="positive",
            assertion_type="correlational",
            causal_type=None,
            scope=Scope(
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
                disease=[],
                condition=None,
                developmental_stage=None,
                in_vitro=True,
            ),
            conditions=[],
            hedging=Hedging(
                verbatim_hedge="confirming that Rai14 interacts with Ii in a complex with MHC II",
                certainty="high",
                generalizability="medium",
                causality_hedge="correlational",
            ),
            epistemic_status=EpistemicStatus(
                section="results",
                function="novel_finding",
                is_primary=True,
                cited_source=None,
            ),
            evidence_unit_ids=["e_004"],
            parent_assertion_ids=["a_001"],
            provenance=Provenance(
                source_sentence="confirming that Rai14 interacts with Ii in a complex with MHC II.",
                section_name="Results: Rai14 depletion retains MHC II at the plasma membrane",
                char_offset_start=None,
                char_offset_end=None,
            ),
        ),
        # a_004: Rai14 required for MHC II internalization
        AssertionDraft(
            draft_id="a_004",
            natural_language="Rai14 is required for MHC II internalization and trafficking to endocytic compartments",
            canonical_form="RAI14 — is_required_for — MHC_II_internalization",
            negatable_form="Rai14 is NOT required for MHC II internalization",
            subject_entity=Entity(
                surface_form="Rai14",
                canonical_name="RAI14",
                ontology_id="UniProt:Q9UHD9",
                ontology_source="UniProt",
                entity_type="protein",
                aliases=["retinoic acid induced 14"],
            ),
            object_entity=Entity(
                surface_form="MHC II internalization",
                canonical_name="MHC II internalization",
                ontology_id=None,
                ontology_source=None,
                entity_type="other",
                aliases=["HLA-DR internalization", "MHC class II trafficking"],
            ),
            predicate="is_required_for",
            direction="positive",
            assertion_type="mechanistic_causal",
            causal_type="necessary",
            scope=Scope(
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
                disease=[],
                condition=None,
                developmental_stage=None,
                in_vitro=True,
            ),
            conditions=[],
            hedging=Hedging(
                verbatim_hedge="Altogether, these results indicate that Rai14 is required for MHC II internalization",
                certainty="high",
                generalizability="medium",
                causality_hedge="causal",
            ),
            epistemic_status=EpistemicStatus(
                section="results",
                function="novel_finding",
                is_primary=True,
                cited_source=None,
            ),
            evidence_unit_ids=["e_005"],
            parent_assertion_ids=[],
            provenance=Provenance(
                source_sentence="Altogether, these results indicate that Rai14 is required for MHC II internalization.",
                section_name="Results: Rai14 depletion retains MHC II at the plasma membrane",
                char_offset_start=None,
                char_offset_end=None,
            ),
        ),
        # a_005: Rai14 depletion impairs MHC II uptake kinetics
        AssertionDraft(
            draft_id="a_005",
            natural_language="Rai14 knockdown delays the kinetics of MHC II uptake into endosomal compartments",
            canonical_form="RAI14_knockdown — delays — MHC_II_uptake_kinetics",
            negatable_form="Rai14 knockdown does NOT delay MHC II uptake kinetics",
            subject_entity=Entity(
                surface_form="Rai14 knockdown",
                canonical_name="RAI14 knockdown",
                ontology_id=None,
                ontology_source=None,
                entity_type="protein",
                aliases=["Rai14 depletion", "RAI14 siRNA knockdown"],
            ),
            object_entity=Entity(
                surface_form="MHC II uptake",
                canonical_name="MHC II uptake",
                ontology_id=None,
                ontology_source=None,
                entity_type="other",
                aliases=["HLA-DR uptake"],
            ),
            predicate="delays",
            direction="positive",
            assertion_type="mechanistic_causal",
            causal_type="sufficient",
            scope=Scope(
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
                disease=[],
                condition=None,
                developmental_stage=None,
                in_vitro=True,
            ),
            conditions=[],
            hedging=Hedging(
                verbatim_hedge="Rai14 knocked-down cells internalized less MHC II in endosomes than the control cells, with a decrease of almost 40% of MHC II in endosomes 60 minutes after the uptake",
                certainty="high",
                generalizability="medium",
                causality_hedge="causal",
            ),
            epistemic_status=EpistemicStatus(
                section="results",
                function="novel_finding",
                is_primary=True,
                cited_source=None,
            ),
            evidence_unit_ids=["e_006"],
            parent_assertion_ids=[],
            provenance=Provenance(
                source_sentence="Rai14 knocked-down cells internalized less MHC II in endosomes than the control cells, with a decrease of almost 40% of MHC II in endosomes 60 minutes after the uptake",
                section_name="Results: Rai14 depletion retains MHC II at the plasma membrane",
                char_offset_start=None,
                char_offset_end=None,
            ),
        ),
        # a_006: Rai14 positive regulator of macropinocytosis
        AssertionDraft(
            draft_id="a_006",
            natural_language="Rai14 is a positive regulator of macropinocytosis in antigen-presenting cells",
            canonical_form="RAI14 — promotes — macropinocytosis",
            negatable_form="Rai14 does NOT promote macropinocytosis",
            subject_entity=Entity(
                surface_form="Rai14",
                canonical_name="RAI14",
                ontology_id="UniProt:Q9UHD9",
                ontology_source="UniProt",
                entity_type="protein",
                aliases=["retinoic acid induced 14"],
            ),
            object_entity=Entity(
                surface_form="macropinocytosis",
                canonical_name="macropinocytosis",
                ontology_id="GO:0044351",
                ontology_source="GO",
                entity_type="other",
                aliases=["macropinocytotic uptake", "fluid phase endocytosis"],
            ),
            predicate="promotes",
            direction="positive",
            assertion_type="mechanistic_causal",
            causal_type="necessary",
            scope=Scope(
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
                        surface_form="MelJuSo melanoma cells",
                    ),
                    OntologyTerm(
                        term_id="CL:0000451",
                        term_name="dendritic cell",
                        ontology="CL",
                        surface_form="BMDCs",
                    ),
                ],
                disease=[],
                condition=None,
                developmental_stage=None,
                in_vitro=True,
            ),
            conditions=[],
            hedging=Hedging(
                verbatim_hedge="Altogether, these results indicate that Rai14 is required for macropinocytosis",
                certainty="high",
                generalizability="medium",
                causality_hedge="causal",
            ),
            epistemic_status=EpistemicStatus(
                section="results",
                function="novel_finding",
                is_primary=True,
                cited_source=None,
            ),
            evidence_unit_ids=["e_008", "e_010", "e_011"],
            parent_assertion_ids=[],
            provenance=Provenance(
                source_sentence="Altogether, these results indicate that Rai14 is required for macropinocytosis.",
                section_name="Results: Silencing of Rai14 inhibits macropinocytosis",
                char_offset_start=None,
                char_offset_end=None,
            ),
        ),
        # a_007: Rai14 required for macropinosome closure via PtdIns(4,5)P2 depletion
        AssertionDraft(
            draft_id="a_007",
            natural_language="Rai14 is required for phosphatidylinositol 4,5-bisphosphate depletion during macropinosome closure",
            canonical_form="RAI14 — is_required_for — PtdIns(4,5)P2_depletion_at_macropinosome",
            negatable_form="Rai14 is NOT required for PtdIns(4,5)P2 depletion during macropinosome closure",
            subject_entity=Entity(
                surface_form="Rai14",
                canonical_name="RAI14",
                ontology_id="UniProt:Q9UHD9",
                ontology_source="UniProt",
                entity_type="protein",
                aliases=["retinoic acid induced 14"],
            ),
            object_entity=Entity(
                surface_form="PtdIns(4,5)P2 depletion",
                canonical_name="phosphatidylinositol 4,5-bisphosphate depletion",
                ontology_id="GO:0046854",
                ontology_source="GO",
                entity_type="other",
                aliases=["PIP2 depletion"],
            ),
            predicate="is_required_for",
            direction="positive",
            assertion_type="mechanistic_causal",
            causal_type="necessary",
            scope=Scope(
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
                disease=[],
                condition="macropinosome closure",
                developmental_stage=None,
                in_vitro=True,
            ),
            conditions=[],
            hedging=Hedging(
                verbatim_hedge="Intriguingly, in cells silenced for Rai14, the percentage of cells retaining PtdIns(4,5)P2 at the membrane of nascent macropinosomes dramatically increased, suggesting that Rai14 is required for macropinosome closure",
                certainty="medium",
                generalizability="medium",
                causality_hedge="causal",
            ),
            epistemic_status=EpistemicStatus(
                section="results",
                function="novel_finding",
                is_primary=True,
                cited_source=None,
            ),
            evidence_unit_ids=["e_009"],
            parent_assertion_ids=[],
            provenance=Provenance(
                source_sentence="Intriguingly, in cells silenced for Rai14, the percentage of cells retaining PtdIns(4,5)P2 at the membrane of nascent macropinosomes dramatically increased, suggesting that Rai14 is required for macropinosome closure.",
                section_name="Results: Silencing of Rai14 inhibits macropinocytosis",
                char_offset_start=None,
                char_offset_end=None,
            ),
        ),
        # a_008: Rai14 regulates PAK phosphorylation
        AssertionDraft(
            draft_id="a_008",
            natural_language="Rai14 is required for p21-activated kinase (PAK) phosphorylation and activation in dendritic cells",
            canonical_form="RAI14 — is_required_for — PAK_phosphorylation",
            negatable_form="Rai14 is NOT required for PAK phosphorylation",
            subject_entity=Entity(
                surface_form="Rai14",
                canonical_name="RAI14",
                ontology_id="UniProt:Q9UHD9",
                ontology_source="UniProt",
                entity_type="protein",
                aliases=["retinoic acid induced 14"],
            ),
            object_entity=Entity(
                surface_form="PAK phosphorylation",
                canonical_name="PAK",
                ontology_id="UniProt:Q13153",
                ontology_source="UniProt",
                entity_type="protein",
                aliases=["p21-activated kinase", "PAK1"],
            ),
            predicate="is_required_for",
            direction="positive",
            assertion_type="mechanistic_causal",
            causal_type="necessary",
            scope=Scope(
                species=[
                    OntologyTerm(
                        term_id="NCBITaxon:10090",
                        term_name="Mus musculus",
                        ontology="NCBI Taxonomy",
                        surface_form="mouse",
                    )
                ],
                tissue=[],
                cell_type=[
                    OntologyTerm(
                        term_id="CL:0000451",
                        term_name="dendritic cell",
                        ontology="CL",
                        surface_form="BMDCs",
                    )
                ],
                disease=[],
                condition=None,
                developmental_stage=None,
                in_vitro=True,
            ),
            conditions=[],
            hedging=Hedging(
                verbatim_hedge="quantification of the levels of PAK phosphorylation revealed a decrease of almost 50% in BMDCs silenced for Rai14 compared to control cells",
                certainty="high",
                generalizability="medium",
                causality_hedge="unclear",
            ),
            epistemic_status=EpistemicStatus(
                section="results",
                function="novel_finding",
                is_primary=True,
                cited_source=None,
            ),
            evidence_unit_ids=["e_012"],
            parent_assertion_ids=[],
            provenance=Provenance(
                source_sentence="quantification of the levels of PAK phosphorylation revealed a decrease of almost 50% in BMDCs silenced for Rai14 compared to control cells",
                section_name="Results: Silencing of Rai14 inhibits macropinocytosis",
                char_offset_start=None,
                char_offset_end=None,
            ),
        ),
        # a_009: Rai14 negative regulator of migration
        AssertionDraft(
            draft_id="a_009",
            natural_language="Rai14 negatively regulates dendritic cell migration and motility",
            canonical_form="RAI14 — inhibits — dendritic_cell_migration",
            negatable_form="Rai14 does NOT inhibit dendritic cell migration",
            subject_entity=Entity(
                surface_form="Rai14",
                canonical_name="RAI14",
                ontology_id="UniProt:Q9UHD9",
                ontology_source="UniProt",
                entity_type="protein",
                aliases=["retinoic acid induced 14"],
            ),
            object_entity=Entity(
                surface_form="cell migration",
                canonical_name="cell migration",
                ontology_id="GO:0030335",
                ontology_source="GO",
                entity_type="other",
                aliases=["cell motility"],
            ),
            predicate="inhibits",
            direction="positive",
            assertion_type="mechanistic_causal",
            causal_type="necessary",
            scope=Scope(
                species=[
                    OntologyTerm(
                        term_id="NCBITaxon:10090",
                        term_name="Mus musculus",
                        ontology="NCBI Taxonomy",
                        surface_form="mouse",
                    )
                ],
                tissue=[],
                cell_type=[
                    OntologyTerm(
                        term_id="CL:0000451",
                        term_name="dendritic cell",
                        ontology="CL",
                        surface_form="BMDCs",
                    )
                ],
                disease=[],
                condition=None,
                developmental_stage=None,
                in_vitro=True,
            ),
            conditions=[
                Condition(
                    condition_type="biological_context",
                    value="confinement in microfabricated channels",
                )
            ],
            hedging=Hedging(
                verbatim_hedge="BMDCs silenced for Rai14 move faster (mean speed 7.1 µm/min) than control BMDCs (mean speed 5.4 µm/min)",
                certainty="high",
                generalizability="medium",
                causality_hedge="causal",
            ),
            epistemic_status=EpistemicStatus(
                section="results",
                function="novel_finding",
                is_primary=True,
                cited_source=None,
            ),
            evidence_unit_ids=["e_013"],
            parent_assertion_ids=[],
            provenance=Provenance(
                source_sentence="BMDCs silenced for Rai14 move faster (mean speed 7.1 µm/min) than control BMDCs (mean speed 5.4 µm/min)",
                section_name="Results: Rai14 negatively regulates BMDC migration",
                char_offset_start=None,
                char_offset_end=None,
            ),
        ),
        # a_010: Rai14-myosin II interaction
        AssertionDraft(
            draft_id="a_010",
            natural_language="Rai14 directly binds to myosin II motor protein",
            canonical_form="RAI14 — directly_binds — myosin_II",
            negatable_form="Rai14 does NOT directly bind myosin II",
            subject_entity=Entity(
                surface_form="Rai14",
                canonical_name="RAI14",
                ontology_id="UniProt:Q9UHD9",
                ontology_source="UniProt",
                entity_type="protein",
                aliases=["retinoic acid induced 14"],
            ),
            object_entity=Entity(
                surface_form="myosin II",
                canonical_name="MYH9",
                ontology_id="UniProt:P35579",
                ontology_source="UniProt",
                entity_type="protein",
                aliases=["non-muscle myosin IIA", "NMIIA"],
            ),
            predicate="directly_binds",
            direction="positive",
            assertion_type="mechanistic_causal",
            causal_type=None,
            scope=Scope(
                species=[
                    OntologyTerm(
                        term_id="NCBITaxon:10090",
                        term_name="Mus musculus",
                        ontology="NCBI Taxonomy",
                        surface_form="mouse",
                    )
                ],
                tissue=[],
                cell_type=[
                    OntologyTerm(
                        term_id="CL:0000451",
                        term_name="dendritic cell",
                        ontology="CL",
                        surface_form="BMDCs",
                    )
                ],
                disease=[],
                condition=None,
                developmental_stage=None,
                in_vitro=True,
            ),
            conditions=[],
            hedging=Hedging(
                verbatim_hedge="Co-immunoprecipitation experiments revealed that Rai14 is indeed able to bind myosin II",
                certainty="high",
                generalizability="medium",
                causality_hedge="correlational",
            ),
            epistemic_status=EpistemicStatus(
                section="results",
                function="novel_finding",
                is_primary=True,
                cited_source=None,
            ),
            evidence_unit_ids=["e_015"],
            parent_assertion_ids=[],
            provenance=Provenance(
                source_sentence="Co-immunoprecipitation experiments revealed that Rai14 is indeed able to bind myosin II",
                section_name="Results: Rai14 negatively regulates BMDC migration",
                char_offset_start=None,
                char_offset_end=None,
            ),
        ),
        # a_011: Rai14 bridges Ii to myosin II
        AssertionDraft(
            draft_id="a_011",
            natural_language="Rai14 mediates the interaction between Invariant chain and myosin II",
            canonical_form="RAI14 — bridges — Ii_and_myosin_II",
            negatable_form="Rai14 does NOT bridge Invariant chain to myosin II",
            subject_entity=Entity(
                surface_form="Rai14",
                canonical_name="RAI14",
                ontology_id="UniProt:Q9UHD9",
                ontology_source="UniProt",
                entity_type="protein",
                aliases=["retinoic acid induced 14"],
            ),
            object_entity=Entity(
                surface_form="Ii-myosin II interaction",
                canonical_name="Ii-myosin II complex",
                ontology_id=None,
                ontology_source=None,
                entity_type="other",
                aliases=["Invariant chain-myosin II bridge"],
            ),
            predicate="bridges",
            direction="positive",
            assertion_type="mechanistic_causal",
            causal_type="necessary",
            scope=Scope(
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
                disease=[],
                condition=None,
                developmental_stage=None,
                in_vitro=True,
            ),
            conditions=[],
            hedging=Hedging(
                verbatim_hedge="suggesting that Rai14 bridges Ii to myosin II",
                certainty="medium",
                generalizability="medium",
                causality_hedge="unclear",
            ),
            epistemic_status=EpistemicStatus(
                section="results",
                function="novel_finding",
                is_primary=True,
                cited_source=None,
            ),
            evidence_unit_ids=["e_016"],
            parent_assertion_ids=["a_001", "a_010"],
            provenance=Provenance(
                source_sentence="suggesting that Rai14 bridges Ii to myosin II.",
                section_name="Results: Rai14 negatively regulates BMDC migration",
                char_offset_start=None,
                char_offset_end=None,
            ),
        ),
    ]

    # =========================================================================
    # CITATION CONTEXTS
    # =========================================================================

    citation_contexts = [
        # c_001: Ii-MHC II chaperoning
        CitationContext(
            citation_id="c_001",
            citing_sentence="Invariant chain (Ii, CD74) is a type II transmembrane protein that self-associates into trimers and provides a scaffold for the assembly of MHC II heterodimers. Ii interacts with the peptide-binding groove of MHC II to prevent the premature binding of endogenous peptides and chaperones new MHC II molecules to endosomes for the loading of antigenic peptides.",
            cited_source_doi=None,
            cited_source_pmid=None,
            cited_source_ref_key="[2, 3]",
            cited_claim_paraphrase="Invariant chain acts as a scaffold for MHC II assembly, prevents premature peptide binding, and chaperones new MHC II to antigen-loading endosomes.",
            relationship="contextualizes",
            linked_assertion_draft_ids=[],
            section="introduction",
        ),
        # c_002: Ii regulates macropinocytosis via myosin II interaction
        CitationContext(
            citation_id="c_002",
            citing_sentence="By interacting with the actin motor myosin II, Ii regulates the macropinocytic and migratory ability of DCs in an antagonistic manner.",
            cited_source_doi=None,
            cited_source_pmid=None,
            cited_source_ref_key="[7, 8]",
            cited_claim_paraphrase="Invariant chain interacts with myosin II to antagonistically regulate dendritic cell macropinocytosis and migration.",
            relationship="contextualizes",
            linked_assertion_draft_ids=["a_006", "a_009"],
            section="introduction",
        ),
        # c_003: N-Ank protein family and membrane binding/shaping
        CitationContext(
            citation_id="c_003",
            citing_sentence="Rai14, also known as novel retinal pigment epithelial cell gene (NORPEG) and ankycorbin, is a member of a superfamily of ankyrin repeat proteins, termed N-Ank. The N-Ank protein superfamily includes proteins containing a set of ankyrin repeats and an N-terminal amphipathic helix that allows membrane interactions by insertion, senses membrane curvatures, and modulates membrane topologies. Their common function is membrane binding and shaping by combining electrostatic interactions of curvature-sensing ankyrin repeats and electrostatic and salt-insensitive hydrophobic interactions mediated by amphipathic helix insertion into one membrane leaflet.",
            cited_source_doi=None,
            cited_source_pmid=None,
            cited_source_ref_key="[9]",
            cited_claim_paraphrase="N-Ank proteins (including Rai14) bind and shape membranes via ankyrin repeats and amphipathic helices through curvature-sensing and hydrophobic interactions.",
            relationship="contextualizes",
            linked_assertion_draft_ids=[],
            section="introduction",
        ),
        # c_004: Rai14 association with cortical actin and cytoskeleton
        CitationContext(
            citation_id="c_004",
            citing_sentence="The reported intracellular localization of Rai14 suggests that it also associates with the cortical actin cytoskeleton, F-actin stress fibers, and cell-cell adhesions sites. Furthermore, it has been proposed that Rai14 is a cytoskeleton-associated protein linked to actin function and organization.",
            cited_source_doi=None,
            cited_source_pmid=None,
            cited_source_ref_key="[10-14]",
            cited_claim_paraphrase="Rai14 localizes to cortical actin structures, F-actin stress fibers, and cell-cell adhesion sites; proposed to regulate actin function and organization.",
            relationship="contextualizes",
            linked_assertion_draft_ids=["a_002"],
            section="introduction",
        ),
    ]

    # =========================================================================
    # EXTRACTION METADATA
    # =========================================================================

    extraction_metadata = ExtractionMetadata(
        extraction_model="claude-haiku-4-5-20251001",
        extraction_version="0.2.0",
        extraction_timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        paper_char_count=None,
        extraction_duration_seconds=None,
    )

    # =========================================================================
    # ASSEMBLE EXTRACTION RESULT
    # =========================================================================

    result = ExtractionResult(
        paper_provenance=paper_provenance,
        evidence_units=evidence_units,
        assertion_drafts=assertion_drafts,
        citation_contexts=citation_contexts,
        extraction_metadata=extraction_metadata,
    )

    return result


if __name__ == "__main__":
    # Build the extraction
    extraction = build_extraction()

    # Validate against schema
    try:
        _ = extraction.model_validate(extraction.model_dump())
        print("[OK] Extraction result validates against schema.")
    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")
        exit(1)

    # Report totals
    print("\nExtraction Summary:")
    print(f"  Paper: {extraction.paper_provenance.title}")
    print(f"  DOI: {extraction.paper_provenance.doi}")
    print(f"  PMID: {extraction.paper_provenance.pmid}")
    print(f"\n  Evidence Units: {len(extraction.evidence_units)}")
    print(f"  Assertion Drafts (Novel Findings): {len(extraction.assertion_drafts)}")
    print(f"  Citation Contexts: {len(extraction.citation_contexts)}")

    # Dump to JSON
    output_path = Path(__file__).parent / "extraction_test_haiku.json"
    with open(output_path, "w") as f:
        json.dump(extraction.model_dump(mode="json"), f, indent=2)

    print(f"\n[OK] Extraction saved to {output_path}")
    print(f"File size: {output_path.stat().st_size} bytes")
