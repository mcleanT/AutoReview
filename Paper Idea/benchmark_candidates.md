# Benchmark Review Paper Candidates

> 50 candidate reference reviews for evaluating AutoReview, curated via OpenAlex, PubMed, and Semantic Scholar (2026-03-13).
>
> **Tier A**: Landmark reviews, 500+ citations, published 2015–2019. Enable time-gated comparison + retrieval-controlled condition.
> **Tier B**: Contemporary reviews, 50–200+ citations, published 2023–2024. End-to-end evaluation on active frontiers.

## Grading Rubric

Each paper is graded A–D on **Benchmark Suitability** (not paper quality — all are strong reviews):

| Grade | Meaning |
|-------|---------|
| **A** | Ideal benchmark: well-scoped topic, clearly a review, high citation count for tier, from a top venue, covers a domain well-suited to automated review generation |
| **B** | Strong benchmark: good scope and citations, minor concern (e.g., slightly outside year window, topic overlap with another candidate, niche venue) |
| **C** | Acceptable: usable if needed to fill a domain/tier gap, but has a notable limitation (borderline citations, ambiguous scope, heavy methodology focus) |
| **D** | Backup only: significant concern (pre-dates window, too narrow, likely contaminated, or poor fit for automated generation) |

---

## Domain 1: Biomedical (13 papers)

### Tier A (Landmark, 2015–2019)

| # | Title | First Author | Journal | Year | Citations | Grade | Justification |
|---|-------|-------------|---------|------|-----------|-------|---------------|
| 1 | Microglia Function in the Central Nervous System During Health and Neurodegeneration | Colonna M | Annu Rev Immunol | 2017 | ~1,500 | **A** | Perfect Tier A exemplar: canonical Annual Reviews article, well-defined scope (microglia in health + disease), massive citations, covers a topic where synthesis across AD/PD/ALS is critical. Excellent test of cross-disease integration. |
| 2 | Blood-Brain Barrier: From Physiology to Disease and Back | Sweeney MD | Physiol Rev | 2018 | ~2,000 | **A** | Flagship Physiological Reviews article covering BBB molecular physiology through to disease. Extremely well-structured, broad evidence base. Tests the pipeline's ability to handle mechanistic + translational content. |
| 3 | Cancer Immunoediting and Resistance to T Cell-Based Immunotherapy | O'Donnell JS | Nat Rev Clin Oncol | 2019 | ~1,200 | **A** | Nature Reviews Clinical Oncology — top venue for clinical reviews. Three-phase immunoediting framework maps cleanly to a review structure. Strong citation network for retrieval testing. |
| 4 | Approaches to Treat Immune Hot, Altered and Cold Tumours with Combination Immunotherapies | Galon J | Nat Rev Drug Discov | 2019 | ~1,500 | **A** | Defines the hot/cold tumor classification that became standard. Nature Reviews Drug Discovery. Highly structured content ideal for automated synthesis. |
| 5 | CAR T Cell Therapy for Solid Tumors | Newick K | Annu Rev Med | 2017 | ~800 | **B** | Strong Annual Reviews article on a well-defined topic. Slightly lower citations than other Tier A candidates but covers a focused, important area. Good test of the pipeline on a therapeutic intervention review. |
| 6 | Ageing as a Risk Factor for Neurodegenerative Disease | Hou Y | Nat Rev Neurol | 2019 | ~1,000 | **A** | Nature Reviews Neurology. Bridges aging biology with neurodegeneration — tests the pipeline's ability to synthesize across two overlapping fields. Well-scoped. |
| 7 | The Amyloid Hypothesis of Alzheimer's Disease at 25 Years | Selkoe DJ | EMBO Mol Med | 2016 | ~2,000 | **B** | Landmark by the hypothesis founders. Risk: heavily opinionated/perspective-like rather than systematic. Still valuable as it tests whether the pipeline can handle a review with a strong argumentative thread. |

### Tier B (Contemporary, 2023–2024)

| # | Title | First Author | Journal | Year | Citations | Grade | Justification |
|---|-------|-------------|---------|------|-----------|-------|---------------|
| 8 | Mechanisms of Long COVID and the Path Toward Therapeutics | Peluso M | Cell | 2024 | ~200 | **A** | Cell review on the defining health challenge of the 2020s. Rapidly accumulating citations. Excellent benchmark: broad mechanistic scope (viral persistence, immune dysregulation, microbiome), active frontier, minimal contamination risk. |
| 9 | GLP-1 Receptor Agonists: Cardiovascular Benefits and Mechanisms of Action | Ussher JR | Nat Rev Cardiol | 2023 | ~150 | **A** | Nature Reviews Cardiology. GLP-1RAs are the most impactful drug class of 2023–2024. Well-scoped (CV focus), strong evidence base from clinical trials. Tests pipeline on pharmacology + clinical evidence synthesis. |
| 10 | Microbiota–Gut–Brain Axis and Its Therapeutic Applications in Neurodegenerative Diseases | Loh JS | Sig Transduct Target Ther | 2024 | ~120 | **A** | Interdisciplinary topic (microbiology × neuroscience × therapeutics). Tests the pipeline on cross-field synthesis. Well-cited for 2024, top Nature-family journal. |
| 11 | Inflammation and Aging: Signaling Pathways and Intervention Therapies | Li X | Sig Transduct Target Ther | 2023 | ~200 | **B** | Strong citations and scope. Minor concern: some overlap with Tier A aging papers (#6). Still valuable as it tests contemporary inflammaging literature. |
| 12 | Dynamics and Specificities of T Cells in Cancer Immunotherapy | Oliveira G | Nat Rev Cancer | 2023 | ~150 | **B** | Nature Reviews Cancer. Focused on single-cell/TCR technologies revealing T cell biology during immunotherapy. Slightly narrow scope but excellent venue and citation trajectory. |
| 13 | Role of Neuroinflammation in Neurodegeneration Development | Zhang W | Sig Transduct Target Ther | 2023 | ~100 | **B** | Good Tier B candidate covering microglia, astrocytes, cytokines across AD/PD/ALS/MS. Some overlap with Tier A microglia paper (#1) but focuses on contemporary 2020s literature. |

---

## Domain 2: Computer Science / AI (13 papers)

### Tier A (Landmark, 2015–2019)

| # | Title | First Author | Journal | Year | Citations | Grade | Justification |
|---|-------|-------------|---------|------|-----------|-------|---------------|
| 14 | A Survey on Deep Learning in Medical Image Analysis | Litjens G | Med Image Anal | 2017 | ~14,000 | **A** | One of the most cited DL surveys ever. Massive scope (classification, detection, segmentation across all anatomical domains). Perfect Tier A: tests pipeline on a survey that requires organizing hundreds of methods. |
| 15 | Federated Machine Learning: Concept and Applications | Yang Q | ACM TIST | 2019 | ~6,000 | **A** | Foundational FL survey defining horizontal/vertical/transfer FL. Clean taxonomy, massive citations. Ideal benchmark: well-structured topic that maps cleanly to a review outline. |
| 16 | A Survey on Image Data Augmentation for Deep Learning | Shorten C | J Big Data | 2019 | ~5,000 | **A** | Covers geometric, color, GAN-based, and mix-based augmentation. Extremely well-scoped and practical. High citations despite non-traditional venue — tests the pipeline on a clearly delineated methods review. |
| 17 | Deep Learning for Generic Object Detection: A Survey | Liu L | IJCV | 2019 | ~2,500 | **B** | IJCV flagship survey covering 300+ detection frameworks. Excellent scope but overlaps with Zhao 2019 (TNNLS). Pick one; this one has the better venue. |
| 18 | Deep Reinforcement Learning: A Brief Survey | Arulkumaran K | IEEE Signal Proc Mag | 2017 | ~3,000 | **B** | Concise but highly cited DRL survey. Good scope. Minor concern: "brief" format means the reference review is shorter than typical, which may affect comparison fairness. |
| 19 | A Survey on Graph Neural Networks | Zhang Z | IEEE TKDE | 2020 | ~3,500 | **B** | Comprehensive GNN taxonomy. Published 2020 (borderline for Tier A 2015–2019 window). If year flexibility allows, this is an excellent candidate; otherwise, use as strong borderline. |

### Tier B (Contemporary, 2023–2024)

| # | Title | First Author | Journal | Year | Citations | Grade | Justification |
|---|-------|-------------|---------|------|-----------|-------|---------------|
| 20 | Retrieval-Augmented Generation for Large Language Models: A Survey | Gao Y | arXiv | 2023 | ~1,500 | **A** | THE definitive RAG survey. Covers Naive/Advanced/Modular RAG paradigms. Extremely well-cited for 2023. Risk: arXiv preprint (not peer-reviewed venue), but this is the canonical reference. Tests pipeline on an NLP infrastructure topic. |
| 21 | A Survey of Large Language Models | Zhao WX | arXiv | 2023 | ~3,000 | **A** | The most comprehensive LLM survey. Covers pretraining, adaptation, utilization, evaluation. Massive scope and citations. Risk: very broad — may be harder to match as a benchmark. But this IS the review everyone cites. |
| 22 | Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models | Zhang Y | Comput Linguist | 2023 | ~500 | **A** | Peer-reviewed (Computational Linguistics — top NLP venue). Well-scoped taxonomy of LLM hallucination. Strong citation trajectory. Ideal Tier B: tests pipeline on a focused, high-impact contemporary problem. |
| 23 | Text-to-Image Diffusion Models in Generative AI: A Survey | Zhang C | arXiv | 2023 | ~200 | **B** | Good scope covering diffusion foundations + applications. Moderate citations. Tests pipeline on generative AI, a hot frontier. Minor: arXiv only. |
| 24 | A Comprehensive Survey of Retrieval-Augmented Generation (RAG): Evolution, Current Landscape and Future Directions | Gupta S | arXiv | 2024 | ~100 | **B** | More recent RAG survey than Gao 2023. Lower citations but covers 2024 developments. Good complement if both RAG papers are used (one per tier is redundant — pick one). |
| 25 | Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in NLP | Liu P | ACM Comput Surv | 2023 | ~3,000 | **B** | ACM Computing Surveys — top survey venue. Covers prompt-based learning paradigm comprehensively. Very high citations for Tier B (borderline Tier A). Strong benchmark for NLP methods review. |
| 26 | A Comprehensive Survey on Hallucination Mitigation Techniques in Large Language Models | Tonmoy S | arXiv | 2024 | ~150 | **B** | Focused specifically on mitigation (32+ techniques). Complements the hallucination taxonomy paper (#22). Good for testing pipeline on a solution-oriented review. |

---

## Domain 3: Materials Science (12 papers)

### Tier A (Landmark, 2015–2019)

| # | Title | First Author | Journal | Year | Citations | Grade | Justification |
|---|-------|-------------|---------|------|-----------|-------|---------------|
| 27 | Halide Perovskite Photovoltaics: Background, Status, and Future Prospects | Jena AK | Chem Rev | 2019 | ~4,000 | **A** | Chemical Reviews landmark. Covers the entire perovskite PV field: fundamentals, efficiency, stability, commercialization. Ideal benchmark: massive scope, top venue, clean topic boundary. |
| 28 | Metal-Halide Perovskites for Photovoltaic and Light-Emitting Devices | Stranks SD | Nat Nanotechnol | 2015 | ~5,000 | **A** | Nature Nanotechnology. Early landmark covering perovskites for both solar cells and LEDs. Tests the pipeline on a dual-application review from the field's formative period. |
| 29 | Recent Advances in Two-Dimensional Materials beyond Graphene | Bhimanapati GR | ACS Nano | 2015 | ~3,500 | **A** | ACS Nano multi-author review (27 authors). Covers TMDs, MXenes, phosphorene, silicene — the full 2D materials landscape. Excellent test of breadth synthesis. |
| 30 | Applications of 2D MXenes in Energy Conversion and Storage Systems | Pang J | Chem Soc Rev | 2018 | ~1,200 | **A** | Chemical Society Reviews. Focused on MXenes for batteries, supercapacitors, photocatalysis. Well-scoped, high citations. Tests pipeline on an energy materials review. |
| 31 | New Horizons for Inorganic Solid State Ion Conductors | Zhang Z | Energy Environ Sci | 2018 | ~1,500 | **A** | Energy & Environmental Science. Critical review of solid electrolytes for solid-state batteries. Covers conductivity mechanisms, stabilities, and interfaces. High impact, well-defined topic. |
| 32 | A Review on Metal-Organic Frameworks: Synthesis and Applications | Safaei M | TrAC | 2019 | ~800 | **B** | Good MOF overview but TrAC is a less prestigious venue than Chem Rev/Chem Soc Rev for this topic. Usable if the domain needs more Tier A coverage. |
| 33 | Mechanical Properties of High-Entropy Alloys with Emphasis on FCC Alloys | Li Z | Prog Mater Sci | 2018 | ~800 | **A** | Progress in Materials Science — the highest-impact review journal in materials. Systematic treatment of HEA mechanical behavior. Tests pipeline on a focused mechanical properties review. |
| 34 | Additive Manufacturing (3D Printing): A Review of Materials, Methods, Applications and Challenges | Ngo TD | Composites Part B | 2018 | ~3,000 | **B** | Very high citations. Covers metals, polymers, ceramics, concrete for 3D printing. Minor concern: Composites Part B is mid-tier venue, but the paper itself is a de facto standard reference. |

### Tier B (Contemporary, 2023–2024)

| # | Title | First Author | Journal | Year | Citations | Grade | Justification |
|---|-------|-------------|---------|------|-----------|-------|---------------|
| 35 | Chemical Stability of Lead Halide Perovskite Solar Cells | Zhuang J | Nano-Micro Lett | 2023 | ~100 | **A** | Focused on degradation mechanisms (moisture, O₂, heat, light) and stabilization strategies. Well-scoped, complementary to Tier A perovskite papers. Nano-Micro Letters is a rising venue. |
| 36 | Exploring the Potential of High Entropy Alloys: A Comprehensive Review | Arun S | Johnson Matthey Technol Rev | 2023 | ~80 | **B** | Covers HEA microstructure, properties, and applications comprehensively. Minor concern: niche venue. But good complement to Tier A HEA paper (#33) for testing contemporary literature synthesis. |
| 37 | Biodegradable Polymers in Biomedical Applications: A Review | Kurowiak J | Int J Mol Sci | 2023 | ~100 | **B** | Covers tissue engineering, drug delivery, implants. MDPI open-access venue (moderate prestige). Good for testing pipeline on a materials-meets-biomedical topic. |
| 38 | MOFs for Heavy Metal Removal: Synthesis, Applications and Mechanism — A Systematic Review | Lin K | Chem Eng J | 2023 | ~120 | **A** | Chemical Engineering Journal. Well-scoped environmental application of MOFs. Systematic review format. Tests pipeline on an applied/environmental materials topic. |

---

## Domain 4: Social Science (12 papers)

### Tier A (Landmark, 2015–2019)

| # | Title | First Author | Journal | Year | Citations | Grade | Justification |
|---|-------|-------------|---------|------|-----------|-------|---------------|
| 39 | Social Media, Political Polarization, and Political Disinformation: A Review of the Scientific Literature | Tucker JA | SSRN/Hewlett Foundation | 2018 | ~800 | **A** | The canonical multi-topic review linking social media to polarization and disinformation. Widely cited in both academia and policy. Tests pipeline on a politically relevant, interdisciplinary topic. Minor: white paper, not peer-reviewed journal. |
| 40 | A Systematic Review: The Influence of Social Media on Depression, Anxiety and Psychological Distress in Adolescents | Keles B | J Adolescence | 2019 | ~800 | **A** | Defines the adolescent social media–mental health research agenda. Clean systematic review methodology. Ideal benchmark: well-scoped, policy-relevant, strong citations, peer-reviewed. |
| 41 | Strategies for Addressing Vaccine Hesitancy — A Systematic Review | Jarrett C | Vaccine | 2015 | ~1,000 | **A** | Systematic review of interventions across global contexts. Clean scope (strategies, not just determinants). Vaccine is the top venue for this topic. Tests pipeline on a public health intervention review. |
| 42 | Systematic Literature Review on the Spread of Health-Related Misinformation on Social Media | Wang Y | Soc Sci Med | 2019 | ~600 | **A** | Social Science & Medicine. Reviews 57 articles on health misinformation drivers. Well-scoped intersection of health + communication. Strong citations, peer-reviewed. |
| 43 | Beyond Misinformation: Understanding and Coping with the 'Post-Truth' Era | Lewandowsky S | J Appl Res Mem Cogn | 2017 | ~800 | **B** | Influential framework paper. Minor concern: somewhat perspective/framework-oriented rather than systematic review. But the "technocognition" framework became widely adopted. Tests pipeline on a theory-heavy review. |
| 44 | Income Inequality and Depression: A Systematic Review and Meta-Analysis | Patel V | World Psychiatry | 2018 | ~500 | **B** | World Psychiatry (top impact factor in psychiatry). Includes both meta-analysis and scoping review. Good for testing pipeline on quantitative synthesis. Minor: partially methodological (meta-analytic). |

### Tier B (Contemporary, 2023–2024)

| # | Title | First Author | Journal | Year | Citations | Grade | Justification |
|---|-------|-------------|---------|------|-----------|-------|---------------|
| 45 | The Psychological Drivers of Misinformation Belief and Its Resistance to Correction | Ecker UKH | Nat Rev Psychol | 2022 | ~400 | **A** | Nature Reviews Psychology — the premiere review venue for this field. Covers cognitive mechanisms behind misinformation belief. Tests pipeline on a psychology-focused review. Technically 2022 but fits the Tier B use case. |
| 46 | Mental Health in Europe During the COVID-19 Pandemic: A Systematic Review | Ahmed N | Lancet Psychiatry | 2023 | ~200 | **A** | The Lancet Psychiatry. Reviews 177 studies — massive evidence synthesis task. Well-scoped (Europe, COVID-19, longitudinal). Ideal Tier B: tests pipeline on a large systematic review with clear geographic/temporal boundaries. |
| 47 | COVID-19 Vaccine Hesitancy: Umbrella Review of Systematic Reviews and Meta-Analysis | Al Rahbeni T | JMIR | 2024 | ~100 | **A** | Meta-review of 78 meta-analyses. Tests whether the pipeline can handle an umbrella review format. JMIR is the top digital health venue. Good complement to Tier A vaccine hesitancy paper (#41). |
| 48 | The Role of (Social) Media in Political Polarization: A Systematic Review | Kubin E | Ann Int Comm Assoc | 2021 | ~200 | **B** | Systematic review of 94 articles. Technically 2021 (outside strict 2023–2024 Tier B). But fills an important gap on the polarization topic with robust methodology. |
| 49 | Fake News, Disinformation and Misinformation in Social Media: A Review | Aïmeur E | Soc Netw Anal Min | 2023 | ~150 | **B** | Covers detection methods (NLP, ML, deep learning). Good interdisciplinary angle (CS meets social science). Tests pipeline on a methods-oriented social science review. |
| 50 | Echo Chambers on Social Media: A Systematic Review of the Literature | Terren L | Rev Commun Res | 2021 | ~250 | **B** | Reviews 55 studies finding methodology-dependent results. Technically 2021. Strong scope and finding (method matters for conclusions). Tests pipeline on a contested/nuanced topic. |

---

## Summary Statistics

| Domain | Tier A | Tier B | Total |
|--------|--------|--------|-------|
| Biomedical | 7 | 6 | 13 |
| Computer Science / AI | 6 | 7 | 13 |
| Materials Science | 7 | 4 (*) | 11 (+1 borderline) |
| Social Science | 6 | 6 | 12 |
| **Total** | **26** | **23** (+1) | **50** |

(*) Materials Tier B has fewer strong candidates; consider adding 1-2 more from lithium battery cathodes or thermoelectric materials if needed.

## Grade Distribution

| Grade | Count | Notes |
|-------|-------|-------|
| **A** | 28 | Ideal benchmarks — prioritize these for the final 17–23 selection |
| **B** | 22 | Strong backups — use to fill gaps or replace A-graded papers with issues |
| **C** | 0 | None selected (filtered out during curation) |
| **D** | 0 | None selected |

---

## Recommended Final Selection (17–23 papers)

Based on grades, domain balance, and tier requirements from the paper plan:

### Tier A Selection (7 papers, ~2 per domain)

| Domain | Paper # | Short title |
|--------|---------|-------------|
| Biomedical | 1 | Microglia in CNS (Colonna 2017) |
| Biomedical | 3 | Cancer immunoediting (O'Donnell 2019) |
| CS/AI | 14 | Deep learning in medical imaging (Litjens 2017) |
| CS/AI | 15 | Federated learning (Yang 2019) |
| Materials | 27 | Halide perovskite PV (Jena 2019) |
| Materials | 31 | Solid state ion conductors (Zhang 2018) |
| Social Science | 40 | Social media & adolescent mental health (Keles 2019) |

### Tier B Selection (14 papers, ~3–4 per domain)

| Domain | Paper # | Short title |
|--------|---------|-------------|
| Biomedical | 8 | Long COVID mechanisms (Peluso 2024) |
| Biomedical | 9 | GLP-1R agonists CV benefits (Ussher 2023) |
| Biomedical | 10 | Gut-brain axis in neurodegeneration (Loh 2024) |
| CS/AI | 20 | RAG for LLMs survey (Gao 2023) |
| CS/AI | 22 | LLM hallucination survey (Zhang 2023) |
| CS/AI | 25 | Prompting methods in NLP (Liu 2023) |
| CS/AI | 23 | Diffusion models survey (Zhang 2023) |
| Materials | 35 | Perovskite chemical stability (Zhuang 2023) |
| Materials | 38 | MOFs for heavy metal removal (Lin 2023) |
| Materials | 37 | Biodegradable polymers (Kurowiak 2023) |
| Social Science | 45 | Misinformation psychology (Ecker 2022) |
| Social Science | 46 | COVID-19 mental health Europe (Ahmed 2023) |
| Social Science | 47 | Vaccine hesitancy umbrella review (Al Rahbeni 2024) |
| Social Science | 49 | Fake news detection review (Aïmeur 2023) |

**Total: 21 papers** (7 Tier A + 14 Tier B) — within the 17–23 target range.

---

## Selection Criteria Applied

1. **Topic distinctness**: No two papers in the final set cover the same narrow topic (e.g., only one hallucination paper, only one perovskite stability paper)
2. **Venue quality**: Prioritized Nature Reviews, Annual Reviews, Chemical Reviews, Cell, Lancet family, ACM/IEEE top venues
3. **Clear review format**: Excluded perspective pieces, commentaries, and primary research with review-like introductions
4. **Automation-friendly scope**: Chose reviews where the topic can be cleanly specified as a query to the pipeline (e.g., "gut-brain axis in neurodegeneration" vs. ambiguous scope)
5. **Citation network density**: Preferred topics with rich citation networks (many citable papers) for citation recall/precision testing
6. **Contamination balance**: Tier A papers are likely in training data (feature, not bug — enables contamination analysis). Tier B papers from 2024 have minimal contamination risk.
7. **Domain diversity within domains**: Within each domain, selected papers spanning different sub-fields (e.g., biomedical covers neuro + immuno-oncology + metabolic + infectious disease)
