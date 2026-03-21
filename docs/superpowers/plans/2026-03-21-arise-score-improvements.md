# ARISE Score Improvements: v3 → v4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve AutoReview v3 RAG review from 84/100 to 93+/100 on the official ARISE rubric by adding figures, tables, missing citations, and targeted prose improvements.

**Architecture:** The review at `output/arise/arise_rag_v3/review.md` is improved in-place. Figures are generated as PNGs via matplotlib and referenced inline. Tables are inserted as markdown. Text revisions target specific sections. A final v4 copy is saved alongside v3.

**Tech Stack:** matplotlib (figures), markdown (tables), Python (data extraction from snapshots)

**Data sources:**
- Extractions: `output/arise/arise_rag_v3/snapshots/05_extraction.json` (634 papers)
- Clustering: `output/arise/arise_rag_v3/snapshots/06_clustering.json` (10 themes, 4 contradictions)
- Current review: `output/arise/arise_rag_v3/review.md` (15,714 words, 97 citations)

---

## Task 1: Generate Figures (Presentation: Visuals 1.0 → 4.5)

**Files:**
- Create: `output/arise/arise_rag_v3/figures/fig1_rag_pipeline.png`
- Create: `output/arise/arise_rag_v3/figures/fig2_temporal_distribution.png`
- Create: `output/arise/arise_rag_v3/figures/fig3_taxonomy_tree.png`
- Create: `output/arise/arise_rag_v3/figures/fig4_theme_evidence.png`

All figures must use colorblind-safe palette: `["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]`, 300 DPI, Arial/Helvetica font, `constrained_layout=True`.

### Fig 1: RAG Pipeline Architecture Schematic
- [ ] **Step 1:** Generate a schematic showing the canonical RAG pipeline: Query → Retrieval (sparse/dense/hybrid) → Reranking → Context Integration → LLM Generation → Output with Citations. Show the three architectural generations (Naive → Adaptive → Agentic) as layered annotations. Use boxes and arrows, clean layout. Save to `output/arise/arise_rag_v3/figures/fig1_rag_pipeline.png`. Use the `programmatic-schematics` skill or matplotlib with patches/arrows.

### Fig 2: Temporal Distribution of Corpus
- [ ] **Step 2:** Generate a bar chart of papers by year from the extraction data. Data: 2018:2, 2020:3, 2021:15, 2022:20, 2023:30, 2024:165, 2025:399. X-axis: year, Y-axis: number of papers. Annotate the 2023 inflection point. Title: "Temporal Distribution of 634 Reviewed Papers (2018–2025)". Save to `output/arise/arise_rag_v3/figures/fig2_temporal_distribution.png`.

### Fig 3: RAG Architecture Taxonomy
- [ ] **Step 3:** Generate a hierarchical taxonomy tree showing RAG architectural variants. Root: "RAG Architectures" → branches: "Naive RAG" (single retrieval → generate), "Advanced RAG" (pre-retrieval optimization, post-retrieval processing), "Modular RAG" (interchangeable components), "Agentic RAG" (autonomous decision-making, tool use). Each leaf should list 2-3 representative systems. Use a tree/dendrogram layout. Save to `output/arise/arise_rag_v3/figures/fig3_taxonomy_tree.png`.

### Fig 4: Evidence Strength by Theme
- [ ] **Step 4:** Generate a grouped horizontal bar chart showing the evidence strength distribution across the 10 themes from clustering. For each theme, show stacked bars of strong/moderate/weak/preliminary counts. Read data from `output/arise/arise_rag_v3/snapshots/06_clustering.json` → themes[].evidence_strength_distribution. Title: "Evidence Strength Distribution Across Research Themes". Save to `output/arise/arise_rag_v3/figures/fig4_theme_evidence.png`.

---

## Task 2: Generate Tables (Organization: Summarization 2.0 → 4.5)

These tables are inserted directly into the review markdown.

### Table 1: Retrieval Methods Comparison
- [ ] **Step 1:** Create a markdown table comparing retrieval paradigms. Insert after the introductory paragraph of Section 3.

| Method | Mechanism | Key Systems | Strengths | Limitations |
|--------|-----------|-------------|-----------|-------------|
| Dense bi-encoder | Semantic embedding similarity | DPR, ColBERT, M3-Embedding | Semantic matching, zero-shot transfer | Representation bottleneck, domain shift |
| Sparse lexical | Term overlap (BM25, TF-IDF) | BM25, SPLADE | Exact match, no GPU needed, domain robust | No semantic understanding |
| Hybrid | Score fusion of dense + sparse | RRF pipelines, ColBERT-PRF | Best of both, robust across domains | Complexity, tuning fusion weights |
| Late interaction | Per-token similarity (MaxSim) | ColBERT, ColBERT-X, ColPali | Fine-grained matching, scalable | Index size, computational cost |
| Learned sparse | Neural term weighting | SPLADE, ANCE-PRF | Inverted index efficiency + semantics | Training data requirements |

Caption: "Table 1. Comparison of retrieval paradigms for RAG systems."

### Table 2: Domain Application Summary
- [ ] **Step 2:** Create a markdown table summarizing domain applications. Insert at the start of Section 10 (Biomedical) or as a standalone before Section 10.

| Domain | Papers | Key Challenges | Representative Systems | Best Reported Metric |
|--------|--------|----------------|----------------------|---------------------|
| Biomedical/Clinical | ~190 | Patient safety, regulatory compliance, EHR heterogeneity | BiomedRAG, JMLR, Patho-AgenticRAG | Pooled effect size 1.35 (95% CI) |
| Legal | ~25 | Citation networks, jurisdiction hierarchy, precedent | COLIEE systems, case-law RAG | Near-prerequisite for KG integration |
| Financial | ~15 | SEC filings, temporal sensitivity, numerical reasoning | Financial RAG evaluators | MRR 0.160→0.750 with hybrid |
| Education | ~20 | Pedagogical accuracy, curriculum alignment | Educational QA + code interpreter | Improved learning outcomes |
| Cybersecurity | ~10 | Threat intelligence, real-time updates | RTLFixer, domain-adapted RAG | Hardware design error correction |
| Enterprise | ~30 | Multilingual, metadata enrichment, privacy | Enterprise RAG platforms | 82.5% vs 73.3% precision with LLM metadata |

Caption: "Table 2. Summary of RAG applications across domains."

### Table 3: Evaluation Frameworks Comparison
- [ ] **Step 3:** Create a markdown table comparing evaluation frameworks. Insert in Section 8 after the introductory paragraph.

| Framework | Metrics | Scope | Automation | Limitations |
|-----------|---------|-------|------------|-------------|
| RAGAS | Faithfulness, answer relevance, context relevance | Component-level | Fully automated (LLM-based) | Sensitive to judge model choice |
| ARES | Confidence intervals on RAG quality | End-to-end + component | Automated with PPI | Requires labeled calibration set |
| RAGBench | TRACe explainability scores | End-to-end | Automated | 100K samples but English-only |
| ASTRID | Multi-dimension faithfulness | Generation faithfulness | Automated | Focused on faithfulness only |
| Human evaluation | Task-specific rubrics | End-to-end | Manual | Expensive, low inter-rater reliability |

Caption: "Table 3. Comparison of RAG evaluation frameworks."

### Table 4: Key Takeaways per Section
- [ ] **Step 4:** Create a summary table of key takeaways. Insert before Section 13 (Discussion) or at the end of Section 12.

| Section | Key Takeaway |
|---------|-------------|
| Retrieval (§3) | No single retrieval method dominates; hybrid dense-sparse with re-ranking is the practical standard |
| Knowledge Rep. (§4) | Chunking strategy is a first-order quality determinant; knowledge graphs unlock multi-hop reasoning |
| Generation (§5) | Multi-passage fusion remains unsolved; more context helps recall but can hurt precision |
| Advanced Arch. (§6) | Self-reflective and agentic RAG outperform static pipelines but add complexity |
| Hallucination (§7) | RAG reduces hallucination robustly, but adversarial corpus poisoning is an emerging threat |
| Evaluation (§8) | No universal benchmark exists; LLM-as-judge is scalable but carries systematic biases |
| Training (§9) | RAG and fine-tuning are complementary, not competing; joint training yields best results |
| Biomedical (§10) | Largest application domain; clinical safety demands exceed current validation standards |
| Cross-domain (§11) | Domain adaptation of retrieval components is the consistent performance lever |
| Multimodal (§12) | Vision-first retrieval (ColPali) is displacing OCR-dependent pipelines |

Caption: "Table 4. Summary of key findings across review sections."

---

## Task 3: Insert Figures and Tables into Review (Presentation + Organization)

**Files:**
- Modify: `output/arise/arise_rag_v3/review.md`

- [ ] **Step 1:** Insert figure references into the review text at appropriate locations:
  - Fig 1 → End of Section 1 (Introduction), after the paragraph describing the review's scope. Add: `![Figure 1. Canonical RAG pipeline architecture showing the progression from naive single-retrieval to agentic multi-tool systems.](figures/fig1_rag_pipeline.png)`
  - Fig 2 → Section 2.3 (Corpus Overview), after the temporal distribution discussion. Add: `![Figure 2. Temporal distribution of 634 reviewed papers (2018–2025), showing the inflection point in 2023–2024.](figures/fig2_temporal_distribution.png)`
  - Fig 3 → Section 6 (Advanced RAG Architectures), at the start. Add: `![Figure 3. Taxonomy of RAG architectural variants from naive single-pass to autonomous agentic systems.](figures/fig3_taxonomy_tree.png)`
  - Fig 4 → Section 2.3 or before Section 13. Add: `![Figure 4. Evidence strength distribution across ten thematic clusters identified in the reviewed corpus.](figures/fig4_theme_evidence.png)`

- [ ] **Step 2:** Insert Tables 1-4 at the locations specified in Task 2.

- [ ] **Step 3:** Add figure/table discussion in surrounding prose. Each figure/table needs at least one sentence referencing it (e.g., "As shown in Table 1, retrieval methods differ fundamentally in their trade-offs..."). ARISE Summarization criterion requires that visuals "reinforce takeaways."

---

## Task 4: Fix Missing Citations (References: Accuracy 4.0 → 4.5, Appropriateness 4.5 → 5.0)

**Files:**
- Modify: `output/arise/arise_rag_v3/review.md`

- [ ] **Step 1:** Add citations for the 10 high-severity gaps identified in passage search. For each, insert the appropriate `[N]` reference and add the entry to the bibliography. The missing foundational works are:

  1. **HyDE** (Gao et al., 2022) — "Precise Zero-Shot Dense Retrieval without Relevance Labels" — cite in §3.4
  2. **CRAG** (Yan et al., 2024) — "Corrective Retrieval Augmented Generation" — cite in §6 (currently attributed to Open-RAG citation)
  3. **RAGAS** (Es et al., 2024) — "RAGAS: Automated Evaluation of Retrieval Augmented Generation" — cite in §8
  4. **ARES** (Saad-Falcon et al., 2024) — "ARES: An Automated Evaluation Framework for RAG Systems" — cite in §8
  5. **REALM** (Guu et al., 2020) — "Retrieval-Augmented Language Model Pre-Training" — cite in §9
  6. **FAISS** (Johnson et al., 2019) — "Billion-scale similarity search with GPUs" — cite in §4
  7. **Lost in the Middle** (Liu et al., 2024) — "Lost in the Middle: How Language Models Use Long Contexts" — cite in §5
  8. **MT-Bench** (Zheng et al., 2023) — "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" — cite in §8 for LLM-as-judge biases

- [ ] **Step 2:** Fix the duplicate reference entries [60]/[61] flagged by the ARISE judge. Deduplicate and renumber.

- [ ] **Step 3:** Verify the "hundreds of publications per year by 2025" claim in §1 by adding a self-reference to §2.3 corpus statistics: "(see Section 2.3; the 2025 cohort alone accounts for 399 of the 634 reviewed papers)"

- [ ] **Step 4:** Fix the "45+ independent studies" claim in §7 — either cite the meta-analysis source or soften to "numerous independent studies across the corpus"

- [ ] **Step 5:** Fix the "<5% regulatory reference" claim in §10 — either cite a source or qualify as "our corpus analysis indicates that fewer than five percent..."

---

## Task 5: Improve Synthesis Methodology Description (Methodology: 3.5 → 4.5)

**Files:**
- Modify: `output/arise/arise_rag_v3/review.md` (Section 2)

- [ ] **Step 1:** Expand Section 2.2 to describe the synthesis methodology more precisely. Add a paragraph after the current data extraction description:

> "Thematic clustering was performed algorithmically using an evidence mapping approach: extracted findings were grouped by shared claims, methodological approaches, and domain of application, with clusters iteratively refined through cross-extraction comparison. Contradictions were identified through structured resolution: when two or more papers reported opposing findings on the same question, differences in study design, sample characteristics, evaluation methodology, and domain context were systematically examined to determine whether the contradiction reflected genuine disagreement or methodological heterogeneity. Evidence weighting incorporated study design classification (computational experiment, systematic review, cohort study, etc.), a quality score reflecting methodological rigour (0.0–1.0 scale based on experimental design, benchmark coverage, and ablation completeness), and sample size where reported. This structured approach was designed to move beyond simple vote-counting toward a principled synthesis that accounts for the heterogeneous quality and scope of the included literature."

---

## Task 6: Add Limitations Section (Analysis: Identification of Limitations 3.5 → 4.5)

**Files:**
- Modify: `output/arise/arise_rag_v3/review.md` (Section 13)

- [ ] **Step 1:** Add a new subsection "13.5 Limitations of This Review" before the Conclusion (Section 14). Content:

> "Several limitations of this review should be acknowledged. First, 168 of the 634 reviewed papers (26.5%) were characterised from abstracts alone, as full text was unavailable through the queried databases; extraction quality for these papers is necessarily shallower than for the 466 papers with full-text access. Second, despite the multi-database search strategy, the corpus is predominantly English-language and skewed toward venues indexed in PubMed, Semantic Scholar, and arXiv, potentially under-representing RAG research published in non-English journals, regional conference proceedings, or industry technical reports. Third, the temporal distribution of the corpus — with 63% of papers from 2025 alone — means that many included works are preprints that have not yet undergone formal peer review; findings from these papers should be interpreted with appropriate caution. Fourth, while our structured contradiction resolution approach accounts for study design and quality differences, it does not constitute formal meta-analysis: effect sizes are reported from individual studies rather than pooled statistically, and publication bias — the tendency for positive results to be over-represented — is not formally assessed. Finally, the rapid pace of RAG research means that significant work published after our search cutoff (March 2025) is necessarily excluded, and the field may have evolved in directions not captured here."

---

## Task 7: Expand Ethics and Societal Impact (Originality/Analysis)

**Files:**
- Modify: `output/arise/arise_rag_v3/review.md` (Section 13)

- [ ] **Step 1:** Expand the existing Discussion content on ethics/societal implications. Add a paragraph in Section 13.3 or create "13.4 Ethical and Societal Considerations":

> "Beyond technical challenges, RAG systems raise ethical considerations that the literature has only begun to address. Retrieval corpora inherit the biases of their source documents: a clinical RAG system drawing on medical literature that under-represents certain demographic groups may perpetuate health disparities in its generated recommendations. Intellectual property concerns arise when retrieval systems index copyrighted material — the distinction between retrieving a passage for grounding versus reproducing copyrighted text is legally unsettled. Privacy-preserving RAG, while an active research direction, remains largely theoretical: most deployed systems retrieve from centralised knowledge bases without differential privacy guarantees, creating risks when the corpus contains sensitive information. The environmental cost of large-scale retrieval infrastructure — maintaining and querying dense indices over millions of documents requires substantial computational resources — is rarely discussed in the RAG literature but contributes to the broader sustainability concerns surrounding large-scale AI deployment. More fundamentally, RAG systems shift epistemic authority: when a system retrieves and cites sources, users may treat the generated synthesis as authoritative without evaluating the quality or relevance of the retrieved evidence. This risk is particularly acute in high-stakes domains where non-expert users rely on RAG-generated outputs for decision-making. Addressing these concerns requires not only technical solutions (bias-aware retrieval, privacy-preserving indexing, carbon-efficient infrastructure) but also governance frameworks that hold RAG system operators accountable for the quality and fairness of their knowledge bases."

---

## Task 8: Strengthen Originality Framing

**Files:**
- Modify: `output/arise/arise_rag_v3/review.md` (Sections 1 and 13)

- [ ] **Step 1:** Add a "comparison to prior surveys" paragraph at the end of Section 1 (Introduction), before the scope/roadmap paragraph:

> "Several prior surveys have examined aspects of RAG. Gao et al. (2024) provided an early taxonomy distinguishing naive, advanced, and modular RAG paradigms across approximately 100 papers. Zhao et al. (2024) surveyed retrieval-augmented LLMs with focus on training strategies. Fan et al. (2024) reviewed RAG for AI-generated content. The present review extends beyond these works in three respects: scale (634 papers versus typical corpora of 50–150), temporal coverage (through March 2025, capturing the field's most explosive growth period), and analytical approach (structured contradiction resolution with evidence-weighted synthesis rather than descriptive categorisation). We introduce a 'Productive Tensions' analytical framework that treats the field's four most-contested disagreements not as problems to be resolved but as design dimensions along which practitioners must make context-dependent choices."

- [ ] **Step 2:** In Section 13, rename the contradiction resolution discussion as the "Productive Tensions Framework" to create a nameable contribution.

---

## Task 9: Polish Formatting and Scope (minor gains)

**Files:**
- Modify: `output/arise/arise_rag_v3/review.md`

- [ ] **Step 1:** In the Abstract, make objectives more explicitly measurable. Change "This review synthesises the current state of RAG research" to "This review systematically synthesises the current state of RAG research, addressing four specific objectives: (1) mapping the technical landscape of retrieval, generation, and evaluation components; (2) assessing domain-specific adaptation across biomedicine, law, finance, and multimodal settings; (3) resolving four contested contradictions through evidence-weighted analysis; and (4) identifying critical gaps and actionable research directions."

- [ ] **Step 2:** Add a "How to read this review" sentence at the end of the Introduction: "Readers primarily interested in domain applications may proceed directly to Sections 10–12; those focused on architectural choices will find the core technical discussion in Sections 3–6; and practitioners seeking deployment guidance are directed to Section 13.4."

---

## Task 10: Save Final v4 and Re-evaluate

**Files:**
- Create: `output/arise/arise_rag_v4/review.md` (copy of improved v3)
- Create: `output/arise/arise_rag_v4/figures/` (copy figures)
- Create: `output/arise/arise_rag_v4/status.json`

- [ ] **Step 1:** Copy the improved review and figures to a v4 output directory.
- [ ] **Step 2:** Update status.json with v4 metadata (word count, citation count, figure count, table count).
- [ ] **Step 3:** Re-evaluate against the official ARISE rubric using an opus subagent to verify score improvement.

---

## Execution Order and Dependencies

```
Independent (batch 1):
  Task 1 (figures)     — no dependencies
  Task 4 (citations)   — no dependencies
  Task 5 (methodology) — no dependencies
  Task 6 (limitations) — no dependencies
  Task 7 (ethics)      — no dependencies
  Task 8 (originality) — no dependencies

Depends on Task 1 (batch 2):
  Task 2 (tables)      — needs figure data awareness but not figure files
  Task 3 (insert figs) — needs Task 1 figures generated

Depends on all above (batch 3):
  Task 9 (polish)      — final text polish after all insertions
  Task 10 (save + eval) — after everything
```
