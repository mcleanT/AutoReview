"""Pipeline node definitions — wires DAG nodes to implementation modules."""
from __future__ import annotations

from typing import Any

import structlog

from autoreview.analysis.clustering import ThematicClusterer
from autoreview.analysis.comprehensiveness import (
    BenchmarkValidator,
    BorderlineRescreener,
    CoverageAnomalyChecker,
    PostGapRevalidator,
    QueryCoverageChecker,
)
from autoreview.analysis.evidence_map import GapSeverity
from autoreview.analysis.gap_detector import GapDetector
from autoreview.config.models import DomainConfig
from autoreview.critique.holistic_critic import HolisticCritic, holistic_critique_loop
from autoreview.critique.outline_critic import OutlineCritic
from autoreview.critique.revision import outline_critique_loop
from autoreview.critique.section_critic import SectionCritic, section_critique_loop
from autoreview.extraction.extractor import PaperExtractor, PaperScreener
from autoreview.llm.prompts.outline import ReviewOutline
from autoreview.models.knowledge_base import KnowledgeBase, PipelinePhase
from autoreview.models.paper import CandidatePaper
from autoreview.writing.assembler import DraftAssembler
from autoreview.writing.outliner import OutlineGenerator
from autoreview.writing.section_writer import SectionWriter

logger = structlog.get_logger()


class PipelineNodes:
    """Collection of pipeline node functions."""

    def __init__(self, llm: Any, config: DomainConfig) -> None:
        self.llm = llm
        self.config = config

    async def query_expansion(self, kb: KnowledgeBase) -> None:
        """Node: Generate search queries and scope document."""
        from autoreview.llm.prompts.query_expansion import build_query_expansion_prompt

        prompt = build_query_expansion_prompt(
            kb.topic, kb.domain, self.config.search.date_range,
        )

        from autoreview.models.base import AutoReviewModel
        from pydantic import Field

        class QueryExpansionResult(AutoReviewModel):
            pubmed_queries: list[str] = Field(default_factory=list)
            semantic_scholar_queries: list[str] = Field(default_factory=list)
            openalex_queries: list[str] = Field(default_factory=list)
            perplexity_questions: list[str] = Field(default_factory=list)
            scope_document: str = ""

        response = await self.llm.generate_structured(
            prompt=prompt,
            response_model=QueryExpansionResult,
            system="You are an expert research librarian generating search queries for a literature review.",
        )
        result: QueryExpansionResult = response.parsed

        kb.search_queries = {
            "pubmed": result.pubmed_queries,
            "semantic_scholar": result.semantic_scholar_queries,
            "openalex": result.openalex_queries,
            "perplexity": result.perplexity_questions,
        }
        kb.scope_document = result.scope_document
        kb.current_phase = PipelinePhase.QUERY_EXPANSION
        kb.add_audit_entry(
            "query_expansion", "generated",
            f"Queries: {sum(len(v) for v in kb.search_queries.values())}",
            {"input_tokens": response.input_tokens, "output_tokens": response.output_tokens},
        )

        # Comprehensiveness: query coverage check
        query_checker = QueryCoverageChecker(self.llm)
        qc_result = await query_checker.check(kb.search_queries, kb.scope_document or "")
        kb.comprehensiveness_checks.append(qc_result)

    async def search(self, kb: KnowledgeBase) -> None:
        """Node: Execute multi-source search."""
        from autoreview.search.aggregator import SearchAggregator

        sources = []
        all_dbs = (
            self.config.databases.get("primary", [])
            + self.config.databases.get("secondary", [])
            + self.config.databases.get("discovery", [])
        )

        for db in all_dbs:
            try:
                if db == "pubmed":
                    from autoreview.search.pubmed import PubMedSearch
                    sources.append(PubMedSearch())
                elif db == "semantic_scholar":
                    from autoreview.search.semantic_scholar import SemanticScholarSearch
                    sources.append(SemanticScholarSearch())
                elif db == "openalex":
                    from autoreview.search.openalex import OpenAlexSearch
                    sources.append(OpenAlexSearch())
                elif db == "perplexity":
                    from autoreview.search.perplexity import PerplexitySearch
                    sources.append(PerplexitySearch())
            except Exception as e:
                logger.warning("search.source_init_failed", source=db, error=str(e))

        agg = SearchAggregator(sources=sources)
        kb.candidate_papers = await agg.search(
            kb.search_queries,
            max_results_per_source=self.config.search.max_results_per_source,
        )
        kb.current_phase = PipelinePhase.SEARCH
        kb.add_audit_entry("search", "complete", f"Found {len(kb.candidate_papers)} papers")

    async def screening(self, kb: KnowledgeBase) -> None:
        """Node: Screen papers for relevance."""
        screener = PaperScreener(self.llm, batch_size=self.config.search.screening_batch_size)
        kb.screened_papers = await screener.screen(
            kb.candidate_papers,
            scope_document=kb.scope_document or "",
            threshold=self.config.search.relevance_threshold,
        )
        kb.current_phase = PipelinePhase.SCREENING
        kb.add_audit_entry("screening", "complete", f"Screened to {len(kb.screened_papers)} papers")

        # Comprehensiveness: coverage anomaly check
        all_dbs = (
            self.config.databases.get("primary", [])
            + self.config.databases.get("secondary", [])
            + self.config.databases.get("discovery", [])
        )
        anomaly_checker = CoverageAnomalyChecker()
        anomaly_result = anomaly_checker.check(
            kb.candidate_papers, kb.screened_papers, expected_sources=all_dbs,
        )
        kb.comprehensiveness_checks.append(anomaly_result)

        # Comprehensiveness: borderline re-screening
        rescreener = BorderlineRescreener(self.llm)
        rescreen_result, promoted = await rescreener.rescreen(
            screener.borderline_papers,
            scope_document=kb.scope_document or "",
            threshold=self.config.search.relevance_threshold,
        )
        kb.comprehensiveness_checks.append(rescreen_result)
        if promoted:
            kb.screened_papers.extend(promoted)
            kb.add_audit_entry(
                "screening", "borderline_promoted",
                f"Promoted {len(promoted)} borderline papers",
            )

    async def extraction(self, kb: KnowledgeBase) -> None:
        """Node: Extract structured information from papers."""
        extractor = PaperExtractor(
            self.llm,
            domain_fields=self.config.extraction.domain_fields,
            max_concurrent=self.config.extraction.max_concurrent,
        )
        papers = [sp.paper for sp in kb.screened_papers]
        kb.extractions = await extractor.extract_batch(papers)
        kb.current_phase = PipelinePhase.EXTRACTION
        kb.add_audit_entry("extraction", "complete", f"Extracted {len(kb.extractions)} papers")

    async def clustering(self, kb: KnowledgeBase) -> None:
        """Node: Thematic clustering + contradiction detection + gap analysis."""
        clusterer = ThematicClusterer(self.llm)
        gap_detector = GapDetector(self.llm)

        evidence_map = await clusterer.build_evidence_map(
            kb.extractions, kb.scope_document or "",
        )

        gaps, coverage = await gap_detector.detect_gaps(
            evidence_map.themes, kb.scope_document or "",
        )
        evidence_map.gaps = gaps
        evidence_map.coverage_score = coverage

        kb.evidence_map = evidence_map
        kb.current_phase = PipelinePhase.CLUSTERING
        kb.add_audit_entry(
            "clustering", "complete",
            f"Themes: {len(evidence_map.themes)}, Gaps: {len(gaps)}, Coverage: {coverage:.2f}",
        )

    async def gap_search(self, kb: KnowledgeBase) -> None:
        """Node: Gap-aware supplementary search (conditional)."""
        # Store pre-gap state for revalidation
        pre_gaps = list(kb.evidence_map.gaps) if kb.evidence_map and kb.evidence_map.gaps else []
        pre_coverage = kb.evidence_map.coverage_score if kb.evidence_map else 0.0

        if not kb.evidence_map or not kb.evidence_map.gaps:
            await self._run_benchmark_validation(kb)
            return

        major_gaps = [g for g in kb.evidence_map.gaps if g.severity == "major"]
        if not major_gaps:
            await self._run_benchmark_validation(kb)
            return

        logger.info("gap_search.triggered", major_gaps=len(major_gaps))

        # Generate queries from gaps
        gap_queries: dict[str, list[str]] = {}
        for db in self.config.databases.get("primary", []):
            gap_queries[db] = []
            for gap in major_gaps:
                gap_queries[db].extend(gap.suggested_queries)

        # Re-use search infrastructure
        from autoreview.search.aggregator import SearchAggregator
        sources = []
        for db in self.config.databases.get("primary", []):
            try:
                if db == "semantic_scholar":
                    from autoreview.search.semantic_scholar import SemanticScholarSearch
                    sources.append(SemanticScholarSearch())
                elif db == "pubmed":
                    from autoreview.search.pubmed import PubMedSearch
                    sources.append(PubMedSearch())
            except Exception:
                pass

        if not sources:
            await self._run_benchmark_validation(kb)
            return

        agg = SearchAggregator(sources=sources)
        new_papers = await agg.search(gap_queries, max_results_per_source=100)

        # Screen and extract new papers
        screener = PaperScreener(self.llm)
        new_screened = await screener.screen(
            new_papers, scope_document=kb.scope_document or "",
        )

        extractor = PaperExtractor(self.llm, domain_fields=self.config.extraction.domain_fields)
        new_papers_list = [sp.paper for sp in new_screened]
        new_extractions = await extractor.extract_batch(new_papers_list)

        # Merge into existing state
        kb.candidate_papers.extend(new_papers)
        kb.screened_papers.extend(new_screened)
        kb.extractions.update(new_extractions)

        kb.current_phase = PipelinePhase.GAP_SEARCH
        kb.add_audit_entry(
            "gap_search", "complete",
            f"Added {len(new_screened)} papers, {len(new_extractions)} extractions",
        )

        # Comprehensiveness: post-gap re-validation
        revalidator = PostGapRevalidator(self.llm)
        reval_result = await revalidator.check(
            kb.evidence_map.themes if kb.evidence_map else [],
            kb.scope_document or "",
            pre_gaps=pre_gaps,
            pre_coverage=pre_coverage,
        )
        kb.comprehensiveness_checks.append(reval_result)

        # Update evidence map with new coverage info
        if kb.evidence_map and reval_result.metrics.get("post_coverage"):
            kb.evidence_map.coverage_score = reval_result.metrics["post_coverage"]

        await self._run_benchmark_validation(kb)

    async def _run_benchmark_validation(self, kb: KnowledgeBase) -> None:
        """Run benchmark validation check."""
        pipeline_dois = set()
        for p in kb.candidate_papers:
            if p.doi:
                pipeline_dois.add(p.doi.lower().strip())
        for sp in kb.screened_papers:
            if sp.paper.doi:
                pipeline_dois.add(sp.paper.doi.lower().strip())

        validator = BenchmarkValidator()
        bench_result = await validator.check(kb.topic, pipeline_dois)
        kb.comprehensiveness_checks.append(bench_result)

    async def outline(self, kb: KnowledgeBase) -> None:
        """Node: Generate and critique the outline."""
        generator = OutlineGenerator(self.llm)
        critic = OutlineCritic(self.llm)

        review_outline, critiques = await outline_critique_loop(
            llm=self.llm,
            outline_generator=generator,
            outline_critic=critic,
            evidence_map=kb.evidence_map,
            scope_document=kb.scope_document or "",
            required_sections=self.config.outline.required_sections,
            max_cycles=self.config.outline.max_critique_cycles,
            threshold=self.config.critique.score_threshold,
        )

        kb.outline = review_outline.model_dump()
        kb.critique_history.extend(critiques)
        kb.current_phase = PipelinePhase.OUTLINE
        kb.add_audit_entry("outline", "complete", f"Sections: {len(review_outline.sections)}")

    async def section_writing(self, kb: KnowledgeBase) -> None:
        """Node: Write and critique all sections."""
        outline = ReviewOutline.model_validate(kb.outline)
        writer = SectionWriter(self.llm)
        critic = SectionCritic(self.llm)

        drafts = await writer.write_all_sections(outline, kb.extractions, kb.evidence_map)

        # Critique each section
        for section_id, draft in drafts.items():
            final_draft, critiques = await section_critique_loop(
                llm=self.llm,
                critic=critic,
                draft=draft,
                outline=outline,
                max_cycles=self.config.critique.max_revision_cycles,
                threshold=self.config.critique.score_threshold,
            )
            drafts[section_id] = final_draft
            kb.critique_history.extend(critiques)

        kb.section_drafts = {sid: d.text for sid, d in drafts.items()}
        kb.current_phase = PipelinePhase.SECTION_CRITIQUE
        kb.add_audit_entry("section_writing", "complete", f"Sections: {len(drafts)}")

    async def assembly(self, kb: KnowledgeBase) -> None:
        """Node: Assemble draft and run holistic critique."""
        outline = ReviewOutline.model_validate(kb.outline)
        from autoreview.writing.section_writer import SectionDraft

        section_drafts = {
            sid: SectionDraft(section_id=sid, title=sid, text=text)
            for sid, text in kb.section_drafts.items()
        }

        assembler = DraftAssembler()
        full_draft = assembler.assemble(outline, section_drafts)

        # Holistic critique loop
        critic = HolisticCritic(self.llm)
        final_draft, critiques = await holistic_critique_loop(
            llm=self.llm,
            critic=critic,
            full_draft=full_draft,
            scope_document=kb.scope_document or "",
            max_cycles=self.config.critique.max_revision_cycles,
            threshold=self.config.critique.score_threshold,
            convergence_delta=self.config.critique.convergence_delta,
        )

        kb.full_draft = final_draft
        kb.critique_history.extend(critiques)
        kb.current_phase = PipelinePhase.HOLISTIC_CRITIQUE
        kb.add_audit_entry("assembly", "complete", f"Words: {len(final_draft.split())}")

    async def final_polish(self, kb: KnowledgeBase) -> None:
        """Node: Language polish and consistency pass."""
        if not kb.full_draft:
            return

        response = await self.llm.generate(
            prompt=f"Polish this review paper for language, terminology consistency, and flow. "
            f"Maintain all citation markers [@paper_id]. Do not change the structure.\n\n"
            f"{kb.full_draft}",
            system="You are an expert scientific editor performing final language polish.",
            temperature=0.3,
        )

        kb.full_draft = response.content
        kb.current_phase = PipelinePhase.FINAL_POLISH
        kb.add_audit_entry(
            "final_polish", "complete",
            token_usage={"input_tokens": response.input_tokens, "output_tokens": response.output_tokens},
        )
