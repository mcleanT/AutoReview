"""Pipeline node definitions — wires DAG nodes to implementation modules."""

from __future__ import annotations

import os
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
from autoreview.analysis.contextual_enricher import ContextualEnricher
from autoreview.analysis.gap_detector import GapDetector
from autoreview.config.models import DomainConfig
from autoreview.critique.holistic_critic import HolisticCritic, holistic_critique_loop
from autoreview.critique.outline_critic import OutlineCritic
from autoreview.critique.revision import outline_critique_loop
from autoreview.critique.section_critic import SectionCritic, section_critique_loop
from autoreview.extraction.extractor import PaperExtractor, PaperScreener
from autoreview.llm.prompts.outline import ReviewOutline
from autoreview.models.enrichment import CorpusExpansionResult, SectionEnrichment
from autoreview.models.knowledge_base import KnowledgeBase, PipelinePhase
from autoreview.models.paper import CandidatePaper
from autoreview.validation.citation_validator import CitationValidator
from autoreview.writing.assembler import DraftAssembler
from autoreview.writing.narrative_architect import NarrativeArchitect
from autoreview.writing.outliner import OutlineGenerator
from autoreview.writing.section_writer import SectionWriter

logger = structlog.get_logger()


class TokenBudgetExceeded(Exception):
    """Raised when cumulative token usage exceeds the configured budget."""

    def __init__(self, used: int, budget: int) -> None:
        self.used = used
        self.budget = budget
        super().__init__(f"Token budget exceeded: {used:,} used vs {budget:,} budget")


class _TokenAccumulator:
    """Wraps an LLM provider to accumulate token counts across multiple calls.

    Create a local instance per node invocation. Pass it in place of
    ``self.llm`` to helper classes so that every ``generate`` /
    ``generate_structured`` call is transparently counted.  After the
    node finishes, read ``usage`` to get the aggregate totals.

    When ``global_accumulator`` is set, token counts are also added to it
    so that a pipeline-wide budget can be enforced.
    """

    def __init__(
        self,
        llm: Any,
        global_accumulator: _GlobalTokenAccumulator | None = None,
    ) -> None:
        self._llm = llm
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cache_creation_input_tokens: int = 0
        self.cache_read_input_tokens: int = 0
        self._global = global_accumulator

    def _track(self, response: Any) -> None:
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        self.cache_creation_input_tokens += getattr(response, "cache_creation_input_tokens", 0)
        self.cache_read_input_tokens += getattr(response, "cache_read_input_tokens", 0)
        if self._global:
            self._global.add(response.input_tokens, response.output_tokens)

    async def generate(self, *args: Any, **kwargs: Any) -> Any:
        response = await self._llm.generate(*args, **kwargs)
        self._track(response)
        return response

    async def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
        response = await self._llm.generate_structured(*args, **kwargs)
        self._track(response)
        return response

    @property
    def usage(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
        }


class _GlobalTokenAccumulator:
    """Pipeline-wide token counter for budget enforcement."""

    def __init__(self, budget: int | None = None) -> None:
        self.total_input: int = 0
        self.total_output: int = 0
        self.budget = budget

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.total_input += input_tokens
        self.total_output += output_tokens
        if self.budget and (self.total_input + self.total_output) > self.budget:
            raise TokenBudgetExceeded(
                self.total_input + self.total_output,
                self.budget,
            )

    @property
    def total(self) -> int:
        return self.total_input + self.total_output


class PipelineNodes:
    """Collection of pipeline node functions."""

    def __init__(self, llm: Any, config: DomainConfig) -> None:
        self.llm = llm
        self.config = config
        self._global_tokens = _GlobalTokenAccumulator(
            budget=config.llm.token_budget,
        )
        from autoreview.pipeline.remediation import RemediationDispatcher

        self.dispatcher = RemediationDispatcher(llm, config)

    async def query_expansion(self, kb: KnowledgeBase) -> None:
        """Node: Generate search queries and scope document."""
        from autoreview.llm.prompts.query_expansion import build_query_expansion_prompt

        prompt = build_query_expansion_prompt(
            kb.topic,
            kb.domain,
            self.config.search.date_range,
        )

        from pydantic import Field

        from autoreview.models.base import AutoReviewModel

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
            "query_expansion",
            "generated",
            f"Queries: {sum(len(v) for v in kb.search_queries.values())}",
            {"input_tokens": response.input_tokens, "output_tokens": response.output_tokens},
        )

        # Comprehensiveness: query coverage check
        query_checker = QueryCoverageChecker(self.llm)
        qc_result = await query_checker.check(kb.search_queries, kb.scope_document or "")
        kb.comprehensiveness_checks.append(qc_result)

        # Auto-remediate: expand queries for uncovered topics
        if qc_result.remediation:
            await self.dispatcher.execute(kb, qc_result)
            # Re-check after expansion
            qc_recheck = await query_checker.check(kb.search_queries, kb.scope_document or "")
            kb.comprehensiveness_checks.append(qc_recheck)
            # Second round if still warning
            if qc_recheck.remediation:
                await self.dispatcher.execute(kb, qc_recheck)

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
        tracker = _TokenAccumulator(self.llm, self._global_tokens)
        screener = PaperScreener(tracker, batch_size=self.config.search.screening_batch_size)
        kb.screened_papers = await screener.screen(
            kb.candidate_papers,
            scope_document=kb.scope_document or "",
            threshold=self.config.search.relevance_threshold,
        )

        # Log threshold and score distribution
        from collections import Counter

        score_dist = Counter(sp.relevance_score for sp in kb.screened_papers)
        logger.info(
            "screening.score_distribution",
            threshold=self.config.search.relevance_threshold,
            total_passed=len(kb.screened_papers),
            distribution=dict(sorted(score_dist.items())),
        )

        kb.current_phase = PipelinePhase.SCREENING

        # Comprehensiveness: coverage anomaly check
        all_dbs = (
            self.config.databases.get("primary", [])
            + self.config.databases.get("secondary", [])
            + self.config.databases.get("discovery", [])
        )
        anomaly_checker = CoverageAnomalyChecker()
        anomaly_result = anomaly_checker.check(
            kb.candidate_papers,
            kb.screened_papers,
            expected_sources=all_dbs,
        )
        kb.comprehensiveness_checks.append(anomaly_result)

        # Auto-remediate: lower threshold or expand queries
        if anomaly_result.remediation:
            await self.dispatcher.execute(kb, anomaly_result)

        # Comprehensiveness: borderline re-screening
        rescreener = BorderlineRescreener(tracker)
        rescreen_result, promoted = await rescreener.rescreen(
            screener.borderline_papers,
            scope_document=kb.scope_document or "",
            threshold=self.config.search.relevance_threshold,
        )
        kb.comprehensiveness_checks.append(rescreen_result)
        if promoted:
            kb.screened_papers.extend(promoted)
            kb.add_audit_entry(
                "screening",
                "borderline_promoted",
                f"Promoted {len(promoted)} borderline papers",
            )

        kb.add_audit_entry(
            "screening",
            "complete",
            f"Screened to {len(kb.screened_papers)} papers",
            tracker.usage,
        )

    async def full_text_retrieval(self, kb: KnowledgeBase) -> None:
        """Node: Retrieve full text from multiple sources.

        Chains strategies in priority order:
          1. Elsevier ScienceDirect API (requires ELSEVIER_API_KEY)
          2. Semantic Scholar openAccessPdf
          3. PubMed Central (JATS XML via PMCID)
          4. arXiv / bioRxiv / medRxiv PDFs
          5. Unpaywall (DOI-based lookup, tries all available URLs)
          6. Springer Nature API (requires SPRINGER_API_KEY)
        """
        from autoreview.search.full_text import FullTextResolver

        resolver = FullTextResolver(
            unpaywall_email=os.environ.get("UNPAYWALL_EMAIL"),
            entrez_email=os.environ.get("ENTREZ_EMAIL"),
            elsevier_api_key=os.environ.get("ELSEVIER_API_KEY"),
            springer_api_key=os.environ.get("SPRINGER_API_KEY"),
        )
        try:
            source_counts = await resolver.resolve(kb.screened_papers)
        finally:
            await resolver.close()

        total_enriched = sum(source_counts.values())
        total_papers = len(kb.screened_papers)
        abstract_only = sum(
            1 for sp in kb.screened_papers if not sp.paper.full_text and sp.paper.abstract
        )
        title_only = sum(
            1 for sp in kb.screened_papers if not sp.paper.full_text and not sp.paper.abstract
        )
        details = ", ".join(f"{k}: {v}" for k, v in sorted(source_counts.items()))

        logger.info(
            "full_text.coverage",
            total=total_papers,
            full_text=total_enriched,
            abstract_only=abstract_only,
            title_only=title_only,
            pct_full_text=round(total_enriched / total_papers * 100, 1) if total_papers else 0,
        )

        kb.current_phase = PipelinePhase.FULL_TEXT_RETRIEVAL
        kb.add_audit_entry(
            "full_text_retrieval",
            "complete",
            f"Enriched {total_enriched}/{total_papers} papers with full text "
            f"({details}); {abstract_only} abstract-only, {title_only} title-only",
        )

    def _effective_max_concurrent(self) -> int:
        """Return extraction concurrency, capped for Ollama providers."""
        from autoreview.llm.ollama import OllamaLLMProvider

        llm = self.llm
        if isinstance(llm, OllamaLLMProvider):
            return self.config.extraction.ollama_max_concurrent
        return self.config.extraction.max_concurrent

    async def extraction(self, kb: KnowledgeBase) -> None:
        """Node: Extract structured information from papers in batches."""
        tracker = _TokenAccumulator(self.llm, self._global_tokens)
        extractor = PaperExtractor(
            tracker,
            domain_fields=self.config.extraction.domain_fields,
            max_concurrent=self._effective_max_concurrent(),
            full_text_max_chars=self.config.extraction.full_text_max_chars,
            tiered_models=self.config.extraction.tiered_models,
            section_truncation=self.config.extraction.section_truncation,
        )
        batch_size = self.config.extraction.extraction_batch_size
        papers = kb.screened_papers
        total_batches = (len(papers) + batch_size - 1) // batch_size

        for i in range(0, len(papers), batch_size):
            batch = papers[i : i + batch_size]
            batch_num = i // batch_size + 1
            logger.info(
                "extraction.batch_start",
                batch=batch_num,
                total=total_batches,
                papers=len(batch),
            )
            results = await extractor.extract_batch(batch)
            kb.extractions.update(results)
            kb.save_snapshot(f"extraction_batch_{batch_num}")

        kb.current_phase = PipelinePhase.EXTRACTION
        kb.add_audit_entry(
            "extraction",
            "complete",
            f"Extracted {len(kb.extractions)} papers in {total_batches} batches",
            tracker.usage,
        )

    async def clustering(self, kb: KnowledgeBase) -> None:
        """Node: Thematic clustering + contradiction detection + gap analysis + evidence chains."""
        from autoreview.analysis.evidence_chains import EvidenceChainBuilder

        tracker = _TokenAccumulator(self.llm, self._global_tokens)
        clusterer = ThematicClusterer(tracker)
        gap_detector = GapDetector(tracker)

        evidence_map = await clusterer.build_evidence_map(
            kb.extractions,
            kb.scope_document or "",
        )

        gaps, coverage = await gap_detector.detect_gaps(
            evidence_map.themes,
            kb.scope_document or "",
        )
        evidence_map.gaps = gaps
        evidence_map.coverage_score = coverage

        # Build evidence chains and enrich contradictions
        chain_builder = EvidenceChainBuilder(tracker)
        paper_years: dict[str, int] = {}
        for sp in kb.screened_papers:
            if sp.paper.year:
                paper_years[sp.paper.id] = sp.paper.year

        chains = await chain_builder.build_chains(
            kb.extractions,
            evidence_map.themes,
            paper_years,
        )
        evidence_map.evidence_chains = [c.model_dump() for c in chains]

        enriched = await chain_builder.enrich_contradictions(
            evidence_map.contradictions,
            kb.extractions,
        )
        evidence_map.enriched_contradictions = [e.model_dump() for e in enriched]

        progressions = chain_builder.detect_temporal_progressions(
            list(kb.extractions.keys()),
            kb.extractions,
            paper_years,
        )
        evidence_map.temporal_progressions = [t.model_dump() for t in progressions]

        kb.evidence_map = evidence_map
        kb.current_phase = PipelinePhase.CLUSTERING
        kb.add_audit_entry(
            "clustering",
            "complete",
            f"Themes: {len(evidence_map.themes)}, Gaps: {len(gaps)}, "
            f"Coverage: {coverage:.2f}, Chains: {len(chains)}",
            tracker.usage,
        )

    async def gap_search(self, kb: KnowledgeBase) -> None:
        """Node: Gap-aware supplementary search (conditional)."""
        tracker = _TokenAccumulator(self.llm, self._global_tokens)
        # Store pre-gap state for revalidation
        pre_gaps = list(kb.evidence_map.gaps) if kb.evidence_map and kb.evidence_map.gaps else []
        pre_coverage = kb.evidence_map.coverage_score if kb.evidence_map else 0.0

        if not kb.evidence_map or not kb.evidence_map.gaps:
            await self._run_benchmark_validation(kb)
            return

        coverage_score = kb.evidence_map.coverage_score if kb.evidence_map else 0.0
        coverage_threshold = getattr(self.config.search, "min_coverage_threshold", 0.75)
        all_gaps = kb.evidence_map.gaps
        major_gaps = [g for g in all_gaps if g.severity == "major"]

        if not all_gaps and coverage_score >= coverage_threshold:
            await self._run_benchmark_validation(kb)
            return

        if not major_gaps and coverage_score >= coverage_threshold:
            await self._run_benchmark_validation(kb)
            return

        gaps_to_search = all_gaps  # Include minor gaps in query generation
        logger.info(
            "gap_search.triggered",
            major_gaps=len(major_gaps),
            total_gaps=len(gaps_to_search),
            coverage_score=coverage_score,
        )

        # Generate queries from gaps
        gap_dbs = self.config.databases.get("primary", []) + self.config.databases.get(
            "secondary", []
        )
        gap_queries: dict[str, list[str]] = {}
        for db in gap_dbs:
            gap_queries[db] = []
            for gap in gaps_to_search:
                gap_queries[db].extend(gap.suggested_queries)

        # Re-use search infrastructure
        from autoreview.search.aggregator import SearchAggregator

        sources = []
        for db in gap_dbs:
            try:
                if db == "semantic_scholar":
                    from autoreview.search.semantic_scholar import SemanticScholarSearch

                    sources.append(SemanticScholarSearch())
                elif db == "pubmed":
                    from autoreview.search.pubmed import PubMedSearch

                    sources.append(PubMedSearch())
                elif db == "openalex":
                    from autoreview.search.openalex import OpenAlexSearch

                    sources.append(OpenAlexSearch())
            except Exception:
                pass

        if not sources:
            await self._run_benchmark_validation(kb)
            return

        agg = SearchAggregator(sources=sources)
        new_papers = await agg.search(gap_queries, max_results_per_source=200)

        # Deduplicate by DOI against existing corpus
        existing_dois: set[str] = set()
        for p in kb.candidate_papers:
            if p.doi:
                existing_dois.add(p.doi.lower().strip())
        new_papers = [
            p for p in new_papers if not p.doi or p.doi.lower().strip() not in existing_dois
        ]

        # Screen and extract new papers
        screener = PaperScreener(tracker)
        new_screened = await screener.screen(
            new_papers,
            scope_document=kb.scope_document or "",
        )

        extractor = PaperExtractor(
            tracker,
            domain_fields=self.config.extraction.domain_fields,
            full_text_max_chars=self.config.extraction.full_text_max_chars,
            tiered_models=self.config.extraction.tiered_models,
            section_truncation=self.config.extraction.section_truncation,
        )
        new_to_extract = [sp for sp in new_screened if sp.paper.id not in kb.extractions]
        new_extractions = await extractor.extract_batch(new_to_extract)

        # Merge into existing state
        kb.candidate_papers.extend(new_papers)
        kb.screened_papers.extend(new_screened)
        kb.extractions.update(new_extractions)

        kb.current_phase = PipelinePhase.GAP_SEARCH
        kb.add_audit_entry(
            "gap_search",
            "complete",
            f"Added {len(new_screened)} papers, {len(new_extractions)} extractions",
            tracker.usage,
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

        # Auto-remediate: retry gap search for remaining gaps
        if reval_result.remediation:
            # Pass previous queries so retry uses different terms
            prev_queries = []
            for gap in pre_gaps:
                prev_queries.extend(gap.suggested_queries)
            reval_result.remediation.params["previous_queries"] = prev_queries

            await self.dispatcher.execute(kb, reval_result)
            # Re-validate after retry
            reval_recheck = await revalidator.check(
                kb.evidence_map.themes if kb.evidence_map else [],
                kb.scope_document or "",
                pre_gaps=pre_gaps,
                pre_coverage=pre_coverage,
            )
            kb.comprehensiveness_checks.append(reval_recheck)
            # Second round if still warning
            if reval_recheck.remediation:
                reval_recheck.remediation.params["previous_queries"] = prev_queries
                await self.dispatcher.execute(kb, reval_recheck)

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
        tracker = _TokenAccumulator(self.llm, self._global_tokens)
        generator = OutlineGenerator(tracker)
        critic = OutlineCritic(tracker)

        review_outline, critiques = await outline_critique_loop(
            llm=tracker,
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
        kb.add_audit_entry(
            "outline",
            "complete",
            f"Sections: {len(review_outline.sections)}",
            tracker.usage,
        )

    async def narrative_planning(self, kb: KnowledgeBase) -> None:
        """Node: Plan narrative architecture before section writing."""
        tracker = _TokenAccumulator(self.llm, self._global_tokens)
        outline = ReviewOutline.model_validate(kb.outline)
        architect = NarrativeArchitect(tracker)

        plan = await architect.plan(
            outline=outline,
            evidence_map=kb.evidence_map,
            scope_document=kb.scope_document or "",
        )

        kb.narrative_plan = plan
        kb.current_phase = PipelinePhase.NARRATIVE_PLANNING
        kb.add_audit_entry(
            "narrative_planning",
            "complete",
            f"Sections planned: {len(plan.section_directives)}",
            tracker.usage,
        )

    async def contextual_enrichment(self, kb: KnowledgeBase) -> None:
        """Node: Retrieve adjacent contextual material to broaden sections."""
        tracker = _TokenAccumulator(self.llm, self._global_tokens)
        outline = ReviewOutline.model_validate(kb.outline)
        enricher = ContextualEnricher(tracker)

        # Generate enrichment queries for each section
        queries_map = await enricher.generate_queries(
            outline=outline,
            narrative_plan=kb.narrative_plan,
            scope_document=kb.scope_document or "",
        )

        # Initialize search sources
        from autoreview.search.aggregator import SearchAggregator

        all_dbs = (
            self.config.databases.get("primary", [])
            + self.config.databases.get("secondary", [])
            + self.config.databases.get("discovery", [])
        )
        sources = []
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
                logger.warning("contextual_enrichment.source_init_failed", source=db, error=str(e))

        if not sources:
            kb.current_phase = PipelinePhase.CONTEXTUAL_ENRICHMENT
            kb.add_audit_entry("contextual_enrichment", "skipped", "No search sources available")
            return

        # Collect existing DOIs for deduplication
        existing_dois: set[str] = set()
        for p in kb.candidate_papers:
            if p.doi:
                existing_dois.add(p.doi.lower().strip())

        screener = PaperScreener(tracker)

        for section_id, section_queries in queries_map.items():
            if not section_queries.queries:
                continue

            # Skip enrichment for well-covered sections (15+ papers already assigned)
            section = outline.get_section(section_id)
            if section and len(section.paper_ids) >= 15:
                logger.info(
                    "contextual_enrichment.skipped_well_covered",
                    section_id=section_id,
                    paper_count=len(section.paper_ids),
                )
                continue

            # Search using enrichment queries
            query_strings = [q.query for q in section_queries.queries]
            queries_by_source: dict[str, list[str]] = {db: query_strings for db in all_dbs}

            agg = SearchAggregator(sources=sources)
            new_papers = await agg.search(queries_by_source, max_results_per_source=50)

            # Deduplicate against existing corpus
            unique_papers = [
                p for p in new_papers if not p.doi or p.doi.lower().strip() not in existing_dois
            ]

            # Screen with lower threshold (2 instead of 3) — we want adjacent material
            screened = await screener.screen(
                unique_papers,
                scope_document=kb.scope_document or "",
                threshold=2,
            )

            # Take top 5 screened papers per section
            top_papers = [sp.paper for sp in screened[:5]]

            # Extract contextual information
            section = outline.get_section(section_id)
            section_title = section.title if section else section_queries.section_title
            section_desc = section.description if section else ""

            extractions = await enricher.extract_contextual_batch(
                papers=top_papers,
                section_title=section_title,
                section_description=section_desc,
            )

            kb.contextual_enrichment[section_id] = SectionEnrichment(
                section_id=section_id,
                section_title=section_title,
                queries_generated=section_queries.queries,
                papers_found=len(new_papers),
                papers_screened=len(screened),
                contextual_extractions=extractions,
            )

            # Track new DOIs
            for p in unique_papers:
                if p.doi:
                    existing_dois.add(p.doi.lower().strip())

        kb.current_phase = PipelinePhase.CONTEXTUAL_ENRICHMENT
        kb.add_audit_entry(
            "contextual_enrichment",
            "complete",
            f"Sections enriched: {len(kb.contextual_enrichment)}, "
            f"Total extractions: {sum(len(e.contextual_extractions) for e in kb.contextual_enrichment.values())}",
            tracker.usage,
        )

    async def corpus_expansion(self, kb: KnowledgeBase) -> None:
        """Node: Expand corpus with primary research papers informed by enrichment insights."""
        # Guard: skip if no enrichment data
        if not kb.contextual_enrichment:
            kb.current_phase = PipelinePhase.CORPUS_EXPANSION
            kb.add_audit_entry("corpus_expansion", "skipped", "No contextual enrichment data")
            return

        # Check that at least one section has contextual extractions
        has_extractions = any(e.contextual_extractions for e in kb.contextual_enrichment.values())
        if not has_extractions:
            kb.current_phase = PipelinePhase.CORPUS_EXPANSION
            kb.add_audit_entry(
                "corpus_expansion",
                "skipped",
                "Enrichment exists but no contextual extractions",
            )
            return

        from autoreview.llm.prompts.corpus_expansion import (
            CORPUS_EXPANSION_SYSTEM_PROMPT,
            CorpusExpansionQueryResult,
            build_corpus_expansion_query_prompt,
        )
        from autoreview.search.aggregator import SearchAggregator

        tracker = _TokenAccumulator(self.llm, self._global_tokens)
        outline = ReviewOutline.model_validate(kb.outline)

        # 1. Collect insights and generate queries per section
        all_queries: list[str] = []
        query_to_sections: dict[str, list[str]] = {}
        section_query_counts: dict[str, int] = {}

        for section_id, enrichment in kb.contextual_enrichment.items():
            if not enrichment.contextual_extractions:
                continue

            # Aggregate key_concepts and cross_field_connections (deduplicated)
            key_concepts: list[str] = []
            cross_field_connections: list[str] = []
            seen_concepts: set[str] = set()
            seen_connections: set[str] = set()

            for ext in enrichment.contextual_extractions:
                for c in ext.key_concepts:
                    if c.lower() not in seen_concepts:
                        seen_concepts.add(c.lower())
                        key_concepts.append(c)
                for c in ext.cross_field_connections:
                    if c.lower() not in seen_connections:
                        seen_connections.add(c.lower())
                        cross_field_connections.append(c)

            # Get existing paper_ids for this section from outline
            section = outline.get_section(section_id)
            existing_paper_ids = section.paper_ids if section else []

            prompt = build_corpus_expansion_query_prompt(
                section_id=section_id,
                section_title=enrichment.section_title,
                section_description=section.description if section else "",
                key_concepts=key_concepts,
                cross_field_connections=cross_field_connections,
                existing_paper_ids=existing_paper_ids,
                scope_document=kb.scope_document or "",
            )

            response = await tracker.generate_structured(
                prompt=prompt,
                response_model=CorpusExpansionQueryResult,
                system=CORPUS_EXPANSION_SYSTEM_PROMPT,
            )
            result: CorpusExpansionQueryResult = response.parsed

            section_query_counts[section_id] = len(result.queries)
            for q in result.queries:
                query_str = q.query
                all_queries.append(query_str)
                query_to_sections.setdefault(query_str, []).append(section_id)

            logger.info(
                "corpus_expansion.queries_generated",
                section_id=section_id,
                query_count=len(result.queries),
            )

        if not all_queries:
            kb.current_phase = PipelinePhase.CORPUS_EXPANSION
            kb.add_audit_entry("corpus_expansion", "skipped", "No queries generated")
            return

        # 2. Consolidated search across all sources
        all_dbs = (
            self.config.databases.get("primary", [])
            + self.config.databases.get("secondary", [])
            + self.config.databases.get("discovery", [])
        )
        sources = []
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
                logger.warning("corpus_expansion.source_init_failed", source=db, error=str(e))

        if not sources:
            kb.current_phase = PipelinePhase.CORPUS_EXPANSION
            kb.add_audit_entry("corpus_expansion", "skipped", "No search sources available")
            return

        # Deduplicate queries
        unique_queries = list(dict.fromkeys(all_queries))
        queries_by_source: dict[str, list[str]] = {db: unique_queries for db in all_dbs}

        agg = SearchAggregator(sources=sources)
        new_papers = await agg.search(queries_by_source, max_results_per_source=30)

        # 3. Deduplicate against existing corpus by DOI
        existing_dois: set[str] = set()
        for p in kb.candidate_papers:
            if p.doi:
                existing_dois.add(p.doi.lower().strip())

        unique_papers = [
            p for p in new_papers if not p.doi or p.doi.lower().strip() not in existing_dois
        ]

        # 4. Screen at standard threshold — these are evidence papers
        screener = PaperScreener(tracker)
        new_screened = await screener.screen(
            unique_papers,
            scope_document=kb.scope_document or "",
            threshold=self.config.search.relevance_threshold,
        )

        # 5. Full extraction (skip already-extracted papers)
        extractor = PaperExtractor(
            tracker,
            domain_fields=self.config.extraction.domain_fields,
            max_concurrent=self.config.extraction.max_concurrent,
            full_text_max_chars=self.config.extraction.full_text_max_chars,
            tiered_models=self.config.extraction.tiered_models,
            section_truncation=self.config.extraction.section_truncation,
        )
        new_to_extract = [sp for sp in new_screened if sp.paper.id not in kb.extractions]
        new_extractions = await extractor.extract_batch(new_to_extract)

        # 6. Merge into KB
        kb.candidate_papers.extend(unique_papers)
        kb.screened_papers.extend(new_screened)
        kb.extractions.update(new_extractions)

        # 7. Assign new paper IDs to outline sections
        new_paper_ids = list(new_extractions.keys())
        section_new_papers: dict[str, list[str]] = {}

        for query_str, section_ids in query_to_sections.items():
            for sid in section_ids:
                section_new_papers.setdefault(sid, []).extend(new_paper_ids)

        # Deduplicate per section and update outline
        outline_dict = kb.outline
        if outline_dict and new_paper_ids:
            outline_obj = ReviewOutline.model_validate(outline_dict)
            for sid, pids in section_new_papers.items():
                section = outline_obj.get_section(sid)
                if section:
                    existing_set = set(section.paper_ids)
                    for pid in pids:
                        if pid not in existing_set:
                            section.paper_ids.append(pid)
                            existing_set.add(pid)
            kb.outline = outline_obj.model_dump()

        # 8. Track results per section
        for section_id, enrichment in kb.contextual_enrichment.items():
            section = outline.get_section(section_id)
            section_pids = section_new_papers.get(section_id, [])
            # Deduplicate
            seen: set[str] = set()
            unique_pids: list[str] = []
            for pid in section_pids:
                if pid not in seen:
                    seen.add(pid)
                    unique_pids.append(pid)

            kb.corpus_expansion_results[section_id] = CorpusExpansionResult(
                section_id=section_id,
                section_title=enrichment.section_title,
                queries_generated=section_query_counts.get(section_id, 0),
                papers_found=len(new_papers),
                papers_screened=len(new_screened),
                papers_extracted=len(new_extractions),
                new_paper_ids=unique_pids,
            )

        # 9. Update phase and audit
        kb.current_phase = PipelinePhase.CORPUS_EXPANSION
        kb.add_audit_entry(
            "corpus_expansion",
            "complete",
            f"Queries: {len(unique_queries)}, Found: {len(new_papers)}, "
            f"Screened: {len(new_screened)}, Extracted: {len(new_extractions)}",
            tracker.usage,
        )

    async def section_writing(self, kb: KnowledgeBase) -> None:
        """Node: Write and critique all sections."""
        tracker = _TokenAccumulator(self.llm, self._global_tokens)
        outline = ReviewOutline.model_validate(kb.outline)
        writer = SectionWriter(tracker)
        critic = SectionCritic(tracker)

        drafts = await writer.write_all_sections(
            outline,
            kb.extractions,
            kb.evidence_map,
            narrative_plan=kb.narrative_plan,
            contextual_enrichment=kb.contextual_enrichment or None,
        )

        # Validate citations and critique each section
        citation_validator = CitationValidator()
        for section_id, draft in drafts.items():
            # Run citation validation before critique
            section_obj = outline.get_section(section_id)
            section_paper_ids = section_obj.paper_ids if section_obj else []
            cv_report = citation_validator.validate_section(
                draft.text,
                section_paper_ids,
                kb.extractions,
            )
            cv_issues = CitationValidator.to_critique_issues(cv_report)

            final_draft, critiques = await section_critique_loop(
                llm=tracker,
                critic=critic,
                draft=draft,
                outline=outline,
                max_cycles=self.config.critique.max_revision_cycles,
                threshold=self.config.critique.score_threshold,
                extra_issues=cv_issues,
            )
            drafts[section_id] = final_draft
            kb.critique_history.extend(critiques)

        kb.section_drafts = {sid: d.text for sid, d in drafts.items()}
        kb.current_phase = PipelinePhase.SECTION_CRITIQUE
        kb.add_audit_entry(
            "section_writing",
            "complete",
            f"Sections: {len(drafts)}",
            tracker.usage,
        )

    async def passage_search(self, kb: KnowledgeBase) -> None:
        """Node: Mine written sections for undercited claims and retrieve more papers."""
        if not kb.section_drafts:
            return

        from autoreview.analysis.passage_miner import PassageMiner
        from autoreview.search.aggregator import SearchAggregator

        tracker = _TokenAccumulator(self.llm, self._global_tokens)

        # 1. Mine sections for undercited claims
        miner = PassageMiner(tracker)
        mining_results = await miner.mine_all_sections(kb.section_drafts, kb.extractions)

        # 2. Collect targeted queries from high and medium priority claims
        queries = miner.collect_queries(mining_results, priorities={"high", "medium"})
        if not queries:
            kb.current_phase = PipelinePhase.PASSAGE_SEARCH
            kb.add_audit_entry("passage_search", "skipped", "No high/medium priority claims found")
            return

        # 3. Build queries dict for all databases
        all_dbs = (
            self.config.databases.get("primary", [])
            + self.config.databases.get("secondary", [])
            + self.config.databases.get("discovery", [])
        )
        queries_by_source: dict[str, list[str]] = {db: queries for db in all_dbs}

        # 4. Search across all databases
        sources = []
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
                logger.warning("passage_search.source_init_failed", source=db, error=str(e))

        new_papers: list[CandidatePaper] = []
        if sources:
            agg = SearchAggregator(sources=sources)
            new_papers = await agg.search(queries_by_source, max_results_per_source=50)

        # 5. Citation snowballing from top-10 most-cited S2 papers in corpus
        try:
            from autoreview.search.semantic_scholar import SemanticScholarSearch as S2

            s2 = S2()
            top_s2_ids = sorted(
                [
                    (p.external_ids.get("s2_id", ""), p.citation_count or 0)
                    for p in kb.candidate_papers
                    if p.external_ids.get("s2_id")
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            s2_ids = [pid for pid, _ in top_s2_ids if pid]
            if s2_ids:
                snowballed = await s2.snowball_references(s2_ids, limit_per_paper=20)
                new_papers.extend(snowballed)
        except Exception as e:
            logger.warning("passage_search.snowball_failed", error=str(e))

        # 6. Deduplicate by DOI, then screen and extract new papers
        existing_dois: set[str] = set()
        for p in kb.candidate_papers:
            if p.doi:
                existing_dois.add(p.doi.lower().strip())
        new_papers = [
            p for p in new_papers if not p.doi or p.doi.lower().strip() not in existing_dois
        ]

        screener = PaperScreener(tracker)
        new_screened = await screener.screen(
            new_papers,
            scope_document=kb.scope_document or "",
            threshold=self.config.search.relevance_threshold,
        )

        extractor = PaperExtractor(
            tracker,
            domain_fields=self.config.extraction.domain_fields,
            full_text_max_chars=self.config.extraction.full_text_max_chars,
            tiered_models=self.config.extraction.tiered_models,
            section_truncation=self.config.extraction.section_truncation,
        )
        new_to_extract = [sp for sp in new_screened if sp.paper.id not in kb.extractions]
        new_extractions = await extractor.extract_batch(new_to_extract)

        # 7. Merge into KB
        kb.candidate_papers.extend(new_papers)
        kb.screened_papers.extend(new_screened)
        kb.extractions.update(new_extractions)

        # 8. Revise sections that gained >= 2 new relevant papers
        from autoreview.writing.section_writer import SectionWriter

        writer = SectionWriter(tracker)
        revised_count = 0

        # Build title lookup from outline if available
        section_titles: dict[str, str] = {}
        if kb.outline:
            from autoreview.llm.prompts.outline import ReviewOutline

            try:
                outline_obj = ReviewOutline.model_validate(kb.outline)
                section_titles = {s.id: s.title for s in outline_obj.flatten()}
            except Exception:
                pass

        for result in mining_results:
            if result.undercited_claims and len(new_extractions) >= 2:
                section_title = section_titles.get(result.section_id, result.section_id)
                try:
                    draft = await writer.revise_section_with_evidence(
                        section_id=result.section_id,
                        section_title=section_title,
                        existing_text=kb.section_drafts.get(result.section_id, ""),
                        new_paper_ids=list(new_extractions.keys())[:10],
                        extractions=kb.extractions,
                    )
                    kb.section_drafts[result.section_id] = draft.text
                    revised_count += 1
                except Exception as e:
                    logger.warning(
                        "passage_search.revision_failed",
                        section_id=result.section_id,
                        error=str(e),
                    )

        kb.current_phase = PipelinePhase.PASSAGE_SEARCH
        kb.add_audit_entry(
            "passage_search",
            "complete",
            f"Queries: {len(queries)}, New screened: {len(new_screened)}, "
            f"New extractions: {len(new_extractions)}, Sections revised: {revised_count}",
            tracker.usage,
        )

    async def assembly(self, kb: KnowledgeBase) -> None:
        """Node: Assemble draft and run holistic critique."""
        tracker = _TokenAccumulator(self.llm, self._global_tokens)
        outline = ReviewOutline.model_validate(kb.outline)
        from autoreview.writing.section_writer import SectionDraft

        title_map = {s.id: s.title for s in outline.flatten()}
        section_drafts = {
            sid: SectionDraft(section_id=sid, title=title_map.get(sid, sid), text=text)
            for sid, text in kb.section_drafts.items()
        }

        assembler = DraftAssembler()
        full_draft = assembler.assemble(outline, section_drafts)

        # Citation validation on full draft
        citation_validator = CitationValidator()
        cv_report = citation_validator.validate_full_draft(full_draft, kb.extractions)
        cv_issues = CitationValidator.to_critique_issues(cv_report)

        # Holistic critique loop
        critic = HolisticCritic(tracker)
        final_draft, critiques = await holistic_critique_loop(
            llm=tracker,
            critic=critic,
            full_draft=full_draft,
            scope_document=kb.scope_document or "",
            max_cycles=self.config.critique.max_revision_cycles,
            threshold=self.config.critique.score_threshold,
            convergence_delta=self.config.critique.convergence_delta,
            extra_issues=cv_issues,
        )

        kb.full_draft = final_draft
        kb.critique_history.extend(critiques)
        kb.current_phase = PipelinePhase.HOLISTIC_CRITIQUE
        kb.add_audit_entry(
            "assembly",
            "complete",
            f"Words: {len(final_draft.split())}",
            tracker.usage,
        )

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
            "final_polish",
            "complete",
            token_usage={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )
