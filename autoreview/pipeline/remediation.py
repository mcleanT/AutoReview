"""Remediation dispatcher for actionable comprehensiveness checks."""

from __future__ import annotations

from typing import Any

import structlog

from autoreview.analysis.comprehensiveness import ComprehensiveCheckResult
from autoreview.config.models import DomainConfig
from autoreview.models.knowledge_base import KnowledgeBase
from autoreview.search.base import SearchSource

logger = structlog.get_logger()


class RemediationDispatcher:
    """Executes remediation actions prescribed by comprehensiveness checks.

    Each check result may include a ``RemediationAction`` describing what
    the pipeline should do to fix the detected issue. This dispatcher
    reads the action and calls the appropriate method, tracking round
    counts to enforce iteration caps.
    """

    def __init__(self, llm: Any, config: DomainConfig) -> None:
        self.llm = llm
        self.config = config
        self._round_counts: dict[str, int] = {}

    def _max_rounds(self, action: str) -> int:
        """Get the max rounds for an action from config."""
        action_map = {
            "expand_queries": self.config.search.max_query_expansion_rounds,
            "retry_gap_search": self.config.search.max_gap_search_rounds,
            "lower_screening_threshold": 1,  # Always max 1
        }
        return action_map.get(action, 0)

    async def execute(
        self,
        kb: KnowledgeBase,
        check_result: ComprehensiveCheckResult,
    ) -> bool:
        """Execute the remediation action from a check result.

        Returns True if a remediation was performed, False otherwise.
        """
        if check_result.remediation is None:
            return False

        action = check_result.remediation.action
        params = check_result.remediation.params

        # Check round cap
        current_rounds = self._round_counts.get(action, 0)
        max_rounds = self._max_rounds(action)
        if current_rounds >= max_rounds:
            logger.info(
                "remediation.max_rounds_reached",
                action=action,
                rounds=current_rounds,
                max_rounds=max_rounds,
            )
            return False

        # Dispatch to handler
        handler = {
            "expand_queries": self._expand_queries,
            "retry_gap_search": self._retry_gap_search,
            "lower_screening_threshold": self._lower_screening_threshold,
        }.get(action)

        if handler is None:
            logger.warning("remediation.unknown_action", action=action)
            return False

        logger.info(
            "remediation.executing",
            action=action,
            round=current_rounds + 1,
            max_rounds=max_rounds,
            params=list(params.keys()),
        )

        result = await handler(kb, params)
        self._round_counts[action] = current_rounds + 1
        return result

    async def _expand_queries(
        self,
        kb: KnowledgeBase,
        params: dict[str, Any],
    ) -> bool:
        """Generate additional queries for uncovered sub-topics."""
        from pydantic import Field

        from autoreview.llm.prompts.query_expansion import (
            TARGETED_EXPANSION_SYSTEM_PROMPT,
            build_targeted_query_expansion_prompt,
        )
        from autoreview.models.base import AutoReviewModel

        uncovered_topics = params.get("uncovered_topics", [])
        if not uncovered_topics:
            return False

        class TargetedQueryResult(AutoReviewModel):
            pubmed_queries: list[str] = Field(default_factory=list)
            semantic_scholar_queries: list[str] = Field(default_factory=list)
            openalex_queries: list[str] = Field(default_factory=list)

        prompt = build_targeted_query_expansion_prompt(
            uncovered_topics=uncovered_topics,
            domain=self.config.domain,
            date_range=self.config.search.date_range,
        )

        response = await self.llm.generate_structured(
            prompt=prompt,
            response_model=TargetedQueryResult,
            system=TARGETED_EXPANSION_SYSTEM_PROMPT,
        )
        result: TargetedQueryResult = response.parsed

        # Merge new queries into existing
        added = 0
        for db, new_queries in [
            ("pubmed", result.pubmed_queries),
            ("semantic_scholar", result.semantic_scholar_queries),
            ("openalex", result.openalex_queries),
        ]:
            if db in kb.search_queries and new_queries:
                existing = set(kb.search_queries[db])
                for q in new_queries:
                    if q not in existing:
                        kb.search_queries[db].append(q)
                        added += 1

        kb.add_audit_entry(
            "remediation",
            "expand_queries",
            f"Added {added} queries for {len(uncovered_topics)} uncovered topics",
            {"input_tokens": response.input_tokens, "output_tokens": response.output_tokens},
        )

        logger.info(
            "remediation.queries_expanded",
            uncovered_topics=uncovered_topics,
            queries_added=added,
        )
        return added > 0

    async def _retry_gap_search(
        self,
        kb: KnowledgeBase,
        params: dict[str, Any],
    ) -> bool:
        """Retry gap search with alternative queries for remaining gaps."""
        from pydantic import Field

        from autoreview.llm.prompts.clustering import (
            RETRY_GAP_SEARCH_SYSTEM_PROMPT,
            build_retry_gap_queries_prompt,
        )
        from autoreview.models.base import AutoReviewModel
        from autoreview.search.aggregator import SearchAggregator

        remaining_gaps = params.get("remaining_gaps", [])
        previous_queries = params.get("previous_queries", [])
        if not remaining_gaps:
            return False

        # Generate new queries via LLM
        class RetryGapQueryResult(AutoReviewModel):
            queries: list[str] = Field(default_factory=list)

        prompt = build_retry_gap_queries_prompt(remaining_gaps, previous_queries)
        response = await self.llm.generate_structured(
            prompt=prompt,
            response_model=RetryGapQueryResult,
            system=RETRY_GAP_SEARCH_SYSTEM_PROMPT,
        )
        new_queries = response.parsed.queries

        if not new_queries:
            return False

        # Build search sources (primary + secondary only)
        gap_dbs = self.config.databases.get("primary", []) + self.config.databases.get(
            "secondary", []
        )
        sources: list[SearchSource] = []
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
            return False

        queries_by_source = {db: new_queries for db in gap_dbs}
        agg = SearchAggregator(sources=sources)
        new_papers = await agg.search(queries_by_source, max_results_per_source=200)

        # Screen and extract
        from autoreview.extraction.extractor import PaperExtractor, PaperScreener

        screener = PaperScreener(self.llm)
        new_screened = await screener.screen(
            new_papers,
            scope_document=kb.scope_document or "",
            threshold=self.config.search.relevance_threshold,
        )

        extractor = PaperExtractor(
            self.llm,
            domain_fields=self.config.extraction.domain_fields,
            tiered_models=self.config.extraction.tiered_models,
            section_truncation=self.config.extraction.section_truncation,
        )
        new_extractions = await extractor.extract_batch(new_screened)

        # Merge
        kb.candidate_papers.extend(new_papers)
        kb.screened_papers.extend(new_screened)
        kb.extractions.update(new_extractions)

        kb.add_audit_entry(
            "remediation",
            "retry_gap_search",
            f"Retry found {len(new_screened)} papers, {len(new_extractions)} extractions "
            f"for {len(remaining_gaps)} remaining gaps",
            {"input_tokens": response.input_tokens, "output_tokens": response.output_tokens},
        )

        logger.info(
            "remediation.gap_search_retried",
            remaining_gaps=len(remaining_gaps),
            new_papers=len(new_screened),
            new_extractions=len(new_extractions),
        )
        return len(new_screened) > 0

    async def _lower_screening_threshold(
        self,
        kb: KnowledgeBase,
        params: dict[str, Any],
    ) -> bool:
        """Re-screen rejected papers at a lower threshold."""
        from autoreview.extraction.extractor import PaperScreener

        current_threshold = self.config.search.relevance_threshold
        new_threshold = current_threshold - 1
        if new_threshold < 1:
            return False

        # Find papers that were not screened in (not in screened_papers)
        screened_ids = {sp.paper.id for sp in kb.screened_papers}
        rejected = [p for p in kb.candidate_papers if p.id not in screened_ids]

        if not rejected:
            return False

        screener = PaperScreener(self.llm, batch_size=self.config.search.screening_batch_size)
        newly_screened = await screener.screen(
            rejected,
            scope_document=kb.scope_document or "",
            threshold=new_threshold,
        )

        if newly_screened:
            kb.screened_papers.extend(newly_screened)

        kb.add_audit_entry(
            "remediation",
            "lower_screening_threshold",
            f"Re-screened {len(rejected)} papers at threshold {new_threshold}, "
            f"promoted {len(newly_screened)}",
        )

        logger.info(
            "remediation.threshold_lowered",
            old_threshold=current_threshold,
            new_threshold=new_threshold,
            rejected_count=len(rejected),
            promoted=len(newly_screened),
        )
        return len(newly_screened) > 0
