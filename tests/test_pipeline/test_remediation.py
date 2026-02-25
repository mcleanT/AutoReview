"""Tests for remediation dispatcher."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from autoreview.analysis.comprehensiveness import (
    CheckStatus,
    ComprehensiveCheckResult,
    RemediationAction,
)
from autoreview.config import load_config
from autoreview.models.knowledge_base import KnowledgeBase
from autoreview.pipeline.remediation import RemediationDispatcher


class TestRemediationDispatcher:
    def _make_kb(self) -> KnowledgeBase:
        kb = KnowledgeBase(topic="test topic", domain="biomedical", output_dir="/tmp/test")
        kb.search_queries = {"pubmed": ["q1"], "semantic_scholar": ["q2"]}
        kb.scope_document = "Review covers topic A and topic B."
        return kb

    def _make_check(
        self, action: str, params: dict | None = None, status: str = "warning",
    ) -> ComprehensiveCheckResult:
        return ComprehensiveCheckResult(
            check_name="test_check",
            status=CheckStatus(status),
            score=0.5,
            details="Test",
            metrics={},
            remediation=RemediationAction(action=action, params=params or {}),
        )

    @pytest.mark.asyncio
    async def test_passed_check_no_action(self):
        config = load_config(domain="biomedical")
        dispatcher = RemediationDispatcher(llm=MagicMock(), config=config)
        kb = self._make_kb()

        result = ComprehensiveCheckResult(
            check_name="test",
            status=CheckStatus.PASSED,
            score=1.0,
            details="All good",
            metrics={},
        )
        executed = await dispatcher.execute(kb, result)
        assert executed is False

    @pytest.mark.asyncio
    async def test_no_remediation_no_action(self):
        config = load_config(domain="biomedical")
        dispatcher = RemediationDispatcher(llm=MagicMock(), config=config)
        kb = self._make_kb()

        result = ComprehensiveCheckResult(
            check_name="test",
            status=CheckStatus.WARNING,
            score=0.5,
            details="Warning but no action",
            metrics={},
            remediation=None,
        )
        executed = await dispatcher.execute(kb, result)
        assert executed is False

    @pytest.mark.asyncio
    async def test_expand_queries_respects_max_rounds(self):
        config = load_config(domain="biomedical")
        dispatcher = RemediationDispatcher(llm=MagicMock(), config=config)
        kb = self._make_kb()

        check = self._make_check(
            "expand_queries",
            params={"uncovered_topics": ["topic_x"]},
        )

        # Exhaust rounds
        dispatcher._round_counts["expand_queries"] = config.search.max_query_expansion_rounds
        executed = await dispatcher.execute(kb, check)
        assert executed is False

    @pytest.mark.asyncio
    async def test_unknown_action_returns_false(self):
        config = load_config(domain="biomedical")
        dispatcher = RemediationDispatcher(llm=MagicMock(), config=config)
        kb = self._make_kb()

        check = self._make_check("unknown_action", params={})
        executed = await dispatcher.execute(kb, check)
        assert executed is False

    @pytest.mark.asyncio
    async def test_round_counter_increments(self):
        config = load_config(domain="biomedical")
        mock_llm = MagicMock()
        dispatcher = RemediationDispatcher(llm=mock_llm, config=config)

        # Patch the internal method to avoid real LLM calls
        dispatcher._expand_queries = AsyncMock(return_value=True)

        kb = self._make_kb()
        check = self._make_check(
            "expand_queries",
            params={"uncovered_topics": ["topic_x"]},
        )

        await dispatcher.execute(kb, check)
        assert dispatcher._round_counts["expand_queries"] == 1

        await dispatcher.execute(kb, check)
        assert dispatcher._round_counts["expand_queries"] == 2

    @pytest.mark.asyncio
    async def test_lower_threshold_max_one_round(self):
        config = load_config(domain="biomedical")
        dispatcher = RemediationDispatcher(llm=MagicMock(), config=config)
        kb = self._make_kb()

        check = self._make_check(
            "lower_screening_threshold",
            params={"current_rejection_rate": 0.85},
        )

        # First round should be allowed, max is 1
        dispatcher._lower_screening_threshold = AsyncMock(return_value=True)
        executed = await dispatcher.execute(kb, check)
        assert executed is True

        # Second round should be blocked
        executed = await dispatcher.execute(kb, check)
        assert executed is False
