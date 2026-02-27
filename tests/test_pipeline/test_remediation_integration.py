"""Integration test for remediation flow through pipeline nodes."""
from __future__ import annotations

import pytest

from autoreview.analysis.comprehensiveness import CheckStatus
from autoreview.config import load_config
from autoreview.llm.provider import LLMStructuredResponse
from autoreview.models.knowledge_base import KnowledgeBase
from autoreview.pipeline.nodes import PipelineNodes


class MockRemediationLLM:
    """Mock LLM that supports query expansion and coverage checking."""

    def __init__(self):
        self.call_count = 0

    async def generate_structured(self, prompt, response_model, system="", max_tokens=4096, temperature=0.0, model_override=None):
        self.call_count += 1
        name = response_model.__name__

        if name == "QueryExpansionResult":
            # Return initial query expansion result
            return LLMStructuredResponse(
                parsed=response_model(
                    pubmed_queries=["query1"],
                    semantic_scholar_queries=["query2"],
                    openalex_queries=["query3"],
                    perplexity_questions=["question1"],
                    scope_document="This review covers topic A, topic B, and topic C.",
                ),
                input_tokens=100, output_tokens=200,
            )

        if name == "QueryCoverageResult":
            from autoreview.llm.prompts.comprehensiveness import QueryCoverageResult, SubTopicCoverage
            # First coverage check: missing topic C
            # After expansion (call_count > 3): all covered
            if self.call_count <= 3:
                return LLMStructuredResponse(
                    parsed=QueryCoverageResult(
                        sub_topic_assessments=[
                            SubTopicCoverage(sub_topic="Topic A", covered=True, matching_queries=["q"]),
                            SubTopicCoverage(sub_topic="Topic B", covered=True, matching_queries=["q"]),
                            SubTopicCoverage(sub_topic="Topic C", covered=False, matching_queries=[]),
                        ],
                        overall_coverage_score=0.67,
                    ),
                    input_tokens=100, output_tokens=100,
                )
            else:
                return LLMStructuredResponse(
                    parsed=QueryCoverageResult(
                        sub_topic_assessments=[
                            SubTopicCoverage(sub_topic="Topic A", covered=True, matching_queries=["q"]),
                            SubTopicCoverage(sub_topic="Topic B", covered=True, matching_queries=["q"]),
                            SubTopicCoverage(sub_topic="Topic C", covered=True, matching_queries=["q"]),
                        ],
                        overall_coverage_score=1.0,
                    ),
                    input_tokens=100, output_tokens=100,
                )

        if name == "TargetedQueryResult":
            # Return expanded queries
            return LLMStructuredResponse(
                parsed=response_model(
                    pubmed_queries=["topic C expanded query"],
                    semantic_scholar_queries=["topic C semantic"],
                    openalex_queries=["topic C openalex"],
                ),
                input_tokens=100, output_tokens=100,
            )

        # Fallback for any other model
        raise ValueError(f"Unexpected model: {name}")


class TestQueryExpansionRemediation:
    @pytest.mark.asyncio
    async def test_query_expansion_auto_remediates(self):
        """After query expansion, missing sub-topics trigger targeted expansion."""
        config = load_config(domain="biomedical")
        llm = MockRemediationLLM()
        nodes = PipelineNodes(llm=llm, config=config)
        kb = KnowledgeBase(topic="test topic", domain="biomedical", output_dir="/tmp/test")

        await nodes.query_expansion(kb)

        # Should have original queries plus expanded ones
        assert len(kb.search_queries["pubmed"]) > 1
        assert "topic C expanded query" in kb.search_queries["pubmed"]

        # Should have multiple comprehensiveness checks (initial + recheck)
        qc_checks = [c for c in kb.comprehensiveness_checks if c.check_name == "query_coverage"]
        assert len(qc_checks) >= 2

        # Should have audit entry for remediation
        remediation_entries = [e for e in kb.audit_log if e.node_name == "remediation"]
        assert len(remediation_entries) >= 1
        assert remediation_entries[0].action == "expand_queries"
