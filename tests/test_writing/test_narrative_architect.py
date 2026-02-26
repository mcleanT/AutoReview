"""Tests for the NarrativeArchitect stage and related components."""

from __future__ import annotations

import pytest

from autoreview.analysis.evidence_map import (
    ConsensusClaim,
    Contradiction,
    EvidenceMap,
    IdentifiedGap,
    Theme,
)
from autoreview.config.models import DomainConfig
from autoreview.extraction.models import EvidenceStrength, Finding, PaperExtraction
from autoreview.llm.prompts.critique import (
    HOLISTIC_CRITIQUE_SYSTEM_PROMPT,
    SECTION_CRITIQUE_SYSTEM_PROMPT,
)
from autoreview.llm.prompts.narrative import (
    NARRATIVE_ARCHITECT_SYSTEM_PROMPT,
    build_narrative_planning_prompt,
)
from autoreview.llm.prompts.outline import OutlineSection, ReviewOutline
from autoreview.llm.provider import LLMResponse, LLMStructuredResponse
from autoreview.models.knowledge_base import KnowledgeBase, PipelinePhase
from autoreview.models.narrative import NarrativePlan, SectionNarrativeDirective
from autoreview.pipeline.nodes import PipelineNodes
from autoreview.writing.narrative_architect import NarrativeArchitect
from autoreview.writing.section_writer import SectionWriter


# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------

def _make_outline() -> ReviewOutline:
    return ReviewOutline(
        title="Test Review",
        sections=[
            OutlineSection(
                id="1", title="Introduction", description="Overview",
                paper_ids=["p1"], estimated_word_count=500,
            ),
            OutlineSection(
                id="2", title="Methods", description="Review methodology",
                paper_ids=[], estimated_word_count=300,
            ),
            OutlineSection(
                id="3", title="Results", description="Key findings",
                paper_ids=["p1", "p2"], estimated_word_count=1000,
                subsections=[
                    OutlineSection(
                        id="3.1", title="Theme A", description="First theme",
                        paper_ids=["p1"], estimated_word_count=500,
                    ),
                ],
            ),
        ],
    )


def _make_evidence_map() -> EvidenceMap:
    return EvidenceMap(
        themes=[
            Theme(name="Theme A", description="First theme", paper_ids=["p1", "p2"]),
        ],
        consensus_claims=[
            ConsensusClaim(
                claim="Finding X",
                supporting_paper_ids=["p1", "p2"],
                strength="moderate",
                evidence_count=2,
            ),
        ],
        contradictions=[
            Contradiction(
                claim_a="A", claim_b="B",
                paper_ids_a=["p1"], paper_ids_b=["p2"],
                theme="Theme A",
            ),
        ],
    )


def _make_extractions() -> dict[str, PaperExtraction]:
    return {
        "p1": PaperExtraction(
            paper_id="p1",
            key_findings=[Finding(claim="Claim A", evidence_strength=EvidenceStrength.STRONG, paper_id="p1")],
            methods_summary="Methods A", limitations="Limits A",
        ),
        "p2": PaperExtraction(
            paper_id="p2",
            key_findings=[Finding(claim="Claim B", evidence_strength=EvidenceStrength.MODERATE, paper_id="p2")],
            methods_summary="Methods B", limitations="Limits B",
        ),
    }


def _make_directive(section_id: str = "1") -> SectionNarrativeDirective:
    return SectionNarrativeDirective(
        section_id=section_id,
        narrative_role="Sets up the central tension between mechanistic and epidemiological evidence",
        central_claim="The gut-brain axis plays a causal role in neurodegeneration",
        structural_suggestion="problem-solution: establish the unknown, then trace how the field addressed it",
        key_insights=[
            "Mechanistic studies converge on short-chain fatty acids as mediators",
            "Epidemiological evidence is weaker due to confounding",
            "Animal models show causality but human translation remains uncertain",
        ],
        transition_from_prev="Building on the methodological landscape described above",
        transition_to_next="These findings set the stage for the contradictions examined in the next section",
    )


def _make_narrative_plan(outline: ReviewOutline) -> NarrativePlan:
    section_ids = [s.id for s in outline.flatten()]
    return NarrativePlan(
        central_argument="The gut-brain axis literature reveals a causal mechanism for neurodegeneration",
        narrative_arc=(
            "The paper opens by mapping the landscape of gut-brain research, "
            "then traces how mechanistic and epidemiological literatures diverge, "
            "and concludes by identifying the methodological reforms needed to reconcile them."
        ),
        section_directives=[_make_directive(sid) for sid in section_ids],
    )


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class MockNarrativeLLM:
    """Mock LLM that returns canned narrative planning responses."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate(
        self, prompt: str, system: str = "", max_tokens: int = 4096, temperature: float = 0.3,
    ) -> LLMResponse:
        self.calls.append({"method": "generate", "prompt": prompt, "system": system})
        return LLMResponse(
            content="This section synthesizes findings from multiple studies [@p1] [@p2].",
            input_tokens=500, output_tokens=300,
        )

    async def generate_structured(
        self, prompt: str, response_model: type, system: str = "", max_tokens: int = 4096, temperature: float = 0.0,
    ) -> LLMStructuredResponse:
        self.calls.append({
            "method": "generate_structured",
            "prompt": prompt,
            "system": system,
            "response_model": response_model,
        })
        if response_model == NarrativePlan:
            return LLMStructuredResponse(
                parsed=_make_narrative_plan(_make_outline()),
                input_tokens=800, output_tokens=600,
            )
        raise ValueError(f"Unexpected response_model: {response_model}")


# ---------------------------------------------------------------------------
# 1. TestNarrativeModels — Pure model tests (sync)
# ---------------------------------------------------------------------------

class TestNarrativeModels:
    def test_section_directive_construction(self):
        d = _make_directive()
        assert d.section_id == "1"
        assert d.narrative_role
        assert d.central_claim
        assert d.structural_suggestion
        assert len(d.key_insights) == 3
        assert d.transition_from_prev
        assert d.transition_to_next
        # Roundtrip through model_dump
        dumped = d.model_dump()
        assert dumped["section_id"] == "1"
        assert dumped["key_insights"] == d.key_insights

    def test_narrative_plan_construction(self):
        outline = _make_outline()
        plan = _make_narrative_plan(outline)
        assert plan.central_argument
        assert plan.narrative_arc
        assert len(plan.section_directives) == 4  # 1, 2, 3, 3.1

    def test_narrative_plan_serialization_roundtrip(self):
        outline = _make_outline()
        plan = _make_narrative_plan(outline)
        dumped = plan.model_dump()
        restored = NarrativePlan.model_validate(dumped)
        assert restored.central_argument == plan.central_argument
        assert restored.narrative_arc == plan.narrative_arc
        assert len(restored.section_directives) == len(plan.section_directives)
        for orig, rest in zip(plan.section_directives, restored.section_directives):
            assert orig.section_id == rest.section_id
            assert orig.key_insights == rest.key_insights


# ---------------------------------------------------------------------------
# 2. TestBuildNarrativePlanningPrompt — Prompt builder tests (sync)
# ---------------------------------------------------------------------------

class TestBuildNarrativePlanningPrompt:
    def test_prompt_contains_scope_document(self):
        prompt = build_narrative_planning_prompt(
            _make_outline(), _make_evidence_map(), "This is the review scope.",
        )
        assert "This is the review scope." in prompt

    def test_prompt_contains_outline_sections(self):
        prompt = build_narrative_planning_prompt(
            _make_outline(), _make_evidence_map(), "scope",
        )
        assert "1. Introduction" in prompt
        assert "2. Methods" in prompt
        assert "3. Results" in prompt
        assert "3.1. Theme A" in prompt

    def test_prompt_contains_evidence_summary(self):
        prompt = build_narrative_planning_prompt(
            _make_outline(), _make_evidence_map(), "scope",
        )
        assert "Theme A" in prompt
        assert "Finding X" in prompt
        # Contradictions
        assert "'A' vs 'B'" in prompt

    def test_prompt_with_empty_evidence(self):
        empty_em = EvidenceMap()
        prompt = build_narrative_planning_prompt(
            _make_outline(), empty_em, "scope",
        )
        assert "(none identified)" in prompt

    def test_prompt_section_ids_listed(self):
        prompt = build_narrative_planning_prompt(
            _make_outline(), _make_evidence_map(), "scope",
        )
        # The "Your Task" block should list all section IDs
        assert "1, 2, 3, 3.1" in prompt

    def test_prompt_with_subsections(self):
        prompt = build_narrative_planning_prompt(
            _make_outline(), _make_evidence_map(), "scope",
        )
        # Subsection 3.1 should be indented (depth 2 = 2 spaces)
        assert "  3.1. Theme A" in prompt


# ---------------------------------------------------------------------------
# 3. TestNarrativeArchitect — Core class tests (async)
# ---------------------------------------------------------------------------

class TestNarrativeArchitect:
    async def test_plan_returns_narrative_plan(self):
        llm = MockNarrativeLLM()
        architect = NarrativeArchitect(llm)
        plan = await architect.plan(
            outline=_make_outline(),
            evidence_map=_make_evidence_map(),
            scope_document="Test scope",
        )
        assert isinstance(plan, NarrativePlan)
        assert plan.central_argument
        assert plan.narrative_arc
        assert len(plan.section_directives) > 0

    async def test_plan_calls_generate_structured_with_correct_model(self):
        llm = MockNarrativeLLM()
        architect = NarrativeArchitect(llm)
        await architect.plan(
            outline=_make_outline(),
            evidence_map=_make_evidence_map(),
            scope_document="Test scope",
        )
        structured_calls = [c for c in llm.calls if c["method"] == "generate_structured"]
        assert len(structured_calls) == 1
        assert structured_calls[0]["response_model"] is NarrativePlan

    async def test_plan_uses_correct_system_prompt(self):
        llm = MockNarrativeLLM()
        architect = NarrativeArchitect(llm)
        await architect.plan(
            outline=_make_outline(),
            evidence_map=_make_evidence_map(),
            scope_document="Test scope",
        )
        structured_calls = [c for c in llm.calls if c["method"] == "generate_structured"]
        assert structured_calls[0]["system"] == NARRATIVE_ARCHITECT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# 4. TestSectionWriterDirectiveIntegration — Directive formatting (async)
# ---------------------------------------------------------------------------

class TestSectionWriterDirectiveIntegration:
    async def test_write_section_without_directive(self):
        llm = MockNarrativeLLM()
        writer = SectionWriter(llm)
        outline = _make_outline()
        section = outline.sections[0]
        draft = await writer.write_section(
            section, outline, _make_extractions(), _make_evidence_map(),
            directive=None,
        )
        assert draft.section_id == "1"
        assert draft.text

    async def test_write_section_with_directive(self):
        llm = MockNarrativeLLM()
        writer = SectionWriter(llm)
        outline = _make_outline()
        section = outline.sections[0]
        directive = _make_directive("1")

        await writer.write_section(
            section, outline, _make_extractions(), _make_evidence_map(),
            directive=directive,
        )

        # The prompt sent to the LLM should contain narrative guidance fields
        generate_calls = [c for c in llm.calls if c["method"] == "generate"]
        assert len(generate_calls) == 1
        prompt = generate_calls[0]["prompt"]
        assert "Narrative Guidance" in prompt
        assert "Role in paper" in prompt
        assert directive.central_claim in prompt
        assert "Structural suggestion" in prompt
        assert "Key insights" in prompt
        assert "Opening transition hint" in prompt
        assert "Closing transition hint" in prompt

    async def test_write_all_sections_builds_directive_map(self):
        llm = MockNarrativeLLM()
        writer = SectionWriter(llm)
        outline = _make_outline()
        plan = _make_narrative_plan(outline)

        await writer.write_all_sections(
            outline, _make_extractions(), _make_evidence_map(),
            narrative_plan=plan,
        )

        # Each top-level section should get a write_section call with its directive
        generate_calls = [c for c in llm.calls if c["method"] == "generate"]
        assert len(generate_calls) == len(outline.sections)  # 3 top-level sections
        for call in generate_calls:
            # All should contain narrative guidance since plan has directives for all IDs
            assert "Narrative Guidance" in call["prompt"]

    async def test_write_all_sections_without_plan(self):
        llm = MockNarrativeLLM()
        writer = SectionWriter(llm)
        outline = _make_outline()

        drafts = await writer.write_all_sections(
            outline, _make_extractions(), _make_evidence_map(),
            narrative_plan=None,
        )
        assert len(drafts) == len(outline.sections)
        for sid, draft in drafts.items():
            assert draft.text


# ---------------------------------------------------------------------------
# 5. TestNarrativePlanningNode — Pipeline node integration (async)
# ---------------------------------------------------------------------------

class TestNarrativePlanningNode:
    async def test_narrative_planning_node_populates_kb(self):
        llm = MockNarrativeLLM()
        config = DomainConfig()
        nodes = PipelineNodes(llm, config)

        kb = KnowledgeBase(topic="test", output_dir="/tmp/test_narrative")
        kb.outline = _make_outline().model_dump()
        kb.evidence_map = _make_evidence_map()
        kb.scope_document = "Test scope"

        await nodes.narrative_planning(kb)

        assert kb.narrative_plan is not None
        assert len(kb.narrative_plan.section_directives) == 4  # 1, 2, 3, 3.1

    async def test_narrative_planning_node_updates_phase(self):
        llm = MockNarrativeLLM()
        config = DomainConfig()
        nodes = PipelineNodes(llm, config)

        kb = KnowledgeBase(topic="test", output_dir="/tmp/test_narrative")
        kb.outline = _make_outline().model_dump()
        kb.evidence_map = _make_evidence_map()
        kb.scope_document = "Test scope"

        await nodes.narrative_planning(kb)

        assert kb.current_phase == PipelinePhase.NARRATIVE_PLANNING

    async def test_narrative_planning_node_adds_audit_entry(self):
        llm = MockNarrativeLLM()
        config = DomainConfig()
        nodes = PipelineNodes(llm, config)

        kb = KnowledgeBase(topic="test", output_dir="/tmp/test_narrative")
        kb.outline = _make_outline().model_dump()
        kb.evidence_map = _make_evidence_map()
        kb.scope_document = "Test scope"

        await nodes.narrative_planning(kb)

        assert len(kb.audit_log) >= 1
        last_entry = kb.audit_log[-1]
        assert last_entry.node_name == "narrative_planning"
        assert last_entry.action == "complete"


# ---------------------------------------------------------------------------
# 6. TestCritiqueDimensions — Prompt content verification (sync)
# ---------------------------------------------------------------------------

class TestCritiqueDimensions:
    def test_section_critique_has_structural_variety(self):
        assert "structural_variety" in SECTION_CRITIQUE_SYSTEM_PROMPT

    def test_section_critique_has_paragraph_cohesion(self):
        assert "paragraph_cohesion" in SECTION_CRITIQUE_SYSTEM_PROMPT

    def test_holistic_critique_has_prose_flow(self):
        assert "prose_flow" in HOLISTIC_CRITIQUE_SYSTEM_PROMPT
