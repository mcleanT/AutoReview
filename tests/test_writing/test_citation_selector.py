# tests/test_writing/test_citation_selector.py
from __future__ import annotations


def _make_extraction(paper_id, evidence_strength="moderate", year=2023, source="semantic_scholar"):
    """Helper to build a minimal PaperExtraction-like object."""
    from autoreview.extraction.models import Finding, PaperExtraction

    return PaperExtraction(
        paper_id=paper_id,
        key_findings=[
            Finding(
                claim=f"Claim from {paper_id}",
                evidence_strength=evidence_strength,
                paper_id=paper_id,
            )
        ],
        methods_summary="Method",
        limitations="None",
        relationships=[],
    )


def _make_evidence_map(themes=None, consensus_claims=None, evidence_chains=None):
    """Helper to build a minimal EvidenceMap."""
    from autoreview.analysis.evidence_map import (
        EvidenceMap,
    )

    return EvidenceMap(
        themes=themes or [],
        consensus_claims=consensus_claims or [],
        contradictions=[],
        gaps=[],
        evidence_chains=evidence_chains or [],
    )


def test_score_paper_basic():
    from autoreview.config.citation import CitationConfig
    from autoreview.writing.citation_selector import CitationSelector

    cfg = CitationConfig()
    selector = CitationSelector(cfg)
    ext = _make_extraction("p1", evidence_strength="strong", year=2024)
    evidence_map = _make_evidence_map()
    score = selector.score_paper(
        paper_id="p1",
        extraction=ext,
        section_paper_ids=["p1", "p2"],
        evidence_map=evidence_map,
        paper_year=2024,
        paper_source="semantic_scholar",
        relevance_score=5,
        date_range=(2015, 2025),
    )
    assert 0.0 <= score <= 2.0  # max with seminal boost


def test_compute_section_budget():
    from autoreview.config.citation import CitationConfig
    from autoreview.writing.citation_selector import CitationSelector

    cfg = CitationConfig(min_citations_per_section=5)
    selector = CitationSelector(cfg)
    budget = selector.compute_section_budget(
        section_word_count=1500,
        target_per_1k=9.0,
        num_assigned=100,
    )
    assert budget == 14  # round(1500 * 9.0 / 1000) = 14, > min 5, < 100


def test_compute_section_budget_capped_by_assigned():
    from autoreview.config.citation import CitationConfig
    from autoreview.writing.citation_selector import CitationSelector

    cfg = CitationConfig(min_citations_per_section=8)
    selector = CitationSelector(cfg)
    budget = selector.compute_section_budget(
        section_word_count=1500, target_per_1k=9.0, num_assigned=5
    )
    assert budget == 5  # capped by num_assigned


def test_assign_tiers_basic():
    from autoreview.config.citation import CitationConfig
    from autoreview.writing.citation_selector import CitationSelector

    cfg = CitationConfig()
    selector = CitationSelector(cfg)
    scored = [
        ("p1", 0.9),
        ("p2", 0.8),
        ("p3", 0.7),
        ("p4", 0.6),
        ("p5", 0.5),
        ("p6", 0.4),
        ("p7", 0.3),
        ("p8", 0.2),
        ("p9", 0.1),
        ("p10", 0.05),
    ]
    tiers = selector.assign_tiers(scored, budget=10)
    primary = [pid for pid, _, tier in tiers if tier == "primary"]
    supporting = [pid for pid, _, tier in tiers if tier == "supporting"]
    contextual = [pid for pid, _, tier in tiers if tier == "contextual"]
    assert len(primary) >= 1
    assert len(primary) + len(supporting) + len(contextual) == 10


def test_assign_tiers_small_section():
    """Sections with <3 papers should all be PRIMARY."""
    from autoreview.config.citation import CitationConfig
    from autoreview.writing.citation_selector import CitationSelector

    cfg = CitationConfig()
    selector = CitationSelector(cfg)
    scored = [("p1", 0.9), ("p2", 0.8)]
    tiers = selector.assign_tiers(scored, budget=2)
    assert all(tier == "primary" for _, _, tier in tiers)


def test_select_for_section_returns_citation_plan():
    from autoreview.config.citation import CitationConfig
    from autoreview.writing.citation_selector import CitationSelector, SectionCitationPlan

    cfg = CitationConfig()
    selector = CitationSelector(cfg)
    extractions = {f"p{i}": _make_extraction(f"p{i}") for i in range(1, 21)}
    evidence_map = _make_evidence_map()
    plan = selector.select_for_section(
        section_id="sec_3",
        paper_ids=[f"p{i}" for i in range(1, 21)],
        extractions=extractions,
        evidence_map=evidence_map,
        section_word_count=1500,
        target_per_1k=9.0,
    )
    assert isinstance(plan, SectionCitationPlan)
    assert plan.citation_budget > 0
    assert len(plan.primary_papers) >= 1


def test_select_all_returns_citation_plan():
    """select_all() with a mock outline returns a CitationPlan with correct structure."""
    from types import SimpleNamespace

    from autoreview.config.citation import CitationConfig
    from autoreview.writing.citation_selector import CitationPlan, CitationSelector

    cfg = CitationConfig()
    selector = CitationSelector(cfg)

    extractions = {f"p{i}": _make_extraction(f"p{i}") for i in range(1, 11)}
    evidence_map = _make_evidence_map()

    # Build a mock outline with two sections
    sections = [
        SimpleNamespace(
            id="sec_1",
            paper_ids=["p1", "p2", "p3", "p4", "p5"],
            estimated_word_count=1000,
        ),
        SimpleNamespace(
            id="sec_2",
            paper_ids=["p6", "p7", "p8", "p9", "p10"],
            estimated_word_count=1200,
        ),
    ]
    outline = SimpleNamespace(sections=sections)

    plan = selector.select_all(
        outline=outline,
        extractions=extractions,
        evidence_map=evidence_map,
        target_per_1k=9.0,
    )

    assert isinstance(plan, CitationPlan)
    assert "sec_1" in plan.sections
    assert "sec_2" in plan.sections
    assert plan.total_citation_budget >= 0
    assert 0.0 <= plan.corpus_utilization_target <= 1.0


def test_seminal_boost_fires():
    """Paper in a ConsensusClaim with evidence_count >= 10 gets the seminal boost."""
    from autoreview.analysis.evidence_map import ConsensusClaim, EvidenceMap
    from autoreview.config.citation import CitationConfig
    from autoreview.writing.citation_selector import CitationSelector

    # Build a ConsensusClaim with evidence_count=15 pointing at "seminal_paper"
    claim = ConsensusClaim(
        claim="Test consensus",
        strength="strong",
        evidence_count=15,
        supporting_paper_ids=["seminal_paper"],
    )
    evidence_map = EvidenceMap(
        themes=[],
        consensus_claims=[claim],
        contradictions=[],
        gaps=[],
        evidence_chains=[],
    )

    cfg = CitationConfig(seminal_paper_boost=1.5)
    selector = CitationSelector(cfg)

    ext_seminal = _make_extraction("seminal_paper", evidence_strength="moderate", year=2020)
    ext_control = _make_extraction("control_paper", evidence_strength="moderate", year=2020)

    score_seminal = selector.score_paper(
        paper_id="seminal_paper",
        extraction=ext_seminal,
        section_paper_ids=["seminal_paper", "control_paper"],
        evidence_map=evidence_map,
        paper_year=2020,
        paper_source="semantic_scholar",
        relevance_score=4,
        date_range=(2010, 2024),
    )
    score_control = selector.score_paper(
        paper_id="control_paper",
        extraction=ext_control,
        section_paper_ids=["seminal_paper", "control_paper"],
        evidence_map=evidence_map,
        paper_year=2020,
        paper_source="semantic_scholar",
        relevance_score=4,
        date_range=(2010, 2024),
    )

    # Seminal paper should score strictly higher due to boost
    assert score_seminal > score_control


def test_recency_score_none_year():
    """year=None returns 0.5 (neutral recency)."""
    from autoreview.config.citation import CitationConfig
    from autoreview.writing.citation_selector import CitationSelector

    selector = CitationSelector(CitationConfig())
    score = selector._recency_score(None, (2000, 2024))
    assert score == 0.5


def test_temporal_spread_constraint():
    """temporal_spread=True ensures papers from different years appear in the plan."""
    from autoreview.config.citation import CitationConfig
    from autoreview.writing.citation_selector import CitationSelector

    cfg = CitationConfig(temporal_spread=True, min_citations_per_section=3)
    selector = CitationSelector(cfg)

    # Papers with varied years
    paper_ids = [f"p{i}" for i in range(1, 8)]
    years = {"p1": 2010, "p2": 2010, "p3": 2015, "p4": 2020, "p5": 2022, "p6": 2023, "p7": 2024}
    extractions = {pid: _make_extraction(pid) for pid in paper_ids}
    evidence_map = _make_evidence_map()

    plan = selector.select_for_section(
        section_id="sec_spread",
        paper_ids=paper_ids,
        extractions=extractions,
        evidence_map=evidence_map,
        section_word_count=500,
        target_per_1k=9.0,
        paper_years=years,
    )

    all_assigned = plan.primary_papers + plan.supporting_papers + plan.contextual_papers
    assigned_ids = {pc.paper_id for pc in all_assigned}
    assigned_years = {years[pid] for pid in assigned_ids if pid in years}

    # With 7 papers spanning 2010–2024, we expect multiple distinct years
    assert len(assigned_years) >= 2
