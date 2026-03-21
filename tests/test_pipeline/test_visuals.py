from autoreview.models.visuals import FigureMetadata, TableMetadata, VisualInsertionAnchor


def test_visual_insertion_anchor_defaults():
    anchor = VisualInsertionAnchor(section_id="sec_1")
    assert anchor.position == "after"


def test_visual_insertion_anchor_before():
    anchor = VisualInsertionAnchor(section_id="sec_3_1", position="before")
    assert anchor.position == "before"


def test_figure_metadata():
    fig = FigureMetadata(
        key="fig1_test",
        path="figures/fig1_test.png",
        caption="Test figure.",
        anchor=VisualInsertionAnchor(section_id="sec_1"),
    )
    assert fig.key == "fig1_test"
    assert fig.data_driven is False


def test_table_metadata():
    tbl = TableMetadata(
        key="table1_test",
        markdown="| A | B |\n|---|---|\n| 1 | 2 |",
        caption="Test table.",
        anchor=VisualInsertionAnchor(section_id="sec_3", position="before"),
    )
    assert tbl.anchor.position == "before"
    assert "| A | B |" in tbl.markdown


def test_generate_retrieval_table():
    from autoreview.models.visuals import TableMetadata
    from autoreview.tables.generators import generate_retrieval_table

    result = generate_retrieval_table()
    assert isinstance(result, TableMetadata)
    assert "Dense" in result.markdown
    assert "Sparse" in result.markdown
    assert result.anchor.position == "before"


def test_generate_domain_table():
    from autoreview.tables.generators import generate_domain_table

    theme_counts = {"Biomedical": 190, "Legal": 25, "Financial": 15}
    result = generate_domain_table(theme_counts)
    assert "190" in result.markdown
    assert "Biomedical" in result.markdown


def test_generate_all_tables():
    from unittest.mock import MagicMock

    from autoreview.models.visuals import TableMetadata
    from autoreview.tables.generators import generate_all_tables

    mock_kb = MagicMock()
    mock_kb.evidence_map = MagicMock()
    mock_kb.evidence_map.themes = []
    mock_kb.outline = {
        "sections": [{"id": "sec_3", "title": "Retrieval", "description": "About retrieval"}]
    }

    tables = generate_all_tables(mock_kb, llm=None)
    assert isinstance(tables, dict)
    assert all(isinstance(v, TableMetadata) for v in tables.values())


from unittest.mock import MagicMock


def test_generate_temporal_chart(tmp_path):
    from autoreview.figures.generators import generate_temporal_chart

    year_counts = {2020: 3, 2021: 15, 2022: 20, 2023: 30, 2024: 165, 2025: 399}
    path = generate_temporal_chart(year_counts, tmp_path / "figures")
    assert path.exists()
    assert path.suffix == ".png"


def test_generate_evidence_chart(tmp_path):
    from autoreview.figures.generators import generate_evidence_chart

    themes = [
        {
            "name": "Retrieval",
            "evidence_strength_distribution": {"strong": 10, "moderate": 20, "weak": 5},
        },
        {
            "name": "Generation",
            "evidence_strength_distribution": {"strong": 5, "moderate": 15, "weak": 8},
        },
    ]
    path = generate_evidence_chart(themes, tmp_path / "figures")
    assert path.exists()


def test_generate_all_figures(tmp_path):
    from autoreview.figures.generators import generate_all_figures
    from autoreview.models.visuals import FigureMetadata

    mock_kb = MagicMock()
    mock_kb.output_dir = str(tmp_path)
    mock_kb.screened_papers = []
    mock_kb.evidence_map = MagicMock()
    mock_kb.evidence_map.themes = []

    figures = generate_all_figures(mock_kb)
    assert isinstance(figures, dict)
    assert all(isinstance(v, FigureMetadata) for v in figures.values())
    assert len(figures) == 4
