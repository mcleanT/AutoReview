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
