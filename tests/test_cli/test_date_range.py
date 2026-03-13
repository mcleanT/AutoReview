from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from autoreview.cli import app

runner = CliRunner()


class TestDateRangeCLI:
    @patch("autoreview.pipeline.runner.run_pipeline")
    @patch("autoreview.llm.factory.create_llm_provider")
    @patch("autoreview.config.load_config")
    def test_run_passes_date_range_to_config(self, mock_load, mock_llm, mock_run):
        mock_config = MagicMock()
        mock_config.llm = MagicMock()
        mock_config.writing.citation_format = "apa"
        mock_load.return_value = mock_config
        mock_llm.return_value = MagicMock()
        mock_kb = MagicMock()
        mock_kb.output_dir = "output"
        mock_run.return_value = mock_kb

        result = runner.invoke(app, ["run", "test topic", "--date-range", "-2019"])

        # Verify date_range was set on config
        assert mock_config.search.date_range == "-2019"

    @patch("autoreview.pipeline.runner.run_pipeline")
    @patch("autoreview.llm.factory.create_llm_provider")
    @patch("autoreview.config.load_config")
    @patch("autoreview.models.knowledge_base.KnowledgeBase.load_snapshot")
    def test_resume_passes_date_range(self, mock_snap, mock_load, mock_llm, mock_run):
        mock_kb = MagicMock()
        mock_kb.topic = "test"
        mock_kb.domain = "general"
        mock_kb.current_phase = "search"
        mock_kb.candidate_papers = []
        mock_kb.screened_papers = []
        mock_kb.output_dir = "output"
        mock_snap.return_value = mock_kb

        mock_config = MagicMock()
        mock_config.llm = MagicMock()
        mock_config.writing.citation_format = "apa"
        mock_load.return_value = mock_config
        mock_llm.return_value = MagicMock()
        mock_run.return_value = mock_kb

        result = runner.invoke(app, ["resume", "snapshot.json", "--date-range", "2015-2019"])

        assert mock_config.search.date_range == "2015-2019"
