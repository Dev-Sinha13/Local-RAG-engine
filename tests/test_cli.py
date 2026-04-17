"""CLI tests for retrieval mode plumbing."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from rusty_rag import cli


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_ingest_command_passes_retrieval_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "paper.pdf"
            pdf_path.write_text("placeholder", encoding="utf-8")

            with patch("rusty_rag.rag.ingest") as ingest_mock:
                result = self.runner.invoke(
                    cli.main,
                    ["ingest", str(pdf_path), "--retrieval-mode", "bm25"],
                )

        self.assertEqual(result.exit_code, 0, result.output)
        ingest_mock.assert_called_once_with(str(pdf_path), retrieval_mode="bm25")

    def test_query_command_passes_retrieval_mode(self) -> None:
        with patch("rusty_rag.rag.query", return_value="ready") as query_mock:
            result = self.runner.invoke(
                cli.main,
                ["query", "What is RAG?", "--retrieval-mode", "vector"],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        query_mock.assert_called_once_with("What is RAG?", retrieval_mode="vector")
        self.assertIn("ready", result.output)

    def test_help_mentions_retrieval_mode_option(self) -> None:
        result = self.runner.invoke(cli.main, ["query", "--help"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--retrieval-mode", result.output)


if __name__ == "__main__":
    unittest.main()
