"""Unit tests for retrieval modes in the RAG pipeline."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from rusty_rag import rag, store


class RagPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.original_cache_dir = os.environ.get("RUSTY_RAG_CACHE_DIR")
        os.environ["RUSTY_RAG_CACHE_DIR"] = self.temp_dir.name
        store.clear_bm25_cache()
        self.addCleanup(self._restore_env)

    def _restore_env(self) -> None:
        if self.original_cache_dir is None:
            os.environ.pop("RUSTY_RAG_CACHE_DIR", None)
        else:
            os.environ["RUSTY_RAG_CACHE_DIR"] = self.original_cache_dir
        os.environ.pop("RETRIEVAL_MODE", None)
        os.environ.pop("RETRIEVAL_TOP_K", None)
        os.environ.pop("VECTOR_MIN_SCORE", None)
        store.clear_bm25_cache()

    def test_get_retrieval_mode_uses_explicit_value_or_env(self) -> None:
        os.environ["RETRIEVAL_MODE"] = "bm25"
        self.assertEqual(rag.get_retrieval_mode(), "bm25")
        self.assertEqual(rag.get_retrieval_mode("vector"), "vector")

    def test_get_retrieval_mode_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            rag.get_retrieval_mode("semantic")

    def test_ingest_bm25_mode_skips_vector_dependencies_and_persists_chunks(self) -> None:
        with (
            patch.object(rag, "extract_pdf_text", return_value="alpha beta gamma") as extract_mock,
            patch.object(rag, "chunk_by_tokens", return_value=["alpha beta", "beta gamma"]) as chunk_mock,
            patch.object(rag, "embed_texts") as embed_mock,
            patch.object(rag, "create_client") as client_mock,
            patch.object(rag, "init_collection") as init_mock,
            patch.object(rag, "upsert_chunks") as upsert_mock,
        ):
            rag.ingest("paper.pdf", retrieval_mode="bm25")

        extract_mock.assert_called_once_with("paper.pdf")
        chunk_mock.assert_called_once()
        embed_mock.assert_not_called()
        client_mock.assert_not_called()
        init_mock.assert_not_called()
        upsert_mock.assert_not_called()
        self.assertEqual(store.get_all_chunks(), ["alpha beta", "beta gamma"])

    def test_ingest_hybrid_mode_runs_vector_pipeline_and_local_persistence(self) -> None:
        client = object()
        with (
            patch.object(rag, "extract_pdf_text", return_value="alpha beta gamma"),
            patch.object(rag, "chunk_by_tokens", return_value=["alpha beta", "beta gamma"]),
            patch.object(rag, "embed_texts", return_value=[[0.1, 0.2], [0.3, 0.4]]) as embed_mock,
            patch.object(rag, "create_client", return_value=client) as client_mock,
            patch.object(rag, "init_collection") as init_mock,
            patch.object(rag, "upsert_chunks") as upsert_mock,
        ):
            rag.ingest("paper.pdf", retrieval_mode="hybrid")

        embed_mock.assert_called_once_with(["alpha beta", "beta gamma"])
        client_mock.assert_called_once_with()
        init_mock.assert_called_once_with(client)
        upsert_mock.assert_called_once_with(client, ["alpha beta", "beta gamma"], [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(store.get_all_chunks(), ["alpha beta", "beta gamma"])

    def test_ingest_vector_mode_still_persists_local_chunks(self) -> None:
        with (
            patch.object(rag, "extract_pdf_text", return_value="alpha beta"),
            patch.object(rag, "chunk_by_tokens", return_value=["alpha beta"]),
            patch.object(rag, "embed_texts", return_value=[[0.1]]),
            patch.object(rag, "create_client", return_value=object()),
            patch.object(rag, "init_collection"),
            patch.object(rag, "upsert_chunks"),
        ):
            rag.ingest("vector.pdf", retrieval_mode="vector")

        self.assertEqual(store.get_all_chunks(), ["alpha beta"])

    def test_query_bm25_mode_uses_keyword_search_only(self) -> None:
        store.upsert_document_chunks(
            "paper.pdf",
            [
                "machine learning uses training data",
                "gardening roses need water",
            ],
        )

        with (
            patch.object(rag, "embed_query") as embed_query_mock,
            patch.object(rag, "create_client") as create_client_mock,
            patch.object(rag, "search") as search_mock,
            patch.object(rag, "ask", return_value="keyword answer") as ask_mock,
        ):
            response = rag.query("machine learning", retrieval_mode="bm25")

        self.assertEqual(response, "keyword answer")
        embed_query_mock.assert_not_called()
        create_client_mock.assert_not_called()
        search_mock.assert_not_called()
        context = ask_mock.call_args.kwargs["context"]
        self.assertIn("machine learning uses training data", context)
        self.assertNotIn("gardening roses need water", context)

    def test_query_vector_mode_uses_vector_search_only(self) -> None:
        with (
            patch.object(rag, "embed_query", return_value=[0.1, 0.2]) as embed_query_mock,
            patch.object(rag, "create_client", return_value=object()) as create_client_mock,
            patch.object(rag, "search", return_value=[("semantic result", 0.91)]) as search_mock,
            patch.object(rag, "get_bm25_index") as bm25_index_mock,
            patch.object(rag, "ask", return_value="vector answer") as ask_mock,
        ):
            response = rag.query("paraphrased question", retrieval_mode="vector")

        self.assertEqual(response, "vector answer")
        embed_query_mock.assert_called_once_with("paraphrased question")
        create_client_mock.assert_called_once_with()
        search_mock.assert_called_once()
        bm25_index_mock.assert_not_called()
        self.assertIn("semantic result", ask_mock.call_args.kwargs["context"])

    def test_query_hybrid_mode_fuses_vector_and_bm25_results(self) -> None:
        store.upsert_document_chunks(
            "paper.pdf",
            [
                "machine learning uses training data",
                "retrieval augmented generation combines search and generation",
            ],
        )

        with (
            patch.object(rag, "embed_query", return_value=[0.3, 0.4]),
            patch.object(rag, "create_client", return_value=object()),
            patch.object(rag, "search", return_value=[("semantic vector result", 0.88)]),
            patch.object(rag, "ask", return_value="hybrid answer") as ask_mock,
        ):
            response = rag.query("machine learning retrieval", retrieval_mode="hybrid")

        self.assertEqual(response, "hybrid answer")
        context = ask_mock.call_args.kwargs["context"]
        self.assertIn("semantic vector result", context)
        self.assertIn("machine learning uses training data", context)

    def test_query_reports_missing_local_chunks_for_bm25_mode(self) -> None:
        with patch.object(rag, "ask") as ask_mock:
            response = rag.query("anything", retrieval_mode="bm25")

        self.assertIn("indexed local chunks yet", response)
        ask_mock.assert_not_called()

    def test_reciprocal_rank_fusion_merges_scores_by_rank(self) -> None:
        merged = rag._reciprocal_rank_fusion(
            [("alpha", 0.8), ("beta", 0.7)],
            [("beta", 10.0), ("gamma", 9.0)],
            top_k=3,
            k=1,
        )

        self.assertEqual(merged[0][0], "beta")
        self.assertEqual([item[0] for item in merged], ["beta", "alpha", "gamma"])


if __name__ == "__main__":
    unittest.main()
