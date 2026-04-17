"""Unit tests for local chunk storage and BM25 index caching."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from rusty_rag import store


class ChunkStoreTests(unittest.TestCase):
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
        store.clear_bm25_cache()

    def test_load_chunk_store_returns_empty_store_when_missing(self) -> None:
        self.assertEqual(store.load_chunk_store(), {"version": store.STORE_VERSION, "documents": []})

    def test_load_chunk_store_migrates_legacy_chunk_list(self) -> None:
        path = store.get_chunk_store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(["alpha", "beta"], handle)

        migrated = store.load_chunk_store()

        self.assertEqual(migrated["version"], store.STORE_VERSION)
        self.assertEqual(len(migrated["documents"]), 1)
        self.assertEqual(migrated["documents"][0]["source"], store.UNKNOWN_SOURCE)
        self.assertEqual(migrated["documents"][0]["chunks"], ["alpha", "beta"])

    def test_upsert_document_chunks_replaces_existing_source(self) -> None:
        store.upsert_document_chunks("docs/example.pdf", ["alpha", "beta"])
        updated = store.upsert_document_chunks("docs/example.pdf", ["gamma"])

        self.assertEqual(len(updated["documents"]), 1)
        self.assertEqual(updated["documents"][0]["chunks"], ["gamma"])
        self.assertEqual(
            updated["documents"][0]["source"],
            str((PROJECT_ROOT / "docs" / "example.pdf").resolve(strict=False)),
        )

    def test_get_bm25_index_reuses_cache_until_store_changes(self) -> None:
        store.upsert_document_chunks("one.pdf", ["machine learning systems"])

        first_index, first_chunks = store.get_bm25_index()
        second_index, second_chunks = store.get_bm25_index()

        self.assertIs(first_index, second_index)
        self.assertEqual(first_chunks, second_chunks)

        store.upsert_document_chunks("two.pdf", ["gardening roses"])
        third_index, third_chunks = store.get_bm25_index()

        self.assertIsNot(first_index, third_index)
        self.assertIn("gardening roses", third_chunks)


if __name__ == "__main__":
    unittest.main()
