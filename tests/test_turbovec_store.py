"""Tests for the persistent local turbovec vector backend."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from rusty_rag import turbovec_store


class TurbovecStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.original_cache_dir = os.environ.get("RUSTY_RAG_CACHE_DIR")
        self.original_backend = os.environ.get("VECTOR_BACKEND")
        self.original_bit_width = os.environ.get("TURBOVEC_BIT_WIDTH")
        os.environ["RUSTY_RAG_CACHE_DIR"] = self.temp_dir.name
        os.environ["TURBOVEC_BIT_WIDTH"] = "4"
        self.addCleanup(self._restore_env)

    def _restore_env(self) -> None:
        for key, value in {
            "RUSTY_RAG_CACHE_DIR": self.original_cache_dir,
            "VECTOR_BACKEND": self.original_backend,
            "TURBOVEC_BIT_WIDTH": self.original_bit_width,
        }.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_backend_selection_defaults_to_qdrant(self) -> None:
        os.environ.pop("VECTOR_BACKEND", None)
        self.assertEqual(turbovec_store.get_vector_backend(), "qdrant")
        self.assertEqual(turbovec_store.get_vector_backend("turbovec"), "turbovec")
        with self.assertRaises(ValueError):
            turbovec_store.get_vector_backend("other")

    def test_vectors_persist_search_and_replace_prior_ids(self) -> None:
        first_vectors = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        turbovec_store.upsert_vectors(first_vectors, [101, 102])

        self.assertTrue(turbovec_store.get_index_path().exists())
        initial_results = turbovec_store.search(first_vectors[0], top_k=2)
        self.assertEqual({chunk_id for chunk_id, _ in initial_results}, {101, 102})

        turbovec_store.upsert_vectors([first_vectors[0]], [201], replaced_ids=[101, 102])
        replacement_results = turbovec_store.search(first_vectors[0], top_k=2)
        self.assertEqual([chunk_id for chunk_id, _ in replacement_results], [201])

    def test_search_returns_empty_results_without_an_index(self) -> None:
        self.assertEqual(turbovec_store.search([1.0] * 8), [])


if __name__ == "__main__":
    unittest.main()
