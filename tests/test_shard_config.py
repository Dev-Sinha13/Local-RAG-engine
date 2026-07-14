"""Tests for the no-download SHARD backend configuration."""

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

from rusty_rag import shard
from rusty_rag import llm


class ShardConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original = {
            key: os.environ.get(key)
            for key in ("LLM_BACKEND", "SHARD_MODEL_PATH", "SHARD_STREAM_BITS", "SHARD_MAX_NEW_TOKENS")
        }
        self.addCleanup(self._restore_environment)

    def _restore_environment(self) -> None:
        for key, value in self.original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_backend_defaults_to_ollama_and_validates_values(self) -> None:
        os.environ.pop("LLM_BACKEND", None)
        self.assertEqual(shard.get_llm_backend(), "ollama")
        self.assertEqual(shard.get_llm_backend("shard"), "shard")
        with self.assertRaises(ValueError):
            shard.get_llm_backend("unknown")

    def test_settings_use_local_path_and_validate_bit_width(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            os.environ["SHARD_MODEL_PATH"] = directory
            os.environ["SHARD_STREAM_BITS"] = "8"
            settings = shard.get_shard_settings()

        self.assertEqual(settings.model_path, Path(directory))
        self.assertEqual(settings.stream_bits, 8)
        os.environ["SHARD_STREAM_BITS"] = "5"
        with self.assertRaises(ValueError):
            shard.get_shard_settings()

    def test_shard_backend_never_downloads_a_missing_model(self) -> None:
        os.environ.pop("SHARD_MODEL_PATH", None)
        with self.assertRaisesRegex(RuntimeError, "No model was downloaded"):
            shard.ask_with_shard("question", "system")

    def test_llm_routes_shard_backend_without_importing_ollama(self) -> None:
        os.environ["LLM_BACKEND"] = "shard"
        os.environ.pop("SHARD_MODEL_PATH", None)
        with self.assertRaisesRegex(RuntimeError, "No model was downloaded"):
            llm.ask("question")


if __name__ == "__main__":
    unittest.main()
