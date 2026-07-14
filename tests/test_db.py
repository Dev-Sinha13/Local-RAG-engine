"""Compatibility tests for Qdrant client search APIs."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from rusty_rag import db


class DbSearchTests(unittest.TestCase):
    def test_search_uses_current_query_points_api_when_available(self) -> None:
        class CurrentClient:
            def query_points(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(points=[SimpleNamespace(payload={"text": "match"}, score=0.9)])

        client = CurrentClient()
        self.assertEqual(db.search(client, [0.1, 0.2], top_k=4, min_score=0.3), [("match", 0.9)])
        self.assertEqual(client.kwargs["query"], [0.1, 0.2])
        self.assertEqual(client.kwargs["limit"], 4)

    def test_search_uses_legacy_search_api_when_needed(self) -> None:
        class LegacyClient:
            def search(self, **kwargs):
                self.kwargs = kwargs
                return [SimpleNamespace(payload={"text": "legacy"}, score=0.8)]

        client = LegacyClient()
        self.assertEqual(db.search(client, [0.1, 0.2]), [("legacy", 0.8)])
        self.assertEqual(client.kwargs["query_vector"], [0.1, 0.2])
