"""Persistent local vector search backed by turbovec."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np

from .store import get_cache_dir

VECTOR_BACKENDS = ("qdrant", "turbovec")
DEFAULT_VECTOR_BACKEND = "qdrant"
DEFAULT_BIT_WIDTH = 4


def get_vector_backend(vector_backend: str | None = None) -> str:
    """Resolve the configured vector backend."""
    backend = (vector_backend or os.getenv("VECTOR_BACKEND", DEFAULT_VECTOR_BACKEND)).lower()
    if backend not in VECTOR_BACKENDS:
        supported = ", ".join(VECTOR_BACKENDS)
        raise ValueError(f"Unsupported vector backend '{backend}'. Choose one of: {supported}.")
    return backend


def get_index_path() -> Path:
    """Return the local turbovec index path."""
    return get_cache_dir() / "vectors.tvim"


def get_bit_width() -> int:
    """Return and validate the configured turbovec quantization width."""
    value = int(os.getenv("TURBOVEC_BIT_WIDTH", str(DEFAULT_BIT_WIDTH)))
    if value not in {2, 3, 4}:
        raise ValueError("TURBOVEC_BIT_WIDTH must be 2, 3, or 4.")
    return value


def _index_class():
    try:
        from turbovec import IdMapIndex
    except ImportError as exc:  # pragma: no cover - depends on installation state
        raise RuntimeError(
            "The turbovec backend requires the optional 'turbovec' package. "
            "Reinstall RustyRAG to install project dependencies."
        ) from exc
    return IdMapIndex


def _as_vectors(vectors: Iterable[Iterable[float]]) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(list(vectors), dtype=np.float32))
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("Expected at least one embedding vector.")
    if array.shape[1] % 8:
        raise ValueError(
            f"turbovec requires an embedding dimension divisible by 8; got {array.shape[1]}."
        )
    return array


def _load_or_create_index(dimension: int):
    index_path = get_index_path()
    index_class = _index_class()
    if index_path.exists():
        index = index_class.load(str(index_path))
        if index.dim != dimension:
            raise ValueError(
                f"The persisted turbovec index has dimension {index.dim}, "
                f"but received embeddings with dimension {dimension}. "
                "Use a separate RUSTY_RAG_CACHE_DIR or remove the old local index."
            )
        return index
    return index_class(dim=dimension, bit_width=get_bit_width())


def _write_index(index) -> None:
    index_path = get_index_path()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = index_path.with_suffix(".tvim.tmp")
    index.write(str(temporary_path))
    temporary_path.replace(index_path)


def upsert_vectors(
    vectors: Iterable[Iterable[float]],
    ids: Iterable[int],
    replaced_ids: Iterable[int] = (),
) -> None:
    """Replace a source's vectors and persist the resulting IdMap index."""
    vector_array = _as_vectors(vectors)
    id_array = np.asarray(list(ids), dtype=np.uint64)
    if len(id_array) != len(vector_array):
        raise ValueError("The number of vectors must match the number of chunk ids.")
    if len(np.unique(id_array)) != len(id_array):
        raise ValueError("Chunk ids must be unique within a turbovec upsert.")

    index = _load_or_create_index(vector_array.shape[1])
    for chunk_id in replaced_ids:
        index.remove(int(chunk_id))
    index.add_with_ids(vector_array, id_array)
    _write_index(index)


def search(query_vector: Iterable[float], top_k: int = 3) -> list[tuple[int, float]]:
    """Search the local index and return stable chunk-id/score pairs."""
    index_path = get_index_path()
    if not index_path.exists():
        return []

    query = np.ascontiguousarray(np.asarray(list(query_vector), dtype=np.float32)).reshape(1, -1)
    index = _index_class().load(str(index_path))
    if index.dim != query.shape[1]:
        raise ValueError(
            f"The persisted turbovec index has dimension {index.dim}, "
            f"but the query embedding has dimension {query.shape[1]}."
        )
    if len(index) == 0:
        return []

    scores, ids = index.search(query, k=min(top_k, len(index)))
    return [(int(chunk_id), float(score)) for score, chunk_id in zip(scores[0], ids[0])]
