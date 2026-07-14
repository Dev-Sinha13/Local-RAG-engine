"""Local corpus storage for BM25 and hybrid retrieval modes."""

from __future__ import annotations

import json
import os
from hashlib import blake2b
from pathlib import Path
from typing import TypedDict

from .rusty_rag_core import BM25Index

STORE_VERSION = 3
UNKNOWN_SOURCE = "<unknown>"


class DocumentChunks(TypedDict):
    """Stored chunk payload for a single ingested source document."""

    source: str
    chunk_count: int
    chunks: list[str]
    chunk_ids: list[int]


class ChunkStore(TypedDict):
    """On-disk representation of the local chunk corpus."""

    version: int
    documents: list[DocumentChunks]


_bm25_cache: dict[str, object] = {
    "signature": None,
    "chunks": [],
    "index": None,
}


def get_cache_dir() -> Path:
    """Return the cache directory used for local retrieval state."""
    configured = os.getenv("RUSTY_RAG_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".rusty_rag"


def get_chunk_store_path() -> Path:
    """Return the path of the local chunk store."""
    return get_cache_dir() / "chunks.json"


def empty_chunk_store() -> ChunkStore:
    """Create an empty chunk store payload."""
    return {"version": STORE_VERSION, "documents": []}


def make_chunk_ids(source: str, chunks: list[str]) -> list[int]:
    """Make stable uint64 ids for chunks belonging to one source document."""
    ids: list[int] = []
    used_ids: set[int] = set()
    for ordinal, chunk in enumerate(chunks):
        salt = 0
        while True:
            value = f"{source}\0{ordinal}\0{chunk}\0{salt}".encode("utf-8")
            chunk_id = int.from_bytes(blake2b(value, digest_size=8).digest(), "big")
            if chunk_id not in used_ids:
                break
            salt += 1
        used_ids.add(chunk_id)
        ids.append(chunk_id)
    return ids


def load_chunk_store() -> ChunkStore:
    """Load the on-disk chunk store, migrating older formats when needed."""
    path = get_chunk_store_path()
    if not path.exists():
        return empty_chunk_store()

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        chunk_ids = make_chunk_ids(UNKNOWN_SOURCE, payload)
        return {
            "version": STORE_VERSION,
            "documents": [
                {
                    "source": UNKNOWN_SOURCE,
                    "chunk_count": len(payload),
                    "chunks": payload,
                    "chunk_ids": chunk_ids,
                }
            ],
        }

    documents = payload.get("documents", [])
    normalized_documents: list[DocumentChunks] = []
    for document in documents:
        source = document.get("source", UNKNOWN_SOURCE)
        chunks = list(document.get("chunks", []))
        chunk_ids = list(document.get("chunk_ids", []))
        if len(chunk_ids) != len(chunks):
            chunk_ids = make_chunk_ids(source, chunks)
        normalized_documents.append(
            {
                "source": source,
                "chunk_count": document.get("chunk_count", len(chunks)),
                "chunks": chunks,
                "chunk_ids": chunk_ids,
            }
        )

    return {
        "version": STORE_VERSION,
        "documents": normalized_documents,
    }


def save_chunk_store(store: ChunkStore) -> None:
    """Persist the chunk store to disk."""
    path = get_chunk_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(store, handle, ensure_ascii=False, indent=2)
    clear_bm25_cache()


def normalize_source_path(source: str) -> str:
    """Normalize source paths so re-ingest can replace prior entries cleanly."""
    if source == UNKNOWN_SOURCE:
        return UNKNOWN_SOURCE
    return str(Path(source).expanduser().resolve(strict=False))


def upsert_document_chunks(source: str, chunks: list[str]) -> ChunkStore:
    """Insert or replace chunk data for a source document in local storage."""
    store = load_chunk_store()
    normalized_source = normalize_source_path(source)
    remaining_documents = [
        document
        for document in store["documents"]
        if normalize_source_path(document["source"]) != normalized_source
    ]
    remaining_documents.append(
        {
            "source": normalized_source,
            "chunk_count": len(chunks),
            "chunks": list(chunks),
            "chunk_ids": make_chunk_ids(normalized_source, chunks),
        }
    )
    updated_store: ChunkStore = {
        "version": STORE_VERSION,
        "documents": remaining_documents,
    }
    save_chunk_store(updated_store)
    return updated_store


def get_document_chunks(source: str, store: ChunkStore | None = None) -> DocumentChunks | None:
    """Return the stored document record for a source, if present."""
    normalized_source = normalize_source_path(source)
    active_store = store or load_chunk_store()
    return next(
        (
            document
            for document in active_store["documents"]
            if normalize_source_path(document["source"]) == normalized_source
        ),
        None,
    )


def get_chunks_by_id(store: ChunkStore | None = None) -> dict[int, str]:
    """Return a mapping from stable chunk id to text."""
    active_store = store or load_chunk_store()
    return {
        chunk_id: chunk
        for document in active_store["documents"]
        for chunk_id, chunk in zip(document["chunk_ids"], document["chunks"])
    }


def get_all_chunks(store: ChunkStore | None = None) -> list[str]:
    """Flatten the chunk store into a single chunk list."""
    active_store = store or load_chunk_store()
    return [
        chunk
        for document in active_store["documents"]
        for chunk in document["chunks"]
    ]


def clear_bm25_cache() -> None:
    """Reset the in-process BM25 cache."""
    _bm25_cache["signature"] = None
    _bm25_cache["chunks"] = []
    _bm25_cache["index"] = None


def get_bm25_index(store: ChunkStore | None = None) -> tuple[BM25Index | None, list[str]]:
    """Return a cached BM25 index for the current local corpus."""
    active_store = store or load_chunk_store()
    chunks = get_all_chunks(active_store)
    if not chunks:
        return None, []

    path = get_chunk_store_path()
    signature = (
        path.stat().st_mtime_ns if path.exists() else None,
        len(active_store["documents"]),
        len(chunks),
    )

    if _bm25_cache["signature"] == signature:
        return _bm25_cache["index"], list(_bm25_cache["chunks"])

    index = BM25Index(chunks)
    _bm25_cache["signature"] = signature
    _bm25_cache["chunks"] = list(chunks)
    _bm25_cache["index"] = index
    return index, list(chunks)
