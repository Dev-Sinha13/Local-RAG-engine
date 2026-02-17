"""RAG pipeline orchestration — ties together all components.

Uses hybrid retrieval (vector similarity + BM25 keyword matching) for
higher quality results than either method alone.
"""

import json
import os
from pathlib import Path

from rich.console import Console

from . import extract_pdf_text, chunk_by_tokens, BM25Index
from .embeddings import embed_texts, embed_query
from .llm import ask
from .db import create_client, init_collection, upsert_chunks, search

console = Console()

# Local cache for BM25 index (chunks stored on disk between sessions)
CACHE_DIR = Path.home() / ".rusty_rag"
CHUNK_CACHE = CACHE_DIR / "chunks.json"


def _load_chunk_cache() -> list[str]:
    """Load cached chunks from disk for BM25 indexing."""
    if CHUNK_CACHE.exists():
        with open(CHUNK_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_chunk_cache(chunks: list[str]) -> None:
    """Append new chunks to the local cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    existing = _load_chunk_cache()
    existing.extend(chunks)
    with open(CHUNK_CACHE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False)


def ingest(file_path: str) -> None:
    """Ingest a PDF document into the knowledge base.

    Pipeline:
        Extract text (Rust/mmap)
        → Token-aware chunking (Rust)
        → Generate embeddings (Python/Ollama)
        → Store vectors (Python/Qdrant)
        → Cache chunks for BM25 (local file)
    """
    max_tokens = int(os.getenv("CHUNK_MAX_TOKENS", "256"))
    overlap_tokens = int(os.getenv("CHUNK_OVERLAP_TOKENS", "32"))

    console.print(f"  Extracting text from: [bold]{file_path}[/bold]")
    text = extract_pdf_text(file_path)
    console.print(f"  Extracted [green]{len(text):,}[/green] characters.")

    console.print(
        f"  Chunking text (max_tokens={max_tokens}, overlap={overlap_tokens}) "
        f"[dim]\\[Rust · token-aware][/dim]..."
    )
    chunks = chunk_by_tokens(text, max_tokens, overlap_tokens)
    console.print(f"  Created [green]{len(chunks)}[/green] chunks.")

    console.print("  Generating embeddings [dim]\\[Ollama][/dim]...")
    vectors = embed_texts(chunks)
    console.print(f"  Generated [green]{len(vectors)}[/green] embeddings.")

    console.print("  Connecting to Qdrant...")
    client = create_client()
    init_collection(client)

    console.print("  Upserting chunks to Qdrant...")
    upsert_chunks(client, chunks, vectors)

    console.print("  Caching chunks for BM25 index...")
    _save_chunk_cache(chunks)

    console.print(
        f"  [bold green]✓ Successfully ingested {len(chunks)} chunks "
        f"from '{file_path}'.[/bold green]"
    )


def query(question: str) -> str:
    """Query the knowledge base using hybrid search (vector + BM25).

    Pipeline:
        Embed query (Python/Ollama)
        → Vector search (Python/Qdrant)
        → BM25 keyword search (Rust)
        → Reciprocal Rank Fusion (merge results)
        → Build context
        → LLM response (Python/Ollama)
    """
    console.print(f'  Searching knowledge base for: "[italic]{question}[/italic]"')

    # 1. Vector search via Qdrant
    console.print("  Running vector search [dim]\\[Qdrant][/dim]...")
    query_vector = embed_query(question)
    client = create_client()
    vector_results = search(client, query_vector, top_k=10, min_score=0.2)
    console.print(f"    → {len(vector_results)} vector matches")

    # 2. BM25 keyword search via Rust
    cached_chunks = _load_chunk_cache()
    bm25_results: list[tuple[str, float]] = []

    if cached_chunks:
        console.print("  Running BM25 keyword search [dim]\\[Rust][/dim]...")
        index = BM25Index(cached_chunks)
        bm25_hits = index.search(question, top_k=10)
        bm25_results = [(cached_chunks[idx], score) for idx, score in bm25_hits]
        console.print(f"    → {len(bm25_results)} keyword matches")

    # 3. Merge results using Reciprocal Rank Fusion
    merged = _reciprocal_rank_fusion(vector_results, bm25_results, top_k=3)

    if not merged:
        return (
            "I couldn't find any relevant information in the knowledge base. "
            "Please make sure you've ingested documents first with "
            "`rusty-rag ingest <file>`."
        )

    scores_str = ", ".join(f"{score:.3f}" for _, score in merged)
    console.print(
        f"  Found [green]{len(merged)}[/green] relevant chunks "
        f"(hybrid scores: {scores_str})"
    )

    # 4. Build context from retrieved chunks
    context = "\n\n".join(
        f"[Chunk {i + 1} | Score: {score:.3f}]\n{text}"
        for i, (text, score) in enumerate(merged)
    )

    # 5. Generate LLM response
    console.print("  Generating response [dim]\\[Ollama][/dim]...")
    return ask(question, context=context)


def _reciprocal_rank_fusion(
    vector_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    top_k: int = 3,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge two ranked result lists using Reciprocal Rank Fusion (RRF).

    RRF is a simple, parameter-free method for combining ranked lists:
        RRF_score(d) = Σ 1 / (k + rank_i(d))

    where k=60 is the standard constant and rank_i is the position of
    document d in result list i.
    """
    scores: dict[str, float] = {}

    for rank, (text, _) in enumerate(vector_results):
        scores[text] = scores.get(text, 0.0) + 1.0 / (k + rank + 1)

    for rank, (text, _) in enumerate(bm25_results):
        scores[text] = scores.get(text, 0.0) + 1.0 / (k + rank + 1)

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]
