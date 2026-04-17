"""RAG pipeline orchestration for vector, BM25, and hybrid retrieval."""

from __future__ import annotations

import os

from rich.console import Console

from . import chunk_by_tokens, extract_pdf_text
from .db import create_client, init_collection, search, upsert_chunks
from .embeddings import embed_query, embed_texts
from .llm import ask
from .store import get_all_chunks, get_bm25_index, load_chunk_store, upsert_document_chunks

console = Console()

RETRIEVAL_MODES = ("bm25", "hybrid", "vector")
DEFAULT_RETRIEVAL_MODE = "hybrid"


def get_retrieval_mode(retrieval_mode: str | None = None) -> str:
    """Resolve and validate the active retrieval mode."""
    mode = (retrieval_mode or os.getenv("RETRIEVAL_MODE", DEFAULT_RETRIEVAL_MODE)).lower()
    if mode not in RETRIEVAL_MODES:
        supported = ", ".join(RETRIEVAL_MODES)
        raise ValueError(f"Unsupported retrieval mode '{mode}'. Choose one of: {supported}.")
    return mode


def ingest(file_path: str, retrieval_mode: str | None = None) -> None:
    """Ingest a document for BM25, vector, or hybrid retrieval."""
    mode = get_retrieval_mode(retrieval_mode)
    max_tokens = int(os.getenv("CHUNK_MAX_TOKENS", "256"))
    overlap_tokens = int(os.getenv("CHUNK_OVERLAP_TOKENS", "32"))

    console.print(f"  Extracting text from: [bold]{file_path}[/bold]")
    text = extract_pdf_text(file_path)
    console.print(f"  Extracted [green]{len(text):,}[/green] characters.")

    console.print(
        f"  Chunking text (max_tokens={max_tokens}, overlap={overlap_tokens}) "
        f"[dim]\\[Rust - token-aware][/dim]..."
    )
    chunks = chunk_by_tokens(text, max_tokens, overlap_tokens)
    console.print(f"  Created [green]{len(chunks)}[/green] chunks.")

    console.print("  Persisting local chunk store [dim]\\[BM25][/dim]...")
    upsert_document_chunks(file_path, chunks)

    if mode in {"hybrid", "vector"}:
        console.print("  Generating embeddings [dim]\\[Ollama][/dim]...")
        vectors = embed_texts(chunks)
        console.print(f"  Generated [green]{len(vectors)}[/green] embeddings.")

        console.print("  Connecting to Qdrant...")
        client = create_client()
        init_collection(client)

        console.print("  Upserting chunks to Qdrant...")
        upsert_chunks(client, chunks, vectors)
    else:
        console.print("  Skipping embeddings and Qdrant [dim]\\[BM25-only mode][/dim].")

    console.print(
        f"  [bold green]Successfully ingested {len(chunks)} chunks "
        f"from '{file_path}' using [bold]{mode}[/bold] retrieval.[/bold green]"
    )


def query(question: str, retrieval_mode: str | None = None) -> str:
    """Query the knowledge base using the configured retrieval mode."""
    mode = get_retrieval_mode(retrieval_mode)
    top_k = int(os.getenv("RETRIEVAL_TOP_K", "3"))

    console.print(f'  Searching knowledge base for: "[italic]{question}[/italic]"')
    console.print(f"  Retrieval mode: [bold]{mode}[/bold]")

    vector_results: list[tuple[str, float]] = []
    bm25_results: list[tuple[str, float]] = []

    if mode in {"hybrid", "vector"}:
        vector_results = _run_vector_search(question)

    if mode in {"hybrid", "bm25"}:
        bm25_results = _run_bm25_search(question)

    if mode == "hybrid":
        merged = _reciprocal_rank_fusion(vector_results, bm25_results, top_k=top_k)
    elif mode == "vector":
        merged = vector_results[:top_k]
    else:
        merged = bm25_results[:top_k]

    if not merged:
        if mode in {"bm25", "hybrid"} and not get_all_chunks(load_chunk_store()):
            return (
                "I couldn't find any indexed local chunks yet. "
                "Please ingest a document first with `rusty-rag ingest <file>`."
            )
        return (
            "I couldn't find any relevant information in the knowledge base. "
            "Try rephrasing the question or ingesting additional documents."
        )

    scores_str = ", ".join(f"{score:.3f}" for _, score in merged)
    console.print(
        f"  Found [green]{len(merged)}[/green] relevant chunks "
        f"({mode} scores: {scores_str})"
    )

    context = "\n\n".join(
        f"[Chunk {i + 1} | Score: {score:.3f}]\n{text}"
        for i, (text, score) in enumerate(merged)
    )

    console.print("  Generating response [dim]\\[Ollama][/dim]...")
    return ask(question, context=context)


def _run_vector_search(question: str, top_k: int = 10) -> list[tuple[str, float]]:
    """Run vector similarity search against Qdrant."""
    min_score = float(os.getenv("VECTOR_MIN_SCORE", "0.2"))
    console.print("  Running vector search [dim]\\[Qdrant][/dim]...")
    query_vector = embed_query(question)
    client = create_client()
    results = search(client, query_vector, top_k=top_k, min_score=min_score)
    console.print(f"    -> {len(results)} vector matches")
    return results


def _run_bm25_search(question: str, top_k: int = 10) -> list[tuple[str, float]]:
    """Run BM25 keyword retrieval against the local chunk corpus."""
    console.print("  Running BM25 keyword search [dim]\\[Rust][/dim]...")
    index, chunks = get_bm25_index()
    if index is None:
        console.print("    -> 0 keyword matches")
        return []

    hits = index.search(question, top_k=top_k)
    results = [(chunks[idx], score) for idx, score in hits]
    console.print(f"    -> {len(results)} keyword matches")
    return results


def _reciprocal_rank_fusion(
    vector_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    top_k: int = 3,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge two ranked lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}

    for rank, (text, _) in enumerate(vector_results):
        scores[text] = scores.get(text, 0.0) + 1.0 / (k + rank + 1)

    for rank, (text, _) in enumerate(bm25_results):
        scores[text] = scores.get(text, 0.0) + 1.0 / (k + rank + 1)

    sorted_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_results[:top_k]
