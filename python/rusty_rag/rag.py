"""RAG pipeline orchestration — ties together all components."""

import os
from rich.console import Console

from . import extract_pdf_text, chunk_text_parallel
from .embeddings import embed_texts, embed_query
from .llm import ask
from .db import create_client, init_collection, upsert_chunks, search

console = Console()


def ingest(file_path: str) -> None:
    """Ingest a PDF document into the knowledge base.

    Pipeline: Extract text (Rust/mmap) → Chunk (Rust/Rayon) → Embed (Ollama) → Store (Qdrant)
    """
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))

    console.print(f"  Extracting text from: [bold]{file_path}[/bold]")
    text = extract_pdf_text(file_path)
    console.print(f"  Extracted [green]{len(text):,}[/green] characters.")

    console.print(
        f"  Chunking text (size={chunk_size}, overlap={chunk_overlap}) "
        f"[dim]\\[Rust/Rayon parallel][/dim]..."
    )
    chunks = chunk_text_parallel(text, chunk_size, chunk_overlap)
    console.print(f"  Created [green]{len(chunks)}[/green] chunks.")

    console.print("  Generating embeddings [dim]\\[Ollama][/dim]...")
    vectors = embed_texts(chunks)
    console.print(f"  Generated [green]{len(vectors)}[/green] embeddings.")

    console.print("  Connecting to Qdrant...")
    client = create_client()
    init_collection(client)

    console.print("  Upserting chunks to Qdrant...")
    upsert_chunks(client, chunks, vectors)
    console.print(
        f"  [bold green]✓ Successfully ingested {len(chunks)} chunks "
        f"from '{file_path}'.[/bold green]"
    )


def query(question: str) -> str:
    """Query the knowledge base and generate an AI response.

    Pipeline: Embed query (Ollama) → Search (Qdrant) → Build context → LLM response (Ollama)
    """
    console.print(f'  Searching knowledge base for: "[italic]{question}[/italic]"')

    query_vector = embed_query(question)

    client = create_client()
    results = search(client, query_vector)

    if not results:
        return (
            "I couldn't find any relevant information in the knowledge base. "
            "Please make sure you've ingested documents first with "
            "`rusty-rag ingest <file>`."
        )

    scores = ", ".join(f"{score:.3f}" for _, score in results)
    console.print(
        f"  Found [green]{len(results)}[/green] relevant chunks (scores: {scores})"
    )

    # Build context from retrieved chunks
    context = "\n\n".join(
        f"[Chunk {i + 1} | Score: {score:.3f}]\n{text}"
        for i, (text, score) in enumerate(results)
    )

    console.print("  Generating response [dim]\\[Ollama][/dim]...")
    return ask(question, context=context)
