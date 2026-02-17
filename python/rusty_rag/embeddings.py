"""Ollama embedding generation."""

import os
import ollama


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Generate embedding vectors for a batch of text chunks.

    Uses Ollama's embedding API with batch support for efficiency.
    """
    model = model or os.getenv("EMBEDDING_MODEL", "all-minilm")
    response = ollama.embed(model=model, input=texts)
    return response["embeddings"]


def embed_query(query: str, model: str | None = None) -> list[float]:
    """Generate a single embedding vector for a query string."""
    model = model or os.getenv("EMBEDDING_MODEL", "all-minilm")
    response = ollama.embed(model=model, input=query)
    return response["embeddings"][0]
