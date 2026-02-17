"""RustyRAG — Local-first, privacy-focused RAG CLI tool.

Rust core functions are available directly:
    from rusty_rag import extract_pdf_text, chunk_by_tokens, BM25Index
"""

from .rusty_rag_core import (
    extract_pdf_text,
    chunk_text_parallel,
    chunk_text,
    chunk_by_tokens,
    tokenize,
    token_count,
    BM25Index,
)

__all__ = [
    "extract_pdf_text",
    "chunk_text_parallel",
    "chunk_text",
    "chunk_by_tokens",
    "tokenize",
    "token_count",
    "BM25Index",
]
