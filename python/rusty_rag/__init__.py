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

try:
    from .rusty_rag_core import ShardStreamQuantizer
except ImportError:  # Allows source-tree Python tests before the extension is rebuilt.
    ShardStreamQuantizer = None

__all__ = [
    "extract_pdf_text",
    "chunk_text_parallel",
    "chunk_text",
    "chunk_by_tokens",
    "tokenize",
    "token_count",
    "BM25Index",
]

if ShardStreamQuantizer is not None:
    __all__.append("ShardStreamQuantizer")
