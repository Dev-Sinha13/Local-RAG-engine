"""RustyRAG — Local-first, privacy-focused RAG CLI tool.

Rust core functions are available directly:
    from rusty_rag import extract_pdf_text, chunk_text_parallel
"""

from .rusty_rag_core import extract_pdf_text, chunk_text_parallel, chunk_text

__all__ = ["extract_pdf_text", "chunk_text_parallel", "chunk_text"]
