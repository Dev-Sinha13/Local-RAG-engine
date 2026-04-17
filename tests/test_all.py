"""
RustyRAG smoke test script.

This script exercises:
  1. Rust core functions without external services
  2. PDF extraction without external services
  3. BM25-only ingest/query with Ollama
  4. Hybrid ingest/query with Ollama and Qdrant

Usage:
    pip install fpdf2
    python tests/test_all.py
"""

from __future__ import annotations

import os
import sys
import textwrap
import time
from pathlib import Path

GREEN = ""
RED = ""
YELLOW = ""
BOLD = ""
DIM = ""
RESET = ""

passed = 0
failed = 0
skipped = 0

SAMPLE_DIR = Path(__file__).parent / "sample_data"
SAMPLE_PDF = SAMPLE_DIR / "test_paper.pdf"

SAMPLE_TEXT = textwrap.dedent(
    """\
    Advances in Retrieval-Augmented Generation for Local Document Processing

    Abstract

    This paper presents a local-first retrieval-augmented generation system
    that combines Rust for document parsing and chunking with Python for
    retrieval orchestration and language model inference.

    Methodology

    Our system supports three retrieval modes:

    1. BM25-only retrieval for a fully vectorless workflow
    2. Vector retrieval using local embeddings and Qdrant
    3. Hybrid retrieval using Reciprocal Rank Fusion

    Experiments

    We evaluated the system on the following datasets:

    1. MMLU
    2. HumanEval
    3. TriviaQA

    Results

    The hybrid mode outperformed either single retrieval method.
    BM25-only mode remained strong for exact terminology and benchmark names.
    Memory-mapped I/O reduced peak memory usage to 340 MB.
    """
)


def ok(name: str, detail: str = "") -> None:
    global passed
    passed += 1
    print(f"  [OK] {name}" + (f"  ({detail})" if detail else ""))


def fail(name: str, error: str) -> None:
    global failed
    failed += 1
    print(f"  [FAIL] {name}")
    print(f"    {error}")


def skip(name: str, reason: str) -> None:
    global skipped
    skipped += 1
    print(f"  [SKIP] {name}  ({reason})")


def section(title: str) -> None:
    print(f"\n{'-' * 50}")
    print(f"  {title}")
    print(f"{'-' * 50}\n")


def generate_sample_pdf() -> Path | None:
    """Generate a simple ASCII-safe PDF for smoke tests."""
    try:
        from fpdf import FPDF
    except ImportError:
        print("  [WARN] fpdf2 is not installed")
        return None

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Courier", size=11)

    try:
        for line in SAMPLE_TEXT.splitlines():
            if line.strip() and not line.startswith(" ") and len(line.strip()) < 80:
                pdf.set_font("Courier", "B", size=13)
                pdf.cell(0, 10, line.strip(), new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Courier", size=11)
            else:
                pdf.multi_cell(0, 6, line)
        pdf.output(str(SAMPLE_PDF))
    except Exception as exc:
        print(f"  [WARN] Failed to generate PDF: {exc}")
        return None

    return SAMPLE_PDF


def test_rust_core() -> bool:
    section("Rust Core Functions")

    try:
        from rusty_rag import (
            BM25Index,
            chunk_by_tokens,
            chunk_text,
            chunk_text_parallel,
            token_count,
            tokenize,
        )
    except ImportError:
        fail("Import rusty_rag", "Module not found. Run: maturin develop --release")
        return False

    ok("Import rusty_rag")

    tokens = tokenize("Hello, World! This is a RAG test.")
    assert tokens == ["hello", "world", "this", "is", "a", "rag", "test"]
    ok("tokenize()", f"{len(tokens)} tokens")

    count = token_count("The quick brown fox jumps")
    assert count == 5
    ok("token_count()", f"{count} tokens")

    text = "word " * 1000
    chunks = chunk_text(text, 500, 50)
    assert len(chunks) > 1
    ok("chunk_text()", f"{len(chunks)} chunks")

    parallel = chunk_text_parallel(text, 500, 50)
    assert chunks == parallel
    ok("chunk_text_parallel()", "matches sequential output")

    text2 = "The quick brown fox. " * 100
    token_chunks = chunk_by_tokens(text2, 50, 10)
    assert len(token_chunks) > 1
    for i, chunk in enumerate(token_chunks[:-1]):
        wc = token_count(chunk)
        assert wc <= 50, f"Chunk {i} has {wc} tokens"
    ok("chunk_by_tokens()", f"{len(token_chunks)} token-aware chunks")

    docs = [
        "machine learning and deep neural networks",
        "cooking Italian pasta with fresh tomatoes",
        "deep learning for natural language processing",
        "gardening tips for growing roses in spring",
        "transformer models improve machine translation",
    ]
    index = BM25Index(docs)
    results = index.search("machine learning neural networks", 3)
    assert len(results) > 0
    assert results[0][0] in [0, 2, 4]
    ok("BM25Index.search()", f"top doc: {results[0][0]}")

    return True


def test_pdf_extraction() -> bool:
    section("PDF Extraction")

    try:
        from rusty_rag import chunk_by_tokens, extract_pdf_text, token_count
    except ImportError:
        fail("Import rusty_rag", "Module not found")
        return False

    pdf_path = generate_sample_pdf()
    if not pdf_path:
        skip("Generate sample PDF", "PDF generation failed")
        return False
    ok("Generate sample PDF", str(pdf_path))

    text = extract_pdf_text(str(pdf_path))
    assert len(text) > 100
    ok("extract_pdf_text()", f"{len(text):,} chars extracted")

    lowered = text.lower()
    assert "retrieval" in lowered
    assert "bm25" in lowered
    assert "triviaqa" in lowered
    ok("Content verification", "key terms found")

    chunks = chunk_by_tokens(text, 256, 32)
    total_tokens = token_count(text)
    ok("Full PDF -> chunk pipeline", f"{total_tokens} tokens -> {len(chunks)} chunks")

    try:
        extract_pdf_text("nonexistent_file.pdf")
        fail("Missing file handling", "Expected RuntimeError")
    except RuntimeError:
        ok("Missing file handling", "raises RuntimeError")

    return True


def _check_ollama() -> tuple[bool, object | None]:
    try:
        import ollama

        models = ollama.list()
        ok("Ollama connection", "running")
        return True, models
    except Exception as exc:
        skip("Ollama connection", f"not running - {exc}")
        return False, None


def test_bm25_pipeline() -> bool:
    section("BM25-Only Pipeline")

    ollama_ok, models = _check_ollama()
    if not ollama_ok:
        skip("BM25-only pipeline", "skipped (Ollama unavailable)")
        return False

    llm_model = os.getenv("COMPLETION_MODEL", "llama3.2")
    model_names = [model.model.split(":")[0] for model in models.models]
    if llm_model not in model_names:
        skip("Completion model", f"{llm_model} not pulled. Run: ollama pull {llm_model}")
        return False
    ok("Completion model", f"{llm_model} available")

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from rusty_rag.rag import ingest, query

    pdf_path = str(SAMPLE_PDF)

    try:
        start = time.time()
        ingest(pdf_path, retrieval_mode="bm25")
        ok("BM25 ingest", f"{time.time() - start:.1f}s")
    except Exception as exc:
        fail("BM25 ingest", str(exc))
        return False

    try:
        start = time.time()
        response = query("What datasets were used?", retrieval_mode="bm25")
        elapsed = time.time() - start
        lowered = response.lower()
        expected = ["mmlu", "humaneval", "triviaqa"]
        found = [term for term in expected if term in lowered]
        if found:
            ok("BM25 query", f"{elapsed:.1f}s, found: {', '.join(found)}")
        else:
            fail("BM25 query", f"Response missing expected terms: {expected}")
            print(f"    Response: {response[:200]}...")
    except Exception as exc:
        fail("BM25 query", str(exc))
        return False

    return True


def test_hybrid_pipeline() -> bool:
    section("Hybrid Pipeline")

    try:
        from rusty_rag.db import create_client

        client = create_client()
        client.get_collections()
        ok("Qdrant connection", "connected to localhost:6333")
    except Exception as exc:
        skip("Qdrant connection", f"not running - {exc}")
        skip("Hybrid pipeline", "skipped (Qdrant unavailable)")
        return False

    ollama_ok, models = _check_ollama()
    if not ollama_ok:
        skip("Hybrid pipeline", "skipped (Ollama unavailable)")
        return False

    emb_model = os.getenv("EMBEDDING_MODEL", "all-minilm")
    llm_model = os.getenv("COMPLETION_MODEL", "llama3.2")
    model_names = [model.model.split(":")[0] for model in models.models]

    if emb_model not in model_names:
        skip("Embedding model", f"{emb_model} not pulled. Run: ollama pull {emb_model}")
        return False
    ok("Embedding model", f"{emb_model} available")

    if llm_model not in model_names:
        skip("Completion model", f"{llm_model} not pulled. Run: ollama pull {llm_model}")
        return False
    ok("Completion model", f"{llm_model} available")

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from rusty_rag.rag import ingest, query

    pdf_path = str(SAMPLE_PDF)

    try:
        start = time.time()
        ingest(pdf_path, retrieval_mode="hybrid")
        ok("Hybrid ingest", f"{time.time() - start:.1f}s")
    except Exception as exc:
        fail("Hybrid ingest", str(exc))
        return False

    test_queries = [
        ("What datasets were used?", ["mmlu", "humaneval", "triviaqa"]),
        ("What retrieval modes are supported?", ["bm25", "vector", "hybrid"]),
        ("How much memory was used?", ["340", "mb"]),
    ]

    for question, expected_terms in test_queries:
        try:
            start = time.time()
            response = query(question, retrieval_mode="hybrid")
            elapsed = time.time() - start
            lowered = response.lower()
            found = [term for term in expected_terms if term in lowered]
            if found:
                ok(f'Hybrid query: "{question[:40]}"', f"{elapsed:.1f}s, found: {', '.join(found)}")
            else:
                fail(f'Hybrid query: "{question[:40]}"', f"Response missing: {expected_terms}")
                print(f"    Response: {response[:200]}...")
        except Exception as exc:
            fail(f'Hybrid query: "{question[:40]}"', str(exc))

    return True


def main() -> None:
    print(f"\n{BOLD}{'=' * 50}")
    print("  RustyRAG - Full Test Suite")
    print(f"{'=' * 50}{RESET}")

    core_ok = test_rust_core()
    pdf_ok = test_pdf_extraction()

    if core_ok and pdf_ok:
        test_bm25_pipeline()
        test_hybrid_pipeline()

    section("Results")
    total = passed + failed + skipped
    print(f"  {GREEN}{passed} passed{RESET}", end="")
    if failed:
        print(f"  |  {RED}{failed} failed{RESET}", end="")
    if skipped:
        print(f"  |  {YELLOW}{skipped} skipped{RESET}", end="")
    print(f"  |  {total} total\n")

    if failed:
        print(f"  {RED}{BOLD}Some tests failed.{RESET}\n")
        sys.exit(1)

    if skipped:
        print("  Some tests were skipped because Ollama or Qdrant is unavailable.\n")
        return

    print(f"  {GREEN}{BOLD}All tests passed.{RESET}\n")


if __name__ == "__main__":
    main()
