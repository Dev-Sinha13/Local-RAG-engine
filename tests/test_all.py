"""
RustyRAG Full Test Script
=========================

Generates a sample PDF and tests the entire pipeline:
  1. Rust core functions (tokenizer, chunker, BM25) — no services needed
  2. PDF extraction — no services needed
  3. Full pipeline (ingest + query) — needs Ollama + Qdrant running

Usage:
    pip install fpdf2           # one-time, for generating the sample PDF
    python tests/test_all.py
"""

import sys
import os
import time
import textwrap
from pathlib import Path

# ── Color helpers for terminal output ──

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

passed = 0
failed = 0
skipped = 0


def ok(name: str, detail: str = ""):
    global passed
    passed += 1
    print(f"  {GREEN}✓{RESET} {name}" + (f"  {DIM}{detail}{RESET}" if detail else ""))


def fail(name: str, error: str):
    global failed
    failed += 1
    print(f"  {RED}✗{RESET} {name}")
    print(f"    {RED}{error}{RESET}")


def skip(name: str, reason: str):
    global skipped
    skipped += 1
    print(f"  {YELLOW}○{RESET} {name}  {DIM}({reason}){RESET}")


def section(title: str):
    print(f"\n{BOLD}{CYAN}{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}{RESET}\n")


# ═══════════════════════════════════════════════════
#  STEP 1: Generate sample PDF
# ═══════════════════════════════════════════════════

SAMPLE_DIR = Path(__file__).parent / "sample_data"
SAMPLE_PDF = SAMPLE_DIR / "test_paper.pdf"

SAMPLE_TEXT = textwrap.dedent("""\
    Advances in Retrieval-Augmented Generation for Local Document Processing

    Abstract

    This paper presents a novel approach to retrieval-augmented generation (RAG) that operates
    entirely on local hardware without requiring cloud-based API calls. Our system, RustyRAG,
    combines a Rust-based performance layer for document parsing and text chunking with a
    Python orchestration layer for embedding generation and language model inference.

    Introduction

    Large language models have demonstrated remarkable capabilities in natural language
    understanding and generation. However, their use in document question-answering systems
    typically requires sending sensitive data to cloud-based APIs, raising privacy concerns.
    We propose a fully local alternative that maintains competitive performance while ensuring
    complete data privacy.

    The key innovation of our approach is the hybrid architecture that leverages Rust for
    CPU-bound operations such as PDF parsing with memory-mapped I/O and parallel text chunking
    using Rayon's work-stealing scheduler, while delegating network-bound operations like
    embedding generation and LLM inference to Python.

    Methodology

    Our system implements the following pipeline:

    Document Ingestion: PDF files are processed using memory-mapped file I/O to handle
    documents larger than available RAM. The extracted text is then split into overlapping
    chunks using a token-aware sliding window algorithm. Each chunk is converted into a
    384-dimensional embedding vector using a local embedding model.

    Hybrid Retrieval: At query time, we employ both vector similarity search via Qdrant
    and BM25 keyword matching via a custom Rust implementation. Results from both methods
    are merged using Reciprocal Rank Fusion to produce a final ranked list of relevant chunks.

    Response Generation: The top-ranked chunks are assembled into a context window and
    passed to a local language model along with the user's question. The model is instructed
    to answer solely based on the provided context.

    Experiments

    We evaluated our system on three benchmark datasets:

    1. MMLU - a multi-task benchmark covering 57 academic subjects
    2. HumanEval - a benchmark for evaluating code generation capabilities
    3. TriviaQA - an open-domain question answering benchmark

    Our hybrid retrieval approach (vector + BM25) outperformed pure vector search by 12.3%
    on precision@3 and 8.7% on recall@10 across all three benchmarks.

    The token-aware chunking strategy reduced context window waste by 23% compared to
    character-based chunking, as chunks align with actual model token boundaries.

    Results

    Performance metrics on TriviaQA:
    - Pure vector search: 71.2% accuracy
    - Pure BM25 search: 63.8% accuracy
    - Hybrid (RRF): 79.4% accuracy
    - Improvement over best single method: +8.2%

    Memory usage for a 500MB PDF corpus:
    - Standard file I/O: 2.1 GB peak RAM
    - Memory-mapped I/O: 340 MB peak RAM (83.8% reduction)

    Chunking throughput:
    - Sequential: 12.4 MB/s
    - Parallel (Rayon, 8 cores): 89.2 MB/s (7.2x speedup)

    Conclusion

    We have demonstrated that a fully local RAG system can achieve competitive performance
    with cloud-based alternatives while providing complete data privacy. The hybrid Rust/Python
    architecture provides an effective balance between performance and development velocity.

    Future work includes implementing semantic sentence-boundary detection for smarter chunking,
    adding support for additional file formats, and exploring cross-encoder re-ranking for
    improved retrieval precision.

    References

    [1] Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. 2020.
    [2] Robertson et al. The Probabilistic Relevance Framework: BM25 and Beyond. 2009.
    [3] Cormack et al. Reciprocal Rank Fusion outperforms Condorcet and individual
        Rank Learning Methods. 2009.
""")


def generate_sample_pdf():
    """Generate a sample PDF for testing."""
    try:
        from fpdf import FPDF
    except ImportError:
        print(f"\n  {YELLOW}fpdf2 not installed. Installing...{RESET}")
        os.system(f"{sys.executable} -m pip install fpdf2 -q")
        from fpdf import FPDF

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    for line in SAMPLE_TEXT.split("\n"):
        if line.strip() and not line.startswith(" ") and len(line.strip()) < 80:
            # Looks like a heading
            pdf.set_font("Helvetica", "B", size=13)
            pdf.cell(0, 10, line.strip(), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=11)
        else:
            pdf.multi_cell(0, 6, line)

    pdf.output(str(SAMPLE_PDF))
    return SAMPLE_PDF


# ═══════════════════════════════════════════════════
#  STEP 2: Test Rust core (no services needed)
# ═══════════════════════════════════════════════════

def test_rust_core():
    section("Rust Core Functions (no services needed)")

    try:
        from rusty_rag import (
            tokenize,
            token_count,
            chunk_text,
            chunk_text_parallel,
            chunk_by_tokens,
            BM25Index,
        )
    except ImportError:
        fail("Import rusty_rag", "Module not found. Run: maturin develop --release")
        return False

    ok("Import rusty_rag")

    # ── Tokenizer ──
    tokens = tokenize("Hello, World! This is a RAG test.")
    assert tokens == ["hello", "world", "this", "is", "a", "rag", "test"], f"Got: {tokens}"
    ok("tokenize()", f"{len(tokens)} tokens")

    count = token_count("The quick brown fox jumps")
    assert count == 5, f"Expected 5, got {count}"
    ok("token_count()", f"{count} tokens")

    # ── Character chunking ──
    text = "word " * 1000  # 5000 chars
    chunks = chunk_text(text, 500, 50)
    assert len(chunks) > 1
    ok("chunk_text()", f"{len(chunks)} chunks from 5000 chars")

    parallel = chunk_text_parallel(text, 500, 50)
    assert chunks == parallel
    ok("chunk_text_parallel()", "matches sequential output")

    # ── Token-aware chunking ──
    text2 = "The quick brown fox. " * 100  # ~500 words
    token_chunks = chunk_by_tokens(text2, 50, 10)
    assert len(token_chunks) > 1
    # Verify each chunk has approximately the right number of words
    for i, chunk in enumerate(token_chunks[:-1]):  # last chunk may be shorter
        wc = token_count(chunk)
        assert wc <= 50, f"Chunk {i} has {wc} tokens, expected ≤50"
    ok("chunk_by_tokens()", f"{len(token_chunks)} token-aware chunks")

    # ── BM25 ──
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
    top_idx = results[0][0]
    assert top_idx in [0, 2, 4], f"Expected ML doc, got index {top_idx}"
    ok("BM25Index.search()", f"top result: doc[{top_idx}] (score={results[0][1]:.3f})")

    # BM25 should NOT match cooking/gardening
    result_indices = {idx for idx, _ in results}
    assert 1 not in result_indices, "Cooking doc should not match ML query"
    assert 3 not in result_indices, "Gardening doc should not match ML query"
    ok("BM25 relevance", "irrelevant docs correctly excluded")

    # Empty query
    empty = index.search("xyznonexistent", 5)
    assert len(empty) == 0
    ok("BM25 no-match", "returns empty for unknown terms")

    print(f"\n  {DIM}repr: {repr(index)}{RESET}")
    return True


# ═══════════════════════════════════════════════════
#  STEP 3: Test PDF extraction (no services needed)
# ═══════════════════════════════════════════════════

def test_pdf_extraction():
    section("PDF Extraction (no services needed)")

    try:
        from rusty_rag import extract_pdf_text
    except ImportError:
        fail("Import", "rusty_rag not installed")
        return False

    # Generate sample PDF
    pdf_path = generate_sample_pdf()
    ok("Generate sample PDF", str(pdf_path))

    # Extract text
    text = extract_pdf_text(str(pdf_path))
    assert len(text) > 100, f"Extracted too little text: {len(text)} chars"
    ok("extract_pdf_text()", f"{len(text):,} chars extracted")

    # Verify key content is present
    text_lower = text.lower()
    assert "retrieval" in text_lower, "Missing expected content: 'retrieval'"
    assert "rag" in text_lower, "Missing expected content: 'rag'"
    assert "bm25" in text_lower, "Missing expected content: 'bm25'"
    ok("Content verification", "key terms found in extracted text")

    # Test chunk pipeline on extracted text
    from rusty_rag import chunk_by_tokens, token_count
    chunks = chunk_by_tokens(text, 256, 32)
    total_tokens = token_count(text)
    ok("Full PDF → chunk pipeline", f"{total_tokens} tokens → {len(chunks)} chunks")

    # Test error handling
    try:
        extract_pdf_text("nonexistent_file.pdf")
        fail("Error handling", "Should have raised for missing file")
    except RuntimeError:
        ok("Error handling", "missing file raises RuntimeError")

    try:
        extract_pdf_text("README.md")
        fail("Error handling", "Should have raised for non-PDF")
    except RuntimeError:
        ok("Error handling", "non-PDF file raises RuntimeError")

    return True


# ═══════════════════════════════════════════════════
#  STEP 4: Test full pipeline (needs Ollama + Qdrant)
# ═══════════════════════════════════════════════════

def test_full_pipeline():
    section("Full Pipeline (requires Ollama + Qdrant)")

    # Check if Qdrant is running
    try:
        from rusty_rag.db import create_client
        client = create_client()
        client.get_collections()
        ok("Qdrant connection", "connected to localhost:6333")
    except Exception as e:
        skip("Qdrant connection", f"not running — {e}")
        skip("Full pipeline", "skipped (Qdrant unavailable)")
        return False

    # Check if Ollama is running
    try:
        import ollama
        ollama.list()
        ok("Ollama connection", "running")
    except Exception as e:
        skip("Ollama connection", f"not running — {e}")
        skip("Full pipeline", "skipped (Ollama unavailable)")
        return False

    # Check if models are pulled
    try:
        models = ollama.list()
        model_names = [m.model.split(":")[0] for m in models.models]
        emb_model = os.getenv("EMBEDDING_MODEL", "all-minilm")
        llm_model = os.getenv("COMPLETION_MODEL", "llama3.2")

        if emb_model not in model_names:
            skip("Embedding model", f"{emb_model} not pulled. Run: ollama pull {emb_model}")
            return False
        ok("Embedding model", f"{emb_model} available")

        if llm_model not in model_names:
            skip("Completion model", f"{llm_model} not pulled. Run: ollama pull {llm_model}")
            return False
        ok("Completion model", f"{llm_model} available")
    except Exception as e:
        skip("Model check", str(e))
        return False

    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Run ingestion
    print()
    print(f"  {CYAN}Running ingestion pipeline...{RESET}")
    print()

    from rusty_rag.rag import ingest

    pdf_path = str(SAMPLE_DIR / "test_paper.pdf")
    start = time.time()

    try:
        ingest(pdf_path)
        elapsed = time.time() - start
        ok("Ingest pipeline", f"completed in {elapsed:.1f}s")
    except Exception as e:
        fail("Ingest pipeline", str(e))
        return False

    # Run queries
    print()
    print(f"  {CYAN}Running query pipeline...{RESET}")
    print()

    from rusty_rag.rag import query

    test_queries = [
        ("What datasets were used?", ["mmlu", "humaneval", "triviaqa"]),
        ("What is the hybrid retrieval approach?", ["vector", "bm25", "fusion"]),
        ("How much did memory-mapped I/O reduce RAM usage?", ["83", "340", "memory"]),
    ]

    for question, expected_terms in test_queries:
        try:
            start = time.time()
            response = query(question)
            elapsed = time.time() - start

            response_lower = response.lower()
            found = [t for t in expected_terms if t in response_lower]

            if found:
                ok(
                    f'Query: "{question[:50]}"',
                    f"{elapsed:.1f}s, found: {', '.join(found)}",
                )
            else:
                fail(
                    f'Query: "{question[:50]}"',
                    f"Response didn't contain expected terms: {expected_terms}",
                )
                print(f"    {DIM}Response: {response[:200]}...{RESET}")
        except Exception as e:
            fail(f'Query: "{question[:50]}"', str(e))

    return True


# ═══════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════

def main():
    print(f"\n{BOLD}{'═' * 50}")
    print("  RustyRAG — Full Test Suite")
    print(f"{'═' * 50}{RESET}")

    # Always run these (no external deps needed)
    core_ok = test_rust_core()
    pdf_ok = test_pdf_extraction()

    # Only run if services are available
    if core_ok and pdf_ok:
        test_full_pipeline()

    # Summary
    section("Results")
    total = passed + failed + skipped
    print(f"  {GREEN}{passed} passed{RESET}", end="")
    if failed:
        print(f"  ·  {RED}{failed} failed{RESET}", end="")
    if skipped:
        print(f"  ·  {YELLOW}{skipped} skipped{RESET}", end="")
    print(f"  ·  {total} total\n")

    if failed:
        print(f"  {RED}{BOLD}Some tests failed!{RESET}\n")
        sys.exit(1)
    elif skipped:
        print(
            f"  {YELLOW}Some tests were skipped because Ollama/Qdrant aren't running.{RESET}"
        )
        print(f"  {DIM}To run all tests:{RESET}")
        print(f"  {DIM}  docker-compose up -d{RESET}")
        print(f"  {DIM}  ollama pull all-minilm && ollama pull llama3.2{RESET}")
        print(f"  {DIM}  python tests/test_all.py{RESET}\n")
    else:
        print(f"  {GREEN}{BOLD}All tests passed!{RESET}\n")


if __name__ == "__main__":
    main()
