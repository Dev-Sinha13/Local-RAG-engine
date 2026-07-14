# RustyRAG

Local-first PDF question answering with a hybrid Rust/Python stack.

RustyRAG now supports three first-class retrieval modes:

- `bm25`: fully vectorless local retrieval backed by a Rust BM25 index
- `hybrid`: BM25 plus vector retrieval fused with Reciprocal Rank Fusion
- `vector`: vector search through Qdrant or a persistent local turbovec index

The result is a project that can run as a simple, dependency-light keyword RAG system or as a richer semantic RAG pipeline, without forking the codebase.

## Why This Is Useful

- `bm25` mode removes the embedding and vector database requirements entirely.
- `hybrid` mode remains the best default when you want both keyword precision and semantic recall.
- `vector` mode is still available when you only want embedding-based retrieval.

For local workflows, that means you can choose between:

- minimum setup and fast ingest
- stronger semantic retrieval
- a hybrid path that combines both

## Retrieval Modes

| Mode | Embeddings | Vector backend | Local chunk store | Best for |
| --- | --- | --- | --- | --- |
| `bm25` | No | No | Yes | Fast, simple, exact-term retrieval |
| `hybrid` | Yes | Qdrant or turbovec | Yes | Best overall retrieval quality |
| `vector` | Yes | Qdrant or turbovec | Yes | Semantic retrieval experiments |

Even in `vector` mode, RustyRAG persists a local chunk store so you can switch retrieval strategies later without re-chunking the source documents.

## Architecture

### Ingestion

1. Rust extracts PDF text.
2. Rust performs token-aware chunking.
3. RustyRAG writes chunk data to a local corpus store for BM25 retrieval.
4. In `hybrid` and `vector` modes, RustyRAG also generates embeddings and stores them in the configured vector backend.

### Query

1. RustyRAG resolves the configured retrieval mode.
2. It runs BM25, vector, or both retrieval paths.
3. In `hybrid` mode, it merges ranked results with Reciprocal Rank Fusion.
4. The retrieved chunks become the LLM context window for answer generation.

### Language Split

- Rust handles PDF extraction, token-aware chunking, tokenization, and BM25 indexing.
- Python handles orchestration, storage, CLI behavior, and LLM/vector integrations.

## Project Structure

```text
src/
  bm25.rs                 Rust BM25 retrieval
  chunker.rs              Rust chunking
  lib.rs                  PyO3 exports
  pdf.rs                  Rust PDF extraction
  tokenizer.rs            Rust tokenization

python/rusty_rag/
  cli.py                  CLI entry point
  db.py                   Qdrant operations
  embeddings.py           Ollama embeddings
  llm.py                  Ollama chat completions
  rag.py                  Retrieval-mode orchestration
  store.py                Local chunk store and BM25 cache

tests/
  test_all.py             End-to-end smoke script
  test_cli.py             CLI unit tests
  test_rag_pipeline.py    Retrieval-mode unit tests
  test_store.py           Local store and BM25 cache tests
```

## Getting Started

### Prerequisites

- Rust stable
- Python 3.9+
- Ollama for local model inference
- Docker only if you want Qdrant-backed `hybrid` or `vector` mode
- Maturin for building the Rust extension

### Install

```bash
git clone https://github.com/Dev-Sinha13/Local-RAG-engine.git
cd Local-RAG-engine
cp .env.example .env
pip install maturin
maturin develop --release
```

### Minimal BM25-Only Setup

This is the vectorless path.

```bash
ollama pull llama3.2
rusty-rag ingest path/to/document.pdf --retrieval-mode bm25
rusty-rag query "What does the paper say about benchmark results?" --retrieval-mode bm25
```

You do not need:

- an embedding model
- Qdrant
- Docker

### Hybrid Setup

```bash
docker-compose up -d
ollama pull all-minilm
ollama pull llama3.2
rusty-rag ingest path/to/document.pdf --retrieval-mode hybrid
rusty-rag query "What datasets were used?" --retrieval-mode hybrid
```

### Vector-Only Setup

```bash
docker-compose up -d
ollama pull all-minilm
ollama pull llama3.2
rusty-rag ingest path/to/document.pdf --retrieval-mode vector
rusty-rag query "Summarize the methodology section" --retrieval-mode vector
```

## Usage

### Ingest

```bash
rusty-rag ingest paper.pdf
rusty-rag ingest paper.pdf --retrieval-mode bm25
rusty-rag ingest paper.pdf --retrieval-mode hybrid
rusty-rag ingest paper.pdf --retrieval-mode vector
```

If `--retrieval-mode` is omitted, RustyRAG uses the `RETRIEVAL_MODE` environment variable or falls back to `hybrid`.

### Query

```bash
rusty-rag query "What is the main contribution?"
rusty-rag query "What is the main contribution?" --retrieval-mode bm25
rusty-rag query "What is the main contribution?" --retrieval-mode hybrid
rusty-rag query "What is the main contribution?" --retrieval-mode vector
```

## Configuration

All configuration lives in `.env`.

```env
# Retrieval mode
RETRIEVAL_MODE=hybrid
RETRIEVAL_TOP_K=3

# Local cache directory for chunk store
RUSTY_RAG_CACHE_DIR=

# Qdrant
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=documents
VECTOR_MIN_SCORE=0.2

# Vector backend: qdrant (default) or turbovec (persistent local index)
VECTOR_BACKEND=qdrant
# TURBOVEC_BIT_WIDTH=4

# Ollama
EMBEDDING_MODEL=all-minilm
COMPLETION_MODEL=llama3.2

# Chunking
CHUNK_MAX_TOKENS=256
CHUNK_OVERLAP_TOKENS=32
```

### Local Chunk Store

RustyRAG persists BM25-ready chunks under:

```text
~/.rusty_rag/chunks.json
```

Or, if `RUSTY_RAG_CACHE_DIR` is set:

```text
$RUSTY_RAG_CACHE_DIR/chunks.json
```

The store is structured per source document and replaces prior entries when the same document path is re-ingested.

### Local turbovec Vector Index

Set `VECTOR_BACKEND=turbovec` to use a persistent compressed local index instead of Qdrant. The index is written as `vectors.tvim` beside `chunks.json`, so vector and hybrid retrieval can run without Docker or a Qdrant service.

```env
VECTOR_BACKEND=turbovec
TURBOVEC_BIT_WIDTH=4
```

Re-ingesting a document replaces its prior vectors. If you switch an existing Qdrant-backed corpus to turbovec, re-ingest its documents because RustyRAG deliberately does not retain embedding vectors in the chunk cache.

### Vector Backend Benchmark

Use the reproducible local benchmark to compare turbovec with exact float32 search and embedded Qdrant on a 384-dimensional workload:

```bash
python benchmarks/turbovec_benchmark.py --qdrant
```

It reports index-build time, single-query mean/p50/p95 latency, QPS, Recall@10 against exact float32 ranking, and turbovec index compression. Pass `--output results.json` to save a run.

## Testing

### Unit tests

The new unit suite covers:

- chunk store loading, migration, replacement, and BM25 cache reuse
- retrieval mode validation and routing
- BM25-only ingest/query behavior
- hybrid and vector pipeline behavior
- CLI argument plumbing for retrieval modes

Run:

```bash
python -m unittest tests.test_store tests.test_rag_pipeline tests.test_cli
```

### End-to-end smoke test

The existing integration script is still available:

```bash
python tests/test_all.py
```

That script exercises the Rust core directly and runs the full ingestion/query path when Ollama and Qdrant are available.

## Notes on Retrieval Quality

- `bm25` is strong for exact phrases, acronyms, identifiers, citations, and technical terms.
- `vector` is better for paraphrases and semantic similarity.
- `hybrid` is usually the strongest default because it combines both signals.

## What Changed

This version adds:

- a true vectorless BM25 retrieval path
- retrieval-mode aware ingest and query flows
- a structured local corpus store with migration from the old chunk cache format
- process-local BM25 index reuse
- dedicated unit tests for the new storage and orchestration behavior
- updated documentation for all supported retrieval modes

## License

MIT
