use pyo3::prelude::*;

mod bm25;
mod chunker;
mod pdf;
mod tokenizer;

/// Extract all text from a PDF file using memory-mapped I/O.
///
/// Returns the full text as a single string with normalized whitespace.
/// Uses mmap under the hood so it can handle files larger than available RAM.
#[pyfunction]
fn extract_pdf_text(path: &str) -> PyResult<String> {
    pdf::extract_text(path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:#}", e)))
}

/// Split text into overlapping chunks using a parallel sliding window algorithm.
///
/// Uses Rayon's work-stealing scheduler to extract chunks across all CPU cores.
/// Returns chunks in the same order as sequential processing.
#[pyfunction]
#[pyo3(signature = (text, chunk_size=1000, overlap=100))]
fn chunk_text_parallel(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    chunker::chunk_text_parallel(text, chunk_size, overlap)
}

/// Split text into overlapping chunks using a sequential sliding window algorithm.
///
/// Single-threaded version, useful for small texts or debugging.
#[pyfunction]
#[pyo3(signature = (text, chunk_size=1000, overlap=100))]
fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    chunker::chunk_text(text, chunk_size, overlap)
}

/// Token-aware text chunking with overlap.
///
/// Splits text into chunks where each chunk contains at most `max_tokens` words.
/// Preserves original formatting. Aligns with how LLMs tokenize text.
#[pyfunction]
#[pyo3(signature = (text, max_tokens=256, overlap_tokens=32))]
fn chunk_by_tokens(text: &str, max_tokens: usize, overlap_tokens: usize) -> Vec<String> {
    chunker::chunk_by_tokens(text, max_tokens, overlap_tokens)
}

/// Tokenize text into lowercase word tokens.
///
/// Splits on non-alphanumeric characters (preserving apostrophes).
#[pyfunction]
fn tokenize(text: &str) -> Vec<String> {
    tokenizer::tokenize(text)
}

/// Count the number of word tokens in text.
#[pyfunction]
fn token_count(text: &str) -> usize {
    tokenizer::token_count(text)
}

/// RustyRAG Core — High-performance Rust backend.
///
/// Exposes:
///   - extract_pdf_text: PDF parsing with memory-mapped I/O
///   - chunk_text / chunk_text_parallel: Character-based chunking
///   - chunk_by_tokens: Token-aware chunking
///   - tokenize / token_count: Word-level tokenization
///   - BM25Index: Keyword search index
#[pymodule]
fn rusty_rag_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_pdf_text, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_text_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_text, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_by_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(token_count, m)?)?;
    m.add_class::<bm25::BM25Index>()?;
    Ok(())
}
