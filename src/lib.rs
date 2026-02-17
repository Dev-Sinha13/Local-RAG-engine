use pyo3::prelude::*;

mod chunker;
mod pdf;

/// Extract all text from a PDF file using memory-mapped I/O.
///
/// Returns the full text as a single string with normalized whitespace.
/// Uses mmap under the hood so it can handle files larger than available RAM.
#[pyfunction]
fn extract_pdf_text(path: &str) -> PyResult<String> {
    pdf::extract_text(path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:#}", e))
    })
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

/// RustyRAG Core — High-performance Rust backend for PDF extraction and text chunking.
#[pymodule]
fn rusty_rag_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_pdf_text, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_text_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_text, m)?)?;
    Ok(())
}
