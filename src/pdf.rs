use anyhow::{Context, Result};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// Extracts all text content from a PDF file at the given path.
///
/// Uses memory-mapped file I/O to handle datasets larger than available RAM.
/// Returns the full text as a single `String` with normalized whitespace.
pub fn extract_text(path: &str) -> Result<String> {
    let file_path = Path::new(path);

    if !file_path.exists() {
        anyhow::bail!("File not found: {}", path);
    }

    if file_path
        .extension()
        .map_or(true, |ext| ext.to_ascii_lowercase() != "pdf")
    {
        anyhow::bail!("File is not a PDF: {}", path);
    }

    // Memory-mapped I/O: the OS pages data in/out as needed, enabling
    // processing of files larger than available RAM.
    let file = File::open(file_path)
        .with_context(|| format!("Failed to open file: {}", path))?;
    // SAFETY: The file is opened read-only and we do not modify it.
    // The mmap is dropped before the file handle, and no concurrent
    // writers are expected for PDF ingestion.
    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("Failed to memory-map file: {}", path))?;

    let text = pdf_extract::extract_text_from_mem(&mmap[..])
        .with_context(|| format!("Failed to extract text from PDF: {}", path))?;

    // Normalize whitespace: collapse multiple spaces/newlines
    let cleaned = text
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<&str>>()
        .join("\n");

    if cleaned.is_empty() {
        anyhow::bail!(
            "No text could be extracted from the PDF. It may be image-based or encrypted: {}",
            path
        );
    }

    Ok(cleaned)
}
