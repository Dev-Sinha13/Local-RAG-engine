use anyhow::{Context, Result};
use std::path::Path;

/// Extracts all text content from a PDF file at the given path.
///
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

    let bytes = std::fs::read(file_path)
        .with_context(|| format!("Failed to read file: {}", path))?;

    let text = pdf_extract::extract_text_from_mem(&bytes)
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
