use rayon::prelude::*;

/// Splits text into overlapping chunks using a sliding window algorithm.
///
/// - `chunk_size`: maximum number of characters per chunk
/// - `overlap`: number of characters shared between adjacent chunks
///
/// Returns a `Vec<String>` where each element is one chunk.
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![];
    }

    if chunk_size == 0 {
        return vec![];
    }

    // If the text is shorter than or equal to chunk_size, return it as a single chunk
    if text.len() <= chunk_size {
        return vec![text.to_string()];
    }

    let step = if overlap >= chunk_size {
        1 // Prevent infinite loop if overlap >= chunk_size
    } else {
        chunk_size - overlap
    };

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < text.len() {
        let end = (start + chunk_size).min(text.len());
        chunks.push(text[start..end].to_string());

        if end == text.len() {
            break;
        }

        start += step;
    }

    chunks
}

/// Parallelized version of `chunk_text` using Rayon's work-stealing iterator.
///
/// Pre-computes chunk boundaries sequentially, then extracts all chunks in
/// parallel across available CPU cores. This provides significant speedup
/// when processing large documents with many chunks.
///
/// - `chunk_size`: maximum number of characters per chunk
/// - `overlap`: number of characters shared between adjacent chunks
///
/// Returns a `Vec<String>` where each element is one chunk, in the same
/// order as the sequential version.
pub fn chunk_text_parallel(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() || chunk_size == 0 {
        return vec![];
    }

    if text.len() <= chunk_size {
        return vec![text.to_string()];
    }

    let step = if overlap >= chunk_size {
        1
    } else {
        chunk_size - overlap
    };

    // Pre-compute chunk boundaries (lightweight, sequential)
    let mut boundaries = Vec::new();
    let mut start = 0;

    while start < text.len() {
        let end = (start + chunk_size).min(text.len());
        boundaries.push((start, end));

        if end == text.len() {
            break;
        }

        start += step;
    }

    // Extract chunks in parallel using Rayon's work-stealing scheduler
    boundaries
        .par_iter()
        .map(|&(start, end)| text[start..end].to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_chunking() {
        let text = "a".repeat(2500);
        let chunks = chunk_text(&text, 1000, 100);

        // With 2500 chars, chunk_size=1000, step=900:
        // Chunk 0: [0..1000], Chunk 1: [900..1900], Chunk 2: [1800..2500]
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), 1000);
        assert_eq!(chunks[1].len(), 1000);
        assert_eq!(chunks[2].len(), 700); // Remainder
    }

    #[test]
    fn test_overlap() {
        let text: String = (0..2000).map(|i| char::from(b'A' + (i % 26) as u8)).collect();
        let chunks = chunk_text(&text, 1000, 100);

        // The last 100 characters of chunk 0 should equal the first 100 characters of chunk 1
        let tail_of_first = &chunks[0][900..1000];
        let head_of_second = &chunks[1][0..100];
        assert_eq!(tail_of_first, head_of_second, "Overlap region must match");
    }

    #[test]
    fn test_no_content_lost() {
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(50);
        let chunks = chunk_text(&text, 1000, 100);

        // Reconstruct original text from non-overlapping parts
        let mut reconstructed = String::new();
        for (i, chunk) in chunks.iter().enumerate() {
            if i == 0 {
                reconstructed.push_str(chunk);
            } else {
                // Skip the overlap portion (first `overlap` chars are duplicates)
                let new_content = &chunk[100.min(chunk.len())..];
                reconstructed.push_str(new_content);
            }
        }

        assert_eq!(
            reconstructed, text,
            "Reconstructed text must match original"
        );
    }

    #[test]
    fn test_small_text() {
        let text = "Hello, world!";
        let chunks = chunk_text(text, 1000, 100);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_empty_text() {
        let chunks = chunk_text("", 1000, 100);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_exact_chunk_size() {
        let text = "x".repeat(1000);
        let chunks = chunk_text(&text, 1000, 100);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 1000);
    }

    // --- Parallel chunking tests ---

    #[test]
    fn test_parallel_matches_sequential() {
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
        let sequential = chunk_text(&text, 1000, 100);
        let parallel = chunk_text_parallel(&text, 1000, 100);
        assert_eq!(sequential, parallel, "Parallel output must match sequential");
    }

    #[test]
    fn test_parallel_empty_text() {
        let chunks = chunk_text_parallel("", 1000, 100);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_parallel_small_text() {
        let text = "Hello, world!";
        let chunks = chunk_text_parallel(text, 1000, 100);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_parallel_large_document() {
        let text = "x".repeat(100_000);
        let sequential = chunk_text(&text, 500, 50);
        let parallel = chunk_text_parallel(&text, 500, 50);
        assert_eq!(sequential.len(), parallel.len());
        assert_eq!(sequential, parallel);
    }
}

