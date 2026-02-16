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
}
