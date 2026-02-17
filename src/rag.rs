use anyhow::{Context, Result};

use crate::chunker;
use crate::db;
use crate::embeddings;
use crate::llm;
use crate::pdf;

/// Ingests a PDF document into the knowledge base.
///
/// Pipeline: Extract text → Chunk → Generate embeddings → Upsert to Qdrant.
pub async fn ingest_document(path: &str) -> Result<()> {
    println!("  Extracting text from: {}", path);
    let text = pdf::extract_text(path)?;
    println!("  Extracted {} characters.", text.len());

    let chunk_size: usize = std::env::var("CHUNK_SIZE")
        .unwrap_or_else(|_| "1000".to_string())
        .parse()
        .unwrap_or(1000);
    let chunk_overlap: usize = std::env::var("CHUNK_OVERLAP")
        .unwrap_or_else(|_| "100".to_string())
        .parse()
        .unwrap_or(100);

    println!("  Chunking text (size={}, overlap={}) [parallel]...", chunk_size, chunk_overlap);
    let chunks = chunker::chunk_text_parallel(&text, chunk_size, chunk_overlap);
    println!("  Created {} chunks.", chunks.len());

    println!("  Generating embeddings...");
    let vectors = embeddings::embed_texts(&chunks).await?;
    println!("  Generated {} embeddings.", vectors.len());

    println!("  Connecting to Qdrant...");
    let qdrant = db::create_client()?;
    db::init_collection(&qdrant).await?;

    println!("  Upserting chunks to Qdrant...");
    db::upsert_chunks(&qdrant, &chunks, &vectors)
        .await
        .context("Failed to upsert chunks")?;
    println!("  Successfully ingested {} chunks from '{}'.", chunks.len(), path);

    Ok(())
}

/// Queries the knowledge base and generates an AI response.
///
/// Pipeline: Embed query → Search Qdrant → Build context → Prompt LLM.
pub async fn query_document(question: &str) -> Result<String> {
    println!("  Searching knowledge base for: \"{}\"", question);

    // Generate embedding for the query
    let query_vector = embeddings::embed_query(question)
        .await
        .context("Failed to embed query")?;

    // Search Qdrant for relevant chunks
    let qdrant = db::create_client()?;
    let results = db::search(&qdrant, query_vector, 3, 0.3).await?;

    if results.is_empty() {
        return Ok(
            "I couldn't find any relevant information in the knowledge base. \
             Please make sure you've ingested documents first with `rusty_rag ingest <file>`."
                .to_string(),
        );
    }

    println!(
        "  Found {} relevant chunks (scores: {})",
        results.len(),
        results
            .iter()
            .map(|(_, score)| format!("{:.3}", score))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Build context from retrieved chunks
    let context = results
        .iter()
        .enumerate()
        .map(|(i, (text, score))| format!("[Chunk {} | Score: {:.3}]\n{}", i + 1, score, text))
        .collect::<Vec<_>>()
        .join("\n\n");

    // Ask the LLM with context
    println!("  Generating response...");
    let response = llm::ask_ai(question, &context).await?;

    Ok(response)
}
