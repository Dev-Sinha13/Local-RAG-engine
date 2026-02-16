use anyhow::Result;
use rig::client::Nothing;
use rig::client::EmbeddingsClient;
use rig::providers::ollama;
use rig::embeddings::EmbeddingModel;
use rig::embeddings::Embedding;

/// Creates an Ollama client with explicit type annotation for the HTTP client.
fn create_ollama_client() -> Result<ollama::Client<reqwest::Client>> {
    let client: ollama::Client<reqwest::Client> = ollama::Client::new(Nothing)?;
    Ok(client)
}

/// Generates embedding vectors for a batch of text chunks using Ollama's embedding model.
///
/// Returns a vector of embedding vectors (Vec<f32>).
pub async fn embed_texts(texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let client = create_ollama_client()?;
    let model_name =
        std::env::var("EMBEDDING_MODEL").unwrap_or_else(|_| "all-minilm".to_string());

    let emb_model = client.embedding_model(&model_name);

    let embeddings: Vec<Embedding> = emb_model.embed_texts(texts.to_vec()).await?;

    let vectors: Vec<Vec<f32>> = embeddings
        .into_iter()
        .map(|e| e.vec.iter().map(|&v| v as f32).collect())
        .collect();

    Ok(vectors)
}

/// Generates a single embedding vector for a query string.
pub async fn embed_query(query: &str) -> Result<Vec<f32>> {
    let client = create_ollama_client()?;
    let model_name =
        std::env::var("EMBEDDING_MODEL").unwrap_or_else(|_| "all-minilm".to_string());

    let emb_model = client.embedding_model(&model_name);

    let result = emb_model.embed_text(query).await?;

    Ok(result.vec.iter().map(|&v| v as f32).collect())
}
