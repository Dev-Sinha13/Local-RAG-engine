use anyhow::{Context, Result};
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, PointStruct, SearchPointsBuilder,
    UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::{Payload, Qdrant};
use serde_json::json;
use uuid::Uuid;

/// The vector dimension for the all-minilm model (384-dimensional embeddings).
const VECTOR_SIZE: u64 = 384;

/// Creates a Qdrant client connected to the configured URL.
pub fn create_client() -> Result<Qdrant> {
    let url = std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".to_string());
    let client = Qdrant::from_url(&url)
        .build()
        .context("Failed to connect to Qdrant")?;
    Ok(client)
}

/// Gets the collection name from the environment or uses the default.
fn collection_name() -> String {
    std::env::var("COLLECTION_NAME").unwrap_or_else(|_| "documents".to_string())
}

/// Initializes the "documents" collection in Qdrant.
///
/// If the collection already exists, this is a no-op.
pub async fn init_collection(client: &Qdrant) -> Result<()> {
    let name = collection_name();

    // Check if collection already exists
    let collections = client
        .list_collections()
        .await
        .context("Failed to list Qdrant collections")?;

    let exists = collections
        .collections
        .iter()
        .any(|c| c.name == name);

    if exists {
        println!("  Collection '{}' already exists.", name);
        return Ok(());
    }

    client
        .create_collection(
            CreateCollectionBuilder::new(&name)
                .vectors_config(VectorParamsBuilder::new(VECTOR_SIZE, Distance::Cosine)),
        )
        .await
        .with_context(|| format!("Failed to create collection '{}'", name))?;

    println!("  Created collection '{}'.", name);
    Ok(())
}

/// Upserts text chunks with their corresponding embedding vectors into Qdrant.
pub async fn upsert_chunks(
    client: &Qdrant,
    chunks: &[String],
    vectors: &[Vec<f32>],
) -> Result<()> {
    let name = collection_name();

    let points: Vec<PointStruct> = chunks
        .iter()
        .zip(vectors.iter())
        .map(|(text, vector)| {
            let id = Uuid::new_v4().to_string();
            let payload = json!({
                "text": text,
            });

            PointStruct::new(
                id,
                vector.clone(),
                Payload::try_from(payload).expect("Failed to convert payload"),
            )
        })
        .collect();

    client
        .upsert_points(UpsertPointsBuilder::new(&name, points).wait(true))
        .await
        .context("Failed to upsert points into Qdrant")?;

    Ok(())
}

/// Searches the collection for the most similar vectors to the query.
///
/// Returns the top `top_k` matches as (text, score) pairs, filtered by minimum score.
pub async fn search(
    client: &Qdrant,
    query_vector: Vec<f32>,
    top_k: u64,
    min_score: f32,
) -> Result<Vec<(String, f32)>> {
    let name = collection_name();

    let results = client
        .search_points(
            SearchPointsBuilder::new(&name, query_vector, top_k)
                .with_payload(true)
                .score_threshold(min_score),
        )
        .await
        .context("Failed to search Qdrant")?;

    let matches: Vec<(String, f32)> = results
        .result
        .iter()
        .filter_map(|point| {
            let text = point
                .payload
                .get("text")
                .and_then(|v| {
                    // Extract the string value from the Qdrant Value type
                    if let Some(qdrant_client::qdrant::value::Kind::StringValue(s)) = &v.kind {
                        Some(s.clone())
                    } else {
                        None
                    }
                })?;
            Some((text, point.score))
        })
        .collect();

    Ok(matches)
}
