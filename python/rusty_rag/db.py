"""Qdrant vector database operations."""

import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

VECTOR_SIZE = 384  # Dimension for all-minilm embeddings


def create_client(url: str | None = None) -> QdrantClient:
    """Create a Qdrant client connected to the configured URL."""
    url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
    return QdrantClient(url=url)


def get_collection_name() -> str:
    """Get the collection name from environment or use default."""
    return os.getenv("COLLECTION_NAME", "documents")


def init_collection(client: QdrantClient, name: str | None = None) -> None:
    """Initialize the documents collection in Qdrant.

    If the collection already exists, this is a no-op.
    """
    name = name or get_collection_name()
    collections = [c.name for c in client.get_collections().collections]

    if name in collections:
        return

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


def upsert_chunks(
    client: QdrantClient,
    chunks: list[str],
    vectors: list[list[float]],
    collection: str | None = None,
) -> None:
    """Upsert text chunks with their embedding vectors into Qdrant."""
    collection = collection or get_collection_name()

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": chunk},
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    client.upsert(collection_name=collection, points=points)


def search(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int = 3,
    min_score: float = 0.3,
    collection: str | None = None,
) -> list[tuple[str, float]]:
    """Search for the most similar chunks to the query vector.

    Returns (text, score) pairs filtered by minimum relevance score.
    """
    collection = collection or get_collection_name()

    results = client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        score_threshold=min_score,
    )

    return [(point.payload["text"], point.score) for point in results]
