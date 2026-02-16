"""
Vector database management using Qdrant (local).
"""
import logging
import numpy as np
from typing import List, Dict, Tuple, Any
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    QdrantClient = None
    VectorParams = None
    PointStruct = None
    Distance = None


class QdrantManager:
    """
    Manage Qdrant vector database.
    One collection per (chunking_strategy + embedding_model).
    """

    def __init__(self, db_path: str = "./qdrant_storage"):
        """Initialize Qdrant client with local storage."""
        if QdrantClient is None:
            raise ImportError("qdrant-client is required")

        self.db_path = db_path
        logger.info(f"Initializing Qdrant at {db_path}")
        self.client = QdrantClient(path=db_path)

    def create_collection(self, collection_name: str, vector_size: int) -> None:
        """
        Create a collection with cosine similarity.
        """
        try:
            # Check if collection exists
            existing_collections = self.client.get_collections().collections
            if any(c.name == collection_name for c in existing_collections):
                logger.info(f"Collection exists: {collection_name}, recreating...")
                self.client.delete_collection(collection_name)

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                ),
            )
            logger.info(f"✓ Created collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def add_vectors(self, collection_name: str,
                   embeddings: np.ndarray,
                   metadata_list: List[Dict[str, Any]]) -> None:
        """
        Add vectors and metadata to collection.

        Args:
            collection_name: Collection name
            embeddings: Array of shape (n, dimension)
            metadata_list: List of metadata dicts for each vector
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Embedding and metadata list length mismatch")

        points = []
        for i, (embedding, metadata) in enumerate(zip(embeddings, metadata_list)):
            point = PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload=metadata
            )
            points.append(point)

        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"Added {len(points)} vectors to {collection_name}")
        except Exception as e:
            logger.error(f"Error adding vectors: {e}")
            raise

    def search(self, collection_name: str, query_embedding: np.ndarray,
              limit: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors.

        Returns:
            List of (chunk_id, similarity_score, metadata)
        """
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit,
            )

            output = []
            for result in results:
                output.append((
                    result.payload.get('chunk_id', ''),
                    result.score,
                    result.payload
                ))
            return output

        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")

    def get_collection_names(self) -> List[str]:
        """Get all collection names."""
        try:
            collections = self.client.get_collections().collections
            return [c.name for c in collections]
        except Exception as e:
            logger.error(f"Error getting collections: {e}")
            return []

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get info about a collection."""
        try:
            collection_info = self.client.get_collection(collection_name)
            return {
                'name': collection_name,
                'points_count': collection_info.points_count,
                'vector_size': collection_info.config.params.vectors.size,
                'distance': str(collection_info.config.params.vectors.distance)
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
