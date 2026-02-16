"""
Embedding models for RAG system.
Uses Hugging Face sentence-transformers models.
"""
import logging
import numpy as np
from typing import List, Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class EmbeddingModel(ABC):
    """Base class for embedding models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.dimension = None

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts.
        Returns: normalized embeddings (n_samples, dimension)
        """
        pass


class SentenceTransformerEmbedding(EmbeddingModel):
    """Embedding using Sentence Transformers from Hugging Face."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required")

        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model dimension: {self.dimension}")

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts using Sentence Transformers.
        Normalized to unit vectors for cosine similarity.
        """
        if not texts:
            return np.array([]).reshape(0, self.dimension)

        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings


class EmbeddingPipeline:
    """
    Manage multiple embedding models.
    Ensures consistent preprocessing and normalization.
    """

    # Available models
    AVAILABLE_MODELS = {
        'all-MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2',
        'all-mpnet-base-v2': 'sentence-transformers/all-mpnet-base-v2',
        'e5-base-v2': 'intfloat/e5-base-v2',
        'bge-base-en-v1.5': 'BAAI/bge-base-en-v1.5',
        'e5-large-v2': 'intfloat/e5-large-v2',
    }

    def __init__(self, model_names: List[str] = None):
        """
        Initialize embedding pipeline with specified models.
        If None, use all available models.
        """
        if model_names is None:
            model_names = list(self.AVAILABLE_MODELS.keys())

        self.models: Dict[str, SentenceTransformerEmbedding] = {}
        self._load_models(model_names)

    def _load_models(self, model_names: List[str]) -> None:
        """Load specified models."""
        for short_name in model_names:
            if short_name not in self.AVAILABLE_MODELS:
                logger.warning(f"Unknown model: {short_name}, skipping")
                continue

            full_name = self.AVAILABLE_MODELS[short_name]
            try:
                model = SentenceTransformerEmbedding(full_name)
                self.models[short_name] = model
                logger.info(f"✓ Loaded: {short_name}")
            except Exception as e:
                logger.error(f"Failed to load {short_name}: {e}")

    def embed(self, model_name: str, texts: List[str]) -> np.ndarray:
        """
        Embed texts using specified model.
        Returns normalized embeddings.
        """
        if model_name not in self.models:
            raise ValueError(f"Model not loaded: {model_name}")

        return self.models[model_name].embed(texts)

    def get_all_model_names(self) -> List[str]:
        """Get list of loaded model names."""
        return list(self.models.keys())

    def get_model_dimension(self, model_name: str) -> int:
        """Get embedding dimension for a model."""
        if model_name not in self.models:
            raise ValueError(f"Model not loaded: {model_name}")
        return self.models[model_name].dimension

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for embedding.
        - Strip whitespace
        - Normalize unicode
        - Remove extra spaces
        """
        text = text.strip()
        text = ' '.join(text.split())  # Normalize spaces
        return text

    def batch_embed(self, model_name: str, texts: List[str],
                   batch_size: int = 32) -> np.ndarray:
        """
        Embed texts in batches for memory efficiency.
        """
        if not texts:
            return np.array([]).reshape(0, self.get_model_dimension(model_name))

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embed(model_name, batch)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def compute_similarity(self, embedding1: np.ndarray,
                          embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two normalized embeddings.
        Range: [0, 1] (or [-1, 1] depending on content)
        """
        return float(np.dot(embedding1, embedding2))
