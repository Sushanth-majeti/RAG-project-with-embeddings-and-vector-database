"""
Evaluation metrics for RAG retrieval.
"""
import logging
from typing import List, Dict, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Compute evaluation metrics for retrieval results.
    """

    @staticmethod
    def is_relevant(retrieved_chunk: Dict[str, Any],
                   expected_source_file: str,
                   expected_keywords: List[str]) -> bool:
        """
        Determine if a retrieved chunk is relevant.

        A chunk is relevant if:
        - It comes from the expected source file
        - AND it contains at least one expected keyword

        Args:
            retrieved_chunk: Metadata dict of retrieved chunk
            expected_source_file: Expected source file path
            expected_keywords: List of keywords to look for

        Returns:
            True if chunk is relevant
        """
        # Check source file
        source = retrieved_chunk.get('source_file', '')
        if not source or expected_source_file not in source:
            return False

        # Check keywords
        content = retrieved_chunk.get('content', '').lower()
        for keyword in expected_keywords:
            if keyword.lower() in content:
                return True

        return False

    @staticmethod
    def top_k_accuracy(retrieved_results: List[Tuple[str, float, Dict[str, Any]]],
                      expected_source_file: str,
                      expected_keywords: List[str],
                      k: int = 3) -> float:
        """
        Compute Top-K accuracy.
        Returns 1.0 if any of top-k results are relevant, 0.0 otherwise.
        """
        for i, (chunk_id, similarity, metadata) in enumerate(retrieved_results[:k]):
            if EvaluationMetrics.is_relevant(metadata, expected_source_file, expected_keywords):
                return 1.0
        return 0.0

    @staticmethod
    def mean_reciprocal_rank(retrieved_results: List[Tuple[str, float, Dict[str, Any]]],
                            expected_source_file: str,
                            expected_keywords: List[str]) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).
        Returns 1/rank of first relevant result, or 0.0 if no relevant result.
        """
        for rank, (chunk_id, similarity, metadata) in enumerate(retrieved_results, start=1):
            if EvaluationMetrics.is_relevant(metadata, expected_source_file, expected_keywords):
                return 1.0 / rank
        return 0.0

    @staticmethod
    def average_similarity(retrieved_results: List[Tuple[str, float, Dict[str, Any]]],
                          expected_source_file: str,
                          expected_keywords: List[str],
                          k: int = 5) -> float:
        """
        Compute average cosine similarity of top-k relevant results.
        If no relevant results, return 0.0.
        """
        similarities = []
        for chunk_id, similarity, metadata in retrieved_results[:k]:
            if EvaluationMetrics.is_relevant(metadata, expected_source_file, expected_keywords):
                similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0

    @staticmethod
    def compute_all_metrics(
        retrieved_results: List[Tuple[str, float, Dict[str, Any]]],
        expected_source_file: str,
        expected_keywords: List[str]
    ) -> Dict[str, float]:
        """
        Compute all metrics for a single query.

        Returns:
            Dict with keys: top_1, top_3, mrr, avg_similarity
        """
        return {
            'top_1': EvaluationMetrics.top_k_accuracy(
                retrieved_results, expected_source_file, expected_keywords, k=1
            ),
            'top_3': EvaluationMetrics.top_k_accuracy(
                retrieved_results, expected_source_file, expected_keywords, k=3
            ),
            'mrr': EvaluationMetrics.mean_reciprocal_rank(
                retrieved_results, expected_source_file, expected_keywords
            ),
            'avg_similarity': EvaluationMetrics.average_similarity(
                retrieved_results, expected_source_file, expected_keywords
            ),
        }


class QueryEvaluator:
    """
    Evaluate a single query across all retrieved results.
    """

    def __init__(self, query: str, expected_source_file: str, expected_keywords: List[str]):
        self.query = query
        self.expected_source_file = expected_source_file
        self.expected_keywords = expected_keywords

    def evaluate(self, retrieved_results: List[Tuple[str, float, Dict[str, Any]]],
                query_id: str = None) -> Dict[str, Any]:
        """
        Evaluate retrieved results for this query.

        Returns:
            Dict with query info and metrics
        """
        metrics = EvaluationMetrics.compute_all_metrics(
            retrieved_results,
            self.expected_source_file,
            self.expected_keywords
        )

        return {
            'query_id': query_id or f"query_{hash(self.query) % 10000}",
            'query': self.query,
            'expected_source': self.expected_source_file,
            'expected_keywords': self.expected_keywords,
            **metrics
        }


class ExperimentResults:
    """
    Aggregate and analyze experiment results.
    """

    def __init__(self):
        self.results = []

    def add_result(self, chunker_name: str, model_name: str,
                  metrics_by_query: List[Dict[str, Any]]) -> None:
        """
        Add results for a (chunker, model) combination.

        Args:
            chunker_name: Name of chunking strategy
            model_name: Name of embedding model
            metrics_by_query: List of query evaluation results
        """
        # Aggregate metrics across queries
        top_1_scores = [m.get('top_1', 0) for m in metrics_by_query]
        top_3_scores = [m.get('top_3', 0) for m in metrics_by_query]
        mrr_scores = [m.get('mrr', 0) for m in metrics_by_query]
        similarities = [m.get('avg_similarity', 0) for m in metrics_by_query]

        result = {
            'chunking_strategy': chunker_name,
            'embedding_model': model_name,
            'top_1_accuracy': np.mean(top_1_scores),
            'top_3_accuracy': np.mean(top_3_scores),
            'mrr': np.mean(mrr_scores),
            'avg_similarity': np.mean(similarities),
            'num_queries': len(metrics_by_query),
        }

        self.results.append(result)

    def get_best_by_metric(self, metric: str) -> Dict[str, Any]:
        """Get best configuration by metric."""
        if not self.results:
            return {}
        return max(self.results, key=lambda x: x.get(metric, 0))

    def get_all_results(self) -> List[Dict[str, Any]]:
        """Get all results."""
        return self.results

    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics across all configurations."""
        if not self.results:
            return {}

        df_values = [r['top_3_accuracy'] + r['mrr'] for r in self.results]

        return {
            'mean_top_1': np.mean([r['top_1_accuracy'] for r in self.results]),
            'mean_top_3': np.mean([r['top_3_accuracy'] for r in self.results]),
            'mean_mrr': np.mean([r['mrr'] for r in self.results]),
            'mean_similarity': np.mean([r['avg_similarity'] for r in self.results]),
            'best_combined_score': max(df_values) if df_values else 0,
        }
