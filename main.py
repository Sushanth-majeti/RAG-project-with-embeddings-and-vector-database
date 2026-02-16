"""
Main RAG Evaluation Pipeline
Compares chunking strategies and embedding models for retrieval accuracy.
"""
import os
import sys
import logging
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project modules
from src.document_loader import DocumentLoader
from src.chunking import get_all_chunkers
from src.embeddings import EmbeddingPipeline
from src.vector_db import QdrantManager
from src.evaluation import EvaluationMetrics, QueryEvaluator, ExperimentResults
from src.utils import Chunk, save_json, load_json, get_project_root, ensure_dir


def setup_project_structure():
    """Create necessary project directories."""
    root = get_project_root()
    ensure_dir(os.path.join(root, 'projects'))
    ensure_dir(os.path.join(root, 'results'))
    ensure_dir(os.path.join(root, 'qdrant_storage'))
    logger.info("✓ Project structure ready")


def load_sample_documents(projects_folder: str) -> List[Tuple[str, str]]:
    """Load documents from projects folder."""
    logger.info("=" * 60)
    logger.info("STEP 1: LOADING DOCUMENTS")
    logger.info("=" * 60)

    loader = DocumentLoader()
    documents = loader.load_documents(projects_folder)

    if not documents:
        logger.warning(f"No documents found in {projects_folder}")
        logger.info("Creating sample documents for demonstration...")
        return _create_sample_documents(projects_folder)

    logger.info(f"Total documents loaded: {len(documents)}")
    for source, content in documents:
        logger.info(f"  - {Path(source).name}: {len(content)} chars, ~{len(content.split())} words")

    return documents


def _create_sample_documents(folder: str) -> List[Tuple[str, str]]:
    """Create sample documents for demonstration."""
    ensure_dir(folder)

    sample_docs = {
        'annual_report.md': '''# Annual Financial Report 2025

## Executive Summary
Our annual results show strong financial performance with record revenue and profitability.

### Key Metrics
- Total Revenue: $50 Million
- Operating Profit: $12 Million
- Net Earnings: $10 Million
- Profit Margin: 20%

## Revenue Analysis
The company generated significant revenue across all business segments. Our annual financial results 
demonstrate healthy growth compared to previous year.

### Quarterly Breakdown
Q1 Results: $10M revenue, $2M profit
Q2 Results: $12M revenue, $3M profit
Q3 Results: $13M revenue, $3.5M profit
Q4 Results: $15M revenue, $3.5M profit

## Financial Position
Our financial statements show stable balance sheet with growing assets.

### Assets and Liabilities
- Total Assets: $100 Million
- Total Liabilities: $30 Million
- Net Worth: $70 Million

## Performance Summary
Annual results exceed expectations with year-over-year growth of 15%.
''',

        'quarterly_report_q4.md': '''# Q4 2025 Quarterly Results

## Quarterly Financial Overview
Q4 delivered exceptional quarterly earnings with strong performance across divisions.

### Q4 Key Metrics
- Quarterly Revenue: $15 Million
- Quarterly Profit: $3.5 Million
- Quarterly Growth Rate: 18%
- Earnings Per Share: $5.25

## Quarterly Performance Analysis
The quarterly report shows consistent performance improvements. Our quarterly financial 
metrics indicate healthy business operations.

### Divisional Quarterly Results
- Division A: $6M quarterly revenue
- Division B: $5M quarterly revenue  
- Division C: $4M quarterly revenue

## Quarterly Trends
Sequential quarterly improvement demonstrates positive momentum.
''',

        'half_year_interim.md': '''# Half-Year Interim Financial Report

## H1 Interim Results
Our half-year interim statement provides key metrics for the first half of the year.

### H1 2025 Metrics
- H1 Revenue: $22 Million (half-year total)
- H1 Operating Profit: $5 Million
- H1 Net Income: $4.5 Million
- H1 Interim Growth: 12%

## Interim Statement
The interim report for the half-year period reflects solid performance. Our half-year 
financial results are on track.

### H1 vs H2 Comparison
First half interim results: $22M revenue
Second half forecast: $28M revenue (based on Q3-Q4 performance)

## Interim Analysis
First half performance supports full-year guidance.
''',
    }

    documents = []
    for filename, content in sample_docs.items():
        filepath = os.path.join(folder, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        documents.append((filepath, content))
        logger.info(f"Created sample: {filename}")

    return documents


def load_query_set(data_folder: str) -> List[Dict[str, Any]]:
    """Load query evaluation set."""
    logger.info("=" * 60)
    logger.info("STEP 2: LOADING QUERY SET")
    logger.info("=" * 60)

    queries_file = os.path.join(data_folder, 'queries.json')
    queries = load_json(queries_file)
    logger.info(f"Loaded {len(queries)} evaluation queries")
    return queries


def run_chunking_stage(documents: List[Tuple[str, str]]) -> Dict[str, List[Chunk]]:
    """Run all chunking strategies on documents."""
    logger.info("=" * 60)
    logger.info("STEP 3: CHUNKING DOCUMENTS")
    logger.info("=" * 60)

    chunkers = get_all_chunkers()
    all_chunks = {}

    for strategy_name, chunker in chunkers.items():
        logger.info(f"\n[{strategy_name.upper()}]")
        strategy_chunks = []

        for source_file, content in documents:
            chunks = chunker.chunk(content, source_file)
            strategy_chunks.extend(chunks)

        all_chunks[strategy_name] = strategy_chunks
        logger.info(f"  Total: {len(strategy_chunks)} chunks")

    return all_chunks


def run_embedding_stage(embedding_pipeline: EmbeddingPipeline,
                       all_chunks: Dict[str, List[Chunk]]) -> Dict[str, Dict[str, np.ndarray]]:
    """Generate embeddings for all chunks using all models."""
    logger.info("=" * 60)
    logger.info("STEP 4: GENERATING EMBEDDINGS")
    logger.info("=" * 60)

    embeddings_by_strategy = {}

    for strategy_name, chunks in all_chunks.items():
        logger.info(f"\n[{strategy_name.upper()}]")
        embeddings_by_model = {}

        chunk_texts = [chunk.content for chunk in chunks]

        for model_name in embedding_pipeline.get_all_model_names():
            logger.info(f"  Embedding with {model_name}...")
            embeddings = embedding_pipeline.batch_embed(model_name, chunk_texts, batch_size=32)
            embeddings_by_model[model_name] = embeddings
            logger.info(f"    ✓ Generated {embeddings.shape[0]} embeddings")

        embeddings_by_strategy[strategy_name] = embeddings_by_model

    return embeddings_by_strategy


def run_indexing_stage(vector_db: QdrantManager,
                      embedding_pipeline: EmbeddingPipeline,
                      all_chunks: Dict[str, List[Chunk]],
                      embeddings_by_strategy: Dict[str, Dict[str, np.ndarray]]):
    """Index all chunks in Qdrant."""
    logger.info("=" * 60)
    logger.info("STEP 5: INDEXING TO QDRANT")
    logger.info("=" * 60)

    for strategy_name, chunks in all_chunks.items():
        logger.info(f"\n[{strategy_name.upper()}]")

        for model_name in embedding_pipeline.get_all_model_names():
            embeddings = embeddings_by_strategy[strategy_name][model_name]
            vector_size = embedding_pipeline.get_model_dimension(model_name)

            collection_name = f"{strategy_name}_{model_name}"

            # Create collection
            vector_db.create_collection(collection_name, vector_size)

            # Prepare metadata
            metadata_list = []
            for chunk in chunks:
                metadata_list.append({
                    'chunk_id': chunk.chunk_id,
                    'content': chunk.content,
                    'source_file': chunk.source_file,
                    'chunk_index': chunk.chunk_index,
                })

            # Add vectors
            vector_db.add_vectors(collection_name, embeddings, metadata_list)

    logger.info(f"\n✓ Indexed all configurations")


def run_retrieval_stage(vector_db: QdrantManager,
                       embedding_pipeline: EmbeddingPipeline,
                       queries: List[Dict[str, Any]],
                       all_chunks: Dict[str, List[Chunk]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieve results for all queries across all configurations.
    Returns: results_by_config[config_name] = list of query results
    """
    logger.info("=" * 60)
    logger.info("STEP 6: RETRIEVING RESULTS")
    logger.info("=" * 60)

    results_by_config = {}
    total_configs = len(all_chunks) * len(embedding_pipeline.get_all_model_names())
    current_config = 0

    for strategy_name, chunks in all_chunks.items():
        for model_name in embedding_pipeline.get_all_model_names():
            current_config += 1
            collection_name = f"{strategy_name}_{model_name}"

            logger.info(f"\n[{current_config}/{total_configs}] {collection_name}")

            query_results = []

            for query_item in queries:
                query_text = query_item['query']
                expected_source_patterns = query_item.get('expected_source_patterns', [])
                expected_keywords = query_item.get('expected_keywords', [])

                # Embed query
                query_embedding = embedding_pipeline.embed(model_name, [query_text])[0]

                # Search
                retrieved = vector_db.search(collection_name, query_embedding, limit=10)

                # Find expected source file from chunks
                expected_source = None
                for chunk in chunks:
                    for pattern in expected_source_patterns:
                        if pattern.lower() in chunk.source_file.lower():
                            expected_source = chunk.source_file
                            break
                    if expected_source:
                        break

                if not expected_source and chunks:
                    expected_source = chunks[0].source_file

                # Evaluate
                evaluator = QueryEvaluator(query_text, expected_source or "", expected_keywords)
                result = evaluator.evaluate(retrieved, query_id=query_item.get('query_id'))
                query_results.append(result)

            results_by_config[collection_name] = query_results
            logger.info(f"  Retrieved for {len(query_results)} queries")

    return results_by_config


def run_evaluation_stage(results_by_config: Dict[str, List[Dict[str, Any]]]) -> ExperimentResults:
    """Aggregate results and compute final metrics."""
    logger.info("=" * 60)
    logger.info("STEP 7: EVALUATING RESULTS")
    logger.info("=" * 60)

    all_results = ExperimentResults()

    for config_name, query_results in results_by_config.items():
        strategy_name, model_name = config_name.rsplit('_', 1)

        all_results.add_result(strategy_name, model_name, query_results)

    return all_results


def generate_report(experiment_results: ExperimentResults, results_folder: str):
    """Generate and save final report."""
    logger.info("=" * 60)
    logger.info("STEP 8: GENERATING REPORTS")
    logger.info("=" * 60)

    ensure_dir(results_folder)

    # Create DataFrame
    df = pd.DataFrame(experiment_results.get_all_results())

    # Sort by combined score (top-3 + MRR)
    df['combined_score'] = df['top_3_accuracy'] + df['mrr']
    df = df.sort_values('combined_score', ascending=False)

    # Save CSV
    csv_path = os.path.join(results_folder, 'results.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"✓ Saved: results.csv")

    # Save JSON
    json_path = os.path.join(results_folder, 'results.json')
    json_data = {
        'configurations': experiment_results.get_all_results(),
        'summary_stats': experiment_results.get_summary_stats()
    }
    save_json(json_data, json_path)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT RESULTS SUMMARY")
    logger.info("=" * 60)

    print("\n" + "=" * 80)
    print(df[['chunking_strategy', 'embedding_model', 'top_1_accuracy', 'top_3_accuracy', 'mrr', 'combined_score']].to_string(index=False))
    print("=" * 80)

    # Get best configurations
    best_by_top1 = experiment_results.get_best_by_metric('top_1_accuracy')
    best_by_top3 = experiment_results.get_best_by_metric('top_3_accuracy')
    best_by_mrr = experiment_results.get_best_by_metric('mrr')
    best_by_similarity = experiment_results.get_best_by_metric('avg_similarity')

    # Best overall (top-3 + MRR)
    best_df = df.iloc[0]

    logger.info("\n" + "=" * 60)
    logger.info("BEST PERFORMING CONFIGURATIONS")
    logger.info("=" * 60)

    print(f"\n📊 Best by Top-1 Accuracy: {best_by_top1['chunking_strategy']} + {best_by_top1['embedding_model']} ({best_by_top1['top_1_accuracy']:.4f})")
    print(f"📊 Best by Top-3 Accuracy: {best_by_top3['chunking_strategy']} + {best_by_top3['embedding_model']} ({best_by_top3['top_3_accuracy']:.4f})")
    print(f"📊 Best by MRR: {best_by_mrr['chunking_strategy']} + {best_by_mrr['embedding_model']} ({best_by_mrr['mrr']:.4f})")
    print(f"📊 Best by Avg Similarity: {best_by_similarity['chunking_strategy']} + {best_by_similarity['embedding_model']} ({best_by_similarity['avg_similarity']:.4f})")

    print("\n" + "=" * 80)
    print("🏆 BEST OVERALL COMBINATION (Top-3 + MRR)")
    print("=" * 80)
    print(f"\nChunking Strategy: {best_df['chunking_strategy']}")
    print(f"Embedding Model: {best_df['embedding_model']}")
    print(f"Top-1 Accuracy: {best_df['top_1_accuracy']:.4f}")
    print(f"Top-3 Accuracy: {best_df['top_3_accuracy']:.4f}")
    print(f"Mean Reciprocal Rank: {best_df['mrr']:.4f}")
    print(f"Average Similarity: {best_df['avg_similarity']:.4f}")
    print(f"Combined Score: {best_df['combined_score']:.4f}")

    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)
    print(f"\n✨ Based on the experiments, the best-performing configuration is:")
    print(f"   Chunking Strategy: {best_df['chunking_strategy']}")
    print(f"   Embedding Model: {best_df['embedding_model']}")
    print(f"   Reason: Achieved highest combined score (Top-3 accuracy + MRR)")
    print("=" * 80 + "\n")

    return df


def main():
    """Main orchestration function."""
    logger.info("\n" + "=" * 80)
    logger.info("RAG CHUNKING & EMBEDDING STRATEGY EVALUATION")
    logger.info("=" * 80)

    try:
        # Setup
        setup_project_structure()
        root = get_project_root()
        projects_folder = os.path.join(root, 'projects')
        results_folder = os.path.join(root, 'results')
        data_folder = os.path.join(root, 'data')

        # Step 1: Load documents
        documents = load_sample_documents(projects_folder)

        # Step 2: Load query set
        queries = load_query_set(data_folder)

        # Step 3: Run chunking
        all_chunks = run_chunking_stage(documents)

        # Step 4: Initialize embedding pipeline and generate embeddings
        logger.info("\nInitializing embedding models...")
        embedding_pipeline = EmbeddingPipeline()
        embeddings_by_strategy = run_embedding_stage(embedding_pipeline, all_chunks)

        # Step 5: Initialize vector DB and index
        vector_db = QdrantManager(db_path=os.path.join(root, 'qdrant_storage'))
        run_indexing_stage(vector_db, embedding_pipeline, all_chunks, embeddings_by_strategy)

        # Step 6: Retrieve results
        results_by_config = run_retrieval_stage(vector_db, embedding_pipeline, queries, all_chunks)

        # Step 7: Evaluate
        experiment_results = run_evaluation_stage(results_by_config)

        # Step 8: Generate report
        df = generate_report(experiment_results, results_folder)

        logger.info("\n✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
        logger.info(f"Results saved to {results_folder}")

    except Exception as e:
        logger.error(f"❌ Error in main execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
