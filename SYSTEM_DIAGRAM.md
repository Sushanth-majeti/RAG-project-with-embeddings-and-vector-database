# RAG Evaluation Project - System Overview

## 🏗️ Complete System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         RAG EVALUATION PIPELINE                          │
└──────────────────────────────────────────────────────────────────────────┘

STAGE 1: DOCUMENT INGESTION
┌─────────────────────────────────────────────────────────────────────────┐
│ DocumentLoader (document_loader.py)                                     │
│ ├─ PDF extraction (PyPDF2)                                             │
│ ├─ DOCX parsing (python-docx)                                          │
│ ├─ Excel reading (openpyxl)                                            │
│ ├─ PowerPoint extraction (python-pptx)                                 │
│ └─ Markdown reading (raw text)                                         │
│                                                                         │
│ INPUT:  projects/ folder (any of above formats)                        │
│ OUTPUT: List[(source_file, text_content)]                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
STAGE 2: QUERY LOADING
┌─────────────────────────────────────────────────────────────────────────┐
│ Load data/queries.json                                                  │
│ 15 evaluation queries with:                                            │
│ ├─ query text                                                          │
│ ├─ expected keywords                                                   │
│ └─ expected source patterns                                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
STAGE 3: TEXT CHUNKING (PARALLEL)
┌──────────────────────────────────────────────────────────────────────────┐
│ Execute all 5 strategies independently:                                  │
│                                                                          │
│  Strategy 1       Strategy 2       Strategy 3       Strategy 4           │
│  FixedSize        Recursive        Structure-Aware   Hybrid              │
│  512 tok, 100     Para→Sent→Fxd   By Headings       Struct+Recursive    │
│  overlap          overlap          Merge small       fallback            │
│  ▼                ▼                ▼                ▼                    │
│  Chunks[]         Chunks[]         Chunks[]         Chunks[]             │
│                                                                          │
│  Strategy 5                                                              │
│  Table-Aware                                                             │
│  Preserve tables as atomic chunks                                        │
│  ▼                                                                       │
│  Chunks[]                                                                │
│                                                                          │
│ OUTPUT: Dict[strategy_name] = List[Chunk]                              │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
STAGE 4: EMBEDDING GENERATION
┌──────────────────────────────────────────────────────────────────────────┐
│ For each chunking strategy:                                              │
│   For each chunk:                                                        │
│     For each embedding model:                                            │
│       Generate embedding (normalized L2 vectors)                         │
│                                                                          │
│  Model 1: all-MiniLM-L6-v2        (384-dim) ⚡⚡⚡ fast                 │
│  Model 2: all-mpnet-base-v2       (768-dim) ⚡⚡ balanced              │
│  Model 3: intfloat/e5-base-v2     (768-dim) ⚡⚡ strong                │
│  Model 4: BAAI/bge-base-en-v1.5   (768-dim) ⚡⚡ bilingual            │
│  Model 5: intfloat/e5-large-v2    (1024-dim) ⚡ best quality           │
│                                                                          │
│ Total combinations: 5 strategies × 5 models = 25 embedding configs      │
│                                                                          │
│ OUTPUT: Dict[strategy][model] = np.ndarray (n_chunks, dimension)       │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
STAGE 5: VECTOR INDEXING (Qdrant)
┌──────────────────────────────────────────────────────────────────────────┐
│ Create 25 collections (one per strategy×model combination)               │
│                                                                          │
│ ┌─────────────────────────────────────────────────────────────────┐     │
│ │ fixed_size_all-MiniLM-L6-v2 (cosine similarity)               │     │
│ │ ├─ Point 0: [vector] + {metadata}                            │     │
│ │ ├─ Point 1: [vector] + {metadata}                            │     │
│ │ └─ Point N: [vector] + {metadata}                            │     │
│ └─────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│ ┌─────────────────────────────────────────────────────────────────┐     │
│ │ fixed_size_all-mpnet-base-v2 (cosine similarity)              │     │
│ │ ├─ Point 0: [vector] + {metadata}                            │     │
│ │ └─ ...                                                         │     │
│ └─────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│ ... (23 more collections)                                                │
│                                                                          │
│ Storage: ./qdrant_storage/ (local, in-memory friendly)                  │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
STAGE 6: RETRIEVAL & EVALUATION
┌──────────────────────────────────────────────────────────────────────────┐
│ For each configuration (25 total):                                       │
│   For each query (15 total):                                             │
│     1. Embed query with same model                                       │
│     2. Search Qdrant collection (cosine similarity)                      │
│     3. Retrieve top-10 results                                           │
│     4. Evaluate each result:                                             │
│        - Is source file correct?                                         │
│        - Are keywords present?                                           │
│     5. Calculate metrics:                                                │
│        - Top-1 Accuracy: rank-1 is relevant? (0/1)                      │
│        - Top-3 Accuracy: any of rank 1-3 relevant? (0/1)                │
│        - MRR: 1/rank of first relevant (0-1)                            │
│        - Avg Similarity: mean cosine of relevant (0-1)                  │
│                                                                          │
│ Total queries: 25 configs × 15 queries = 375 retrieval operations       │
│ Total metrics: 25 × 4 = 100 metric values                               │
│                                                                          │
│ OUTPUT: Dict[config] = List[Dict[metric_values]]                       │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
STAGE 7: AGGREGATION & RANKING
┌──────────────────────────────────────────────────────────────────────────┐
│ For each configuration:                                                  │
│   Average metrics across 15 queries:                                     │
│   - avg_top_1_accuracy                                                   │
│   - avg_top_3_accuracy                                                   │
│   - avg_mrr                                                              │
│   - avg_similarity                                                       │
│   - combined_score = avg_top_3 + avg_mrr                                 │
│                                                                          │
│ Create ranking (sorted by combined_score descending):                    │
│   Rank 1: highest score ← BEST CONFIGURATION                            │
│   Rank 2: second highest                                                 │
│   ...                                                                    │
│   Rank 25: lowest score                                                  │
│                                                                          │
│ OUTPUT: DataFrame[25 rows × 7 cols]                                     │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
STAGE 8: REPORTING
┌──────────────────────────────────────────────────────────────────────────┐
│ Generate outputs:                                                        │
│                                                                          │
│ 1. results/results.csv                                                   │
│    ├─ 25 rows (configurations)                                           │
│    ├─ 7 columns (metrics)                                                │
│    └─ Sorted by combined_score                                           │
│                                                                          │
│ 2. results/results.json                                                  │
│    ├─ configurations array                                               │
│    └─ summary_stats object                                               │
│                                                                          │
│ 3. Console output                                                        │
│    ├─ Results table                                                      │
│    ├─ Best by each metric                                                │
│    └─ FINAL RECOMMENDATION                                               │
│                                                                          │
│ Example output:                                                          │
│ ┌─────────────────────────────────────────────────────┐                 │
│ │ BEST OVERALL COMBINATION                            │                 │
│ │ Chunking: recursive                                 │                 │
│ │ Embedding: e5-large-v2                              │                 │
│ │ Top-3: 0.967, MRR: 0.833, Score: 1.800             │                 │
│ │ Reason: Highest combined score                       │                 │
│ └─────────────────────────────────────────────────────┘                 │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Flow Example

```
INPUT TEXT:
"Our annual revenue was $50M with profits of $10M. 
 This represents 20% growth over last year."

     │
     ▼ (Chunking Strategy: Recursive)

CHUNKS:
[
  Chunk {
    id: "file_recursive_0",
    content: "Our annual revenue was $50M with profits of $10M.",
    source_file: "annual.pdf",
    metadata: {level: 'sentence', ...}
  },
  Chunk {
    id: "file_recursive_1", 
    content: "This represents 20% growth over last year.",
    source_file: "annual.pdf",
    metadata: {level: 'sentence', ...}
  }
]

     │
     ▼ (Embedding Model: e5-large-v2)

EMBEDDINGS:
[
  [0.123, -0.456, ..., 0.789],  ← chunk 0 (1024-dim, normalized)
  [0.120, -0.450, ..., 0.785]   ← chunk 1 (1024-dim, normalized)
]

     │
     ▼ (Indexing to Qdrant)

COLLECTION: recursive_e5-large-v2
[
  Point {
    id: 0,
    vector: [0.123, -0.456, ..., 0.789],
    payload: {chunk_id, content, source_file, ...}
  },
  Point {
    id: 1,
    vector: [0.120, -0.450, ..., 0.785],
    payload: {chunk_id, content, source_file, ...}
  }
]

     │
     ▼ (Query: "What was the annual revenue?")

QUERY EMBEDDING:
[0.122, -0.455, ..., 0.788]  ← same dimension, same model

     │
     ▼ (Cosine Similarity Search)

RESULTS:
[
  (chunk_0, score=0.97, metadata),   ← Rank 1 ⭐
  (chunk_1, score=0.85, metadata),   ← Rank 2
  ...
]

     │
     ▼ (Evaluation)

METRICS:
- Is rank-1 from annual.pdf? ✓ YES
- Does rank-1 contain "revenue"? ✓ YES
- Top-1 Accuracy: 1.0 ✓
- Top-3 Accuracy: 1.0 ✓
- MRR: 1/1 = 1.0 ✓
- Avg Similarity: 0.97 ✓

     │
     ▼ (Aggregation across 15 queries)

FINAL METRICS FOR THIS CONFIG:
{
  chunking_strategy: "recursive",
  embedding_model: "e5-large-v2",
  top_1_accuracy: 0.93,
  top_3_accuracy: 0.97,
  mrr: 0.87,
  avg_similarity: 0.71,
  combined_score: 1.84
}
```

---

## 🎯 Configuration Matrix (25 Total)

```
              │   MiniLM   │  MPNet  │  E5-base │  E5-large │   BGE  │
──────────────┼────────────┼─────────┼──────────┼───────────┼────────┤
Fixed-size    │     1      │    2    │    3     │     4     │   5    │
Recursive     │     6      │    7    │    8     │     9     │   10   │
Structure-Aw  │    11      │   12    │   13     │    14     │   15   │
Hybrid        │    16      │   17    │   18     │    19     │   20   │
Table-Aware   │    21      │   22    │   23     │    24     │   25   │
──────────────┴────────────┴─────────┴──────────┴───────────┴────────┘

Each cell = one Qdrant collection with independent evaluation
```

---

## 📈 Metrics Hierarchy

```
Per-Query Metrics (calculated per query)
│
├─ Top-1 Accuracy: Is rank-1 relevant?
├─ Top-3 Accuracy: Any of rank 1-3 relevant?
├─ MRR: 1 / rank_of_first_relevant
└─ Avg Similarity: Mean cosine of relevant results

     │
     ▼ (Average across 15 queries)

Per-Configuration Metrics
│
├─ avg_top_1_accuracy    (0-1)
├─ avg_top_3_accuracy    (0-1)
├─ avg_mrr               (0-1)
├─ avg_similarity        (0-1)
└─ combined_score = avg_top_3 + avg_mrr  (0-2)

     │
     ▼ (Rank by combined_score)

FINAL RANKING (25 configurations sorted)
```

---

## 🔄 Execution Timeline

```
Start
  │
  ├─ [1 sec]   Load documents (from projects/)
  ├─ [<1 sec]  Load queries (from data/queries.json)
  ├─ [1-2 sec] Chunking (5 strategies × documents)
  ├─ [30-60s]  Embedding (5 models × chunks)
  │            ├─ First 10s: Model download (~2GB) [one-time]
  │            └─ Remaining: Inference
  ├─ [5-10 sec] Indexing (25 collections to Qdrant)
  ├─ [10-20 sec] Retrieval & Evaluation (25 × 15 queries)
  ├─ [5-10 sec] Aggregation & Ranking
  ├─ [<1 sec]  Report generation
  │
  └─ COMPLETE
  
  Total: ~1-2 minutes (first run ~12 min with model download)
```

---

## 💾 Storage & Memory

```
Memory Usage:
- Embeddings: ~10-50 MB per model per configuration
  (1000 chunks × 384-1024 dims × 4 bytes)
- Qdrant: ~100-200 MB total
- Results: < 1 MB

Disk Usage:
- Model files: ~2.5 GB (one-time, cached)
- Qdrant storage: 100-200 MB
- Results: < 1 MB
- Project: ~ 2.5 GB total
```

---

## 🔑 Key Components

| Component | File | Purpose |
|-----------|------|---------|
| DocumentLoader | document_loader.py | Extract text from files |
| ChunkingStrategies | chunking.py | Split docs 5 different ways |
| EmbeddingPipeline | embeddings.py | Generate vectors for text |
| QdrantManager | vector_db.py | Index & search vectors |
| EvaluationMetrics | evaluation.py | Calculate accuracy metrics |
| Utils | utils.py | Helpers & data classes |
| Orchestrator | main.py | Run full pipeline |

---

## ✨ Design Patterns Used

1. **Strategy Pattern**: Different chunking/embedding strategies
2. **Pipeline Pattern**: Stage-by-stage execution
3. **Factory Pattern**: Create objects by name
4. **Observer Pattern**: Logging at each stage
5. **Aggregator Pattern**: Combine metrics from queries

---

**This architecture enables comprehensive comparison of RAG strategies!** 🚀
