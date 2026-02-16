# RAG Chunking & Embedding Strategy Evaluation Project

## Overview
This project comprehensively compares **5 chunking strategies** and **5 embedding models** for RAG (Retrieval-Augmented Generation) pipelines.

It evaluates retrieval accuracy and reports the best-performing combination using metrics like Top-1/Top-3 accuracy, Mean Reciprocal Rank (MRR), and similarity scores.

---

## Project Requirements

### Input Data
- Loads documents recursively from `/projects` folder
- Supports: PDF, DOCX, XLSX, PPTX, Markdown (.md)
- Extracts clean text and preserves structure

### Chunking Strategies (5 total)
1. **Fixed-Size Chunking** (512 tokens, 100 overlap)
2. **Recursive Chunking** (Paragraph → Sentence → Fixed-size fallback)
3. **Structure-Aware Chunking** (Chunk by headings H1/H2/H3, merge small sections)
4. **Hybrid Chunking** (Structure-aware + Recursive fallback)
5. **Table-Aware Chunking** (Detects & preserves tables as atomic chunks)

### Embedding Models (5 total)
1. `sentence-transformers/all-MiniLM-L6-v2`
2. `sentence-transformers/all-mpnet-base-v2`
3. `intfloat/e5-base-v2`
4. `BAAI/bge-base-en-v1.5`
5. `intfloat/e5-large-v2`

### Vector Database
- **Qdrant** (local storage)
- Cosine similarity
- One collection per (chunking_strategy × embedding_model) = 25 configurations

### Query Evaluation Set
- 15 pre-defined queries derived from financial documents
- Each query includes:
  - `query`: The question
  - `expected_keywords`: Keywords to match
  - `expected_source_patterns`: Expected document patterns

### Evaluation Metrics
For each query:
- **Top-1 Accuracy**: Retrieved chunk from expected source with keywords (binary)
- **Top-3 Accuracy**: Any of top-3 results from expected source with keywords
- **MRR (Mean Reciprocal Rank)**: 1/rank of first relevant result
- **Avg Similarity**: Average cosine similarity of relevant results

---

## Project Structure

```
rag_project_2.0/
├── src/
│   ├── __init__.py              # Package init
│   ├── document_loader.py       # Load PDF, DOCX, XLSX, PPTX, MD
│   ├── chunking.py              # 5 chunking strategies
│   ├── embeddings.py            # Embedding pipeline (HuggingFace)
│   ├── vector_db.py             # Qdrant vector database
│   ├── evaluation.py            # Metrics calculation
│   └── utils.py                 # Helper functions & data classes
├── main.py                      # Main orchestrator
├── projects/                    # Input documents folder
├── results/                     # Output folder (results.csv, results.json)
├── data/
│   └── queries.json             # Evaluation query set
├── qdrant_storage/              # Qdrant local database
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Input Documents
Place your documents in the `projects/` folder:
```bash
projects/
├── document1.pdf
├── document2.docx
├── report.xlsx
├── presentation.pptx
└── notes.md
```

**Note**: The project auto-creates sample financial documents if the folder is empty.

### 3. Run the Pipeline
```bash
python main.py
```

---

## Execution Flow

The pipeline runs in 8 stages:

1. **LOADING DOCUMENTS** - Reads all supported file types
2. **LOADING QUERY SET** - Loads 15 evaluation queries
3. **CHUNKING DOCUMENTS** - Applies all 5 chunking strategies
4. **GENERATING EMBEDDINGS** - Embeds chunks with all 5 models
5. **INDEXING TO QDRANT** - Creates 25 collections in Qdrant
6. **RETRIEVING RESULTS** - Queries each configuration (5 × 5 = 25)
7. **EVALUATING RESULTS** - Computes metrics for all queries
8. **GENERATING REPORTS** - Creates results.csv and results.json

---

## Output Files

### `results/results.csv`
Tab-separated results with:
- `chunking_strategy`: Strategy name
- `embedding_model`: Model name
- `top_1_accuracy`: Top-1 retrieval accuracy
- `top_3_accuracy`: Top-3 retrieval accuracy
- `mrr`: Mean reciprocal rank
- `avg_similarity`: Average cosine similarity
- `num_queries`: Number of queries evaluated
- `combined_score`: Top-3 + MRR (for ranking)

### `results/results.json`
Structured JSON with:
- `configurations`: Full results array
- `summary_stats`: Aggregate statistics

### Console Output
The script prints:
- Progress logs for each stage
- Final results table (sorted by combined score)
- Best configurations by each metric
- **FINAL RECOMMENDATION** highlighting the best combination

---

## Key Design Decisions

### Modularity
- Each chunking strategy is an independent class
- Embedding pipeline abstracts model selection
- Qdrant management separates vector DB logic
- Evaluation metrics are self-contained

### Reproducibility
- Fixed random seed (42) at startup
- Deterministic embedding order
- Batch processing for consistency
- Results saved in both CSV and JSON

### Scalability
- Batch embedding (32 chunks at a time) for memory efficiency
- Generator-style document loading
- Configurable query set
- Can add new embeddings/chunkers easily

---

## Interpreting Results

The **combined_score** (Top-3 + MRR) is the primary ranking metric because:
- **Top-3**: Measures if the correct document is in the top 3 results
- **MRR**: Rewards retrieving correct documents at higher ranks
- Combined: Balances accuracy with rank position

**Recommendation**: Use the configuration with the highest combined_score.

---

## Extending the Project

### Add New Chunking Strategy
Create a new class in `src/chunking.py`:
```python
class MyChunker(ChunkingStrategy):
    def __init__(self):
        super().__init__(name="my_chunker", chunk_size=512)
    
    def chunk(self, text: str, source_file: str) -> List[Chunk]:
        # Your implementation
        pass

# Register in get_all_chunkers()
```

### Add New Embedding Model
Add to `EmbeddingPipeline.AVAILABLE_MODELS`:
```python
'my-model': 'huggingface/my-model-name'
```

### Add More Queries
Edit `data/queries.json` with new query objects.

---

## Dependencies

See `requirements.txt` for full list. Key packages:
- **sentence-transformers**: HuggingFace embedding models
- **qdrant-client**: Vector database client
- **pandas**: Data analysis
- **numpy**: Numerical computing
- **PyPDF2, python-docx, openpyxl, python-pptx**: Document parsing

---

## Troubleshooting

### Out of Memory
- Reduce batch size in `embeddings.py` (line: `batch_embed`)
- Reduce number of documents
- Use smaller embedding models (e.g., all-MiniLM-L6-v2)

### Qdrant Connection Issues
- Ensure `qdrant_storage/` folder is writable
- Delete `qdrant_storage/` and restart if corrupted

### Missing Documents
- Place files in `projects/` folder
- Ensure file extensions are lowercase (.pdf, not .PDF)
- Supported: .pdf, .docx, .xlsx, .pptx, .md

---

## Performance Notes

Typical execution times:
- **Document Loading**: < 1 second
- **Chunking**: 1-2 seconds
- **Embedding**: 30-60 seconds (depends on model sizes and document count)
- **Indexing**: 5-10 seconds
- **Retrieval & Evaluation**: 10-20 seconds
- **Total**: ~1-2 minutes

---

## License & Credits

Built as a comprehensive RAG evaluation framework demonstrating:
- Multi-strategy comparison
- Multi-model evaluation
- Production-grade code structure
- Reproducible experiments

---

## Contact & Support

For issues or questions, refer to inline code comments and docstrings in the `src/` modules.
