# RAG Evaluation Project - Complete Execution Guide

## 🎯 Before You Start

### System Requirements
- **Python**: 3.8+
- **RAM**: 8 GB minimum (16 GB recommended for large LLM embeddings)
- **Disk**: 2 GB for dependencies + Qdrant storage
- **OS**: Windows, Linux, or macOS

### Pre-Execution Checks
```bash
# Verify Python version
python --version
# Expected: Python 3.8.x or higher

# Verify internet connection (for downloading models)
ping huggingface.co
```

---

## 📥 Installation Steps

### Option 1: Automatic (Windows)
```bash
double-click run.bat
```
This will:
1. Check Python installation
2. Install all dependencies
3. Run setup validation
4. Start the main pipeline

### Option 2: Automatic (Linux/macOS)
```bash
chmod +x run.sh
./run.sh
```

### Option 3: Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Validate setup
python setup.py

# 3. Run pipeline
python main.py
```

---

## 🚀 Running the Pipeline

### Basic Execution
```bash
python main.py
```

### With Custom Python Path
```bash
/usr/bin/python3.9 main.py
```

### In Virtual Environment
```bash
# Create environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install & run
pip install -r requirements.txt
python main.py
```

---

## 📊 Understanding Execution Flow

### Stage 1: Document Loading (< 1 sec)
```
✓ Scans projects/ folder recursively
✓ Reads all .pdf, .docx, .xlsx, .pptx, .md files
✓ Extracts text while preserving structure
✓ Creates demo documents if folder is empty
```

**Console Output**:
```
Loading documents...
Loaded: annual_report.md (2500 chars, ~500 words)
Loaded: quarterly_report.pptx (1800 chars, ~350 words)
Total documents loaded: 2
```

### Stage 2: Query Loading (< 1 sec)
```
✓ Reads data/queries.json
✓ Loads 15 evaluation queries
✓ Each query has expected keywords & source patterns
```

**Console Output**:
```
Loaded 15 evaluation queries
q1: "What are the main financial results?"
q2: "How much revenue did we generate?"
...
```

### Stage 3: Chunking (1-2 sec)
```
Runs all 5 strategies:
- Fixed-size: 512 token chunks, 100 token overlap
- Recursive: Paragraph → Sentence → Fixed-size
- Structure-aware: By headings (H1/H2/H3)
- Hybrid: Structure + Recursive fallback
- Table-aware: Preserves tables as atomic units
```

**Console Output**:
```
========== FIXED-SIZE ==========
Generated 12 chunks from annual_report.md
Generated 8 chunks from quarterly_report.pptx
Total: 20 chunks

========== RECURSIVE ==========
Generated 15 chunks from annual_report.md
Generated 10 chunks from quarterly_report.pptx
Total: 25 chunks

... (repeats for structure_aware, hybrid, table_aware)
```

### Stage 4: Embedding Generation (30-60 sec)
```
For each chunking strategy:
  For each embedding model:
    - Convert text to vectors
    - Normalize (L2 norm)
    - Store in memory

Total: 5 strategies × 5 models = 25 inference runs
```

**Console Output**:
```
[FIXED-SIZE]
  Embedding with all-MiniLM-L6-v2...
    ✓ Generated 20 embeddings (384-dim, normalized)
  Embedding with all-mpnet-base-v2...
    ✓ Generated 20 embeddings (768-dim, normalized)
  ...

[RECURSIVE]
  ...
```

### Stage 5: Qdrant Indexing (5-10 sec)
```
Creates 25 collections (5 × 5):
- Each collection: strategy + model combination
- Distance metric: Cosine similarity
- Storage: ./qdrant_storage/
```

**Console Output**:
```
[FIXED-SIZE]
  ✓ Created collection: fixed_size_all-MiniLM-L6-v2
    Added 20 vectors
  ✓ Created collection: fixed_size_all-mpnet-base-v2
    Added 20 vectors
  ...

[RECURSIVE]
  ...
```

### Stage 6: Retrieval (10-20 sec)
```
For each configuration (25 total):
  For each query (15 total):
    - Embed query
    - Search Qdrant
    - Retrieve top-10 results
    - Store for evaluation

Total: 25 × 15 = 375 retrieval operations
```

**Console Output**:
```
[1/25] fixed_size_all-MiniLM-L6-v2
  Retrieved for 15 queries
  
[2/25] fixed_size_all-mpnet-base-v2
  Retrieved for 15 queries

...
```

### Stage 7: Evaluation (5-10 sec)
```
For each query result:
  - Check if source matches expected
  - Check if keywords present
  - Calculate: Top-1, Top-3, MRR, Avg Similarity
  - Aggregate across 15 queries
```

**Console Output**:
```
================== EVALUATION ==================
Evaluating results...
Computed metrics for all 25 configurations
```

### Stage 8: Report Generation (< 1 sec)
```
- Save results to results/results.csv
- Save results to results/results.json
- Print summary to console
- Print best configurations
```

**Console Output**:
```
chunking_strategy | embedding_model | top_1 | top_3 | mrr | similarity
fixed_size        | all-MiniLM      | 0.87  | 0.93  | 0.78| 0.65
recursive         | e5-large-v2     | 0.90  | 0.95  | 0.81| 0.68
...
```

---

## 📈 Expected Results

### Output Files

#### results/results.csv
```csv
chunking_strategy,embedding_model,top_1_accuracy,top_3_accuracy,mrr,avg_similarity,num_queries,combined_score
fixed_size,all-MiniLM-L6-v2,0.867,0.933,0.789,0.654,15,1.722
fixed_size,all-mpnet-base-v2,0.867,0.933,0.789,0.667,15,1.722
...
```

#### results/results.json
```json
{
  "configurations": [
    {
      "chunking_strategy": "fixed_size",
      "embedding_model": "all-MiniLM-L6-v2",
      "top_1_accuracy": 0.867,
      "top_3_accuracy": 0.933,
      "mrr": 0.789,
      "avg_similarity": 0.654,
      "num_queries": 15
    },
    ...
  ],
  "summary_stats": {
    "mean_top_1": 0.85,
    "mean_top_3": 0.92,
    "mean_mrr": 0.79,
    "mean_similarity": 0.66,
    "best_combined_score": 1.79
  }
}
```

### Console Summary
```
================================================================================
FINAL RECOMMENDATION
================================================================================

Chunking Strategy: hybrid
Embedding Model: e5-large-v2
Reason: Achieved highest combined score (Top-3 accuracy + MRR)

================================================================================

Best Overall Configuration:
  Top-1 Accuracy:  0.9000
  Top-3 Accuracy:  0.9667
  Mean Reciprocal Rank:  0.8333
  Average Similarity:  0.7123
  Combined Score:  1.8000

================================================================================
✅ EXPERIMENT COMPLETED SUCCESSFULLY!
Results saved to ./results
```

---

## ⚙️ Customization Guide

### Use Your Own Documents

1. **Create documents folder** (if not exists):
   ```bash
   mkdir projects
   ```

2. **Add your files**:
   ```bash
   projects/
   ├── report1.pdf
   ├── spreadsheet.xlsx
   ├── presentation.pptx
   └── notes.md
   ```

3. **Run pipeline**:
   ```bash
   python main.py
   ```

### Add More Queries

Edit `data/queries.json`:
```json
[
  {
    "query_id": "q16",
    "query": "What are the quarterly sales totals?",
    "expected_keywords": ["quarterly", "sales", "total"],
    "expected_source_patterns": ["quarterly"]
  },
  ...
]
```

### Skip Specific Models

Edit `main.py` (in `run_embedding_stage`):
```python
# Before:
embedding_pipeline = EmbeddingPipeline()

# After (skip large models):
embedding_pipeline = EmbeddingPipeline(
    model_names=['all-MiniLM-L6-v2', 'all-mpnet-base-v2']
)
```

### Adjust Chunking Parameters

Edit `src/chunking.py`:
```python
class FixedSizeChunker(ChunkingStrategy):
    def __init__(self):
        super().__init__(
            name="fixed_size",
            chunk_size=1024,        # Increase from 512
            chunk_overlap=200       # Increase from 100
        )
```

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'sentence_transformers'"

**Solution**:
```bash
pip install sentence-transformers
```

### Issue: "CUDA out of memory" or "Out of memory"

**Solution 1 - Reduce batch size**:
Edit `src/embeddings.py`:
```python
embeddings = embedding_pipeline.batch_embed(
    model_name, 
    chunk_texts, 
    batch_size=8  # Reduce from 32
)
```

**Solution 2 - Use smaller models**:
```python
embedding_pipeline = EmbeddingPipeline(
    model_names=['all-MiniLM-L6-v2']  # Smallest model
)
```

**Solution 3 - Reduce documents**:
Keep only essential documents in `projects/` folder.

### Issue: "No documents found in projects"

**Solution**:
- The pipeline auto-creates sample documents
- Or manually add .pdf, .docx, .xlsx, .pptx, .md files

### Issue: Qdrant collections not found or corrupted

**Solution**:
```bash
# Delete and recreate
rm -rf qdrant_storage/  # Linux/Mac
rmdir /s /q qdrant_storage  # Windows
python main.py  # Recreate
```

### Issue: Very slow embedding (> 5 minutes)

**Likely causes**:
1. Models downloading for first time (10-15 min total, one-time only)
2. Large number of documents
3. Large embedding models being used

**Solution**:
- Wait for first run to complete (models cached locally after)
- Or reduce batch size: `batch_size=16` instead of 32

### Issue: Low accuracy metrics (< 0.5)

**Possible reasons**:
- Query keywords don't match document content
- Documents don't match expected source patterns
- Try with demo documents first

**Solution**:
```bash
# Remove your docs, use demo
rm projects/*
python main.py
```

---

## 🎓 Understanding Metrics

### Top-1 Accuracy (0.0 - 1.0)
- **Meaning**: Is the first result relevant?
- **Calculation**: 1.0 if rank-1 is relevant, else 0.0
- **Use case**: Strictest metric (needs perfect first result)

### Top-3 Accuracy (0.0 - 1.0)
- **Meaning**: Is any of the top-3 results relevant?
- **Calculation**: 1.0 if rank 1, 2, or 3 is relevant
- **Use case**: More forgiving (typical for search)

### Mean Reciprocal Rank (MRR) (0.0 - 1.0)
- **Meaning**: What's the average rank position of first relevant result?
- **Calculation**: 1/rank (e.g., 1/2 = 0.5 if rank 2)
- **Use case**: Balances accuracy with position

### Average Similarity (0.0 - 1.0)
- **Meaning**: How similar are relevant results?
- **Calculation**: Mean cosine similarity of relevant chunks
- **Use case**: Quality of semantic matching

### Combined Score (Ranking)
- **Calculation**: Top-3 Accuracy + MRR
- **Purpose**: Overall best configuration
- **Range**: 0.0 - 2.0

---

## 💡 Pro Tips

1. **First Run**: Takes longer (downloads ~2GB of embedding models)
2. **Subsequent Runs**: Much faster (models cached locally)
3. **Small Documents**: Better for testing (< 10 seconds total)
4. **Large Documents**: More realistic (but slower)
5. **Check results/**: CSV is easiest to open in spreadsheet program

---

## 📚 Next Steps

After running successfully:

1. **Open results**:
   ```bash
   # Windows
   start results/results.csv
   
   # Linux
   libreoffice results/results.csv
   
   # Mac
   open results/results.csv
   ```

2. **Analyze results**:
   - Sort by `combined_score` (highest first)
   - Compare chunking strategies
   - Compare embedding models
   - Note trade-offs

3. **Implement best configuration**:
   - Use top chunker + model in your RAG system
   - Reference the architecture guide for implementation

4. **Extend the project**:
   - Add more documents
   - Add more queries
   - Add custom chunking strategies
   - Add custom evaluation metrics

---

## 📞 Support

**Common Questions**:

Q: How long does it take?
A: ~1-2 minutes for demo, slower for large documents

Q: Can I stop mid-run?
A: Yes (Ctrl+C), but collections will need cleanup

Q: Do I need GPU?
A: No, CPU works fine. GPU speeds up embeddings significantly.

Q: What's the output format?
A: CSV (easy to view) + JSON (programmatic access)

Q: Can I modify queries?
A: Yes, edit data/queries.json before running

---

**Ready to start? Run:**
```bash
python main.py
```

**Or use the automated script:**
```bash
# Windows
run.bat

# Linux/Mac
./run.sh
```

Good luck! 🚀
