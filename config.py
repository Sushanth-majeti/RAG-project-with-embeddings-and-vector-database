"""
Configuration file for RAG evaluation project.
Adjust these settings to customize the evaluation.
"""

# === DOCUMENT LOADING ===
SUPPORTED_FILE_TYPES = {'.pdf', '.docx', '.xlsx', '.pptx', '.md'}

# === CHUNKING PARAMETERS ===
CHUNKING_CONFIG = {
    'fixed_size': {
        'chunk_size': 512,      # Token size
        'chunk_overlap': 100,   # Token overlap
    },
    'recursive': {
        'chunk_size': 512,
        'chunk_overlap': 100,
    },
    'structure_aware': {
        'chunk_size': 512,
        'chunk_overlap': 50,
        'min_section_size': 100,  # Merge sections smaller than this
    },
    'hybrid': {
        'chunk_size': 512,
        'chunk_overlap': 100,
    },
    'table_aware': {
        'chunk_size': 512,
        'chunk_overlap': 50,
    },
}

# === EMBEDDING MODELS ===
EMBEDDING_MODELS = {
    'all-MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2',
    'all-mpnet-base-v2': 'sentence-transformers/all-mpnet-base-v2',
    'e5-base-v2': 'intfloat/e5-base-v2',
    'bge-base-en-v1.5': 'BAAI/bge-base-en-v1.5',
    'e5-large-v2': 'intfloat/e5-large-v2',
}

# === RETRIEVAL PARAMETERS ===
RETRIEVAL_CONFIG = {
    'top_k': 10,              # Number of results to retrieve per query
    'batch_size': 32,         # Batch size for embedding
    'similarity_metric': 'cosine',
}

# === EVALUATION PARAMETERS ===
EVALUATION_CONFIG = {
    'top_1_k': 1,
    'top_3_k': 3,
    'mrr_consideration': 10,  # Consider top-10 for MRR
}

# === VECTOR DATABASE ===
QDRANT_CONFIG = {
    'path': './qdrant_storage',
    'distance': 'cosine',
}

# === REPRODUCIBILITY ===
RANDOM_SEED = 42

# === LOGGING ===
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# === PATHS ===
PROJECTS_FOLDER = './projects'
RESULTS_FOLDER = './results'
DATA_FOLDER = './data'
QUERIES_FILE = './data/queries.json'
