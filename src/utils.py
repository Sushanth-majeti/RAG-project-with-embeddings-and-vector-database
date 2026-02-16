"""
Utility functions for RAG evaluation project.
"""
import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a document chunk."""
    chunk_id: str
    content: str
    source_file: str
    chunk_index: int
    strategy: str
    metadata: Dict[str, Any]


def get_token_count(text: str) -> int:
    """
    Approximate token count using whitespace-based method.
    OpenAI uses ~1.3 characters per token on average.
    """
    return len(text.split()) + len(text) // 4  # Simple heuristic


def save_json(data: Any, filepath: str) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to {filepath}")


def load_json(filepath: str) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded from {filepath}")
    return data


def get_project_root() -> str:
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dir(dirpath: str) -> None:
    """Ensure directory exists."""
    os.makedirs(dirpath, exist_ok=True)
