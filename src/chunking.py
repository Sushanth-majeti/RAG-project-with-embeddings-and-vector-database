"""
Five different chunking strategies for RAG systems.
"""
import re
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from src.utils import Chunk, get_token_count

logger = logging.getLogger(__name__)


@dataclass
class ChunkingStrategy:
    """Base class for chunking strategies."""
    name: str
    chunk_size: int = 512
    chunk_overlap: int = 100

    def chunk(self, text: str, source_file: str) -> List[Chunk]:
        """
        Chunk text according to strategy.
        Must be implemented by subclasses.
        """
        raise NotImplementedError


class FixedSizeChunker(ChunkingStrategy):
    """
    Fixed-size chunking strategy.
    - Chunk size: 512 tokens
    - Overlap: 100 tokens
    """

    def __init__(self):
        super().__init__(name="fixed_size", chunk_size=512, chunk_overlap=100)

    def chunk(self, text: str, source_file: str) -> List[Chunk]:
        """Split text into fixed-size chunks with overlap."""
        chunks = []
        sentences = self._split_into_sentences(text)

        current_chunk = []
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = get_token_count(sentence)

            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(
                    Chunk(
                        chunk_id=f"{source_file}_{self.name}_{chunk_index}",
                        content=chunk_text,
                        source_file=source_file,
                        chunk_index=chunk_index,
                        strategy=self.name,
                        metadata={
                            'chunk_size': get_token_count(chunk_text),
                            'sentence_count': len(current_chunk),
                        }
                    )
                )
                chunk_index += 1

                # Create overlap: keep last ~100 tokens
                overlap_tokens = 0
                overlap_chunk = []
                for sent in reversed(current_chunk):
                    overlap_tokens += get_token_count(sent)
                    overlap_chunk.insert(0, sent)
                    if overlap_tokens >= self.chunk_overlap:
                        break
                current_chunk = overlap_chunk
                current_tokens = overlap_tokens

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(
                Chunk(
                    chunk_id=f"{source_file}_{self.name}_{chunk_index}",
                    content=chunk_text,
                    source_file=source_file,
                    chunk_index=chunk_index,
                    strategy=self.name,
                    metadata={
                        'chunk_size': get_token_count(chunk_text),
                        'sentence_count': len(current_chunk),
                    }
                )
            )

        logger.info(f"Fixed-size chunking: {len(chunks)} chunks from {source_file}")
        return chunks

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class RecursiveChunker(ChunkingStrategy):
    """
    Recursive chunking: Paragraph → Sentence → Fixed-size fallback.
    Hierarchical chunking that respects document structure.
    """

    def __init__(self):
        super().__init__(name="recursive", chunk_size=512, chunk_overlap=100)

    def chunk(self, text: str, source_file: str) -> List[Chunk]:
        """Chunk recursively: paragraphs → sentences → fixed-size fallback."""
        chunks = []
        paragraphs = text.split('\n\n')
        chunk_index = 0

        for para in paragraphs:
            if not para.strip():
                continue

            para_tokens = get_token_count(para)

            # If paragraph fits, add as single chunk
            if para_tokens <= self.chunk_size:
                chunks.append(
                    Chunk(
                        chunk_id=f"{source_file}_{self.name}_{chunk_index}",
                        content=para.strip(),
                        source_file=source_file,
                        chunk_index=chunk_index,
                        strategy=self.name,
                        metadata={
                            'chunk_size': para_tokens,
                            'level': 'paragraph'
                        }
                    )
                )
                chunk_index += 1
            else:
                # Split into sentences
                sentences = self._split_into_sentences(para)
                sentence_chunks = self._combine_sentences(
                    sentences, source_file, chunk_index
                )
                chunks.extend(sentence_chunks)
                chunk_index += len(sentence_chunks)

        logger.info(f"Recursive chunking: {len(chunks)} chunks from {source_file}")
        return chunks

    def _combine_sentences(self, sentences: List[str], source_file: str,
                          start_index: int) -> List[Chunk]:
        """Combine sentences into chunks respecting chunk_size."""
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = get_token_count(sent)

            if current_tokens + sent_tokens > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(
                    Chunk(
                        chunk_id=f"{source_file}_{self.name}_{start_index + len(chunks)}",
                        content=chunk_text,
                        source_file=source_file,
                        chunk_index=start_index + len(chunks),
                        strategy=self.name,
                        metadata={
                            'chunk_size': get_token_count(chunk_text),
                            'level': 'sentence',
                            'sentence_count': len(current_chunk)
                        }
                    )
                )
                current_chunk = []
                current_tokens = 0

            current_chunk.append(sent)
            current_tokens += sent_tokens

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(
                Chunk(
                    chunk_id=f"{source_file}_{self.name}_{start_index + len(chunks)}",
                    content=chunk_text,
                    source_file=source_file,
                    chunk_index=start_index + len(chunks),
                    strategy=self.name,
                    metadata={
                        'chunk_size': get_token_count(chunk_text),
                        'level': 'sentence',
                        'sentence_count': len(current_chunk)
                    }
                )
            )

        return chunks

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class StructureAwareChunker(ChunkingStrategy):
    """
    Structure-aware chunking: Chunk by headings (H1/H2/H3).
    Preserves document hierarchy. Merges small sections if below threshold.
    """

    def __init__(self):
        super().__init__(name="structure_aware", chunk_size=512, chunk_overlap=50)

    def chunk(self, text: str, source_file: str) -> List[Chunk]:
        """Chunk by document structure (headings)."""
        chunks = []
        sections = self._extract_sections(text)
        chunk_index = 0

        for heading, content, level in sections:
            if not content.strip():
                continue

            content_tokens = get_token_count(content)
            heading_str = f"{'#' * level} {heading}" if heading else ""

            # If content is small, merge with previous chunk if possible
            if content_tokens < 100 and chunks and heading:
                # Append to previous chunk
                prev_chunk = chunks[-1]
                merged_text = f"{prev_chunk.content}\n\n{heading_str}\n{content}"
                if get_token_count(merged_text) <= self.chunk_size * 1.5:
                    prev_chunk.content = merged_text
                    prev_chunk.metadata['merged'] = True
                    continue

            # If content is large, split by sentences
            if content_tokens > self.chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', content)
                current = [heading_str] if heading else []
                current_tokens = get_token_count('\n'.join(current))

                for sent in sentences:
                    sent_tokens = get_token_count(sent)
                    if current_tokens + sent_tokens > self.chunk_size and current:
                        chunk_text = '\n'.join(current).strip()
                        chunks.append(
                            Chunk(
                                chunk_id=f"{source_file}_{self.name}_{chunk_index}",
                                content=chunk_text,
                                source_file=source_file,
                                chunk_index=chunk_index,
                                strategy=self.name,
                                metadata={
                                    'chunk_size': get_token_count(chunk_text),
                                    'heading': heading,
                                    'level': level
                                }
                            )
                        )
                        chunk_index += 1
                        current = []
                        current_tokens = 0

                    current.append(sent)
                    current_tokens += sent_tokens

                if current:
                    chunk_text = '\n'.join(current).strip()
                    chunks.append(
                        Chunk(
                            chunk_id=f"{source_file}_{self.name}_{chunk_index}",
                            content=chunk_text,
                            source_file=source_file,
                            chunk_index=chunk_index,
                            strategy=self.name,
                            metadata={
                                'chunk_size': get_token_count(chunk_text),
                                'heading': heading,
                                'level': level
                            }
                        )
                    )
                    chunk_index += 1
            else:
                chunk_text = f"{heading_str}\n{content}".strip()
                chunks.append(
                    Chunk(
                        chunk_id=f"{source_file}_{self.name}_{chunk_index}",
                        content=chunk_text,
                        source_file=source_file,
                        chunk_index=chunk_index,
                        strategy=self.name,
                        metadata={
                            'chunk_size': content_tokens,
                            'heading': heading,
                            'level': level
                        }
                    )
                )
                chunk_index += 1

        logger.info(f"Structure-aware chunking: {len(chunks)} chunks from {source_file}")
        return chunks

    @staticmethod
    def _extract_sections(text: str) -> List[Tuple[str, str, int]]:
        """Extract sections by markdown headings."""
        sections = []
        heading_pattern = r'^(#{1,3})\s+(.+)$'

        lines = text.split('\n')
        current_heading = None
        current_level = 0
        current_content = []

        for line in lines:
            match = re.match(heading_pattern, line)
            if match:
                # Save previous section
                if current_content:
                    sections.append((current_heading or "", '\n'.join(current_content), current_level))
                current_level = len(match.group(1))
                current_heading = match.group(2)
                current_content = []
            else:
                current_content.append(line)

        # Add final section
        if current_content:
            sections.append((current_heading or "", '\n'.join(current_content), current_level or 1))

        return sections if sections else [("", text, 1)]


class HybridChunker(ChunkingStrategy):
    """
    Hybrid chunking: Structure-aware + Recursive fallback.
    Uses structure when available, falls back to recursive chunking.
    """

    def __init__(self):
        super().__init__(name="hybrid", chunk_size=512, chunk_overlap=100)
        self.structure_chunker = StructureAwareChunker()
        self.recursive_chunker = RecursiveChunker()

    def chunk(self, text: str, source_file: str) -> List[Chunk]:
        """
        Try structure-aware chunking first.
        If not enough structure, use recursive chunking.
        """
        # Try structure-aware first
        structure_chunks = self.structure_chunker.chunk(text, source_file)

        # If many small chunks or no clear structure, use recursive
        if len(structure_chunks) > 50 or all(
            chunk.metadata.get('level', 1) == 1 for chunk in structure_chunks
        ):
            result_chunks = self.recursive_chunker.chunk(text, source_file)
        else:
            result_chunks = structure_chunks

        # Rename to hybrid
        for chunk in result_chunks:
            chunk.strategy = self.name
            chunk.chunk_id = f"{source_file}_{self.name}_{chunk.chunk_index}"

        logger.info(f"Hybrid chunking: {len(result_chunks)} chunks from {source_file}")
        return result_chunks


class TableAwareChunker(ChunkingStrategy):
    """
    Table-aware chunking: Detects tables and keeps them as single semantic chunks.
    Useful for PDFs and Excel files with structured tables.
    """

    def __init__(self):
        super().__init__(name="table_aware", chunk_size=512, chunk_overlap=50)

    def chunk(self, text: str, source_file: str) -> List[Chunk]:
        """Chunk while preserving tables as atomic units."""
        chunks = []
        parts = self._split_by_tables(text)
        chunk_index = 0

        for is_table, content in parts:
            if not content.strip():
                continue

            content_tokens = get_token_count(content)

            if is_table or content_tokens <= self.chunk_size:
                # Keep tables and small content as single chunks
                chunks.append(
                    Chunk(
                        chunk_id=f"{source_file}_{self.name}_{chunk_index}",
                        content=content,
                        source_file=source_file,
                        chunk_index=chunk_index,
                        strategy=self.name,
                        metadata={
                            'chunk_size': content_tokens,
                            'is_table': is_table,
                            'chunk_type': 'table' if is_table else 'text'
                        }
                    )
                )
                chunk_index += 1
            else:
                # Split large non-table content
                sentences = re.split(r'(?<=[.!?])\s+', content)
                current_chunk = []
                current_tokens = 0

                for sent in sentences:
                    sent_tokens = get_token_count(sent)
                    if current_tokens + sent_tokens > self.chunk_size and current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append(
                            Chunk(
                                chunk_id=f"{source_file}_{self.name}_{chunk_index}",
                                content=chunk_text,
                                source_file=source_file,
                                chunk_index=chunk_index,
                                strategy=self.name,
                                metadata={
                                    'chunk_size': get_token_count(chunk_text),
                                    'is_table': False,
                                    'chunk_type': 'text'
                                }
                            )
                        )
                        chunk_index += 1
                        current_chunk = []
                        current_tokens = 0

                    current_chunk.append(sent)
                    current_tokens += sent_tokens

                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(
                        Chunk(
                            chunk_id=f"{source_file}_{self.name}_{chunk_index}",
                            content=chunk_text,
                            source_file=source_file,
                            chunk_index=chunk_index,
                            strategy=self.name,
                            metadata={
                                'chunk_size': get_token_count(chunk_text),
                                'is_table': False,
                                'chunk_type': 'text'
                            }
                        )
                    )
                    chunk_index += 1

        logger.info(f"Table-aware chunking: {len(chunks)} chunks from {source_file}")
        return chunks

    @staticmethod
    def _split_by_tables(text: str) -> List[Tuple[bool, str]]:
        """
        Split text by table markers.
        Tables are detected as lines with | or common table patterns.
        """
        parts = []
        lines = text.split('\n')

        current_block = []
        is_table = False

        for line in lines:
            # Simple table detection: lines containing | and multiple cells
            line_is_table = '|' in line and line.count('|') >= 2

            if line_is_table != is_table:
                # Boundary between table and non-table
                if current_block:
                    parts.append((is_table, '\n'.join(current_block)))
                    current_block = []
                is_table = line_is_table

            current_block.append(line)

        if current_block:
            parts.append((is_table, '\n'.join(current_block)))

        return parts


def get_all_chunkers() -> Dict[str, ChunkingStrategy]:
    """Get all available chunking strategies."""
    return {
        'fixed_size': FixedSizeChunker(),
        'recursive': RecursiveChunker(),
        'structure_aware': StructureAwareChunker(),
        'hybrid': HybridChunker(),
        'table_aware': TableAwareChunker(),
    }
