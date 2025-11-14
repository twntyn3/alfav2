"""Semantic-structural chunking with table and heading detection."""
import re
import logging
from typing import Dict, List, Tuple

try:
    import chonkie
except ImportError:
    raise ImportError("chonkie is required for tokenization")

from src.utils import logger
from src.text_processor import clean_text, merge_short_paragraphs
from src.table_processor import detect_table, extract_table_structure, preserve_table_in_chunk


class SemanticChunker:
    """
    Semantic-structural chunker that respects document structure.
    
    Features:
    - Detects and preserves table structures
    - Respects heading hierarchy (h1-h3 patterns)
    - Chunks by semantic boundaries (paragraphs, sections)
    - Dynamic chunk sizing (400-800 tokens) with 10-20% overlap
    """
    
    def __init__(
        self,
        chunk_size: int = 600,
        overlap_ratio: float = 0.15,  # 15% overlap
        tokenizer: str = "o200k_base",
        add_title_prefix: bool = True,
        min_chunk_size: int = 200,
        max_chunk_size: int = 800
    ):
        """
        Initialize semantic chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            overlap_ratio: Overlap as ratio of chunk_size (0.1-0.2)
            tokenizer: Tokenizer name
            add_title_prefix: Add title to each chunk
            min_chunk_size: Minimum chunk size in tokens
            max_chunk_size: Maximum chunk size in tokens
        """
        self.chunk_size = chunk_size
        self.overlap = int(chunk_size * overlap_ratio)
        self.add_title_prefix = add_title_prefix
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Initialize token chunker
        self.chunker = chonkie.TokenChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size
        )
        
        logger.info(f"Initialized semantic chunker: size={chunk_size}, overlap={self.overlap} ({overlap_ratio*100:.0f}%)")
    
    def _detect_headings(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Detect heading patterns in text.
        
        Returns:
            List of (position, level, heading_text)
        """
        headings = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Markdown headings: # ## ###
            if line_stripped.startswith('#'):
                level = len(line_stripped) - len(line_stripped.lstrip('#'))
                heading_text = line_stripped.lstrip('#').strip()
                if heading_text and level <= 3:
                    headings.append((i, level, heading_text))
            
            # ALL CAPS lines (likely headings)
            elif line_stripped.isupper() and len(line_stripped) > 3 and len(line_stripped) < 100:
                headings.append((i, 2, line_stripped))
            
            # Lines ending with colon (likely section headers)
            elif line_stripped.endswith(':') and len(line_stripped) < 80:
                headings.append((i, 3, line_stripped))
        
        return headings
    
    def _split_by_structure(self, text: str) -> List[str]:
        """
        Split text by semantic structure (headings, tables, paragraphs).
        
        Returns:
            List of text segments
        """
        segments = []
        lines = text.split('\n')
        
        # Detect headings
        headings = self._detect_headings(text)
        heading_positions = {pos: (level, text) for pos, level, text in headings}
        
        # Detect tables
        current_segment = []
        in_table = False
        
        for i, line in enumerate(lines):
            is_heading = i in heading_positions
            is_table_line = detect_table(line)
            
            # Start new segment on heading
            if is_heading:
                if current_segment:
                    segments.append('\n'.join(current_segment))
                current_segment = [line]
                in_table = False
            # Table boundaries
            elif is_table_line:
                if not in_table and current_segment:
                    # End previous segment before table
                    segments.append('\n'.join(current_segment))
                    current_segment = []
                in_table = True
                current_segment.append(line)
            # End of table
            elif in_table and not is_table_line and line.strip():
                # Table ended, start new segment
                if current_segment:
                    segments.append('\n'.join(current_segment))
                current_segment = [line]
                in_table = False
            else:
                current_segment.append(line)
        
        # Add remaining segment
        if current_segment:
            segments.append('\n'.join(current_segment))
        
        # If no structure detected, return original text
        if not segments:
            return [text]
        
        return segments
    
    def chunk_document(self, doc: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Chunk document with semantic-structural awareness.
        
        Args:
            doc: Document with "id", "title", "text" fields
            
        Returns:
            List of chunks with semantic structure preserved
        """
        # Safely extract doc_id
        doc_id = str(doc.get("id", doc.get("doc_id", "unknown")))
        if doc_id == "unknown":
            logger.warning("Document without id field, skipping")
            return []
        
        title = doc.get("title", "").strip()
        text = doc.get("text", "").strip()
        
        if not title and not text:
            return []
        
        if not text:
            text = title
            title = ""
        
        # Clean text
        text = clean_text(text, preserve_structure=True)
        
        # Process tables
        text, has_tables = extract_table_structure(text)
        
        # Split by semantic structure
        segments = self._split_by_structure(text)
        
        # Chunk each segment
        all_chunks = []
        for seg_idx, segment in enumerate(segments):
            if not segment.strip():
                continue
            
            # Chunk segment
            chunk_objects = self.chunker.chunk(segment)
            
            for chunk_idx, chunk_obj in enumerate(chunk_objects):
                chunk_text = chunk_obj.text.strip()
                if not chunk_text:
                    continue
                
                # Preserve table structure if present
                if has_tables:
                    chunk_text = preserve_table_in_chunk(segment, chunk_text)
                
                # Additional cleaning
                chunk_text = clean_text(chunk_text, preserve_structure=True)
                if not chunk_text:
                    continue
                
                # Build chunk contents
                if self.add_title_prefix and title:
                    chunk_contents = f"{title}\n{chunk_text}"
                else:
                    chunk_contents = chunk_text
                
                chunk = {
                    "id": f"{doc_id}_{seg_idx}_{chunk_idx}",
                    "doc_id": doc_id,
                    "title": title if self.add_title_prefix else "",
                    "text": chunk_text,
                    "contents": chunk_contents,
                    "has_table": has_tables and detect_table(chunk_text)
                }
                
                all_chunks.append(chunk)
        
        return all_chunks

