"""Chunk documents into smaller pieces with proper tokenization and smart paragraph merging."""
import logging
from typing import Dict, List, Optional

try:
    import chonkie
except ImportError:
    raise ImportError(
        "chonkie is required for tokenization. Install with: pip install chonkie"
    )

from src.utils import logger, load_jsonl, save_jsonl
from src.text_processor import merge_short_paragraphs, clean_text
from src.semantic_chunker import SemanticChunker as SemanticChunkerBase


class DocumentChunker:
    """
    Chunk documents using token-based chunking with overlap.
    
    Algorithm: Uses chonkie.TokenChunker for proper tokenization.
    Time Complexity: O(n * m) where n is number of docs, m is avg tokens per doc
    Space Complexity: O(n * k) where k is avg chunks per doc
    """
    
    def __init__(
        self,
        chunk_size: int = 600,
        overlap: int = 150,
        tokenizer: str = "o200k_base",
        add_title_prefix: bool = True,
        merge_short_paragraphs: bool = True,
        min_paragraph_length: int = 50,
        use_semantic_chunking: bool = True
    ):
        """
        Initialize chunker with advanced text processing.
        
        Args:
            chunk_size: Target number of tokens per chunk
            overlap: Number of overlapping tokens between chunks
            tokenizer: Tokenizer name (o200k_base, word, or model path)
            add_title_prefix: Whether to prepend title to each chunk
            merge_short_paragraphs: Whether to merge short paragraphs before chunking
            min_paragraph_length: Minimum character length for standalone paragraph
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.add_title_prefix = add_title_prefix
        self.merge_short_paragraphs = merge_short_paragraphs
        self.min_paragraph_length = min_paragraph_length
        self.use_semantic_chunking = use_semantic_chunking
        
        # initialize chunker (semantic or token-based)
        if use_semantic_chunking:
            overlap_ratio = overlap / chunk_size if chunk_size > 0 else 0.15
            self.semantic_chunker = SemanticChunkerBase(
                chunk_size=chunk_size,
                overlap_ratio=overlap_ratio,
                tokenizer=tokenizer,
                add_title_prefix=add_title_prefix
            )
            self.chunker = None
        else:
            # initialize token chunker
            self.chunker = chonkie.TokenChunker(
                tokenizer=tokenizer,
                chunk_size=chunk_size
            )
            self.semantic_chunker = None
        
        logger.info(f"Initialized chunker: size={chunk_size}, overlap={overlap}, tokenizer={tokenizer}, semantic={use_semantic_chunking}")
    
    def _extract_title_and_text(self, doc: Dict[str, str]) -> tuple:
        """
        Extract title and text from document.
        Supports both new format (separate fields) and old format (contents field).
        """
        # new format: separate title and text fields
        if "title" in doc and "text" in doc:
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()
            return title, text
        
        # old format: contents field with title\ntext
        if "contents" in doc:
            contents = doc.get("contents", "")
            if not isinstance(contents, str):
                contents = str(contents) if contents is not None else ""
            try:
                lines = contents.split("\n", 1)
                title = lines[0].strip() if lines else ""
                text = lines[1].strip() if len(lines) > 1 else ""
                return title, text
            except (AttributeError, IndexError):
                return "", ""
        
        return "", ""
    
    def chunk_document(self, doc: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Chunk a single document with advanced processing.
        
        Args:
            doc: Document with "id", "title", "text" (or "contents") fields
            
        Returns:
            List of chunks with "id", "doc_id", "title", "text", "contents" fields
        """
        # Safely extract doc_id
        doc_id = str(doc.get("id", doc.get("doc_id", "unknown")))
        if doc_id == "unknown":
            logger.warning("Document without id field, skipping")
            return []
        
        title, text = self._extract_title_and_text(doc)
        
        if not title and not text:
            logger.warning(f"Empty document {doc_id}, skipping")
            return []
        
        if not text:
            # if no text, use title as text
            text = title
            title = ""
        
        # use semantic chunking if enabled
        if self.use_semantic_chunking and self.semantic_chunker:
            return self.semantic_chunker.chunk_document(doc)
        
        # merge short paragraphs for better chunking
        if self.merge_short_paragraphs and text:
            text = merge_short_paragraphs(text, min_length=self.min_paragraph_length)
        
        # additional cleaning before chunking
        text = clean_text(text, preserve_structure=True)
        
        if not text.strip():
            logger.warning(f"Empty text after cleaning for doc {doc_id}, skipping")
            return []
        
        # chunk the text
        if not self.chunker:
            raise ValueError("Chunker not initialized")
        chunks = self.chunker.chunk(text)
        
        result = []
        
        for idx, chunk_obj in enumerate(chunks):
            chunk_text = chunk_obj.text.strip()
            if not chunk_text:
                continue
            
            # additional cleaning of chunk
            chunk_text = clean_text(chunk_text, preserve_structure=True)
            if not chunk_text:
                continue
            
            # build chunk contents (for FlashRAG compatibility)
            if self.add_title_prefix and title:
                chunk_contents = f"{title}\n{chunk_text}"
            else:
                chunk_contents = chunk_text
            
            # create chunk with separate title field for LLM processing
            chunk = {
                "id": f"{doc_id}_{idx}",
                "doc_id": doc_id,
                "title": title if self.add_title_prefix else "",
                "text": chunk_text,
                "contents": chunk_contents  # for FlashRAG compatibility
            }
            
            result.append(chunk)
        
        return result
    
    def chunk_corpus(self, input_jsonl: str, output_jsonl: str) -> None:
        """
        Chunk entire corpus from JSONL file.
        
        Args:
            input_jsonl: Path to corpus.jsonl
            output_jsonl: Path to output chunks.jsonl
        """
        logger.info(f"Loading corpus from {input_jsonl}")
        documents = load_jsonl(input_jsonl)
        
        logger.info(f"Chunking {len(documents)} documents...")
        all_chunks = []
        total_chunks = 0
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            total_chunks += len(chunks)
        
        logger.info(f"Generated {total_chunks} chunks from {len(documents)} documents")
        save_jsonl(all_chunks, output_jsonl)
        logger.info(f"Saved chunks to {output_jsonl}")


def chunk_corpus_file(
    input_jsonl: str,
    output_jsonl: str,
    chunk_size: int = 600,
    overlap: int = 150,
    tokenizer: str = "o200k_base",
    add_title_prefix: bool = True,
    merge_short_paragraphs: bool = True,
    min_paragraph_length: int = 50,
    use_semantic_chunking: bool = True
) -> None:
    """
    Convenience function to chunk corpus file.
    
    Args:
        input_jsonl: Path to corpus.jsonl
        output_jsonl: Path to output chunks.jsonl
        chunk_size: Target tokens per chunk
        overlap: Overlapping tokens
        tokenizer: Tokenizer name
        add_title_prefix: Prepend title to chunks
    """
    chunker = DocumentChunker(
        chunk_size=chunk_size,
        overlap=overlap,
        tokenizer=tokenizer,
        add_title_prefix=add_title_prefix,
        merge_short_paragraphs=merge_short_paragraphs,
        min_paragraph_length=min_paragraph_length,
        use_semantic_chunking=use_semantic_chunking
    )
    chunker.chunk_corpus(input_jsonl, output_jsonl)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chunk corpus JSONL into smaller pieces")
    parser.add_argument("--input", required=True, help="Path to corpus.jsonl")
    parser.add_argument("--output", required=True, help="Path to output chunks.jsonl")
    parser.add_argument("--chunk_size", type=int, default=600, help="Chunk size in tokens")
    parser.add_argument("--overlap", type=int, default=150, help="Overlap in tokens")
    parser.add_argument("--tokenizer", default="o200k_base", help="Tokenizer name")
    parser.add_argument("--add_title_prefix", action="store_true", default=True, help="Add title prefix")
    args = parser.parse_args()
    
    chunk_corpus_file(
        args.input,
        args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        tokenizer=args.tokenizer,
        add_title_prefix=args.add_title_prefix
    )

