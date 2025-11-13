#!/usr/bin/env python3
"""
Chunk corpus JSONL into smaller pieces with proper tokenization.

Usage:
    python scripts/chunk_corpus.py --config configs/base.yaml
    python scripts/chunk_corpus.py --input data/processed/corpus.jsonl --output data/processed/chunks.jsonl
"""
import argparse
import sys
from pathlib import Path

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import warnings

import yaml
from src.chunker import chunk_corpus_file


warnings.filterwarnings("ignore", message="Could not find 'tokenizers'. Falling back to 'tiktoken'.")


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Chunk corpus JSONL into smaller pieces")
    parser.add_argument(
        "--config",
        help="Path to YAML config file (if provided, uses params from config)"
    )
    parser.add_argument(
        "--input",
        help="Path to corpus.jsonl (overrides config)"
    )
    parser.add_argument(
        "--output",
        help="Path to output chunks.jsonl (overrides config)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        help="Chunk size in tokens (overrides config)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        help="Overlap in tokens (overrides config)"
    )
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer name (overrides config)"
    )
    args = parser.parse_args()
    
    if args.config:
        config = load_config(args.config)
        # try to load models config, fallback to default if not found
        try:
            models_config_path = project_root / "configs" / "models.yaml"
            if models_config_path.exists():
                models_config = load_config(str(models_config_path))
                default_tokenizer = models_config.get("tokenizer", {}).get("name", "o200k_base")
            else:
                default_tokenizer = "o200k_base"
        except Exception:
            default_tokenizer = "o200k_base"
        
        input_jsonl = args.input or config["data"]["corpus_jsonl"]
        output_jsonl = args.output or config["data"]["chunks_jsonl"]
        chunk_size = args.chunk_size or config["chunking"]["size_tokens"]
        overlap = args.overlap or config["chunking"]["overlap_tokens"]
        tokenizer = args.tokenizer or config["chunking"].get("tokenizer", default_tokenizer)
        add_title_prefix = config["chunking"].get("add_title_prefix", True)
    else:
        if not args.input or not args.output:
            parser.error("Either --config or both --input and --output must be provided")
        input_jsonl = args.input
        output_jsonl = args.output
        chunk_size = args.chunk_size or 600
        overlap = args.overlap or 150
        tokenizer = args.tokenizer or "o200k_base"
        add_title_prefix = True
    
    # get additional parameters from config
    merge_short = config.get("chunking", {}).get("merge_short_paragraphs", True) if args.config else True
    min_para_length = config.get("chunking", {}).get("min_paragraph_length", 50) if args.config else 50
    use_semantic = config.get("chunking", {}).get("use_semantic_chunking", True) if args.config else True
    
    chunk_corpus_file(
        input_jsonl,
        output_jsonl,
        chunk_size=chunk_size,
        overlap=overlap,
        tokenizer=tokenizer,
        add_title_prefix=add_title_prefix,
        merge_short_paragraphs=merge_short,
        min_paragraph_length=min_para_length,
        use_semantic_chunking=use_semantic
    )


if __name__ == "__main__":
    main()
