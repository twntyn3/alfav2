#!/usr/bin/env python3
"""
Build corpus JSONL from websites CSV.

Usage:
    python scripts/build_corpus.py --config configs/base.yaml
    python scripts/build_corpus.py --input data/raw/websites_updated.csv --output data/processed/corpus.jsonl
"""
import argparse
import sys
from pathlib import Path

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from src.ingest import build_corpus


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Build corpus JSONL from websites CSV")
    parser.add_argument(
        "--config",
        help="Path to YAML config file (if provided, uses paths from config)"
    )
    parser.add_argument(
        "--input",
        help="Path to websites_updated.csv (overrides config)"
    )
    parser.add_argument(
        "--output",
        help="Path to output corpus.jsonl (overrides config)"
    )
    args = parser.parse_args()
    
    if args.config:
        config = load_config(args.config)
        input_csv = args.input or config["data"]["raw_websites"]
        output_jsonl = args.output or config["data"]["corpus_jsonl"]
        # Get normalization settings from config
        text_processing = config.get("text_processing", {})
        normalize_for_search = text_processing.get("normalize_for_retrieval", True)
        normalization_mode = text_processing.get("normalization_mode", "smart")
    else:
        if not args.input or not args.output:
            parser.error("Either --config or both --input and --output must be provided")
        input_csv = args.input
        output_jsonl = args.output
        normalize_for_search = True
        normalization_mode = "letters_numbers"
    
    build_corpus(input_csv, output_jsonl, normalize_for_search=normalize_for_search, normalization_mode=normalization_mode)


if __name__ == "__main__":
    main()
