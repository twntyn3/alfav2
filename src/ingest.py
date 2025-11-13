"""Ingest raw CSV data and convert to corpus JSONL format with advanced text processing."""
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

from src.utils import logger, save_jsonl
from src.text_processor import (
    extract_and_clean_title,
    extract_and_clean_text,
    validate_document
)
from src.table_processor import extract_table_structure

# increase csv field size limit for large text fields
limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(limit)
        break
    except OverflowError:
        limit = limit // 10


def read_websites_csv(csv_path: str, validate: bool = True) -> list[Dict[str, str]]:
    """
    Read websites CSV and convert to corpus format with advanced processing.
    
    Args:
        csv_path: Path to websites_updated.csv
        validate: Whether to validate document quality
        
    Returns:
        List of documents in format:
        {
            "id": web_id,
            "title": cleaned_title,
            "text": cleaned_text,
            "contents": title + "\\n" + text  # for compatibility
        }
        
    Time Complexity: O(n) where n is number of rows
    Space Complexity: O(n) for storing all documents
    """
    corpus = []
    skipped = 0
    skipped_reasons = {}
    
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):  # start=2 because header is row 1
            try:
                web_id = str(row.get("web_id", "")).strip()
                raw_title = (row.get("title") or "").strip()
                raw_text = (row.get("text") or "").strip()
                
                if not web_id:
                    skipped += 1
                    skipped_reasons["missing_web_id"] = skipped_reasons.get("missing_web_id", 0) + 1
                    continue
                
                # clean and normalize text
                title = extract_and_clean_title(raw_title)
                text = extract_and_clean_text(raw_text)
                
                # process tables to preserve structure
                text, has_table = extract_table_structure(text)
                
                # validate document quality
                if validate:
                    is_valid, error_msg = validate_document(title, text)
                    if not is_valid:
                        skipped += 1
                        reason = error_msg or "validation_failed"
                        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                        if row_num <= 10:  # log first few for debugging
                            logger.debug(f"Row {row_num} (web_id={web_id}): {error_msg}")
                        continue
                
                # if no title but has text, use first sentence as title
                if not title and text:
                    first_sentence = text.split(".", 1)[0].strip()
                    if len(first_sentence) < 100:
                        title = first_sentence
                        text = text[len(first_sentence):].lstrip(". ")
                
                # ensure we have content
                if not title and not text:
                    skipped += 1
                    skipped_reasons["empty_content"] = skipped_reasons.get("empty_content", 0) + 1
                    continue
                
                # combine for compatibility (FlashRAG format)
                contents = (title + "\n" + text).strip() if title else text
                
                corpus.append({
                    "id": web_id,
                    "title": title,
                    "text": text,
                    "contents": contents  # for FlashRAG compatibility
                })
                
            except Exception as e:
                skipped += 1
                skipped_reasons[f"error_{type(e).__name__}"] = skipped_reasons.get(f"error_{type(e).__name__}", 0) + 1
                logger.warning(f"Error processing row {row_num}: {e}")
                continue
    
    if skipped > 0:
        logger.info(f"Skipped {skipped} rows. Reasons: {dict(skipped_reasons)}")
    
    logger.info(f"Loaded {len(corpus)} documents from {csv_path}")
    return corpus


def build_corpus(input_csv: str, output_jsonl: str) -> None:
    """
    Build corpus JSONL from websites CSV.
    
    Args:
        input_csv: Path to websites_updated.csv
        output_jsonl: Path to output corpus.jsonl
    """
    logger.info(f"Building corpus from {input_csv}")
    corpus = read_websites_csv(input_csv)
    save_jsonl(corpus, output_jsonl)
    logger.info(f"Saved {len(corpus)} documents to {output_jsonl}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build corpus JSONL from websites CSV")
    parser.add_argument("--input", required=True, help="Path to websites_updated.csv")
    parser.add_argument("--output", required=True, help="Path to output corpus.jsonl")
    args = parser.parse_args()
    
    build_corpus(args.input, args.output)

