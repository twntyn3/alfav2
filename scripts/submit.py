#!/usr/bin/env python3
"""
Generate submit.csv from retrieval results.

Usage:
    python scripts/submit.py --config configs/base.yaml
"""
import argparse
import csv
import json
import sys
from pathlib import Path

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "FlashRAG"))

import yaml
from src.retriever import HybridRetriever
from src.reranker import Reranker
from src.utils import logger, get_timestamp


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_questions(csv_path: str) -> list[tuple[int, str]]:
    """Read questions from CSV file."""
    import csv as csv_module
    questions = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            q_id = int(row.get("q_id", 0))
            query = row.get("query", "").strip()
            if q_id and query:
                questions.append((q_id, query))
    return questions


def main():
    parser = argparse.ArgumentParser(description="Generate submit.csv")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to config file")
    parser.add_argument("--output", help="Path to output submit.csv (overrides config)")
    args = parser.parse_args()
    
    # load configs
    config = load_config(args.config)
    models_config = load_config(project_root / "configs" / "models.yaml")
    
    # get paths
    questions_path = config["data"]["raw_questions"]
    faiss_dir = config["indexes"]["faiss_dir"]
    bm25_dir = config["indexes"]["bm25_dir"]
    submits_dir = config["outputs"]["submits_dir"]
    
    # get parameters
    retrieval_config = config["retrieval"]
    k_retrieve = retrieval_config["k_retrieve"]
    k_final = retrieval_config["k_final"]
    use_reranker = retrieval_config.get("use_reranker", True)
    weight_dense = retrieval_config["hybrid_weight_dense"]
    weight_bm25 = retrieval_config["hybrid_weight_bm25"]
    
    embeddings_config = models_config["embeddings"]
    reranker_config = models_config["reranker"]
    
    # read questions
    logger.info(f"Reading questions from {questions_path}")
    questions = read_questions(questions_path)
    logger.info(f"Loaded {len(questions)} questions")
    
    # find FAISS index file
    faiss_path = Path(faiss_dir)
    faiss_index_files = list(faiss_path.glob("*.index"))
    if not faiss_index_files:
        raise FileNotFoundError(f"No FAISS index found in {faiss_dir}")
    faiss_index_path = faiss_index_files[0]
    faiss_meta_path = faiss_path / "faiss_meta.json"
    
    # initialize retriever
    logger.info("Initializing hybrid retriever...")
    retriever = HybridRetriever(
        faiss_index_path=str(faiss_index_path),
        faiss_meta_path=str(faiss_meta_path) if faiss_meta_path.exists() else None,
        bm25_index_path=bm25_dir,
        embedding_model_name=embeddings_config["model_name"],
        weight_dense=weight_dense,
        weight_bm25=weight_bm25,
        device=embeddings_config["device"],
        normalize_embeddings=embeddings_config.get("normalize_embeddings", True)
    )
    
    # initialize reranker if enabled
    reranker = None
    if use_reranker:
        logger.info("Initializing reranker...")
        reranker = Reranker(
            model_name=reranker_config["model_name"],
            device=reranker_config["device"],
            batch_size=reranker_config["batch_size"]
        )
    
    # process questions with progress bar
    logger.info("Processing questions...")
    results = {}
    
    try:
        from tqdm import tqdm
        question_iter = tqdm(questions, desc="Processing questions")
    except ImportError:
        question_iter = questions
        logger.warning("tqdm not available, progress bar disabled")
    
    for q_id, query in question_iter:
        # retrieve
        candidates = retriever.retrieve(query, k=k_retrieve, return_scores=False)
        
        # rerank if enabled
        if reranker and candidates:
            candidates = reranker.rerank(query, candidates, top_k=k_final, return_scores=False)
        
        # extract web_ids (doc_ids) - ensure exactly 5
        web_ids = [int(c["doc_id"]) for c in candidates[:k_final]]
        
        # pad or truncate to exactly 5
        while len(web_ids) < k_final:
            web_ids.append(web_ids[-1] if web_ids else 1)  # repeat last or use default
        web_ids = web_ids[:k_final]
        
        results[q_id] = web_ids
    
    # generate submit.csv
    timestamp = get_timestamp()
    output_path = Path(args.output) if args.output else Path(submits_dir) / f"submit_{timestamp}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Writing submit.csv to {output_path}")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["q_id", "web_list"])
        for q_id in sorted(results.keys()):
            web_list_str = json.dumps(results[q_id], ensure_ascii=False)
            writer.writerow([q_id, web_list_str])
    
    logger.info(f"Submit file written: {output_path}")
    logger.info(f"Total questions: {len(results)}")
    logger.info(f"Web IDs per question: {k_final}")


if __name__ == "__main__":
    main()
