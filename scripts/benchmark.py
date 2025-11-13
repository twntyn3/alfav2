#!/usr/bin/env python3
"""
Quick benchmark script to test retrieval speed.

Usage:
    python scripts/benchmark.py --config configs/base.yaml --n 100
"""
import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "FlashRAG"))

import yaml
import csv
from src.retriever import HybridRetriever
from src.reranker import Reranker
from src.reranker_ensemble import RerankerEnsemble
from src.utils import logger, resolve_device


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Benchmark retrieval speed")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to config file")
    parser.add_argument("--n", type=int, default=100, help="Number of questions to test")
    args = parser.parse_args()
    
    config = load_config(args.config)
    models_config = load_config(project_root / "configs" / "models.yaml")
    
    # get paths
    questions_path = config["data"]["raw_questions"]
    bm25_dir = config["indexes"]["bm25_dir"]
    
    # get parameters
    retrieval_config = config["retrieval"]
    k_retrieve = retrieval_config["k_retrieve"]
    k_final = retrieval_config["k_final"]
    use_reranker = retrieval_config.get("use_reranker", True)
    
    reranker_config = models_config["reranker"]
    faiss_config = models_config.get("faiss", {})
    bm25_corpus_override = retrieval_config.get("bm25_corpus_path")
    
    # read questions
    questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_id = int(row.get("q_id", 0))
            query = row.get("query", "").strip()
            if q_id and query:
                questions.append((q_id, query))
    
    # limit to n questions
    questions = questions[:args.n]
    logger.info(f"Testing with {len(questions)} questions")
    
    # initialize retriever
    logger.info("Initializing hybrid retriever (FAISS + FlashRAG bm25s)...")
    faiss_path = Path(config["indexes"]["faiss_dir"])
    faiss_index_files = list(faiss_path.glob("*.index"))
    if not faiss_index_files:
        raise FileNotFoundError(f"No FAISS index found in {faiss_path}")
    faiss_index_path = faiss_index_files[0]
    faiss_meta_path = faiss_path / "faiss_meta.json"
    if not faiss_meta_path.exists():
        faiss_meta_path = None

    bm25_corpus_path = bm25_corpus_override or str(Path(bm25_dir) / "chunks.jsonl")
    embeddings_config = models_config["embeddings"]
    embedding_device = resolve_device(embeddings_config.get("device", "auto"))
    # Get normalization mode from config
    text_processing = config.get("text_processing", {})
    normalization_mode = text_processing.get("normalization_mode", "smart")
    
    retriever = HybridRetriever(
        faiss_index_path=str(faiss_index_path),
        faiss_meta_path=str(faiss_meta_path) if faiss_meta_path else None,
        bm25_index_dir=bm25_dir,
        bm25_corpus_path=bm25_corpus_path,
        embedding_model_name=embeddings_config["model_name"],
        device=embedding_device,
        normalize_embeddings=embeddings_config.get("normalize_embeddings", True),
        query_batch_size=retrieval_config.get("batch_size", embeddings_config.get("batch_size", 32)),
        weight_dense=retrieval_config.get("hybrid_weight_dense", 0.6),
        weight_bm25=retrieval_config.get("hybrid_weight_bm25", 0.4),
        fusion_method=retrieval_config.get("fusion_method", "weighted"),
        rrf_k=retrieval_config.get("rrf_k", 60),
        enhance_numerics=retrieval_config.get("enhance_numerics", True),
        normalization_mode=normalization_mode,
        min_score_threshold=retrieval_config.get("min_score_threshold", 0.0),
        filter_by_document_type=retrieval_config.get("filter_by_document_type", False),
        prefer_table_chunks=retrieval_config.get("prefer_table_chunks", False),
        embedding_fp16=embeddings_config.get("use_fp16", False),
        faiss_use_gpu=faiss_config.get("use_gpu", False),
        faiss_gpu_device=faiss_config.get("gpu_device"),
        faiss_use_fp16=faiss_config.get("use_float16", False),
    )
    
    # initialize reranker
    reranker = None
    if use_reranker:
        logger.info("Initializing reranker...")
        reranker_device = resolve_device(reranker_config.get("device", "auto"))
        reranker = Reranker(
            model_name=reranker_config["model_name"],
            device=reranker_device,
            batch_size=reranker_config.get("batch_size", 16),
            max_length=reranker_config.get("max_length", 512),
            use_fp16=reranker_config.get("use_fp16", False),
        )
    
    retrieval_batch_size = max(1, retrieval_config.get("batch_size", 32))
    k_rerank = retrieval_config.get("k_rerank", min(k_retrieve, 20))
    
    # benchmark retrieval only
    logger.info("Benchmarking retrieval (without reranking)...")
    start = time.time()
    for batch_start in range(0, len(questions), retrieval_batch_size):
        batch = questions[batch_start:batch_start + retrieval_batch_size]
        batch_queries = [query for _, query in batch]
        retriever.retrieve_batch(
            batch_queries,
            k=k_retrieve,
            return_scores=False,
        )
    retrieval_time = time.time() - start
    retrieval_per_q = retrieval_time / len(questions) if questions else 0.0
    
    # benchmark with reranking
    if reranker:
        logger.info("Benchmarking retrieval + reranking...")
        start = time.time()
        for batch_start in range(0, len(questions), retrieval_batch_size):
            batch = questions[batch_start:batch_start + retrieval_batch_size]
            batch_queries = [query for _, query in batch]
            batch_candidates = retriever.retrieve_batch(
                batch_queries,
                k=k_retrieve,
                return_scores=False,
            )
            rerank_inputs = [
                [dict(candidate) for candidate in candidates[:k_rerank]]
                for candidates in batch_candidates
            ]
            reranker.batch_rerank(
                batch_queries,
                rerank_inputs,
                top_k=k_final,
                return_scores=False
            )
        full_time = time.time() - start
        full_per_q = full_time / len(questions) if questions else 0.0
        rerank_per_q = max(0.0, full_per_q - retrieval_per_q)
    else:
        full_time = retrieval_time
        full_per_q = retrieval_per_q
        rerank_per_q = 0.0
    
    # estimate time for full dataset
    total_questions = 6977
    estimated_full = full_per_q * total_questions
    estimated_retrieval = retrieval_per_q * total_questions
    
    # print results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Tested questions: {len(questions)}")
    print(f"\nRetrieval only:")
    print(f"  Time: {retrieval_time:.1f}s ({retrieval_per_q:.3f}s per question)")
    print(f"\nRetrieval + Reranking:")
    print(f"  Time: {full_time:.1f}s ({full_per_q:.3f}s per question)")
    if reranker:
        print(f"  - Retrieval: {retrieval_per_q:.3f}s per question")
        print(f"  - Reranking: {rerank_per_q:.3f}s per question")
    print(f"\nEstimated time for {total_questions} questions:")
    print(f"  Retrieval only: {estimated_retrieval/60:.1f} minutes")
    print(f"  Full pipeline: {estimated_full/60:.1f} minutes ({estimated_full/3600:.2f} hours)")
    print("="*60)
    
    # recommendations
    if estimated_full > 3600:  # more than 1 hour
        print("\n⚠️  PERFORMANCE WARNING")
        print("Estimated time exceeds 1 hour. Recommendations:")
        print("  1. Reduce k_retrieve in configs/base.yaml (currently {})".format(k_retrieve))
        print("  2. Use GPU: set device: 'cuda' in configs/models.yaml")
        print("  3. Increase reranker batch_size (currently {})".format(reranker_config.get("batch_size", 16)))
        if embeddings_config.get("device") == "cpu":
            print("  4. Consider using GPU for embeddings (5-10x speedup)")


if __name__ == "__main__":
    main()

