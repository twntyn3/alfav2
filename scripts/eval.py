#!/usr/bin/env python3
"""
Evaluate retrieval pipeline: hybrid retrieval + reranking + metrics.

Usage:
    python scripts/eval.py --config configs/base.yaml
"""
import argparse
import copy
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "FlashRAG"))

import yaml
from src.batch_processor import optimize_batch_size
from src.evaluator import RetrievalEvaluator
from src.failure_logger import FailureLogger
from src.reranker import Reranker
from src.retriever import HybridRetriever
from src.utils import get_timestamp, logger, resolve_device


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_questions(csv_path: str) -> list[tuple[int, str]]:
    """Read questions from CSV file."""
    import csv
    questions = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_id = int(row.get("q_id", 0))
            query = row.get("query", "").strip()
            if q_id and query:
                questions.append((q_id, query))
    return questions


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval pipeline")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to config file")
    parser.add_argument("--ground_truth", help="Path to ground truth JSON file (optional)")
    args = parser.parse_args()
    
    # load configs
    config = load_config(args.config)
    models_config = load_config(project_root / "configs" / "models.yaml")
    
    # get paths
    questions_path = config["data"]["raw_questions"]
    bm25_dir = config["indexes"]["bm25_dir"]
    reports_dir = config["outputs"]["reports_dir"]
    logs_dir = config["outputs"]["logs_dir"]
    
    # get parameters
    retrieval_config = config["retrieval"]
    k_retrieve = retrieval_config["k_retrieve"]
    k_final = retrieval_config["k_final"]
    k_rerank = retrieval_config.get("k_rerank", min(k_retrieve, 20))
    use_reranker = retrieval_config.get("use_reranker", True)
    
    reranker_config = models_config["reranker"]
    faiss_config = models_config.get("faiss", {})
    bm25_corpus_override = retrieval_config.get("bm25_corpus_path")
    
    # read questions
    logger.info(f"Reading questions from {questions_path}")
    questions = read_questions(questions_path)
    logger.info(f"Loaded {len(questions)} questions")
    
    # initialize retriever
    logger.info("Initializing hybrid retriever (FAISS + FlashRAG bm25s)...")
    # find FAISS index and optional meta
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
        enhance_numerics=retrieval_config.get("enhance_numerics", True),
        embedding_fp16=embeddings_config.get("use_fp16", False),
        faiss_use_gpu=faiss_config.get("use_gpu", False),
        faiss_gpu_device=faiss_config.get("gpu_device"),
        faiss_use_fp16=faiss_config.get("use_float16", False),
    )
    
    # initialize reranker if enabled
    reranker = None
    if use_reranker:
        logger.info("Initializing reranker...")
        reranker_device = resolve_device(reranker_config.get("device", "auto"))
        reranker = Reranker(
            model_name=reranker_config["model_name"],
            device=reranker_device,
            batch_size=reranker_config.get("batch_size", 16),
            use_fp16=reranker_config.get("use_fp16", False),
        )
    
    # initialize evaluator
    evaluator = RetrievalEvaluator()
    if args.ground_truth:
        evaluator.load_ground_truth(args.ground_truth)
    
    # initialize failure logger
    failure_logger = FailureLogger(evaluator.ground_truth)
    
    # process questions
    total_questions = len(questions)
    logger.info(f"Processing {total_questions} questions...")
    sys.stdout.flush()
    results: dict[int, list[int]] = {}
    candidate_logs = []
    
    start_time = time.time()
    last_log_time = start_time
    processed = 0
    first_question_logged = False
    
    base_batch_size = max(1, retrieval_config.get("batch_size", 64))
    num_workers = max(1, retrieval_config.get("num_workers", 4))
    retrieval_batch_size = optimize_batch_size(base_batch_size, total_questions)
    logger.info(f"Optimized batch size: {retrieval_batch_size}, workers: {num_workers}")
    
    # progress bar configured not to interfere with logging
    progress_bar = None
    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=total_questions, desc="Processing", unit="q", file=sys.stderr, mininterval=5.0)
    except ImportError:
        logger.warning("tqdm not available, progress bar disabled")
    
    logger.info("Starting multi-threaded question processing loop...")
    sys.stdout.flush()
    
    def process_batch(batch_data: Tuple[int, List[Tuple[int, str]]]) -> Tuple[Dict[int, List[int]], List[Dict], int]:
        """Process a single batch and return results, logs, and count."""
        batch_start, batch = batch_data
        q_ids = [q for q, _ in batch]
        query_texts = [query for _, query in batch]
        batch_results: Dict[int, List[int]] = {}
        batch_logs: List[Dict] = []
        
        batch_candidates = [[] for _ in batch]
        batch_scores = [{} for _ in batch]
        
        try:
            batch_candidates, batch_scores = retriever.retrieve_batch(
                query_texts,
                k=k_retrieve,
                return_scores=True,
            )
        except Exception as e:
            logger.error(f"Batch retrieval failed for questions {q_ids[0]}-{q_ids[-1] if q_ids else 'N/A'}: {e}")
            batch_candidates = [[] for _ in batch]
            batch_scores = [{} for _ in batch]
        
        original_candidates = [copy.deepcopy(cands) for cands in batch_candidates]
        reranked_batch = [cands[:k_final] for cands in batch_candidates]
        
        if reranker:
            rerank_inputs = [
                [dict(candidate) for candidate in candidates[:k_rerank]]
                for candidates in batch_candidates
            ]
            try:
                reranked_batch = reranker.batch_rerank(
                    query_texts,
                    rerank_inputs,
                    top_k=k_final,
                    return_scores=False
                )
            except Exception as e:
                logger.error(f"Batch reranking failed for questions {q_ids[0]}-{q_ids[-1] if q_ids else 'N/A'}: {e}")
                reranked_batch = rerank_inputs
        
        for local_idx, (q_id, query_text) in enumerate(batch):
            candidates = batch_candidates[local_idx] if local_idx < len(batch_candidates) else []
            scores = batch_scores[local_idx] if local_idx < len(batch_scores) else {}
            candidates_before_rerank_list = original_candidates[local_idx] if local_idx < len(original_candidates) else []
            reranked_candidates = reranked_batch[local_idx] if local_idx < len(reranked_batch) else []
            
            log_entry_before = {
                "q_id": q_id,
                "query": query_text,
                "stage": "before_rerank",
                "candidates": [
                    {
                        "chunk_id": c["chunk_id"],
                        "doc_id": int(c["doc_id"]),
                        "score": c["score"],
                        "dense_score": c.get("dense_score", 0.0),
                        "bm25_score": c.get("bm25_score", 0.0)
                    }
                    for c in candidates[:k_retrieve]
                ]
            }
            batch_logs.append(log_entry_before)
            
            candidates_after_rerank = reranked_candidates if reranker else None
            
            if reranker and reranked_candidates:
                log_entry_after = {
                    "q_id": q_id,
                    "query": query_text,
                    "stage": "after_rerank",
                    "candidates": [
                        {
                            "chunk_id": c["chunk_id"],
                            "doc_id": int(c["doc_id"]),
                            "score": c.get("rerank_score", c.get("score", 0.0))
                        }
                        for c in reranked_candidates[:k_final]
                    ]
                }
                batch_logs.append(log_entry_after)
            
            final_candidates = reranked_candidates if reranker else candidates[:k_final]
            web_ids = [int(c["doc_id"]) for c in final_candidates[:k_final]]
            if len(web_ids) < k_final and candidates:
                additional = [int(c["doc_id"]) for c in candidates[k_final:]]
                web_ids.extend(additional[: k_final - len(web_ids)])
            while len(web_ids) < k_final:
                web_ids.append(web_ids[-1] if web_ids else 1)
            web_ids = web_ids[:k_final]
            batch_results[q_id] = web_ids
            
            failure_logger.log_retrieval_failure(
                q_id=q_id,
                query=query_text,
                retrieved_web_ids=web_ids,
                candidates_before_rerank=candidates_before_rerank_list,
                candidates_after_rerank=candidates_after_rerank,
                k=k_final
            )
        
        return batch_results, batch_logs, len(batch)
    
    batches = [
        (start, questions[start:start + retrieval_batch_size])
        for start in range(0, total_questions, retrieval_batch_size)
    ]
    
    with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="eval_worker") as executor:
        future_to_batch = {
            executor.submit(process_batch, (batch_start, batch)): (batch_start, batch)
            for batch_start, batch in batches
        }
        
        for future in as_completed(future_to_batch):
            batch_start, batch = future_to_batch[future]
            try:
                batch_results, batch_logs, batch_size = future.result(timeout=600.0)
                results.update(batch_results)
                candidate_logs.extend(batch_logs)
                processed += batch_size
                
                current_time = time.time()
                should_log = False
                if processed == 1:
                    should_log = True
                elif processed < 50 and processed % 10 == 0:
                    should_log = True
                elif processed % 50 == 0:
                    should_log = True
                elif (current_time - last_log_time) > 60:
                    should_log = True
                
                if should_log:
                    elapsed = current_time - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = (total_questions - processed) / rate if rate > 0 else 0
                    logger.info(
                        f"Progress: {processed}/{total_questions} questions "
                        f"({processed*100/total_questions:.1f}%), "
                        f"elapsed: {elapsed/60:.1f}min, "
                        f"rate: {rate:.2f} q/s, "
                        f"ETA: {remaining/60:.1f}min"
                    )
                    sys.stdout.flush()
                    last_log_time = current_time
                
                if progress_bar:
                    progress_bar.update(batch_size)
            except Exception as exc:
                logger.error(f"Batch {batch_start} failed: {exc}")
                processed += len(batch)
                if progress_bar:
                    progress_bar.update(len(batch))
    
    if progress_bar:
        progress_bar.close()
    
    elapsed_time = time.time() - start_time
    avg_time_per_question = elapsed_time / len(questions) if questions else 0
    logger.info(
        f"✓ Processed {len(questions)} questions in {elapsed_time/60:.1f} minutes "
        f"({elapsed_time:.1f}s total, {avg_time_per_question:.2f}s per question)"
    )
    
    # evaluate if ground truth available
    metrics = {}
    if evaluator.ground_truth:
        logger.info("Computing metrics...")
        eval_metrics = config["evaluation"]["metrics"]
        metrics = evaluator.evaluate(results, metrics=eval_metrics)
        logger.info(f"Hit@5: {metrics.get('hit@5', 0):.4f}")
        logger.info(f"Recall@5: {metrics.get('recall@5', 0):.4f}")
        logger.info(f"MRR: {metrics.get('mrr', 0):.4f}")
    else:
        logger.warning("No ground truth provided, skipping evaluation")
    
    # save report
    timestamp = get_timestamp()
    report_path = Path(reports_dir) / f"report_{timestamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": timestamp,
        "config": {
            "k_retrieve": k_retrieve,
            "k_rerank": k_rerank,
            "k_final": k_final,
            "use_reranker": use_reranker,
            "bm25_backend": models_config["bm25"]["backend"],
            "retrieval_batch_size": retrieval_batch_size,
            "enhance_numerics": retrieval_config.get("enhance_numerics", True),
            "reranker_model": reranker_config["model_name"] if use_reranker else None
        },
        "metrics": metrics,
        "num_queries": len(questions),
        "processing_time_seconds": elapsed_time
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Report saved to {report_path}")
    
    # save candidate logs
    if config["evaluation"].get("save_candidate_logs", True):
        log_path = Path(logs_dir) / f"candidates_{timestamp}.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            for log_entry in candidate_logs:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        logger.info(f"Candidate logs saved to {log_path}")
    
    # save failure logs
    failure_summary = failure_logger.get_failure_summary()
    if failure_summary["total"] > 0:
        failure_path = Path(logs_dir) / f"failures_{timestamp}.json"
        failure_logger.save_failures(str(failure_path))
        logger.warning(f"Found {failure_summary['total']} retrieval failures")
        logger.info(f"Failure summary: {failure_summary}")
        
        # add failure summary to report
        report["failure_summary"] = failure_summary
    
    return results, metrics


if __name__ == "__main__":
    main()
