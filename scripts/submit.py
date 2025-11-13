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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "FlashRAG"))

import yaml
from src.batch_processor import BatchProcessor, optimize_batch_size
from src.retriever import HybridRetriever
from src.reranker import Reranker
from src.utils import logger, get_timestamp, resolve_device
def validate_against_sample(results: Dict[int, List[int]], sample_path: Path, k_final: int) -> None:
    """Ensure submission matches the sample format (q_ids set and list length)."""
    if not sample_path.exists():
        logger.warning("Sample submission not found at %s; skipping shape validation", sample_path)
        return

    with sample_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        sample_qids = []
        for row in reader:
            try:
                sample_qids.append(int(row["q_id"]))
            except (ValueError, KeyError):
                raise ValueError(f"Malformed row in sample submission: {row}") from None

    sample_set = set(sample_qids)
    result_set = set(results.keys())
    if sample_set != result_set:
        missing = sample_set - result_set
        extra = result_set - sample_set
        raise ValueError(
            f"Submission q_id set mismatch with sample (missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]})"
        )

    for q_id, web_ids in results.items():
        if len(web_ids) != k_final:
            raise ValueError(
                f"Question {q_id} has {len(web_ids)} web ids, expected {k_final}. "
                "Ensure candidates are padded/truncated properly."
            )


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
    bm25_dir = config["indexes"]["bm25_dir"]
    submits_dir = config["outputs"]["submits_dir"]
    
    # get parameters
    retrieval_config = config["retrieval"]
    k_retrieve = retrieval_config["k_retrieve"]
    k_final = retrieval_config["k_final"]
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
    
    total_questions = len(questions)
    logger.info(f"Processing {total_questions} questions with multi-threaded batch processing...")
    results = {}
    
    base_batch_size = max(1, retrieval_config.get("batch_size", 64))
    num_workers = max(1, retrieval_config.get("num_workers", 4))
    k_rerank = retrieval_config.get("k_rerank", min(k_retrieve, 20))
    
    retrieval_batch_size = optimize_batch_size(base_batch_size, total_questions)
    logger.info(f"Optimized batch size: {retrieval_batch_size}, workers: {num_workers}")
    
    progress_bar = None
    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=total_questions, desc="Processing", unit="q", mininterval=5.0)
    except ImportError:
        logger.warning("tqdm not available, progress bar disabled")
    
    start_time = time.time()
    last_log_time = start_time
    processed = 0
    
    def process_batch(batch_data: Tuple[int, List[Tuple[int, str]]]) -> Dict[int, List[int]]:
        """Process a single batch and return results."""
        batch_start, batch = batch_data
        q_ids = [q for q, _ in batch]
        query_texts = [query for _, query in batch]
        batch_results: Dict[int, List[int]] = {}
        
        try:
            batch_candidates = retriever.retrieve_batch(
                query_texts,
                k=k_retrieve,
                return_scores=False,
            )
        except Exception as e:
            logger.error(f"Batch retrieval failed for questions {q_ids[0]}-{q_ids[-1] if q_ids else 'N/A'}: {e}")
            batch_candidates = [[] for _ in batch]
        
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
        else:
            reranked_batch = [candidates[:k_final] for candidates in batch_candidates]
        
        for local_idx, (q_id, _) in enumerate(batch):
            candidates = batch_candidates[local_idx] if local_idx < len(batch_candidates) else []
            final_candidates = reranked_batch[local_idx] if local_idx < len(reranked_batch) else []
            if not final_candidates:
                final_candidates = candidates[:k_final]
            
            web_ids = [int(c["doc_id"]) for c in final_candidates[:k_final]]
            if len(web_ids) < k_final:
                for candidate in candidates:
                    doc_id = int(candidate["doc_id"])
                    if doc_id not in web_ids:
                        web_ids.append(doc_id)
                    if len(web_ids) >= k_final:
                        break
            while len(web_ids) < k_final:
                web_ids.append(web_ids[-1] if web_ids else 1)
            web_ids = web_ids[:k_final]
            batch_results[q_id] = web_ids
        
        return batch_results
    
    batches = [
        (start, questions[start:start + retrieval_batch_size])
        for start in range(0, total_questions, retrieval_batch_size)
    ]
    
    with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="submit_worker") as executor:
        future_to_batch = {
            executor.submit(process_batch, (batch_start, batch)): (batch_start, batch)
            for batch_start, batch in batches
        }
        
        for future in as_completed(future_to_batch):
            batch_start, batch = future_to_batch[future]
            try:
                batch_results = future.result(timeout=600.0)
                results.update(batch_results)
                processed += len(batch)
                
                current_time = time.time()
                if processed > 0 and (processed % 100 == 0 or (current_time - last_log_time) > 60):
                    elapsed = current_time - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = (total_questions - processed) / rate if rate > 0 else 0
                    logger.info(
                        f"Progress: {processed}/{total_questions} ({processed*100/total_questions:.1f}%), "
                        f"elapsed: {elapsed/60:.1f}min, rate: {rate:.2f} q/s, ETA: {remaining/60:.1f}min"
                    )
                    sys.stdout.flush()
                    last_log_time = current_time
                
                if progress_bar:
                    progress_bar.update(len(batch))
            except Exception as exc:
                logger.error(f"Batch {batch_start} failed: {exc}")
                processed += len(batch)
                if progress_bar:
                    progress_bar.update(len(batch))
    
    if progress_bar:
        progress_bar.close()
    
    elapsed_time = time.time() - start_time
    logger.info(
        f"✓ Processed {total_questions} questions in {elapsed_time/60:.1f} minutes "
        f"({elapsed_time:.1f}s total, {elapsed_time/max(1, total_questions):.2f}s per question)"
    )
    
    timestamp = get_timestamp()
    output_path = Path(args.output) if args.output else Path(submits_dir) / f"submit_{timestamp}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    validate_against_sample(results, project_root / "sample_submission.csv", k_final)

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
