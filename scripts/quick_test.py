#!/usr/bin/env python3
"""
Quick test script to verify the pipeline works correctly on a small subset.

Usage:
    python scripts/quick_test.py --config configs/base.yaml
"""
import argparse
import sys
import time
from pathlib import Path

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "FlashRAG"))

import yaml
from src.retriever import HybridRetriever
from src.reranker import Reranker
from src.reranker_ensemble import RerankerEnsemble
from src.utils import logger, resolve_device


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_imports():
    """Test that all imports work."""
    logger.info("=" * 60)
    logger.info("TEST 1: Testing imports...")
    try:
        import torch
        import faiss
        from sentence_transformers import SentenceTransformer
        from flashrag.retriever.retriever import MultiRetrieverRouter
        logger.info("✓ All imports successful")
        logger.info(f"  - PyTorch: {torch.__version__}")
        logger.info(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  - CUDA device: {torch.cuda.get_device_name()}")
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_device():
    """Test device resolution."""
    logger.info("=" * 60)
    logger.info("TEST 2: Testing device resolution...")
    try:
        device = resolve_device("auto")
        logger.info(f"✓ Device resolved: {device}")
        return True
    except Exception as e:
        logger.error(f"✗ Device resolution failed: {e}")
        return False


def test_retriever_init(config: dict):
    """Test retriever initialization."""
    logger.info("=" * 60)
    logger.info("TEST 3: Testing retriever initialization...")
    try:
        # Load models config from separate file
        project_root = Path(__file__).parent.parent
        models_config_path = project_root / "configs" / "models.yaml"
        if models_config_path.exists():
            with open(models_config_path, "r", encoding="utf-8") as f:
                models_config = yaml.safe_load(f)
        else:
            models_config = config.get("models", {})
        
        indexes = config["indexes"]
        embeddings_config = models_config.get("embeddings", {})
        faiss_config = models_config.get("faiss", {})
        retrieval_config = config.get("retrieval", {})
        text_processing = config.get("text_processing", {})
        
        # Find FAISS index file (may have different names like e5_Flat.index)
        faiss_dir = Path(indexes["faiss_dir"])
        faiss_index_files = list(faiss_dir.glob("*.index"))
        if not faiss_index_files:
            logger.warning(f"⚠ FAISS index not found in {faiss_dir}")
            logger.warning("  Run 'make build_index' first")
            return False, None
        faiss_index_path = faiss_index_files[0]  # Use first .index file found
        logger.info(f"  Found FAISS index: {faiss_index_path.name}")
        
        bm25_index_dir = Path(indexes["bm25_dir"])
        bm25_corpus_path = Path(config["data"]["chunks_jsonl"])
        
        embedding_device = resolve_device(embeddings_config.get("device", "auto"))
        normalization_mode = text_processing.get("normalization_mode", "smart")
        
        retriever = HybridRetriever(
            faiss_index_path=str(faiss_index_path),
            faiss_meta_path=None,
            bm25_index_dir=str(bm25_index_dir),
            bm25_corpus_path=str(bm25_corpus_path),
            embedding_model_name=embeddings_config["model_name"],
            device=embedding_device,
            query_batch_size=4,  # small batch for testing
            fusion_method=retrieval_config.get("fusion_method", "rrf"),
            rrf_k=retrieval_config.get("rrf_k", 60),
            enhance_numerics=retrieval_config.get("enhance_numerics", True),
            normalization_mode=normalization_mode,
            min_score_threshold=retrieval_config.get("min_score_threshold", 0.0),
            filter_by_document_type=retrieval_config.get("filter_by_document_type", False),
            prefer_table_chunks=retrieval_config.get("prefer_table_chunks", False),
            embedding_fp16=embeddings_config.get("use_fp16", False),
            faiss_use_gpu=faiss_config.get("use_gpu", False),
            faiss_use_fp16=faiss_config.get("use_float16", False),
        )
        logger.info("✓ Retriever initialized successfully")
        logger.info(f"  - Model: {embeddings_config['model_name']}")
        logger.info(f"  - Device: {embedding_device}")
        return True, retriever
    except Exception as e:
        logger.error(f"✗ Retriever initialization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, None


def test_retrieval(retriever: HybridRetriever):
    """Test retrieval on a few queries."""
    logger.info("=" * 60)
    logger.info("TEST 4: Testing retrieval...")
    try:
        test_queries = [
            "как узнать номер счета",
            "процентная ставка по вкладу",
            "где найти реквизиты банка",
        ]
        
        start_time = time.time()
        results = retriever.retrieve_batch(test_queries, k=10)
        elapsed = time.time() - start_time
        
        logger.info(f"✓ Retrieval successful ({elapsed:.2f}s)")
        for i, (query, candidates) in enumerate(zip(test_queries, results)):
            logger.info(f"  Query {i+1}: '{query}' -> {len(candidates)} candidates")
            if candidates:
                logger.info(f"    Top candidate: {candidates[0].get('doc_id', 'N/A')} (score: {candidates[0].get('score', 0):.4f})")
        return True
    except Exception as e:
        logger.error(f"✗ Retrieval failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_reranker_init(config: dict):
    """Test reranker initialization."""
    logger.info("=" * 60)
    logger.info("TEST 5: Testing reranker initialization...")
    try:
        # Load models config from separate file
        project_root = Path(__file__).parent.parent
        models_config_path = project_root / "configs" / "models.yaml"
        if models_config_path.exists():
            with open(models_config_path, "r", encoding="utf-8") as f:
                models_config = yaml.safe_load(f)
        else:
            models_config = config.get("models", {})
        
        reranker_config = models_config.get("reranker", {})
        if not reranker_config:
            logger.warning("⚠ Reranker config not found in models.yaml")
            return False, None
        
        model_name = reranker_config.get("model_name")
        if not model_name:
            logger.warning("⚠ model_name not found in reranker config")
            return False, None
        
        retrieval_config = config.get("retrieval", {})
        reranker_device = resolve_device(reranker_config.get("device", "auto"))
        
        reranker = Reranker(
            model_name=model_name,
            device=reranker_device,
            batch_size=4,  # small batch for testing
            max_length=reranker_config.get("max_length", 512),
            use_fp16=reranker_config.get("use_fp16", False),
        )
        logger.info("✓ Reranker initialized successfully")
        logger.info(f"  - Model: {reranker_config['model_name']}")
        logger.info(f"  - Device: {reranker_device}")
        return True, reranker
    except Exception as e:
        logger.error(f"✗ Reranker initialization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, None


def test_reranking(reranker: Reranker, retriever: HybridRetriever):
    """Test reranking on a few queries."""
    logger.info("=" * 60)
    logger.info("TEST 6: Testing reranking...")
    try:
        test_query = "как узнать номер счета"
        
        # Retrieve candidates
        candidates = retriever.retrieve(test_query, k=20)
        if not candidates:
            logger.warning("⚠ No candidates retrieved, skipping reranking test")
            return False
        
        # Rerank
        start_time = time.time()
        reranked = reranker.rerank(test_query, candidates, top_k=5)
        elapsed = time.time() - start_time
        
        logger.info(f"✓ Reranking successful ({elapsed:.2f}s)")
        logger.info(f"  Query: '{test_query}'")
        logger.info(f"  Reranked {len(candidates)} -> {len(reranked)} candidates")
        if reranked:
            logger.info(f"  Top reranked: {reranked[0].get('doc_id', 'N/A')} (score: {reranked[0].get('rerank_score', 0):.4f})")
        return True
    except Exception as e:
        logger.error(f"✗ Reranking failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_ensemble_reranker(config: dict):
    """Test ensemble reranker initialization (if enabled)."""
    logger.info("=" * 60)
    logger.info("TEST 7: Testing ensemble reranker (if enabled)...")
    try:
        # Load models config from separate file
        project_root = Path(__file__).parent.parent
        models_config_path = project_root / "configs" / "models.yaml"
        if models_config_path.exists():
            with open(models_config_path, "r", encoding="utf-8") as f:
                models_config = yaml.safe_load(f)
        else:
            models_config = config.get("models", {})
        
        ensemble_config = models_config.get("reranker_ensemble", {})
        
        if not ensemble_config.get("enabled", False):
            logger.info("⚠ Ensemble reranker disabled in config, skipping")
            return True
        
        reranker_config = models_config.get("reranker", {})
        retrieval_config = config.get("retrieval", {})
        reranker_device = resolve_device(reranker_config.get("device", "auto"))
        
        ensemble = RerankerEnsemble(
            model_names=ensemble_config.get("models", [reranker_config["model_name"]]),
            weights=ensemble_config.get("weights"),
            device=reranker_device,
            batch_size=4,
            max_length=reranker_config.get("max_length", 512),
            use_fp16=reranker_config.get("use_fp16", False),
            second_pass=ensemble_config.get("second_pass", False),
            second_pass_topk=ensemble_config.get("second_pass_topk", 20),
        )
        logger.info("✓ Ensemble reranker initialized successfully")
        logger.info(f"  - Models: {ensemble_config.get('models', [])}")
        return True
    except Exception as e:
        logger.error(f"✗ Ensemble reranker initialization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description="Quick test of RAG pipeline")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to config file")
    args = parser.parse_args()
    
    logger.info("Starting quick test of RAG pipeline...")
    logger.info(f"Config: {args.config}")
    
    config = load_config(args.config)
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    # Test 2: Device
    results.append(("Device", test_device()))
    
    # Test 3: Retriever
    retriever_ok, retriever = test_retriever_init(config)
    results.append(("Retriever Init", retriever_ok))
    
    # Test 4: Retrieval (only if retriever initialized)
    if retriever_ok and retriever:
        results.append(("Retrieval", test_retrieval(retriever)))
    else:
        logger.warning("⚠ Skipping retrieval test (retriever not initialized)")
        results.append(("Retrieval", None))  # None = skipped
    
    # Test 5: Reranker
    reranker_ok, reranker = test_reranker_init(config)
    results.append(("Reranker Init", reranker_ok))
    
    # Test 6: Reranking (only if both retriever and reranker initialized)
    if reranker_ok and reranker and retriever_ok and retriever:
        results.append(("Reranking", test_reranking(reranker, retriever)))
    else:
        logger.warning("⚠ Skipping reranking test (retriever or reranker not initialized)")
        results.append(("Reranking", None))  # None = skipped
    
    # Test 7: Ensemble
    results.append(("Ensemble Reranker", test_ensemble_reranker(config)))
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, ok in results if ok is True)
    failed = sum(1 for _, ok in results if ok is False)
    skipped = sum(1 for _, ok in results if ok is None)
    total = len(results)
    
    for test_name, ok in results:
        if ok is True:
            status = "✓ PASS"
        elif ok is False:
            status = "✗ FAIL"
        else:
            status = "⊘ SKIP"
        logger.info(f"{status}: {test_name}")
    
    logger.info("=" * 60)
    logger.info(f"Results: {passed} passed, {failed} failed, {skipped} skipped (out of {total} tests)")
    
    if failed == 0:
        if skipped > 0:
            logger.info("✓ All runnable tests passed! (Some tests were skipped - build indexes to run them)")
            logger.info("  To run all tests: make build_index")
        else:
            logger.info("✓ All tests passed! Pipeline is ready.")
        return 0
    else:
        logger.error("✗ Some tests failed. Please fix issues before running full pipeline.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

