"""FlashRAG-backed hybrid retriever."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLASHRAG_PATH = PROJECT_ROOT / "FlashRAG"
if str(FLASHRAG_PATH) not in sys.path:
    sys.path.insert(0, str(FLASHRAG_PATH))

from flashrag.retriever.retriever import MultiRetrieverRouter  # type: ignore

from src.table_processor import enhance_query_for_numerics
from src.utils import logger, resolve_device


class HybridRetriever:
    """Wrapper around FlashRAG's `MultiRetrieverRouter` tuned for bm25s + dense fusion."""

    def __init__(
        self,
        faiss_index_path: str,
        faiss_meta_path: Optional[str],  # kept for backwards compatibility; unused
        bm25_index_dir: str,
        bm25_corpus_path: str,
        embedding_model_name: str,
        device: str = "cuda",
        normalize_embeddings: bool = True,  # kept for API compatibility
        query_batch_size: int = 32,
        weight_dense: float = 0.6,
        weight_bm25: float = 0.4,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        enhance_numerics: bool = True,
        normalization_mode: str = "smart",  # normalization mode for queries
        min_score_threshold: float = 0.0,  # minimum score threshold before reranking
        filter_by_document_type: bool = False,  # filter candidates by document type
        prefer_table_chunks: bool = False,  # prefer chunks with tables for numeric queries
        embedding_fp16: bool = True,
        faiss_use_gpu: bool = True,
        faiss_gpu_device: Optional[int] = None,  # kept for API compatibility
        faiss_use_fp16: bool = True,
    ) -> None:
        del faiss_meta_path  # flashrag handles corpus metadata internally
        del normalize_embeddings
        del faiss_gpu_device

        self.device = resolve_device(device)
        self.embedding_model_name = embedding_model_name
        self.query_batch_size = max(1, int(query_batch_size))
        self.fusion_method = fusion_method
        self.rrf_k = int(rrf_k)
        self.weight_dense = float(weight_dense)
        self.weight_bm25 = float(weight_bm25)
        self.enhance_numerics = enhance_numerics
        self.normalization_mode = normalization_mode
        self.min_score_threshold = float(min_score_threshold)
        self.filter_by_document_type = bool(filter_by_document_type)
        self.prefer_table_chunks = bool(prefer_table_chunks)
        self.embedding_fp16 = bool(embedding_fp16)
        self.faiss_use_gpu = bool(faiss_use_gpu)
        self.faiss_use_fp16 = bool(faiss_use_fp16)
        self.default_topk = 200  # Increased to support k_retrieve=180

        self.faiss_index_path = Path(faiss_index_path)
        if not self.faiss_index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.faiss_index_path}")
        
        # Load FAISS metadata to get pooling_method and other settings
        self.faiss_pooling_method = None
        faiss_meta_path = self.faiss_index_path.parent / "faiss_meta.json"
        if faiss_meta_path.exists():
            try:
                import json
                with open(faiss_meta_path, "r", encoding="utf-8") as f:
                    faiss_meta = json.load(f)
                    self.faiss_pooling_method = faiss_meta.get("pooling_method", "mean")
            except Exception as e:
                logger.warning(f"Could not load FAISS metadata: {e}, using default pooling_method='mean'")
                self.faiss_pooling_method = "mean"
        else:
            logger.warning(f"FAISS metadata not found at {faiss_meta_path}, using default pooling_method='mean'")
            self.faiss_pooling_method = "mean"

        self.bm25_index_dir = Path(bm25_index_dir)
        if not self.bm25_index_dir.exists():
            raise FileNotFoundError(f"BM25 index directory not found: {self.bm25_index_dir}")

        self.bm25_corpus_path = Path(bm25_corpus_path)
        if not self.bm25_corpus_path.exists():
            raise FileNotFoundError(f"BM25 corpus JSONL not found: {self.bm25_corpus_path}")

        router_config = self._build_router_config()
        self.router = MultiRetrieverRouter(router_config)
        self._retriever_map = {getattr(r, "source_name", r.retrieval_method): r for r in self.router.retriever_list}
        logger.info(
            "Hybrid retriever initialised via FlashRAG: dense=%s, bm25=%s, fusion=%s",
            embedding_model_name,
            self._resolve_bm25_index_path(),
            self.fusion_method,
        )

    def retrieve(
        self,
        query: str,
        k: int = 50,
        return_scores: bool = False,
        fusion_method: Optional[str] = None,
    ):
        results, scores = self.retrieve_batch([query], k=k, return_scores=True, fusion_method=fusion_method)
        if return_scores:
            return (results[0] if results else [], scores[0] if scores else {})
        return results[0] if results else []

    def retrieve_batch(
        self,
        queries: List[str],
        k: int = 50,
        return_scores: bool = False,
        fusion_method: Optional[str] = None,
    ):
        if not queries:
            return ([], []) if return_scores else []

        topk = max(1, int(k))
        method = fusion_method or self.fusion_method
        prepared_queries = self._prepare_queries(queries)

        self.router.final_topk = topk
        self.router.merge_method = method
        self._update_retriever_topk(topk)

        if method == "weighted":
            self.router.weights = self._normalized_weights()

        results, score_lists = self.router.batch_search(prepared_queries, num=topk, return_score=True)

        processed_results: List[List[Dict[str, Any]]] = []
        processed_scores: List[Dict[str, Dict[str, float]]] = []

        for idx, docs in enumerate(results):
            raw_scores: List[float] = []
            if idx < len(score_lists):
                candidate_scores = score_lists[idx]
                if isinstance(candidate_scores, list):
                    raw_scores = candidate_scores
            candidates: List[Dict[str, Any]] = []
            # Get original query (before normalization) for numeric detection
            original_query = queries[idx] if idx < len(queries) else ""
            is_numeric_query = self._is_numeric_query(original_query)
            
            for doc_idx, doc in enumerate(docs[:topk]):
                fallback_score = raw_scores[doc_idx] if doc_idx < len(raw_scores) else None
                candidate = self._to_candidate(doc, fallback_score)
                
                # Filter by minimum score threshold if enabled
                if self.min_score_threshold > 0.0 and candidate["score"] < self.min_score_threshold:
                    continue
                
                # Filter/prefer by document type if enabled
                if self.filter_by_document_type or self.prefer_table_chunks:
                    has_table = candidate.get("has_table", False) or self._detect_table_in_candidate(candidate)
                    candidate["has_table"] = has_table
                    
                    # Prefer table chunks for numeric queries
                    if self.prefer_table_chunks and is_numeric_query and has_table:
                        candidate["score"] = candidate["score"] * 1.2  # Boost score by 20%
                
                candidates.append(candidate)
            processed_results.append(candidates)
            if return_scores:
                processed_scores.append(
                    {
                        "hybrid": {cand["chunk_id"]: cand["score"] for cand in candidates},
                        "dense": {cand["chunk_id"]: cand["dense_score"] for cand in candidates},
                        "bm25": {cand["chunk_id"]: cand["bm25_score"] for cand in candidates},
                    }
                )

        if return_scores:
            return processed_results, processed_scores
        return processed_results

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _prepare_queries(self, queries: List[str]) -> List[str]:
        from src.text_processor import normalize_for_retrieval
        
        prepared = []
        for query in queries:
            # Apply numeric enhancement if enabled
            if self.enhance_numerics:
                query = enhance_query_for_numerics(query)
            # Normalize query to match corpus normalization
            query = normalize_for_retrieval(query, mode=self.normalization_mode)
            prepared.append(query)
        return prepared

    def _normalized_weights(self) -> Dict[str, float]:
        raw = {
            "dense": max(0.0, self.weight_dense),
            "bm25": max(0.0, self.weight_bm25),
        }
        total = sum(raw.values())
        if total <= 0:
            return {name: 0.5 for name in raw}
        return {name: value / total for name, value in raw.items()}

    def _update_retriever_topk(self, topk: int) -> None:
        for retriever in self.router.retriever_list:
            retriever.topk = topk
            if hasattr(retriever, "batch_size"):
                retriever.batch_size = self.query_batch_size

    def _build_router_config(self) -> Dict[str, Any]:
        merge_method = self.fusion_method if self.fusion_method != "weighted" else "weighted"
        weights = self._normalized_weights() if merge_method == "weighted" else {}

        bm25_config = {
            "name": "bm25",
            "retrieval_method": "bm25",
            "retrieval_model_path": "",  # Not used for BM25, but required by FlashRAG
            "retrieval_topk": self.default_topk,
            "retrieval_batch_size": self.query_batch_size,
            "retrieval_query_max_length": 256,
            "retrieval_use_fp16": False,
            "retrieval_pooling_method": "mean",
            "instruction": None,
            "index_path": str(self._resolve_bm25_index_path()),
            "corpus_path": str(self.bm25_corpus_path),
            "bm25_backend": "bm25s",
            "save_retrieval_cache": False,
            "use_retrieval_cache": False,
            "retrieval_cache_path": None,
            "use_reranker": False,
            "silent_retrieval": True,
        }

        dense_config = {
            "name": "dense",
            "retrieval_method": self.embedding_model_name,
            "retrieval_model_path": self.embedding_model_name,
            "retrieval_topk": self.default_topk,
            "retrieval_batch_size": self.query_batch_size,
            "retrieval_query_max_length": 256,  # Queries are shorter than documents
            "retrieval_use_fp16": self.embedding_fp16 or self.faiss_use_fp16,
            "retrieval_pooling_method": self.faiss_pooling_method,  # Use pooling_method from index metadata
            "instruction": None,  # Auto-detect (will use "query: " for E5, "passage: " was used during indexing)
            "index_path": str(self.faiss_index_path),
            "corpus_path": str(self.bm25_corpus_path),
            "save_retrieval_cache": False,
            "use_retrieval_cache": False,
            "retrieval_cache_path": None,
            "use_reranker": False,
            "silent_retrieval": True,
            "use_sentence_transformer": True,
            "faiss_gpu": self.faiss_use_gpu,
        }

        multi_retriever_setting = {
            "merge_method": merge_method,
            "topk": self.default_topk,
            "retriever_list": [bm25_config, dense_config],
        }
        if merge_method == "weighted":
            multi_retriever_setting["weights"] = weights
        elif merge_method == "rrf":
            # FlashRAG uses k=60 by default, but we allow customization
            # The rrf_k is passed via the router's internal rrf_merge method
            pass  # RRF k is handled internally by FlashRAG
        
        return {
            "device": self.device,
            "silent_retrieval": True,
            "multi_retriever_setting": multi_retriever_setting,
        }

    def _resolve_bm25_index_path(self) -> Path:
        candidate = self.bm25_index_dir / "bm25"
        if candidate.exists():
            return candidate
        return self.bm25_index_dir

    def _to_candidate(self, doc: Dict[str, Any], fallback_score: Optional[float]) -> Dict[str, Any]:
        chunk_id = str(
            doc.get(
                "chunk_id",
                doc.get("id", doc.get("doc_id", doc.get("document_id", "unknown"))),
            )
        )
        doc_id = doc.get("doc_id")
        if doc_id is None:
            doc_id = chunk_id.split("_")[0] if "_" in chunk_id else chunk_id

        source_scores = doc.get("source_scores") or {}
        if not isinstance(source_scores, dict):
            source_scores = {}

        dense_score = float(doc.get("dense_score", source_scores.get("dense", 0.0)))
        bm25_score = float(doc.get("bm25_score", source_scores.get("bm25", 0.0)))
        score = doc.get("score")
        if score is None:
            score = fallback_score
        if score is None:
            score = max(dense_score, bm25_score)
        score = float(score or 0.0)

        candidate = {
            "chunk_id": chunk_id,
            "doc_id": str(doc_id),
            "score": score,
            "dense_score": dense_score,
            "bm25_score": bm25_score,
            "contents": doc.get("contents") or doc.get("text") or "",
            "source_scores": {key: float(value) for key, value in source_scores.items()},
            "sources": doc.get("sources", []),
            "has_table": doc.get("has_table", False),
        }
        return candidate
    
    @staticmethod
    def _is_numeric_query(query: str) -> bool:
        """Check if query contains numeric patterns (numbers, rates, amounts, etc.)."""
        import re
        # Check for numbers
        if re.search(r'\d+', query):
            return True
        # Check for numeric-related terms
        numeric_terms = ["процент", "ставка", "сумма", "лимит", "комиссия", "курс", "цена", "стоимость"]
        query_lower = query.lower()
        return any(term in query_lower for term in numeric_terms)
    
    @staticmethod
    def _detect_table_in_candidate(candidate: Dict[str, Any]) -> bool:
        """Detect if candidate contains table structure."""
        from src.table_processor import detect_table
        contents = candidate.get("contents") or candidate.get("text") or ""
        return detect_table(contents)

