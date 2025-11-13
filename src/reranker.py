"""FlashRAG CrossReranker wrapper."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLASHRAG_PATH = PROJECT_ROOT / "FlashRAG"
if str(FLASHRAG_PATH) not in sys.path:
    sys.path.insert(0, str(FLASHRAG_PATH))

from flashrag.retriever.reranker import CrossReranker  # type: ignore

from src.utils import logger, resolve_device


class Reranker:
    """Thin wrapper over FlashRAG's `CrossReranker` with project-friendly helpers."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 512,
        use_fp16: bool = True,
    ) -> None:
        self.device = resolve_device(device)
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(1, int(max_length))
        self.default_topk = 5

        config = {
            "rerank_model_name": model_name,
            "rerank_model_path": model_name,
            "rerank_topk": self.default_topk,
            "rerank_max_length": self.max_length,
            "rerank_batch_size": self.batch_size,
            "rerank_use_fp16": bool(use_fp16),
            "device": self.device,
        }

        logger.info("Loading FlashRAG CrossReranker: %s", model_name)
        self._reranker = CrossReranker(config)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        return_scores: bool = False,
    ):
        if not candidates:
            return ([], []) if return_scores else []

        results, scores = self.batch_rerank(
            [query],
            [candidates],
            top_k=top_k,
            return_scores=True,
        )
        if return_scores:
            return (results[0], scores[0])
        return results[0]

    def batch_rerank(
        self,
        queries: List[str],
        candidates_list: List[List[Dict[str, Any]]],
        top_k: int = 5,
        return_scores: bool = False,
    ) -> List[List[Dict[str, Any]]] | Tuple[List[List[Dict[str, Any]]], List[List[float]]]:
        if not queries:
            return ([], []) if return_scores else []

        top_k = max(1, int(top_k))
        self._reranker.topk = top_k

        normalized_candidates = [self._prepare_candidates(cands) for cands in candidates_list]
        reranked_docs, score_lists = self._reranker.rerank(
            queries,
            normalized_candidates,
            batch_size=self.batch_size,
            topk=top_k,
        )

        processed_results: List[List[Dict[str, Any]]] = []
        processed_scores: List[List[float]] = []

        for docs, scores in zip(reranked_docs, score_lists):
            candidates: List[Dict[str, Any]] = []
            for doc, score in zip(docs, scores):
                candidates.append(self._to_candidate(doc, float(score)))
            processed_results.append(candidates)
            processed_scores.append([float(score) for score in scores])

        if return_scores:
            return processed_results, processed_scores
        return processed_results

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _prepare_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        for candidate in candidates:
            text = candidate.get("contents") or candidate.get("text") or ""
            entry = dict(candidate)
            entry["contents"] = text
            prepared.append(entry)
        return prepared

    @staticmethod
    def _to_candidate(doc: Dict[str, Any], score: float) -> Dict[str, Any]:
        chunk_id = str(
            doc.get(
                "chunk_id",
                doc.get("id", doc.get("doc_id", doc.get("document_id", "unknown"))),
            )
        )
        doc_id = doc.get("doc_id")
        if doc_id is None:
            doc_id = chunk_id.split("_")[0] if "_" in chunk_id else chunk_id

        candidate = dict(doc)
        candidate["chunk_id"] = chunk_id
        candidate["doc_id"] = str(doc_id)
        candidate["rerank_score"] = float(score)
        candidate["score"] = float(score if candidate.get("score") is None else candidate["score"])
        return candidate

