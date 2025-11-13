"""Ensemble reranker with second-pass reranking support."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.reranker import Reranker
from src.utils import logger


class RerankerEnsemble:
    """
    Ensemble of multiple rerankers with optional second-pass reranking.
    
    Combines scores from multiple rerankers using weighted averaging,
    and optionally performs a second pass on top candidates.
    """

    def __init__(
        self,
        model_names: List[str],
        weights: List[float] | None = None,
        device: str = "cuda",
        batch_size: int = 64,
        max_length: int = 512,
        use_fp16: bool = True,
        second_pass: bool = False,
        second_pass_topk: int = 20,
    ) -> None:
        """
        Initialize ensemble reranker.
        
        Args:
            model_names: List of reranker model names
            weights: Weights for each reranker (must sum to 1.0). If None, equal weights.
            device: Device to run rerankers on
            batch_size: Batch size for reranking
            max_length: Max sequence length
            use_fp16: Use FP16 precision
            second_pass: Enable second-pass reranking on top candidates
            second_pass_topk: Number of top candidates for second pass
        """
        if not model_names:
            raise ValueError("At least one reranker model must be provided")
        
        if weights is None:
            weights = [1.0 / len(model_names)] * len(model_names)
        
        if len(weights) != len(model_names):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(model_names)})")
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(model_names)] * len(model_names)
        
        self.model_names = model_names
        self.weights = weights
        self.second_pass = second_pass
        self.second_pass_topk = second_pass_topk
        
        logger.info(f"Initializing ensemble reranker with {len(model_names)} models")
        logger.info(f"Models: {model_names}")
        logger.info(f"Weights: {weights}")
        logger.info(f"Second pass: {second_pass} (topk={second_pass_topk})")
        
        # Initialize individual rerankers
        self.rerankers: List[Reranker] = []
        for model_name in model_names:
            reranker = Reranker(
                model_name=model_name,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
                use_fp16=use_fp16,
            )
            self.rerankers.append(reranker)
        
        logger.info("Ensemble reranker initialized successfully")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        return_scores: bool = False,
    ) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], List[float]]:
        """Rerank candidates for a single query."""
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
        """
        Batch rerank candidates using ensemble.
        
        Args:
            queries: List of queries
            candidates_list: List of candidate lists (one per query)
            top_k: Number of top candidates to return
            return_scores: Whether to return scores
            
        Returns:
            Reranked candidates (and scores if return_scores=True)
        """
        if not queries:
            return ([], []) if return_scores else []
        
        top_k = max(1, int(top_k))
        
        # First pass: rerank with all models
        all_reranked: List[List[List[Dict[str, Any]]]] = []
        all_scores: List[List[List[float]]] = []
        
        for reranker in self.rerankers:
            reranked, scores = reranker.batch_rerank(
                queries,
                candidates_list,
                top_k=len(candidates_list[0]) if candidates_list else top_k,
                return_scores=True,
            )
            all_reranked.append(reranked)
            all_scores.append(scores)
        
        # Combine scores using weighted average
        ensemble_results: List[List[Dict[str, Any]]] = []
        ensemble_scores: List[List[float]] = []
        
        for query_idx in range(len(queries)):
            # Create score map for each candidate
            candidate_scores: Dict[str, List[float]] = {}
            candidate_data: Dict[str, Dict[str, Any]] = {}
            
            # Collect scores from all rerankers
            for reranker_idx, (reranked, scores) in enumerate(zip(all_reranked, all_scores)):
                weight = self.weights[reranker_idx]
                for candidate, score in zip(reranked[query_idx], scores[query_idx]):
                    chunk_id = candidate.get("chunk_id", str(candidate.get("id", "")))
                    if chunk_id not in candidate_scores:
                        candidate_scores[chunk_id] = []
                        candidate_data[chunk_id] = candidate
                    candidate_scores[chunk_id].append(float(score) * weight)
            
            # Compute weighted average scores
            final_candidates: List[Tuple[float, Dict[str, Any]]] = []
            for chunk_id, scores in candidate_scores.items():
                avg_score = sum(scores) / len(scores) if scores else 0.0
                candidate = candidate_data[chunk_id].copy()
                candidate["ensemble_score"] = avg_score
                candidate["rerank_score"] = avg_score
                candidate["score"] = avg_score
                final_candidates.append((avg_score, candidate))
            
            # Sort by ensemble score
            final_candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Second pass: rerank top candidates again if enabled
            if self.second_pass and len(final_candidates) > self.second_pass_topk:
                top_candidates = [cand for _, cand in final_candidates[:self.second_pass_topk]]
                # Use first reranker for second pass
                second_pass_reranked, second_pass_scores = self.rerankers[0].batch_rerank(
                    [queries[query_idx]],
                    [top_candidates],
                    top_k=top_k,
                    return_scores=True,
                )
                # Replace top candidates with second-pass results
                final_candidates = [
                    (score, cand)
                    for cand, score in zip(second_pass_reranked[0], second_pass_scores[0])
                ] + final_candidates[self.second_pass_topk:]
            
            # Take top_k
            top_candidates = [cand for _, cand in final_candidates[:top_k]]
            top_scores = [score for score, _ in final_candidates[:top_k]]
            
            ensemble_results.append(top_candidates)
            ensemble_scores.append(top_scores)
        
        if return_scores:
            return ensemble_results, ensemble_scores
        return ensemble_results

