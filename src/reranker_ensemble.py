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
        self.batch_size = max(1, batch_size)
        
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
                batch_size=self.batch_size,
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
        # Safe access with validation
        if not results or not scores:
            return ([], []) if return_scores else []
        if return_scores:
            return (results[0] if results else [], scores[0] if scores else [])
        return results[0] if results else []

    def shrink_batch_size(self, factor: float = 0.5, min_batch: int = 4) -> bool:
        """
        Reduce batch size across ensemble rerankers (used when CUDA OOM occurs).
        """
        changed = False
        for reranker in self.rerankers:
            changed = reranker.shrink_batch_size(factor=factor, min_batch=min_batch) or changed
        if changed:
            self.batch_size = min(r.batch_size for r in self.rerankers)
            logger.warning(
                "Ensemble reranker batch size reduced to %s (factor=%s)",
                self.batch_size,
                factor,
            )
        return changed

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
            try:
                # Determine safe top_k value
                safe_topk = top_k
                if candidates_list and len(candidates_list) > 0:
                    if len(candidates_list[0]) > 0:
                        safe_topk = min(len(candidates_list[0]), top_k * 2)  # Get more than needed for better ensemble
                    else:
                        safe_topk = top_k
                else:
                    safe_topk = top_k
                
                reranked, scores = reranker.batch_rerank(
                    queries,
                    candidates_list,
                    top_k=safe_topk,
                    return_scores=True,
                )
                
                # Validate results
                if not reranked or not scores:
                    logger.warning("Reranker returned empty results")
                    continue
                if len(reranked) != len(queries) or len(scores) != len(queries):
                    logger.warning(f"Reranker results length mismatch: reranked={len(reranked)}, scores={len(scores)}, queries={len(queries)}")
                    continue
                
                all_reranked.append(reranked)
                all_scores.append(scores)
            except Exception as e:
                logger.error(f"Error in reranker batch_rerank: {e}")
                continue
        
        # Check if we have any reranked results
        if not all_reranked or not all_scores:
            logger.warning("No reranked results available, returning empty results")
            # Return candidates as-is if available
            if candidates_list and len(candidates_list) > 0:
                fallback_results = [cands[:top_k] for cands in candidates_list]
                fallback_scores = [[0.0] * min(len(cands), top_k) for cands in candidates_list]
                if return_scores:
                    return fallback_results, fallback_scores
                return fallback_results
            return ([], []) if return_scores else []
        
        # Combine scores using weighted average
        ensemble_results: List[List[Dict[str, Any]]] = []
        ensemble_scores: List[List[float]] = []
        
        for query_idx in range(len(queries)):
            # Create score map for each candidate
            candidate_scores: Dict[str, List[float]] = {}
            candidate_data: Dict[str, Dict[str, Any]] = {}
            
            # Collect scores from all rerankers
            for reranker_idx, (reranked, scores) in enumerate(zip(all_reranked, all_scores)):
                if reranker_idx >= len(self.weights):
                    logger.warning(f"Reranker index {reranker_idx} exceeds weights length, skipping")
                    continue
                weight = self.weights[reranker_idx]
                
                # Safe access to reranked and scores
                if query_idx >= len(reranked) or query_idx >= len(scores):
                    logger.warning(f"Query index {query_idx} out of range for reranker {reranker_idx}")
                    continue
                
                query_reranked = reranked[query_idx]
                query_scores = scores[query_idx]
                
                if not query_reranked or not query_scores:
                    continue
                
                for candidate, score in zip(query_reranked, query_scores):
                    try:
                        chunk_id = candidate.get("chunk_id", str(candidate.get("id", "")))
                        if not chunk_id:
                            continue
                        if chunk_id not in candidate_scores:
                            candidate_scores[chunk_id] = []
                            candidate_data[chunk_id] = candidate
                        score_val = float(score) if score is not None else 0.0
                        candidate_scores[chunk_id].append(score_val * weight)
                    except (ValueError, TypeError, AttributeError) as e:
                        logger.warning(f"Error processing candidate score: {e}")
                        continue
            
            # Compute weighted average scores
            final_candidates: List[Tuple[float, Dict[str, Any]]] = []
            for chunk_id, scores in candidate_scores.items():
                avg_score = sum(scores) / len(scores) if scores else 0.0
                candidate = candidate_data[chunk_id].copy()
                candidate["ensemble_score"] = avg_score
                candidate["rerank_score"] = avg_score
                # Preserve original hybrid score if better than ensemble average
                original_score = candidate.get("score", 0.0)
                if isinstance(original_score, (int, float)) and original_score > avg_score * 0.8:
                    candidate["score"] = max(original_score, avg_score)
                else:
                    candidate["score"] = avg_score
                # Normalize ensemble score for better comparison
                candidate["normalized_score"] = 1.0 / (1.0 + pow(2.718, -avg_score / 2.0))
                final_candidates.append((avg_score, candidate))
            
            # Sort by ensemble score
            final_candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Second pass: rerank top candidates again if enabled
            if self.second_pass and len(final_candidates) > self.second_pass_topk and self.rerankers:
                try:
                    top_candidates = [cand for _, cand in final_candidates[:self.second_pass_topk]]
                    # Use first reranker for second pass
                    if not top_candidates:
                        logger.warning(f"No candidates for second pass on query {query_idx}")
                    else:
                        second_pass_reranked, second_pass_scores = self.rerankers[0].batch_rerank(
                            [queries[query_idx]],
                            [top_candidates],
                            top_k=top_k,
                            return_scores=True,
                        )
                        # Safe access to second pass results
                        if second_pass_reranked and second_pass_scores and len(second_pass_reranked) > 0 and len(second_pass_scores) > 0:
                            second_pass_results = second_pass_reranked[0]
                            second_pass_scores_list = second_pass_scores[0]
                            
                            # Replace top candidates with second-pass results
                            second_pass_with_scores = []
                            for cand, score in zip(second_pass_results, second_pass_scores_list):
                                try:
                                    score_val = float(score) if score is not None else 0.0
                                    second_pass_with_scores.append((score_val, cand))
                                    # Update scores in second-pass candidates
                                    cand["rerank_score"] = score_val
                                    cand["ensemble_score"] = score_val
                                    cand["normalized_score"] = 1.0 / (1.0 + pow(2.718, -score_val / 2.0))
                                except (ValueError, TypeError, ZeroDivisionError):
                                    logger.warning(f"Error processing second pass score: {score}")
                                    continue
                            
                            if second_pass_with_scores:
                                final_candidates = second_pass_with_scores + final_candidates[self.second_pass_topk:]
                except (IndexError, AttributeError, TypeError) as e:
                    logger.warning(f"Error in second pass reranking: {e}, using first pass results")
                    # Continue with first pass results
            
            # Take top_k
            top_candidates = [cand for _, cand in final_candidates[:top_k]]
            top_scores = [score for score, _ in final_candidates[:top_k]]
            
            ensemble_results.append(top_candidates)
            ensemble_scores.append(top_scores)
        
        if return_scores:
            return ensemble_results, ensemble_scores
        return ensemble_results

