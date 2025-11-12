"""Cross-encoder reranker for reordering retrieval candidates."""
import logging
from typing import List, Dict, Optional, Tuple

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    raise ImportError("sentence-transformers is required for reranking")

from src.utils import logger


class Reranker:
    """
    Cross-encoder reranker for reordering retrieval candidates.
    
    Algorithm: Uses cross-encoder to score query-document pairs
    Time Complexity: O(n * m) where n is candidates, m is model forward pass
    Space Complexity: O(n) for storing scores
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512
    ):
        """
        Initialize reranker.
        
        Args:
            model_name: Name of cross-encoder model
            device: Device for model ("cpu" or "cuda")
            batch_size: Batch size for inference
            max_length: Maximum sequence length
        """
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        self.batch_size = batch_size
        logger.info(f"Reranker initialized on {device}")
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
        return_scores: bool = False
    ) -> List[Dict] | Tuple[List[Dict], List[float]]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: Query string
            candidates: List of candidate chunks (must have "contents" or "text" field)
            top_k: Number of top candidates to return
            return_scores: Whether to return reranking scores
            
        Returns:
            Reranked candidates, or (candidates, scores) if return_scores
        """
        if not candidates:
            return [] if not return_scores else ([], [])
        
        # prepare pairs for cross-encoder
        pairs = []
        for candidate in candidates:
            # get text content
            text = candidate.get("contents", candidate.get("text", ""))
            if not text:
                # fallback: use chunk_id
                text = str(candidate.get("chunk_id", ""))
            pairs.append([query, text])
        
        # score pairs in batches
        scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            batch_scores = self.model.predict(batch, show_progress_bar=False)
            scores.extend(batch_scores.tolist())
        
        # combine candidates with scores
        scored_candidates = list(zip(candidates, scores))
        
        # sort by score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # get top-k
        top_candidates = [cand for cand, _ in scored_candidates[:top_k]]
        
        # add rerank scores to candidates
        for i, (cand, score) in enumerate(scored_candidates[:top_k]):
            top_candidates[i]["rerank_score"] = float(score)
            if "score" not in top_candidates[i]:
                top_candidates[i]["score"] = float(score)
        
        if return_scores:
            top_scores = [float(score) for _, score in scored_candidates[:top_k]]
            return top_candidates, top_scores
        return top_candidates
    
    def batch_rerank(
        self,
        queries: List[str],
        candidates_list: List[List[Dict]],
        top_k: int = 5,
        return_scores: bool = False
    ) -> List[List[Dict]] | Tuple[List[List[Dict]], List[List[float]]]:
        """
        Batch rerank for multiple queries.
        
        Args:
            queries: List of query strings
            candidates_list: List of candidate lists (one per query)
            top_k: Number of top candidates per query
            return_scores: Whether to return scores
            
        Returns:
            List of reranked candidates per query, or (results, scores_list) if return_scores
        """
        results = []
        scores_list = []
        
        for query, candidates in zip(queries, candidates_list):
            if return_scores:
                reranked, scores = self.rerank(query, candidates, top_k=top_k, return_scores=True)
                results.append(reranked)
                scores_list.append(scores)
            else:
                reranked = self.rerank(query, candidates, top_k=top_k, return_scores=False)
                results.append(reranked)
        
        if return_scores:
            return results, scores_list
        return results

