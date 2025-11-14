"""Evaluation metrics for retrieval: Hit@K, Recall@K, MRR, NDCG."""
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.utils import logger


class RetrievalEvaluator:
    """
    Evaluator for retrieval quality metrics.
    
    Metrics:
    - Hit@K: Fraction of queries with at least one relevant doc in top-K
    - Recall@K: Average fraction of relevant docs found in top-K
    - MRR: Mean Reciprocal Rank of first relevant doc
    - NDCG@K: Normalized Discounted Cumulative Gain at K
    """
    
    def __init__(self, ground_truth: Optional[Dict[int, Set[int]]] = None):
        """
        Initialize evaluator.
        
        Args:
            ground_truth: Dict mapping q_id to set of relevant web_ids
                         If None, will try to load from file or use empty dict
        """
        self.ground_truth = ground_truth or {}
    
    def load_ground_truth(self, file_path: str) -> None:
        """Load ground truth from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            self.ground_truth = {
                int(q_id): set(web_ids) if isinstance(web_ids, list) else {web_ids}
                for q_id, web_ids in json.load(f).items()
            }
        logger.info(f"Loaded ground truth for {len(self.ground_truth)} queries")
    
    def hit_at_k(self, q_id: int, retrieved_web_ids: List[int], k: int) -> int:
        """
        Check if at least one relevant doc is in top-K.
        
        Returns:
            1 if hit, 0 otherwise
        """
        if q_id not in self.ground_truth:
            return 0
        
        relevant = self.ground_truth[q_id]
        top_k = retrieved_web_ids[:k]
        return 1 if any(web_id in relevant for web_id in top_k) else 0
    
    def recall_at_k(self, q_id: int, retrieved_web_ids: List[int], k: int) -> float:
        """
        Compute Recall@K for a query.
        
        Returns:
            Fraction of relevant docs found in top-K
        """
        if q_id not in self.ground_truth:
            return 0.0
        
        relevant = self.ground_truth[q_id]
        if len(relevant) == 0:
            return 0.0
        
        top_k = retrieved_web_ids[:k]
        found = sum(1 for web_id in top_k if web_id in relevant)
        return found / len(relevant)
    
    def mrr(self, q_id: int, retrieved_web_ids: List[int]) -> float:
        """
        Compute Mean Reciprocal Rank for a query.
        
        Returns:
            Reciprocal rank of first relevant doc, or 0.0 if none found
        """
        if q_id not in self.ground_truth:
            return 0.0
        
        relevant = self.ground_truth[q_id]
        if len(relevant) == 0:
            return 0.0
        
        for rank, web_id in enumerate(retrieved_web_ids, start=1):
            if web_id in relevant:
                return 1.0 / rank
        return 0.0
    
    def ndcg_at_k(self, q_id: int, retrieved_web_ids: List[int], k: int) -> float:
        """
        Compute NDCG@K for a query.
        
        Returns:
            Normalized Discounted Cumulative Gain at K
        """
        if q_id not in self.ground_truth:
            return 0.0
        
        relevant = self.ground_truth[q_id]
        if len(relevant) == 0:
            return 0.0
        
        # compute DCG@K
        dcg = 0.0
        top_k = retrieved_web_ids[:k]
        for rank, web_id in enumerate(top_k, start=1):
            if web_id in relevant:
                # relevance = 1 for relevant docs
                dcg += 1.0 / np.log2(rank + 1)
        
        # compute IDCG@K (ideal DCG with all relevant docs at top)
        num_relevant = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate(
        self,
        results: Dict[int, List[int]],
        k_values: List[int] = [5, 10],
        metrics: List[str] = ["hit@5", "recall@5", "recall@10", "mrr", "ndcg@5"]
    ) -> Dict:
        """
        Evaluate retrieval results.
        
        Args:
            results: Dict mapping q_id to list of retrieved web_ids
            k_values: List of K values for metrics
            metrics: List of metric names to compute
            
        Returns:
            Dictionary with metric values
        """
        metrics_dict = {}
        
        # compute per-query metrics
        hit_scores = {k: [] for k in k_values}
        recall_scores = {k: [] for k in k_values}
        mrr_scores = []
        ndcg_scores = {k: [] for k in k_values}
        
        for q_id, retrieved_web_ids in results.items():
            # Safely convert to int if needed
            if not retrieved_web_ids:
                # Skip empty results
                continue
            if isinstance(retrieved_web_ids[0], str):
                try:
                    retrieved_web_ids = [int(wid) for wid in retrieved_web_ids if wid]
                except (ValueError, TypeError):
                    # If conversion fails, skip this query
                    continue
            
            for k in k_values:
                hit_scores[k].append(self.hit_at_k(q_id, retrieved_web_ids, k))
                recall_scores[k].append(self.recall_at_k(q_id, retrieved_web_ids, k))
                if k == 5:  # compute NDCG@5
                    ndcg_scores[k].append(self.ndcg_at_k(q_id, retrieved_web_ids, k))
            
            mrr_scores.append(self.mrr(q_id, retrieved_web_ids))
        
        # aggregate metrics
        for metric in metrics:
            if metric.startswith("hit@"):
                k = int(metric.split("@")[1])
                if k in hit_scores:
                    metrics_dict[metric] = np.mean(hit_scores[k])
            elif metric.startswith("recall@"):
                k = int(metric.split("@")[1])
                if k in recall_scores:
                    metrics_dict[metric] = np.mean(recall_scores[k])
            elif metric == "mrr":
                metrics_dict["mrr"] = np.mean(mrr_scores)
            elif metric.startswith("ndcg@"):
                k = int(metric.split("@")[1])
                if k in ndcg_scores:
                    metrics_dict[metric] = np.mean(ndcg_scores[k])
        
        # add per-query breakdown for debugging
        metrics_dict["num_queries"] = len(results)
        metrics_dict["num_queries_with_gt"] = sum(
            1 for q_id in results.keys() if q_id in self.ground_truth
        )
        
        return metrics_dict

