"""Failure logging and analysis for retrieval improvements."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional

from src.utils import logger
from src.table_processor import extract_numeric_values, enhance_query_for_numerics


class FailureLogger:
    """
    Log retrieval failures for analysis and improvement.
    
    Tracks:
    - Queries where no relevant docs found in top-K
    - Queries where reranking didn't improve results
    - Missing numeric values in retrieved chunks
    - Suggestions for parameter tuning
    """
    
    def __init__(self, ground_truth: Optional[Dict[int, Set[int]]] = None):
        """
        Initialize failure logger.
        
        Args:
            ground_truth: Dict mapping q_id to set of relevant web_ids
        """
        self.ground_truth = ground_truth or {}
        self.failures = []
    
    def log_retrieval_failure(
        self,
        q_id: int,
        query: str,
        retrieved_web_ids: List[int],
        candidates_before_rerank: List[Dict],
        candidates_after_rerank: Optional[List[Dict]] = None,
        k: int = 5
    ):
        """
        Log retrieval failure: no relevant docs in top-K.
        
        Args:
            q_id: Question ID
            query: Query text
            retrieved_web_ids: Retrieved web_ids (top-K)
            candidates_before_rerank: Full candidate list before reranking
            candidates_after_rerank: Full candidate list after reranking (optional)
            k: Top-K value
        """
        if q_id not in self.ground_truth:
            return  # No ground truth, can't determine failure
        
        relevant = self.ground_truth[q_id]
        top_k_web_ids = set(retrieved_web_ids[:k])
        
        # Check if any relevant in top-K
        found_relevant = bool(top_k_web_ids & relevant)
        
        if not found_relevant:
            # Check if relevant exists in full candidate list
            all_retrieved = {c.get("doc_id") for c in candidates_before_rerank}
            relevant_in_candidates = bool(all_retrieved & relevant)
            
            # Extract numeric values from query
            query_numerics = extract_numeric_values(query)
            
            failure = {
                "q_id": q_id,
                "query": query,
                "failure_type": "no_relevant_in_topk",
                "top_k_web_ids": retrieved_web_ids[:k],
                "relevant_web_ids": list(relevant),
                "relevant_found_in_candidates": relevant_in_candidates,
                "query_numerics": query_numerics,
                "candidates_count": len(candidates_before_rerank),
                "suggestions": self._generate_suggestions(
                    query, query_numerics, relevant_in_candidates, candidates_before_rerank
                )
            }
            
            # Add reranking info if available
            if candidates_after_rerank:
                rerank_improved = self._check_rerank_improvement(
                    candidates_before_rerank, candidates_after_rerank, relevant, k
                )
                failure["rerank_improved"] = rerank_improved
                failure["candidates_after_rerank"] = [
                    {"doc_id": c.get("doc_id"), "score": c.get("score", 0)}
                    for c in candidates_after_rerank[:k]
                ]
            
            self.failures.append(failure)
            logger.warning(f"Retrieval failure for q_id={q_id}: no relevant in top-{k}")
    
    def _check_rerank_improvement(
        self,
        before: List[Dict],
        after: List[Dict],
        relevant: Set[int],
        k: int
    ) -> bool:
        """Check if reranking improved results."""
        before_top_k = {c.get("doc_id") for c in before[:k]} & relevant
        after_top_k = {c.get("doc_id") for c in after[:k]} & relevant
        
        return len(after_top_k) > len(before_top_k)
    
    def _generate_suggestions(
        self,
        query: str,
        query_numerics: List[Dict],
        relevant_in_candidates: bool,
        candidates: List[Dict]
    ) -> List[str]:
        """Generate suggestions for improving retrieval."""
        suggestions = []
        
        # Check for numeric values
        if query_numerics:
            suggestions.append(
                "Query contains numeric values - consider enhancing BM25 weight "
                "or using enhanced query with numeric variations"
            )
            
            # Check if numerics appear in candidates
            candidate_texts = [c.get("contents", c.get("text", "")) for c in candidates[:10]]
            numerics_found = False
            for num_info in query_numerics:
                num_value = num_info["value"]
                if any(num_value in text for text in candidate_texts):
                    numerics_found = True
                    break
            
            if not numerics_found and query_numerics:
                suggestions.append(
                    f"Numeric value {query_numerics[0].get('value', 'unknown')} not found in top candidates - "
                    "may need better numeric matching or query expansion"
                )
        
        # If relevant not in candidates at all
        if not relevant_in_candidates:
            suggestions.append(
                "Relevant documents not in candidate list - consider: "
                "1) Increase k_retrieve, 2) Adjust hybrid weights, 3) Try RRF fusion"
            )
        else:
            suggestions.append(
                "Relevant documents in candidates but not in top-K - consider: "
                "1) Improve reranker, 2) Adjust rerank top-K, 3) Check reranker model"
            )
        
        # Query length analysis (safe string split)
        if query and isinstance(query, str):
            query_words = query.split()
            if len(query_words) < 3:
                suggestions.append("Short query - consider query expansion or synonym matching")
            elif len(query_words) > 20:
                suggestions.append("Long query - consider query summarization or key phrase extraction")
        
        return suggestions
    
    def check_numeric_matching(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5
    ) -> Dict:
        """
        Check if numeric values from query appear in retrieved chunks.
        
        Returns:
            Dict with numeric matching analysis
        """
        query_numerics = extract_numeric_values(query)
        if not query_numerics:
            return {"has_numerics": False}
        
        top_candidates = candidates[:top_k]
        matching_analysis = {
            "has_numerics": True,
            "query_numerics": query_numerics,
            "matches": []
        }
        
        for num_info in query_numerics:
            num_value = num_info["value"]
            matches = []
            for cand in top_candidates:
                cand_text = cand.get("contents", cand.get("text", ""))
                if num_value in cand_text:
                    matches.append({
                        "doc_id": cand.get("doc_id"),
                        "chunk_id": cand.get("chunk_id"),
                        "context": cand_text[:200]  # First 200 chars
                    })
            
            matching_analysis["matches"].append({
                "value": num_value,
                "type": num_info["type"],
                "found_in_chunks": len(matches),
                "chunks": matches
            })
        
        return matching_analysis
    
    def save_failures(self, output_path: str):
        """Save failure logs to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "total_failures": len(self.failures),
                "failures": self.failures
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(self.failures)} failure logs to {output_path}")
    
    def get_failure_summary(self) -> Dict:
        """Get summary statistics of failures."""
        if not self.failures:
            return {"total": 0}
        
        summary = {
            "total": len(self.failures),
            "with_numerics": sum(1 for f in self.failures if f.get("query_numerics")),
            "relevant_in_candidates": sum(
                1 for f in self.failures if f.get("relevant_found_in_candidates", False)
            ),
            "rerank_improved": sum(
                1 for f in self.failures if f.get("rerank_improved", False)
            ),
            "common_suggestions": self._get_common_suggestions()
        }
        
        return summary
    
    def _get_common_suggestions(self) -> Dict[str, int]:
        """Count frequency of suggestions."""
        suggestion_counts = {}
        for failure in self.failures:
            for suggestion in failure.get("suggestions", []):
                # Extract key phrase from suggestion
                key = suggestion.split(":")[0] if ":" in suggestion else suggestion[:50]
                suggestion_counts[key] = suggestion_counts.get(key, 0) + 1
        
        return dict(sorted(suggestion_counts.items(), key=lambda x: x[1], reverse=True)[:5])

