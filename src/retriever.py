"""Hybrid retriever combining dense (FAISS) and sparse (BM25) search."""
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import faiss
except ImportError:
    raise ImportError("faiss is required. Install with: conda install -c pytorch faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

try:
    import bm25s
except ImportError:
    raise ImportError("bm25s is required. Install with: pip install bm25s[core]")

from src.utils import logger, load_jsonl
from src.table_processor import enhance_query_for_numerics


class HybridRetriever:
    """
    Hybrid retriever combining dense (FAISS) and sparse (BM25) search.
    
    Algorithm: 
    - Dense retrieval: FAISS index with sentence-transformers embeddings
    - Sparse retrieval: BM25 index
    - Hybrid scoring: weighted combination of normalized scores
    
    Time Complexity: O(k * log(n)) for FAISS, O(k) for BM25
    Space Complexity: O(n * d) for FAISS index, O(n * vocab) for BM25
    """
    
    def __init__(
        self,
        faiss_index_path: str,
        faiss_meta_path: str,
        bm25_index_path: str,
        embedding_model_name: str,
        weight_dense: float = 0.6,
        weight_bm25: float = 0.4,
        device: str = "cpu",
        normalize_embeddings: bool = True
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            faiss_index_path: Path to FAISS index file
            faiss_meta_path: Path to FAISS metadata JSON file
            bm25_index_path: Path to BM25 index directory
            embedding_model_name: Name of sentence-transformers model
            weight_dense: Weight for dense retrieval scores
            weight_bm25: Weight for BM25 scores
            device: Device for embedding model ("cpu" or "cuda")
            normalize_embeddings: Whether to normalize embeddings
        """
        self.weight_dense = weight_dense
        self.weight_bm25 = weight_bm25
        self.normalize_embeddings = normalize_embeddings
        
        # load FAISS index
        logger.info(f"Loading FAISS index from {faiss_index_path}")
        self.faiss_index = faiss.read_index(faiss_index_path)
        
        # load FAISS metadata
        if faiss_meta_path and Path(faiss_meta_path).exists():
            with open(faiss_meta_path, "r", encoding="utf-8") as f:
                self.faiss_meta = json.load(f)
            self.chunk_ids = self.faiss_meta.get("chunk_ids", [])
            self.doc_ids = self.faiss_meta.get("doc_ids", [])  # web_id for each chunk
        else:
            # infer from chunks if metadata not available
            logger.warning("FAISS metadata not found, inferring from chunks")
            self.faiss_meta = {}
            self.chunk_ids = []
            self.doc_ids = []
        
        # load embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        if normalize_embeddings:
            # normalize existing index embeddings if needed
            pass  # FAISS handles this
        
        # load BM25 index
        logger.info(f"Loading BM25 index from {bm25_index_path}")
        bm25_path = Path(bm25_index_path)
        
        # bm25s saves index in subdirectory "bm25"
        bm25_index_dir = bm25_path / "bm25" if (bm25_path / "bm25").exists() else bm25_path
        
        # load BM25 index
        try:
            import bm25s
            self.bm25_searcher = bm25s.BM25.load(str(bm25_index_dir), mmap=True, load_corpus=False)
            
            # load tokenizer if available
            try:
                import Stemmer
                stemmer = Stemmer.Stemmer("english")
                self.bm25_tokenizer = bm25s.tokenization.Tokenizer(stopwords="en", stemmer=stemmer)
                self.bm25_tokenizer.load_stopwords(str(bm25_index_dir))
                self.bm25_tokenizer.load_vocab(str(bm25_index_dir))
            except:
                # fallback to simple tokenization
                self.bm25_tokenizer = None
                logger.warning("BM25 tokenizer not available, using simple tokenization")
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            self.bm25_searcher = None
            self.bm25_tokenizer = None
        
        # load chunks for BM25
        chunks_file = bm25_path / "chunks.jsonl"
        if not chunks_file.exists():
            chunks_file = bm25_index_dir / "chunks.jsonl"
        
        if chunks_file.exists():
            self.chunks = load_jsonl(str(chunks_file))
            # set corpus for bm25s
            if self.bm25_searcher:
                self.bm25_searcher.corpus = self.chunks
        else:
            logger.warning("BM25 chunks.jsonl not found")
            self.chunks = None

        # build chunk metadata mappings
        self.chunk_id_to_index: Dict[str, int] = {}
        if self.chunks:
            for idx, chunk in enumerate(self.chunks):
                chunk_id = chunk.get("id") or chunk.get("chunk_id")
                if chunk_id is not None:
                    self.chunk_id_to_index[str(chunk_id)] = idx

            # ensure chunk_ids align with chunk order if missing
            if not self.chunk_ids:
                self.chunk_ids = [chunk.get("id", f"chunk_{idx}") for idx, chunk in enumerate(self.chunks)]

            # ensure doc_ids align with chunk order if missing
            if not self.doc_ids:
                self.doc_ids = [
                    chunk.get("doc_id") or (chunk.get("id", "").split("_")[0] if chunk.get("id") else None)
                    for chunk in self.chunks
                ]
        else:
            if not self.chunk_ids:
                self.chunk_ids = []
            if not self.doc_ids:
                self.doc_ids = []
        
        logger.info(f"Initialized hybrid retriever: dense_weight={weight_dense}, bm25_weight={weight_bm25}")
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        if len(scores) == 0:
            return scores
        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)
    
    def _retrieve_dense(self, query: str, k: int) -> Tuple[List[int], np.ndarray]:
        """
        Retrieve using dense (FAISS) search.
        
        Returns:
            (indices, scores) - chunk indices and similarity scores
        """
        # encode query
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False
        )[0]
        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        
        # search in FAISS
        k = min(k, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(query_embedding, k)
        
        # FAISS returns squared L2 distances for inner product, convert to similarities
        # For inner product: higher is better, so we use scores as-is
        # For L2: lower is better, so we'd need to invert
        
        return indices[0].tolist(), scores[0]
    
    def _retrieve_bm25(self, query: str, k: int, enhance_numerics: bool = True) -> Tuple[List[int], np.ndarray]:
        """
        Retrieve using sparse (BM25) search.
        
        Args:
            query: Query string
            k: Number of results
            enhance_numerics: Whether to enhance query for numeric matching
        
        Returns:
            (indices, scores) - chunk indices and BM25 scores
        """
        if not self.bm25_searcher:
            # fallback: return empty results
            return [], np.array([])
        
        try:
            import bm25s
            # Enhance query for numeric values if enabled
            enhanced_query = enhance_query_for_numerics(query) if enhance_numerics else query
            
            # tokenize query using bm25s
            if self.bm25_tokenizer:
                query_tokens = self.bm25_tokenizer.tokenize(
                    [enhanced_query],
                    return_as="ids",
                    update_vocab=False
                )
            else:
                # simple tokenization using default tokenizer
                tokenized = bm25s.tokenize(
                    [enhanced_query],
                    return_ids=True,
                    show_progress=False,
                    leave=False
                )
                query_tokens = tokenized.ids if hasattr(tokenized, "ids") else tokenized[0]
            
            # retrieve using bm25s
            results, scores = self.bm25_searcher.retrieve(query_tokens, k=k)

            # convert results to chunk indices
            indices: List[int] = []
            filtered_scores: List[float] = []
            docs: List = []
            if results is not None and len(results) > 0:
                docs = results[0]
                if isinstance(docs, np.ndarray):
                    docs = docs.tolist()
            score_entries: List = []
            if scores is not None and len(scores) > 0:
                score_entries = scores[0]
                if isinstance(score_entries, np.ndarray):
                    score_entries = score_entries.tolist()
                else:
                    score_entries = list(score_entries)
            if score_entries and len(score_entries) != len(docs):
                score_entries = score_entries[: len(docs)]
            if not score_entries:
                score_entries = [0.0] * len(docs)

            for doc, bm25_score in zip(docs, score_entries):
                appended = False
                if isinstance(doc, dict):
                    chunk_id = doc.get("id") or doc.get("chunk_id")
                    if chunk_id is not None:
                        chunk_id_str = str(chunk_id)
                        if chunk_id_str in self.chunk_id_to_index:
                            indices.append(self.chunk_id_to_index[chunk_id_str])
                            appended = True
                        elif self.chunk_ids and chunk_id_str in self.chunk_ids:
                            indices.append(self.chunk_ids.index(chunk_id_str))
                            appended = True
                    doc_id = doc.get("doc_id")
                    if not appended and doc_id is not None and self.doc_ids:
                        doc_id_str = str(doc_id)
                        for idx, existing_doc_id in enumerate(self.doc_ids):
                            if str(existing_doc_id) == doc_id_str:
                                indices.append(idx)
                                appended = True
                                break
                else:
                    try:
                        indices.append(int(doc))
                        appended = True
                    except (TypeError, ValueError):
                        appended = False

                if appended:
                    filtered_scores.append(float(bm25_score))
                else:
                    logger.debug("Skipped BM25 result without valid chunk mapping")
            
            score_array = np.array(filtered_scores, dtype=float)
            
            return indices, score_array
        except Exception as e:
            logger.warning(f"BM25 retrieval failed: {e}, returning empty results")
            return [], np.array([])
    
    def _rrf_fusion(self, dense_ranks: Dict[int, int], bm25_ranks: Dict[int, int], k: int = 60) -> Dict[int, float]:
        """
        Reciprocal Rank Fusion (RRF) for combining retrieval results.
        
        RRF formula: score = sum(1 / (k + rank)) for each retrieval method
        
        Args:
            dense_ranks: Dict mapping chunk_idx to rank (1-based)
            bm25_ranks: Dict mapping chunk_idx to rank (1-based)
            k: RRF constant (typically 60)
            
        Returns:
            Dict mapping chunk_idx to RRF score
        """
        rrf_scores = {}
        all_indices = set(dense_ranks.keys()) | set(bm25_ranks.keys())
        
        for idx in all_indices:
            score = 0.0
            if idx in dense_ranks:
                score += 1.0 / (k + dense_ranks[idx])
            if idx in bm25_ranks:
                score += 1.0 / (k + bm25_ranks[idx])
            rrf_scores[idx] = score
        
        return rrf_scores
    
    def retrieve(
        self,
        query: str,
        k: int = 50,
        return_scores: bool = False,
        fusion_method: str = "weighted"  # "weighted" or "rrf"
    ) -> List[Dict] | Tuple[List[Dict], Dict]:
        """
        Hybrid retrieval combining dense and sparse search.
        
        Args:
            query: Query string
            k: Number of candidates to retrieve
            return_scores: Whether to return scores
            
        Returns:
            List of candidate chunks with doc_id (web_id), or (chunks, scores_dict) if return_scores
        """
        # retrieve from both indices
        dense_indices, dense_scores = self._retrieve_dense(query, k)
        bm25_indices, bm25_scores = self._retrieve_bm25(query, k, enhance_numerics=True)
        
        # normalize scores
        dense_scores_norm = self._normalize_scores(dense_scores)
        bm25_scores_norm = self._normalize_scores(bm25_scores)
        
        # create score maps and rank maps
        dense_score_map = {idx: score for idx, score in zip(dense_indices, dense_scores_norm)}
        bm25_score_map = {idx: score for idx, score in zip(bm25_indices, bm25_scores_norm)}
        
        # create rank maps (1-based) for RRF
        dense_rank_map = {idx: rank + 1 for rank, idx in enumerate(dense_indices)}
        bm25_rank_map = {idx: rank + 1 for rank, idx in enumerate(bm25_indices)}
        
        # combine candidates
        all_indices = set(dense_indices) | set(bm25_indices)
        
        # compute hybrid scores
        if fusion_method == "rrf":
            # Reciprocal Rank Fusion
            hybrid_scores = self._rrf_fusion(dense_rank_map, bm25_rank_map, k=60)
        else:
            # Weighted sum (default)
            hybrid_scores = {}
            for idx in all_indices:
                dense_score = dense_score_map.get(idx, 0.0)
                bm25_score = bm25_score_map.get(idx, 0.0)
                hybrid_score = self.weight_dense * dense_score + self.weight_bm25 * bm25_score
                hybrid_scores[idx] = hybrid_score
        
        # get top-k by hybrid score
        sorted_indices = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # build result chunks
        results = []
        for idx, score in sorted_indices:
            # get doc_id - try multiple sources
            if self.doc_ids and idx < len(self.doc_ids):
                doc_id = self.doc_ids[idx]
            elif self.chunks and idx < len(self.chunks):
                doc_id = self.chunks[idx].get("doc_id", self.chunks[idx].get("id", "").split("_")[0])
            else:
                # fallback: extract from chunk_id
                chunk_id = self.chunk_ids[idx] if self.chunk_ids and idx < len(self.chunk_ids) else f"chunk_{idx}"
                doc_id = chunk_id.split("_")[0] if "_" in chunk_id else chunk_id
            
            chunk_id = self.chunk_ids[idx] if self.chunk_ids and idx < len(self.chunk_ids) else f"chunk_{idx}"
            
            chunk_data = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,  # web_id
                "score": float(score),
                "dense_score": float(dense_score_map.get(idx, 0.0)),
                "bm25_score": float(bm25_score_map.get(idx, 0.0))
            }
            
            # add chunk content if available
            if self.chunks and idx < len(self.chunks):
                chunk_data["contents"] = self.chunks[idx].get("contents", self.chunks[idx].get("text", ""))
            
            results.append(chunk_data)
        
        if return_scores:
            scores_dict = {
                "dense": {idx: float(s) for idx, s in zip(dense_indices, dense_scores)},
                "bm25": {idx: float(s) for idx, s in zip(bm25_indices, bm25_scores)},
                "hybrid": {idx: float(s) for idx, s in hybrid_scores.items()}
            }
            return results, scores_dict
        return results
    
    def batch_retrieve(
        self,
        queries: List[str],
        k: int = 50,
        return_scores: bool = False
    ) -> List[List[Dict]] | Tuple[List[List[Dict]], List[Dict]]:
        """
        Batch retrieval for multiple queries.
        
        Args:
            queries: List of query strings
            k: Number of candidates per query
            return_scores: Whether to return scores
            
        Returns:
            List of results per query, or (results, scores_list) if return_scores
        """
        results = []
        scores_list = []
        
        for query in queries:
            if return_scores:
                result, scores = self.retrieve(query, k=k, return_scores=True)
                results.append(result)
                scores_list.append(scores)
            else:
                result = self.retrieve(query, k=k, return_scores=False)
                results.append(result)
        
        if return_scores:
            return results, scores_list
        return results

