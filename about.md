# About This Project

This document expands on `README.md` and captures every technical decision behind the AlfaBank RAG pipeline. The guiding ideas were:

- Embrace **FlashRAG** for retrieval, fusion, and reranking instead of custom plumbing.
- Enforce **GPU-only** execution with explicit kernel checks.
- Optimize for **accuracy** with advanced techniques (ensemble reranking, query expansion, text normalization).
- Improve **observability** (timings, diagnostics, failure logs).
- Guarantee **submission shape** parity with the provided sample.

---

## 1. Data & Processing Flow

```
raw CSV ──► ingest (clean, normalize, remove footers) ──► corpus.jsonl
                │
                └──► semantic chunker (preserve tables/headings) ──► chunks.jsonl
                                     │
                                     ├──► FlashRAG bm25s index (sparse, Russian-optimized)
                                     └──► FlashRAG FAISS index (dense, multilingual-e5-large)

questions.csv ──► query normalization ──► query expansion (banking synonyms)
                                          │
                                          └──► FlashRAG MultiRetrieverRouter
                                               (dense + bm25s + RRF fusion)
                                               │
                                               ├──► score filtering (optional)
                                               ├──► document type filtering (optional)
                                               └──► FlashRAG CrossReranker (fp16 on CUDA)
                                                    │
                                                    ├──► Optional: Ensemble reranker
                                                    ├──► Optional: Second-pass reranking
                                                    └──► deduplication by web_id
                                                         │
                                                         └──► hit@k metrics / submit.csv
```

### Chunking & Enrichment

- **Semantic chunking**: `src/chunker.py` / `src/semantic_chunker.py` preserve tables and headings intact.
- **Text normalization**: `src/text_processor.py` applies configurable normalization (smart mode preserves important symbols like `|`, `%`, `.`, `,`, `-`).
- **Footer removal**: Automatic removal of repetitive legal/copyright footers from documents.
- **Query expansion**: `src/table_processor.py` enriches queries with banking domain synonyms and numeric patterns.
- **Shared corpus**: The same `chunks.jsonl` is shared by both FAISS and bm25s indexes to avoid mismatched doc IDs.

---

## 2. FlashRAG Integration Details

| Component | Notes |
|-----------|-------|
| `flashrag/retriever/BM25Retriever` | Patched for Russian language support with custom stopwords list (no stemming). Materializes `chunk_id`, `doc_id`, `contents` for every hit (bm25s backend). |
| `flashrag/retriever/MultiRetrieverRouter` | Supports `merge_method="rrf"` (Reciprocal Rank Fusion) and `"weighted"`. Normalized per-source scores, tracked provenance (`source_scores`, `sources`). |
| `flashrag/config/config.Config` | Device initialization calls `get_device()` (CUDA kernel sanity-check) and raises `GPUMisconfigurationError` if no GPU. |
| `src/retriever.HybridRetriever` | Builds FlashRAG config on the fly, maps retriever names to weights, applies query normalization, supports filtering and score thresholds. |
| `src/reranker.Reranker` | Wraps FlashRAG `CrossReranker`, keeps fp16 + batch configuration, emits reranked candidates with scores. |
| `src/reranker_ensemble.RerankerEnsemble` | Combines multiple rerankers with weighted averaging, optional second-pass reranking on top candidates. |

### Fusion Methods

**RRF (Reciprocal Rank Fusion)** - Default, more robust:
```python
score = sum(1 / (rrf_k + rank)) for each retriever
```

**Weighted Fusion** - Normalizes each retriever's scores, applies configured weights:
```json
{
  "chunk_id": "42_0_0",
  "doc_id": "42",
  "score": 0.88,
  "dense_score": 0.74,
  "bm25_score": 0.93,
  "source_scores": {"dense": 0.74, "bm25": 0.93},
  "sources": ["bm25", "dense"],
  "contents": "... original chunk text ..."
}
```

---

## 3. GPU Enforcement

- `src/utils.resolve_device()` forbids `"cpu"` and delegates to FlashRAG's `get_device()` which:
  1. Calls `torch.cuda.is_available()`.
  2. Allocates a 1-element CUDA tensor and synchronizes.
  3. Inspects emitted warnings for "no kernel image" / "capability mismatch".
  4. Raises `GPUMisconfigurationError` if anything fails.

- FlashRAG's config class performs the same check during initialization so any CLI (`make build_index`, `make eval`, etc.) aborts early if the GPU build is incompatible.
- SentenceTransformer + CrossReranker run in fp16 when the device capability ≥ 6.0 (Pascal).
- FAISS index is built on GPU with fp16 for faster search.

---

## 4. Accuracy Optimizations

### Model Upgrades
- **Embeddings**: Upgraded from `multilingual-e5-base` to `multilingual-e5-large` for better semantic understanding.
- **Reranker**: Upgraded from `ms-marco-MiniLM-L-12-v2` (English-only) to `BAAI/bge-reranker-base` (multilingual).

### Retrieval Parameters
- **k_retrieve**: Increased to 180 (from 30) for better recall before reranking.
- **k_rerank**: Increased to 55 (from 20) for more candidates to rerank.
- **RRF k**: Set to 70 for better precision (higher = more weight to top ranks).

### Chunking Strategy
- **Chunk size**: Reduced to 280 tokens (from 600) for better precision on short banking facts.
- **Overlap**: Increased to 110 tokens (from 150) for better context preservation.
- **Semantic chunking**: Enabled to preserve table structure and headings.

### Text Processing
- **Normalization mode**: "smart" preserves important symbols (|, %, ., ,, -) while normalizing text.
- **Footer removal**: Automatic removal of repetitive legal/copyright footers.
- **Title normalization**: Titles are normalized consistently with text content.

### Query Enhancement
- **Banking synonyms**: Extended synonym dictionary for banking domain terms.
- **Query rewriting**: Common question reformulations (как, где, когда, сколько, можно, нужно).
- **Numeric patterns**: Enhanced matching for numeric values and variations.

### Advanced Techniques
- **Ensemble reranking**: Optional combination of multiple reranker models with weighted averaging.
- **Second-pass reranking**: Optional second reranking pass on top candidates for refinement.
- **Score filtering**: Optional minimum score threshold before reranking.
- **Document type filtering**: Optional filtering by document type (has_table, etc.).
- **Table preference**: Optional boosting of table chunks for numeric queries.

---

## 5. Multi-threaded Batched Execution & Logging

- **Parallel batch processing**: `ThreadPoolExecutor` with configurable `num_workers` (default: 4) processes multiple batches concurrently.
- **Optimized batch sizing**: `src/batch_processor.optimize_batch_size()` adjusts batch sizes based on dataset size and hardware capabilities.
- **Resource efficiency**:
  - Prefetching batches to keep GPU utilization high
  - Memory-efficient candidate processing (avoiding unnecessary copies)
  - Thread-safe result aggregation
- Retrieval and reranking execute in configurable batches (`retrieval.batch_size`, `models.embeddings.batch_size`, `models.reranker.batch_size`).
- `scripts/eval.py` records:
  - Batch timings (retrieval + rerank) with ETA estimation.
  - Diagnostic Recall@50 before reranking for the first batch.
  - Candidate snapshots before and after rerank to `outputs/logs/candidates_*.jsonl`.
  - Failure summaries (missing gold in top-k) with full candidate context.
- `scripts/benchmark.py --n 200` prints throughput per stage and extrapolated run time.
- `scripts/quick_test.py` provides fast validation (~15 minutes) without full rebuild.

---

## 6. Russian Language Support

### BM25 Tokenization
- **Language detection**: Automatic detection of Russian text using `langid`.
- **Custom stopwords**: Russian-specific stopwords list (no stemming support in bm25s for Russian).
- **Tokenizer consistency**: Same tokenizer used for indexing and searching to avoid mismatches.

### Text Normalization
- **Cyrillic support**: Normalization preserves Cyrillic characters (а-яё).
- **Smart mode**: Preserves important symbols while normalizing (useful for banking data with percentages, amounts, etc.).

---

## 7. Submission Guarantees

`scripts/submit.py` enforces:

1. Generated `q_id` set matches exactly the one in `sample_submission.csv`.
2. Each `web_list` has `k_final` entries (with deterministic padding fallback).
3. **Deduplication by web_id**: Only the best chunk per document is kept in final results.

The CSV format is therefore stable and leaderboard-ready.

---

## 8. Configuration Cheatsheet

### Retrieval
- `retrieval.fusion_method`: `"rrf"` (default, more robust) or `"weighted"`.
- `retrieval.rrf_k`: RRF hyperparameter (default: 70, higher = more weight to top ranks).
- `retrieval.hybrid_weight_dense` / `hybrid_weight_bm25`: Control score fusion balance (used only if `fusion_method="weighted"`).
- `retrieval.enhance_numerics`: Toggles numeric synonym expansion for bm25s queries.
- `retrieval.min_score_threshold`: Minimum score threshold before reranking (0.0 = no filtering).
- `retrieval.filter_by_document_type`: Filter candidates by document type (has_table, etc.).
- `retrieval.prefer_table_chunks`: Prefer chunks with tables for numeric queries.

### Text Processing
- `text_processing.normalize_for_retrieval`: Apply text normalization (default: true).
- `text_processing.normalization_mode`: `"letters_numbers"` (default), `"smart"`, or `"aggressive"`.

### Models
- `models.embeddings.device` / `models.reranker.device`: Keep as `"auto"`; the resolver guarantees CUDA.
- `models.faiss.use_gpu`: Keeps FAISS index on GPU; fp16 search is optional (`use_float16`).
- `models.reranker_ensemble.enabled`: Enable ensemble reranking (default: false).
- `models.reranker_ensemble.second_pass`: Enable second-pass reranking (default: false).

---

## 9. Performance Recommendations

### Accuracy vs Speed Trade-offs

**For Maximum Accuracy:**
- `k_retrieve`: 180-200
- `k_rerank`: 50-60
- `chunk_size`: 250-300 tokens
- `overlap`: 100-125 tokens
- Use `multilingual-e5-large` for embeddings
- Use `BAAI/bge-reranker-base` for reranking
- Enable ensemble reranker
- Enable second-pass reranking
- Use RRF fusion with `rrf_k=70-80`

**For Faster Processing:**
- `k_retrieve`: 100-150
- `k_rerank`: 30-40
- `chunk_size`: 400-500 tokens
- `overlap`: 50-100 tokens
- Use `multilingual-e5-base` for embeddings
- Use smaller cross-encoder for reranking
- Disable ensemble reranker
- Use weighted fusion

### Multi-threading
- Adjust `retrieval.num_workers` (default: 4) based on hardware:
  - 4-6 workers for single GPU setups
  - 6-8 workers for multi-GPU or high-VRAM configurations
  - Monitor GPU utilization; aim for >80% during processing

### Batch Sizes
- Optimized defaults (embeddings: 64, reranker: 64) work well for most GPUs.
- Increase gradually while monitoring VRAM.
- For large models (e5-large), reduce batch size if OOM occurs.

### GPU Utilization
- The pipeline is designed to keep GPU busy with parallel batch processing and prefetching.
- RRF fusion is tuned for the provided corpus; if exploring other corpora, log `source_scores` to adjust weights.
- Rebuild indices only when `chunks.jsonl` changes; otherwise reuse persisted artefacts under `indexes/`.

---

## 10. Troubleshooting Checklist

- **CUDA error: no kernel image** → Install a torch build matching GPU compute capability (e.g. `pip install torch==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118` for compute 6.1).
- **FAISS GPU API missing** → Install `faiss-gpu` with proper CUDA support: `conda install -c conda-forge faiss-gpu=1.7.4 cudatoolkit=11.8`.
- **bm25 index missing** → Ensure `make build_index` completed; check `indexes/bm25/bm25/`.
- **Dimension mismatch in FAISS** → Delete `indexes/faiss/`, rebuild with `make build_index` (corpus/model changed).
- **Reranker OOM** → Lower `models.reranker.batch_size` or switch to a smaller cross-encoder.
- **Empty stopwords/vocab for Russian** → This is expected: Russian tokenization uses custom stopwords without stemming.
- **Pooling method warning** → This is informational; default pooling (mean) is used if not detected.
- **Ground truth not found** → Provide via `--ground_truth` argument or `evaluation.ground_truth_path` in config.

---

## 11. Extensibility Ideas

- Add FlashRAG refiner/compressor modules for selective context.
- Support alternative ANN layouts (IVF, HNSW) by swapping dense config.
- Persist retrieval caches (currently disabled for determinism) for active-learning scenarios.
- Build a simple dashboard by parsing `outputs/reports/*.json`.
- Add more sophisticated query rewriting using LLMs.
- Implement adaptive chunking based on document structure.
- Add support for more languages with language-specific tokenization.

---

## 12. Git & Repo Hygiene

- Index artefacts, logs, and outputs are ignored via `.gitignore`.
- Vendored FlashRAG lives under `FlashRAG/`; reinstall with `pip install -e FlashRAG` after pulling updates involving patched modules.
- Line endings normalized through `.gitattributes` (if present).
- Data files (`*.jsonl`, `*.csv`) are ignored but directory structure is preserved with `.gitkeep` files.

---

## 13. Key Improvements Applied

This pipeline includes numerous optimizations for accuracy:

1. **Model upgrades**: multilingual-e5-large, BAAI/bge-reranker-base
2. **Retrieval parameters**: k_retrieve=180, k_rerank=55, rrf_k=70
3. **Chunking**: size=280, overlap=110, semantic chunking enabled
4. **Text processing**: smart normalization, footer removal, title normalization
5. **Query enhancement**: banking synonyms, query rewriting, numeric patterns
6. **Advanced techniques**: ensemble reranking, second-pass reranking, filtering options
7. **Russian support**: optimized tokenization with custom stopwords
8. **GPU optimization**: strict CUDA validation, FP16 support, multi-threaded batching

---

This refactor delivers a maintainable, transparent, and accurate GPU-native pipeline by leaning on FlashRAG's proven components while keeping configuration and observability squarely in project code. For API details refer to `src/retriever.py`, `src/reranker.py`, `src/reranker_ensemble.py`, and the FlashRAG documentation bundled under `FlashRAG/docs/`.
