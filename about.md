# About This Project

This document expands on `README.md` and captures every technical decision behind the AlfaBank RAG pipeline refactor. The guiding ideas were:

- Embrace **FlashRAG** for retrieval, fusion, and reranking instead of custom plumbing.
- Enforce **GPU-only** execution with explicit kernel checks.
- Improve **observability** (timings, diagnostics, failure logs).
- Guarantee **submission shape** parity with the provided sample.

---

## 1. Data & Processing Flow

```
raw CSV ──► ingest ──► corpus.jsonl
                │
                └──► chunker + semantic_chunker ──► chunks.jsonl
                                     │
                                     ├──► FlashRAG bm25s index (sparse)
                                     └──► FlashRAG FAISS index (dense)

questions.csv ──► FlashRAG MultiRetrieverRouter (dense + bm25s + weighted fusion)
                                          │
                                          └──► FlashRAG CrossReranker (fp16 on CUDA)
                                                   │
                                                   └──► hit@k metrics / submit.csv
```

### Chunking & Enrichment

- `src/chunker.py` / `src/semantic_chunker.py` keep tables and headings intact.
- `src/table_processor.py` enriches queries with numeric synonyms (rates, amounts) before sparse retrieval.
- The same `chunks.jsonl` is shared by both FAISS and bm25s indexes to avoid mismatched doc IDs.

---

## 2. FlashRAG Integration Details

| Component | Notes |
|-----------|-------|
| `flashrag/retriever/BM25Retriever` | Patched to materialise `chunk_id`, `doc_id`, `contents` for every hit (bm25s backend). |
| `flashrag/retriever/MultiRetrieverRouter` | Introduced `merge_method="weighted"`, normalised per-source scores, tracked provenance (`source_scores`, `sources`). |
| `flashrag/config/config.Config` | Device initialisation now calls `get_device()` (CUDA kernel sanity-check) and raises if no GPU. |
| `src/retriever.HybridRetriever` | Builds FlashRAG config on the fly, maps retriever names to weights, and exposes a lightweight Pythonic API. |
| `src/reranker.Reranker` | Wraps FlashRAG `CrossReranker`, keeps fp16 + batch configuration, and emits reranked candidates with scores. |

Weighted fusion normalises each retriever’s scores, applies configured weights, and keeps raw contributions:

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

- `src/utils.resolve_device()` forbids `"cpu"` and delegates to FlashRAG’s `get_device()` which:
  1. Calls `torch.cuda.is_available()`.
  2. Allocates a 1-element CUDA tensor and synchronises.
  3. Inspects emitted warnings for “no kernel image” / “capability mismatch”.
  4. Raises `GPUMisconfigurationError` if anything fails.

- FlashRAG’s config class now performs the same check during initialisation so any CLI (`make build_index`, `make eval`, etc.) aborts early if the GPU build is incompatible.
- SentenceTransformer + CrossReranker run in fp16 when the device capability ≥ 6.0 (Pascal).

---

## 4. Multi-threaded Batched Execution & Logging

- **Parallel batch processing**: `ThreadPoolExecutor` with configurable `num_workers` (default: 4) processes multiple batches concurrently.
- **Optimized batch sizing**: `src/batch_processor.optimize_batch_size()` adjusts batch sizes based on dataset size and hardware capabilities.
- **Resource efficiency**:
  - Prefetching batches to keep GPU utilization high
  - Memory-efficient candidate processing (avoiding unnecessary copies)
  - Thread-safe result aggregation
- Retrieval and reranking execute in configurable batches (`retrieval.batch_size`, `models.embeddings.batch_size`, `models.reranker.batch_size`).
- `scripts/eval.py` records:
  - Batch timings (retrieval + rerank) with ETA estimation.
  - Candidate snapshots before and after rerank to `outputs/logs/candidates_*.jsonl`.
  - Failure summaries (missing gold in top-k) with full candidate context.
- `scripts/benchmark.py --n 200` prints throughput per stage and extrapolated run time.

---

## 5. Submission Guarantees

`scripts/submit.py` enforces:

1. Generated `q_id` set matches exactly the one in `sample_submission.csv`.
2. Each `web_list` has `k_final` entries (with deterministic padding fallback).

The CSV format is therefore stable and leaderboard-ready.

---

## 6. Configuration Cheatsheet

- `retrieval.fusion_method`: `"weighted"` (default), `"rrf"`, `"concat"`.
- `retrieval.hybrid_weight_dense` / `hybrid_weight_bm25`: control score fusion balance.
- `retrieval.enhance_numerics`: toggles numeric synonym expansion for bm25s queries.
- `models.embeddings.device` / `models.reranker.device`: keep as `"auto"`; the resolver guarantees CUDA.
- `models.faiss.use_gpu`: keeps FAISS index on GPU; fp16 search is optional (`use_float16`).

---

## 7. Performance Recommendations

- **k_retrieve** ≈ 25–30 gives a good trade-off between sparse recall and rerank latency.
- **Multi-threading**: Adjust `retrieval.num_workers` (default: 4) based on your hardware:
  - 4-6 workers for single GPU setups
  - 6-8 workers for multi-GPU or high-VRAM configurations
  - Monitor GPU utilization; aim for >80% during processing
- **Batch sizes**: Optimized defaults (embeddings: 128, reranker: 64) work well for most GPUs. Increase gradually while monitoring VRAM.
- **GPU utilization**: The pipeline is designed to keep GPU busy with parallel batch processing and prefetching.
- Weighted fusion is tuned for the provided corpus; if exploring other corpora, log `source_scores` to adjust weights.
- Rebuild indices only when `chunks.jsonl` changes; otherwise reuse persisted artefacts under `indexes/`.

---

## 8. Troubleshooting Checklist

- **CUDA error: no kernel image** → Install a torch build matching GPU compute capability (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu121` for compute ≥ 8.0).
- **bm25 index missing** → ensure `make build_index_bm25` completed; check `indexes/bm25/bm25/`.
- **Dimension mismatch in FAISS** → delete `indexes/faiss`, rebuild with `make build_index` (corpus/model changed).
- **Reranker OOM** → lower `models.reranker.batch_size` or switch to a smaller cross-encoder.

---

## 9. Extensibility Ideas

- Add FlashRAG refiner/compressor modules for selective context.
- Support alternative ANN layouts (IVF, HNSW) by swapping dense config.
- Persist retrieval caches (currently disabled for determinism) for active-learning scenarios.
- Build a simple dashboard by parsing `outputs/reports/*.json`.

---

## 10. Git & Repo Hygiene

- Index artefacts, logs, and outputs are ignored via `.gitignore`.
- Vendored FlashRAG lives under `FlashRAG/`; reinstall with `pip install -e FlashRAG` after pulling updates involving patched modules.
- Line endings normalised through `.gitattributes`.

---

This refactor delivers a maintainable, transparent, and fast GPU-native pipeline by leaning on FlashRAG’s proven components while keeping configuration and observability squarely in project code. For API details refer to `src/retriever.py`, `src/reranker.py`, and the FlashRAG documentation bundled under `FlashRAG/docs/`.
