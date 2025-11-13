# RAG Pipeline for Q&A System

GPU-first Retrieval-Augmented Generation pipeline for AlfaBank public data.  
The entire stack now delegates retrieval, indexing and reranking to **FlashRAG** components (bm25s + dense FAISS) and aborts early if no CUDA runtime is available.

---

## 1. Project Structure

```
alfa/
├── configs/              # YAML configuration
│   ├── base.yaml         # pipeline hyperparameters
│   └── models.yaml       # model + device settings
├── data/
│   ├── raw/              # source CSV data
│   └── processed/        # corpus + chunks
├── indexes/              # FAISS + bm25s artefacts
├── outputs/              # reports, logs, submissions
├── scripts/              # CLI entry points
├── src/                  # project modules
│   ├── retriever.py      # FlashRAG hybrid retriever
│   ├── reranker.py       # FlashRAG CrossReranker wrapper
│   ├── chunker.py        # semantic + structural chunking
│   ├── table_processor.py# numeric-aware query enrichment
│   ├── evaluator.py      # metrics & failure analysis
│   └── utils.py          # logging, device guards, helpers
├── FlashRAG/             # vendored FlashRAG (editable)
├── requirements.txt
├── Makefile
├── README.md
└── about.md              # extended documentation
```

---

## 2. Installation (GPU required)

> **Important:** CPU execution is disabled. `resolve_device` halts with `GPUMisconfigurationError` if CUDA kernels are missing or incompatible.

```bash
# Create env (example)
python -m venv .venv
source .venv/bin/activate

# Core deps (torch / faiss-gpu must match your CUDA runtime)
pip install -r requirements.txt

# FlashRAG (editable for local patches)
cd FlashRAG
pip install -e .
pip install flashrag-dev[full]
cd ..
```

Validate GPU availability before running the pipeline:

```bash
python - <<'PY'
import torch
from flashrag.utils import get_device
print("torch.cuda:", torch.cuda.is_available(), "resolved:", get_device())
PY
```

---

## 3. Pipeline Overview

| Stage | Description | Key scripts |
|-------|-------------|-------------|
| **Ingest / Chunk** | Load CSV, clean text, semantic chunking, numeric-aware augmentation | `scripts/build_corpus.py`, `scripts/chunk_corpus.py` |
| **Index build** | Dense FAISS index + bm25s sparse index through FlashRAG builders | `scripts/build_index_dense.py`, `scripts/build_index_bm25.py`, `make build_index` |
| **Hybrid retrieval** | FlashRAG `MultiRetrieverRouter` (dense + bm25s) with weighted fusion | `src/retriever.py` |
| **Cross rerank** | FlashRAG `CrossReranker` (fp16 on CUDA) | `src/reranker.py` |
| **Evaluation** | Batched retrieval + rerank, metrics, failure logs | `scripts/eval.py` |
| **Submission** | Generates submit CSV; validates shape against `sample_submission.csv` | `scripts/submit.py` |

Run the full flow:

```bash
make build_corpus
make chunk_corpus
make build_index      # builds FAISS + bm25s via FlashRAG
make eval             # retrieval + rerank + metrics
make submit           # creates outputs/submits/submit_*.csv
```

All Make targets execute in GPU mode and will abort immediately if CUDA kernels are missing.

---

## 4. Configuration Highlights

`configs/base.yaml`

```yaml
retrieval:
  k_retrieve: 30
  k_rerank: 20
  k_final: 5
  batch_size: 32
  fusion_method: "weighted"     # "weighted", "rrf" or "concat"
  hybrid_weight_dense: 0.6
  hybrid_weight_bm25: 0.4
  enhance_numerics: true        # enrich numeric queries before bm25s

chunking:
  size_tokens: 600
  overlap_tokens: 150
  use_semantic_chunking: true
```

`configs/models.yaml`

```yaml
embeddings:
  model_name: "intfloat/multilingual-e5-base"
  device: "auto"        # resolves to CUDA via utils.resolve_device
  batch_size: 64
  use_fp16: true

reranker:
  model_name: "cross-encoder/ms-marco-MiniLM-L-12-v2"
  device: "auto"
  batch_size: 32
  use_fp16: true

faiss:
  index_type: "Flat"
  use_gpu: true
  use_float16: true
```

---

## 5. Key Features

- **FlashRAG-first retrieval** — hybrid pipeline is entirely powered by FlashRAG `MultiRetrieverRouter`, `BM25Retriever` (bm25s backend) and FAISS dense retriever; we only wrap configuration and post-processing.
- **GPU-only guardrails** — `resolve_device` and FlashRAG’s `get_device()` validate CUDA kernels by executing a tiny tensor op. Any mismatch (wrong build / no kernels) raises `GPUMisconfigurationError` before heavy work starts.
- **Weighted fusion** — custom extension to FlashRAG adds deterministic weighted score fusion (dense/bm25) with per-source normalisation and provenance tracking (`source_scores`, `sources`).
- **Multi-threaded batched processing** — parallel batch execution with `ThreadPoolExecutor` (configurable `num_workers`) for optimal GPU utilization; logs capture timing, ETAs and per-stage throughput.
- **Numeric-aware queries** — optional enrichment for amounts, rates and dates before sparse retrieval (`table_processor.enhance_query_for_numerics`).
- **Sample alignment** — `scripts/submit.py` checks that produced `submit.csv` exactly matches the q_id set of `sample_submission.csv` and enforces fixed list length before writing.

---

## 6. Outputs

- `outputs/submits/submit_*.csv` — submission files (`q_id`, JSON-formatted `web_list`)
- `outputs/reports/report_*.json` — metrics snapshot (Hit@5, Recall@K, MRR, NDCG@5, timings)
- `outputs/logs/candidates_*.jsonl` — candidates pre/post rerank
- `outputs/logs/failures_*.json` — structured diagnostics for missed queries

---

## 7. Metrics

- **Hit@5** — primary leaderboard metric
- **Recall@K**
- **MRR**
- **NDCG@K**

`scripts/eval.py` stores results under `outputs/reports` and failure summaries in `outputs/logs`.

---

## 8. Further Reading

- **[about.md](about.md)** — deep dive into architecture, FlashRAG modifications, optimisation notes, benchmarking methodology.

---

## 9. License / Attribution

All dependencies are open-source. FlashRAG is vendored under its original license; see `FlashRAG/LICENSE` for details.
