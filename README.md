# RAG Pipeline for Q&A System

GPU-first Retrieval-Augmented Generation pipeline for AlfaBank public data.  
The entire stack delegates retrieval, indexing and reranking to **FlashRAG** components (bm25s + dense FAISS) and aborts early if no CUDA runtime is available.

**Key Features:**
- Hybrid retrieval (FAISS dense + BM25 sparse) with RRF fusion
- Multilingual support (Russian + English) with optimized tokenization
- Advanced reranking with optional ensemble and second-pass
- Text normalization and query expansion for banking domain
- GPU-accelerated with strict CUDA validation
- Comprehensive evaluation and failure analysis

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
│   ├── quick_test.py     # quick pipeline validation (~15 min)
│   ├── build_corpus.py   # corpus building
│   ├── chunk_corpus.py   # semantic chunking
│   ├── build_index_dense.py  # FAISS index
│   ├── build_index_bm25.py   # BM25 index
│   ├── eval.py           # evaluation with metrics
│   └── submit.py         # submission generation
├── src/                  # project modules
│   ├── retriever.py      # FlashRAG hybrid retriever wrapper
│   ├── reranker.py       # FlashRAG CrossReranker wrapper
│   ├── reranker_ensemble.py  # ensemble reranker support
│   ├── chunker.py        # semantic + structural chunking
│   ├── table_processor.py# numeric-aware query enrichment
│   ├── text_processor.py # text normalization
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
# Create environment (example)
python -m venv .venv
source .venv/bin/activate  # or: conda create -n alfa python=3.10

# Install dependencies (torch / faiss-gpu must match your CUDA runtime)
pip install -r requirements.txt

# FlashRAG (editable for local patches)
cd FlashRAG
pip install -e .
pip install flashrag-dev[full]
cd ..
```

**For CUDA 11.8 (GTX 1070, compute capability 6.1):**
```bash
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install faiss-gpu==1.7.4 -c conda-forge  # or via conda: conda install -c conda-forge faiss-gpu=1.7.4 cudatoolkit=11.8
```

Validate GPU availability before running the pipeline:

```bash
python - <<'PY'
import torch
from flashrag.utils import get_device
print("torch.cuda:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A")
print("Resolved device:", get_device())
PY
```

---

## 3. Quick Start

### Quick Test (Recommended First Step)
Test the pipeline without full rebuild (~15 minutes):

```bash
make quick_test
```

This validates:
- GPU availability and CUDA compatibility
- Retriever initialization
- Retrieval functionality
- Reranker initialization
- Ensemble reranker (if enabled)

### Full Pipeline

```bash
# Step 1: Build corpus and chunks
make build_corpus
make chunk_corpus

# Step 2: Build indexes (FAISS + BM25)
make build_index

# Step 3: Evaluate (with ground truth if available)
make eval

# Step 4: Generate submission
make submit
```

All Make targets execute in GPU mode and will abort immediately if CUDA kernels are missing.

---

## 4. Configuration Highlights

### `configs/base.yaml`

**Retrieval Settings:**
```yaml
retrieval:
  k_retrieve: 180        # retrieve top-k candidates before reranking
  k_rerank: 55           # number of candidates passed to reranker
  k_final: 5             # final top-k for submission
  fusion_method: "rrf"   # "rrf" (Reciprocal Rank Fusion) or "weighted"
  rrf_k: 70              # RRF hyperparameter (higher = more weight to top ranks)
  enhance_numerics: true # enrich queries for numeric value matching
  min_score_threshold: 0.0  # minimum score threshold before reranking
  filter_by_document_type: false  # filter candidates by document type
  prefer_table_chunks: false      # prefer chunks with tables for numeric queries
  batch_size: 32
  reranker_batch_size: 64
  num_workers: 4
```

**Chunking Settings:**
```yaml
chunking:
  size_tokens: 280       # reduced for better precision on short banking facts
  overlap_tokens: 110    # increased for better context preservation
  use_semantic_chunking: true  # semantic-structural chunking (tables, headings)
```

**Text Processing:**
```yaml
text_processing:
  normalize_for_retrieval: true
  normalization_mode: "smart"  # "letters_numbers", "smart", or "aggressive"
  # smart: keeps letters, digits, spaces, and important symbols (|, %, ., ,, -)
```

### `configs/models.yaml`

```yaml
embeddings:
  model_name: "intfloat/multilingual-e5-large"  # upgraded for better embeddings
  device: "auto"        # resolves to CUDA via utils.resolve_device
  batch_size: 64
  use_fp16: true

reranker:
  model_name: "BAAI/bge-reranker-base"  # multilingual reranker
  device: "auto"
  batch_size: 64
  max_length: 512
  use_fp16: true

# Optional: Ensemble rerankers for advanced accuracy
reranker_ensemble:
  enabled: false  # set to true to use ensemble
  models:
    - "BAAI/bge-reranker-base"
    - "cross-encoder/ms-marco-MiniLM-L-12-v2"
  weights: [0.7, 0.3]
  second_pass: false  # enable second pass reranking
  second_pass_topk: 20

faiss:
  index_type: "Flat"
  use_gpu: true
  use_float16: true
```

---

## 5. Key Features

### Retrieval & Reranking
- **Hybrid retrieval** — FAISS dense + BM25 sparse with RRF (Reciprocal Rank Fusion) or weighted fusion
- **Multilingual support** — optimized Russian tokenization for BM25 with custom stopwords
- **Advanced reranking** — cross-encoder reranker with optional ensemble and second-pass
- **Query expansion** — banking domain synonyms and query rewriting for numeric queries

### Text Processing
- **Smart normalization** — configurable text normalization (smart mode preserves important symbols)
- **Footer removal** — automatic removal of repetitive legal/copyright footers
- **Semantic chunking** — preserves table structure and headings

### GPU Acceleration
- **Strict GPU enforcement** — validates CUDA kernels before execution, raises `GPUMisconfigurationError` if incompatible
- **FP16 support** — half-precision for faster computation and reduced memory usage
- **Multi-threaded batching** — parallel batch processing for optimal GPU utilization

### Evaluation & Debugging
- **Comprehensive metrics** — Hit@5, Recall@K, MRR, NDCG@K
- **Failure analysis** — detailed logs for missed queries with candidate context
- **Candidate logging** — pre/post rerank candidate snapshots for analysis
- **Quick test** — fast validation without full rebuild

---

## 6. Pipeline Overview

| Stage | Description | Key scripts |
|-------|-------------|-------------|
| **Ingest** | Load CSV, clean text, remove footers, normalize | `scripts/build_corpus.py` |
| **Chunk** | Semantic chunking with table preservation | `scripts/chunk_corpus.py` |
| **Index build** | Dense FAISS index + BM25 sparse index | `scripts/build_index_dense.py`, `scripts/build_index_bm25.py` |
| **Hybrid retrieval** | FAISS + BM25 with RRF fusion | `src/retriever.py` |
| **Rerank** | Cross-encoder reranker (optional ensemble) | `src/reranker.py`, `src/reranker_ensemble.py` |
| **Evaluate** | Metrics, failure logs, candidate analysis | `scripts/eval.py` |
| **Submit** | Generate submission CSV with validation | `scripts/submit.py` |

---

## 7. Outputs

- `outputs/submits/submit_*.csv` — submission files (`q_id`, JSON-formatted `web_list`)
- `outputs/reports/report_*.json` — metrics snapshot (Hit@5, Recall@K, MRR, NDCG@K, timings)
- `outputs/logs/candidates_*.jsonl` — candidates pre/post rerank for analysis
- `outputs/logs/failures_*.json` — structured diagnostics for missed queries

---

## 8. Evaluation

### Metrics
- **Hit@5** — primary leaderboard metric
- **Recall@5, Recall@10** — recall at different K
- **MRR** — Mean Reciprocal Rank
- **NDCG@5** — Normalized Discounted Cumulative Gain

### Ground Truth
Provide ground truth via:
- Command line: `python scripts/eval.py --config configs/base.yaml --ground_truth path/to/ground_truth.json`
- Config file: Set `evaluation.ground_truth_path` in `configs/base.yaml`

Format: JSON file with `{"q_id": [web_id, ...]}` mapping

---

## 9. Troubleshooting

### Common Issues

**CUDA error: no kernel image**
- Install PyTorch build matching your GPU compute capability
- For GTX 1070 (sm_61): `pip install torch==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118`

**FAISS GPU API missing**
- Install `faiss-gpu` with proper CUDA support: `conda install -c conda-forge faiss-gpu=1.7.4 cudatoolkit=11.8`

**BM25 index missing**
- Run `make build_index` to build both FAISS and BM25 indexes

**Dimension mismatch in FAISS**
- Delete `indexes/faiss/` and rebuild: `make build_index`

**Reranker OOM**
- Lower `models.reranker.batch_size` in `configs/models.yaml`
- Or switch to a smaller cross-encoder model

**Empty stopwords/vocab for Russian**
- This is expected: Russian tokenization uses custom stopwords without stemming

---

## 10. Performance Tuning

### Accuracy vs Speed Trade-offs

**For Higher Accuracy:**
- Increase `k_retrieve` (180-200)
- Increase `k_rerank` (50-60)
- Use `multilingual-e5-large` for embeddings
- Enable ensemble reranker
- Enable second-pass reranking
- Use smaller chunks (250-300 tokens)

**For Faster Processing:**
- Decrease `k_retrieve` (100-150)
- Decrease `k_rerank` (30-40)
- Use `multilingual-e5-base` for embeddings
- Disable ensemble reranker
- Use larger chunks (400-500 tokens)

### GPU Utilization
- Adjust `retrieval.num_workers` (default: 4) based on hardware
- Monitor GPU utilization; aim for >80% during processing
- Increase batch sizes gradually while monitoring VRAM

---

## 11. Further Reading

- **[about.md](about.md)** — deep dive into architecture, FlashRAG modifications, optimization notes, benchmarking methodology

---

## 12. License / Attribution

All dependencies are open-source. FlashRAG is vendored under its original license; see `FlashRAG/LICENSE` for details.
