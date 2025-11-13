#!/usr/bin/env python3
"""
Build FAISS dense index using FlashRAG.

Usage:
    python scripts/build_index_dense.py --config configs/base.yaml
"""
import argparse
import sys
from pathlib import Path

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "FlashRAG"))

import yaml
from flashrag.utils.utils import GPUMisconfigurationError

try:
    from flashrag.retriever.index_builder import Index_Builder
except GPUMisconfigurationError as exc:
    print(f"[ERROR] {exc}", file=sys.stderr)
    sys.exit(1)

from src.utils import load_jsonl, resolve_device


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Build FAISS dense index using FlashRAG")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config file")
    parser.add_argument("--model_path", help="Path to embedding model (overrides config)")
    parser.add_argument("--corpus_path", help="Path to chunks.jsonl (overrides config)")
    parser.add_argument("--save_dir", help="Path to save index (overrides config)")
    args = parser.parse_args()
    
    # load configs
    config = load_config(args.config)
    models_config = load_config(project_root / "configs" / "models.yaml")
    
    # get paths
    corpus_path = args.corpus_path or config["data"]["chunks_jsonl"]
    save_dir = args.save_dir or config["indexes"]["faiss_dir"]
    model_path = args.model_path or models_config["embeddings"]["model_name"]
    
    # get model config
    embeddings_config = models_config["embeddings"]
    faiss_config = models_config["faiss"]
    embedding_device = resolve_device(embeddings_config.get("device", "auto"))
    use_fp16 = embeddings_config.get("use_fp16", False) and embedding_device.startswith("cuda")
    faiss_gpu_enabled = faiss_config.get("use_gpu", False) and embedding_device.startswith("cuda")
    
    print(f"Building FAISS index...")
    print(f"  Corpus: {corpus_path}")
    print(f"  Model: {model_path}")
    print(f"  Save dir: {save_dir}")
    print(f"  FAISS type: {faiss_config['index_type']}")
    
    # create index builder
    index_builder = Index_Builder(
        retrieval_method="e5",  # will be auto-detected from model
        model_path=model_path,
        corpus_path=corpus_path,
        save_dir=save_dir,
        max_length=512,
        batch_size=embeddings_config.get("batch_size", 32),
        use_fp16=use_fp16,
        pooling_method=None,  # auto-detect
        instruction=None,  # auto-detect for E5/BGE
        faiss_type=faiss_config["index_type"],
        faiss_gpu=faiss_gpu_enabled,
        use_sentence_transformer=True,  # use sentence-transformers for easier setup
        bm25_backend="bm25s"
    )
    
    # build index
    index_builder.build_index()
    
    # save metadata (chunk_ids, doc_ids mapping)
    # load chunks to extract doc_ids
    chunks = load_jsonl(corpus_path)
    
    # extract doc_ids from chunks (web_id)
    chunk_ids = []
    doc_ids = []
    for chunk in chunks:
        chunk_id = chunk.get("id", "")
        doc_id = chunk.get("doc_id", chunk.get("id", "").split("_")[0])  # extract web_id from chunk_id
        chunk_ids.append(chunk_id)
        doc_ids.append(doc_id)
    
    # save metadata
    import json
    meta_path = Path(save_dir) / "faiss_meta.json"
    metadata = {
        "chunk_ids": chunk_ids,
        "doc_ids": doc_ids,
        "num_chunks": len(chunk_ids),
        "model_name": model_path,
        "faiss_type": faiss_config["index_type"]
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Index built successfully in {save_dir}")
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    try:
        main()
    except GPUMisconfigurationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

