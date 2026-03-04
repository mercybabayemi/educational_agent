"""Standalone ChromaDB ingest — no langchain dependency.

Reads cached Parquet files and ingests into ChromaDB directly.
Compatible with the datasci conda env (conda run -n datasci python ...).

Usage:
    conda run -n datasci python research/datasets/ingest_standalone.py
    conda run -n datasci python research/datasets/ingest_standalone.py --status
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CHROMA_DIR = PROJECT_ROOT / "data" / "vectorstore"
COLLECTION_NAME = "enumverse_curriculum"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nigerian_education"

REGISTRY: dict[str, dict] = {
    "vocational":            {"board": "NABTEB", "subject": "trade_skill",      "level": "VOCATIONAL_ADVANCED"},
    "continuous_assessment": {"board": "NERDC",  "subject": "general",          "level": "SSS1"},
    "digital_learning":      {"board": "NERDC",  "subject": "digital_literacy", "level": "SSS1"},
    "online_exams":          {"board": "WAEC",   "subject": "general",          "level": "SSS2"},
    "learning_materials":    {"board": "NERDC",  "subject": "general",          "level": "JSS3"},
    "teacher_training":      {"board": "NERDC",  "subject": "general",          "level": "TEACHER_TRAINING"},
    "special_needs":         {"board": "UBEC",   "subject": "general",          "level": "PRIMARY_4_6"},
}

CHUNK_SIZE = 400   # characters
CHUNK_OVERLAP = 80


def _doc_id(text: str, source: str, board: str) -> str:
    key = text[:200] + source + board
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _simple_chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping character chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if len(c) >= 20]


def get_collection():
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        print("Install chromadb: conda run -n datasci pip install chromadb")
        sys.exit(1)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(COLLECTION_NAME)


def get_embeddings(texts: List[str]) -> List[List[float]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Install sentence-transformers: conda run -n datasci pip install sentence-transformers")
        sys.exit(1)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_list=True)
    return embeddings


def ingest_dataset(key: str, collection) -> int:
    try:
        import pandas as pd
    except ImportError:
        print("Install pandas: conda run -n datasci pip install pandas pyarrow")
        return 0

    parquet_path = RAW_DIR / key / f"{key}.parquet"
    if not parquet_path.exists():
        print(f"  SKIP {key}: no parquet at {parquet_path}")
        return 0

    cfg = REGISTRY[key]
    df = pd.read_parquet(parquet_path)
    print(f"  {key}: {len(df):,} rows → chunking...")

    texts, metadatas, ids = [], [], []
    for _, row in df.iterrows():
        text = str(row.get("text", ""))
        if len(text) < 20:
            continue
        for chunk in _simple_chunk(text):
            meta = {
                "source": f"nigerian_education:{key}",
                "board": str(row.get("board", cfg["board"])),
                "subject": str(row.get("subject", cfg["subject"])),
                "level": str(row.get("level", cfg["level"])),
                "topic": str(row.get("topic", "")),
                "dataset_key": key,
            }
            doc_id = _doc_id(chunk, meta["source"], meta["board"])
            texts.append(chunk)
            metadatas.append(meta)
            ids.append(doc_id)

    if not texts:
        print(f"  SKIP {key}: no valid chunks")
        return 0

    print(f"  {key}: {len(texts):,} chunks → computing embeddings...")
    embeddings = get_embeddings(texts)

    # Batch upsert (chromadb accepts max ~5000 per call)
    batch = 2000
    for i in range(0, len(texts), batch):
        collection.upsert(
            documents=texts[i:i+batch],
            embeddings=embeddings[i:i+batch],
            metadatas=metadatas[i:i+batch],
            ids=ids[i:i+batch],
        )
    print(f"  {key}: ingested {len(texts):,} chunks ✓")
    return len(texts)


def print_status(collection) -> None:
    count = collection.count()
    print(f"\nChromaDB collection '{COLLECTION_NAME}': {count:,} documents")
    print(f"Path: {CHROMA_DIR}\n")
    print(f"{'Dataset':<25} {'Parquet cached'}")
    print("-" * 45)
    for key in REGISTRY:
        path = RAW_DIR / key / f"{key}.parquet"
        status = f"✓ {path.stat().st_size // 1024} KB" if path.exists() else "✗ not cached"
        print(f"  {key:<23} {status}")


def main() -> None:
    p = argparse.ArgumentParser(description="Standalone ChromaDB ingest for Nigerian education datasets")
    p.add_argument("--status", action="store_true", help="Show current ChromaDB status")
    p.add_argument("--datasets", nargs="+", choices=list(REGISTRY), help="Specific datasets (default: all)")
    args = p.parse_args()

    collection = get_collection()

    if args.status:
        print_status(collection)
        return

    keys = args.datasets or list(REGISTRY.keys())
    total = 0
    print(f"\nIngesting {len(keys)} dataset(s) into ChromaDB at {CHROMA_DIR}\n")
    for key in keys:
        total += ingest_dataset(key, collection)

    print(f"\n{'='*50}")
    print(f"Total chunks ingested: {total:,}")
    print(f"Collection now has: {collection.count():,} documents")


if __name__ == "__main__":
    main()
