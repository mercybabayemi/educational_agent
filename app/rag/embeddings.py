"""Embedding wrapper: intfloat/multilingual-e5-large (1024-dim)."""
from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from app.core.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a singleton HuggingFaceEmbeddings instance.

    multilingual-e5-large requires the query prefix "query: " and
    passage prefix "passage: " to be prepended at encode time.
    LangChain's HuggingFaceEmbeddings does NOT do this automatically,
    so we pass model_kwargs to enable mean-pooling and use the
    encode_kwargs to normalise embeddings.
    """
    logger.info(f"Loading embedding model: {settings.embedding_model_id}")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_id,
        model_kwargs={"device": "cuda" if _cuda_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("Embedding model loaded")
    return embeddings


def embed_query(text: str) -> List[float]:
    """Embed a single query string with the required 'query: ' prefix."""
    emb = get_embeddings()
    return emb.embed_query(f"query: {text}")


def embed_documents(texts: List[str]) -> List[List[float]]:
    """Embed a batch of passages with the required 'passage: ' prefix."""
    emb = get_embeddings()
    prefixed = [f"passage: {t}" for t in texts]
    return emb.embed_documents(prefixed)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
