from .base import VectorStore, VectorQueryResult, ScoredPoint
from .qdrant_client import QdrantVectorStore
from .milvus_client import MilvusVectorStore


def create_vector_store(provider: str | None = None) -> VectorStore:
    """Instantiate the vector store backend from settings.

    Args:
        provider: Override for ``settings.vector_store_provider``.
                  Accepts ``"qdrant"`` or ``"milvus"``.
    """
    from src.settings import settings

    provider = (provider or settings.vector_store_provider).lower()

    if provider == "qdrant":
        return QdrantVectorStore(
            uri=settings.qdrant_uri,
            api_key=settings.qdrant_api_key,
        )
    elif provider == "milvus":
        return MilvusVectorStore(
            uri=settings.milvus_uri,
            token=settings.milvus_token,
        )
    else:
        raise ValueError(
            f"Unknown vector_store_provider '{provider}'. "
            "Supported: 'qdrant', 'milvus'."
        )


__all__ = [
    "VectorStore",
    "VectorQueryResult",
    "ScoredPoint",
    "QdrantVectorStore",
    "MilvusVectorStore",
    "create_vector_store",
]
