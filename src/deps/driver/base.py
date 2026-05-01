from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Protocol, Union, runtime_checkable


@dataclass
class ScoredPoint:
    payload: dict[str, Any]
    score: float


@dataclass
class VectorQueryResult:
    points: list[ScoredPoint] = field(default_factory=list)


@runtime_checkable
class VectorStore(Protocol):
    def create_collection(
        self,
        collection_name: str,
        embedding_size: int,
        distance: str = "cosine",
    ) -> None: ...

    def list_collections(self) -> list[str]: ...

    def get_collection_info(self, collection_name: str) -> dict | None: ...

    def delete_collection(self, collection_name: str) -> None: ...

    def add_embeddings(
        self,
        collection_name: str,
        embeddings: List[List[float]],
        payloads: Optional[List[Dict]] = None,
        ids: Optional[List[Union[str, int]]] = None,
        batch_size: int = 32,
    ) -> None: ...

    def query(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
    ) -> VectorQueryResult: ...
