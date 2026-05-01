from uuid import uuid4
from typing import List, Dict, Optional, Union

from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

from .base import VectorQueryResult, ScoredPoint

DISTANCE_METRIC_MAP = {
    "cosine": "COSINE",
    "l2": "L2",
    "ip": "IP",
}

_VECTOR_FIELD = "vector"
_ID_FIELD = "id"
_PAYLOAD_FIELD = "payload"


class MilvusVectorStore:
    def __init__(self, uri: str, token: str | None = None):
        self.uri = uri
        self.token = token
        self.client = MilvusClient(uri=self.uri, token=self.token or "")

    def create_collection(
        self,
        collection_name: str,
        embedding_size: int,
        distance: str = "cosine",
    ) -> None:
        if self.client.has_collection(collection_name):
            return

        metric_type = DISTANCE_METRIC_MAP.get(distance.lower())
        if metric_type is None:
            raise ValueError(
                f"Unsupported distance metric: '{distance}'. "
                f"Supported: {list(DISTANCE_METRIC_MAP.keys())}"
            )

        schema = CollectionSchema(fields=[
            FieldSchema(name=_ID_FIELD, dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name=_VECTOR_FIELD, dtype=DataType.FLOAT_VECTOR, dim=embedding_size),
            FieldSchema(name=_PAYLOAD_FIELD, dtype=DataType.JSON),
        ])

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=_VECTOR_FIELD,
            index_type="AUTOINDEX",
            metric_type=metric_type,
        )

        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

    def list_collections(self) -> list[str]:
        return self.client.list_collections()

    def get_collection_info(self, collection_name: str) -> dict | None:
        if not self.client.has_collection(collection_name):
            return None

        desc = self.client.describe_collection(collection_name)
        stats = self.client.get_collection_stats(collection_name)
        row_count = int(stats.get("row_count", 0))

        dimensions = 0
        distance = "unknown"
        for f in desc.get("fields", []):
            if f.get("name") == _VECTOR_FIELD:
                params = f.get("params", {})
                dimensions = params.get("dim", 0)
                break

        indexes = self.client.list_indexes(collection_name)
        if indexes:
            idx_info = self.client.describe_index(collection_name, index_name=indexes[0])
            distance = idx_info.get("metric_type", "unknown").lower()

        return {
            "vectors_count": row_count,
            "dimensions": dimensions,
            "distance": distance,
            "status": "ready",
        }

    def delete_collection(self, collection_name: str) -> None:
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)

    def add_embeddings(
        self,
        collection_name: str,
        embeddings: List[List[float]],
        payloads: Optional[List[Dict]] = None,
        ids: Optional[List[Union[str, int]]] = None,
        batch_size: int = 32,
    ) -> None:
        if payloads and len(payloads) != len(embeddings):
            raise ValueError("Payloads must match the number of embeddings.")
        if ids and len(ids) != len(embeddings):
            raise ValueError("IDs must match the number of embeddings.")

        data = [
            {
                _ID_FIELD: str(ids[i]) if ids else str(uuid4()),
                _VECTOR_FIELD: embedding,
                _PAYLOAD_FIELD: payloads[i] if payloads else {},
            }
            for i, embedding in enumerate(embeddings)
        ]

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            self.client.insert(collection_name=collection_name, data=batch)

    def query(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
    ) -> VectorQueryResult:
        results = self.client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=[_PAYLOAD_FIELD],
        )

        points: list[ScoredPoint] = []
        if results:
            for hit in results[0]:
                payload = hit.get("entity", {}).get(_PAYLOAD_FIELD, {})
                score = hit.get("distance", 0.0)
                points.append(ScoredPoint(payload=payload, score=score))

        return VectorQueryResult(points=points)
