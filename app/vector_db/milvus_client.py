cat > app/vector_db/milvus_client.py << 'EOF'
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from pymilvus import Collection, connections, utility

from .schemas import rai_schema, customer_schema

DEFAULT_RAI_COLLECTION = os.getenv("MILVUS_RAI_COLLECTION", "rai_docs_test")
DEFAULT_CUSTOMER_COLLECTION = os.getenv("MILVUS_CUSTOMER_COLLECTION", "customer_context_test")


class MilvusVectorDB:
    """
    Minimal Milvus/Zilliz wrapper for:
      - connect
      - ensure collections exist
      - insert
      - vector search (+ optional filtering)
    Designed for unit tests (Jira DoD).
    """

    def __init__(self, uri: str, token: str, alias: str = "default"):
        self.uri = uri
        self.token = token
        self.alias = alias

    def connect(self) -> None:
        connections.connect(
            alias=self.alias,
            uri=self.uri,
            token=self.token,
        )

    def ensure_collections(
        self,
        rai_name: str = DEFAULT_RAI_COLLECTION,
        customer_name: str = DEFAULT_CUSTOMER_COLLECTION,
    ) -> None:
        # Create collections if missing (idempotent)
        if not utility.has_collection(rai_name, using=self.alias):
            Collection(name=rai_name, schema=rai_schema(), using=self.alias)

        if not utility.has_collection(customer_name, using=self.alias):
            Collection(name=customer_name, schema=customer_schema(), using=self.alias)

        # Ensure index exists and load collections
        self._ensure_index_and_load(rai_name)
        self._ensure_index_and_load(customer_name)

    def _ensure_index_and_load(self, collection_name: str) -> None:
        col = Collection(collection_name, using=self.alias)

        # If an index already exists, don't recreate it
        if not col.indexes:
            # AUTOINDEX is compatible with Zilliz Cloud Serverless.
            col.create_index(
                field_name="embedding",
                index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE"},
            )

        col.load()

    def upsert_rai(self, items: List[Dict[str, Any]], collection_name: str = DEFAULT_RAI_COLLECTION) -> None:
        """
        items: [{id, text, source, doc_id, embedding}]
        """
        col = Collection(collection_name, using=self.alias)
        data = [
            [i["id"] for i in items],
            [i["text"] for i in items],
            [i.get("source", "") for i in items],
            [i.get("doc_id", "") for i in items],
            [i["embedding"] for i in items],
        ]
        col.insert(data)
        col.flush()

    def upsert_customer(
        self, items: List[Dict[str, Any]], collection_name: str = DEFAULT_CUSTOMER_COLLECTION
    ) -> None:
        """
        items: [{id, customer_id, text, embedding}]
        """
        col = Collection(collection_name, using=self.alias)
        data = [
            [i["id"] for i in items],
            [i["customer_id"] for i in items],
            [i["text"] for i in items],
            [i["embedding"] for i in items],
        ]
        col.insert(data)
        col.flush()

    def search(
        self,
        query_embedding: List[float],
        collection_name: str,
        top_k: int = 3,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        col = Collection(collection_name, using=self.alias)
        col.load()

        results = col.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE"},
            limit=top_k,
            expr=filter_expr,
            output_fields=["id", "text", "source", "doc_id", "customer_id"],
        )

        out: List[Dict[str, Any]] = []
        for hit in results[0]:
            entity = hit.entity
            out.append(
                {
                    "id": entity.get("id"),
                    "score": float(hit.score),
                    "text": entity.get("text", None),
                    "source": entity.get("source", None),
                    "doc_id": entity.get("doc_id", None),
                    "customer_id": entity.get("customer_id", None),
                }
            )
        return out

    def drop_test_collections(
        self,
        rai_name: str = DEFAULT_RAI_COLLECTION,
        customer_name: str = DEFAULT_CUSTOMER_COLLECTION,
    ) -> None:
        """
        Optional cleanup helper for tests.
        Only use with *_test collections.
        """
        if utility.has_collection(rai_name, using=self.alias):
            utility.drop_collection(rai_name, using=self.alias)
        if utility.has_collection(customer_name, using=self.alias):
            utility.drop_collection(customer_name, using=self.alias)
EOF
