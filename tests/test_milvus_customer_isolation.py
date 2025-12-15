import os
import pytest

from app.vector_db.milvus_client import MilvusVectorDB, DEFAULT_CUSTOMER_COLLECTION
from app.vector_db.embedder import DeterministicEmbedder


@pytest.mark.unit
def test_customer_context_filtered_search_isolated():
    uri = os.environ["MILVUS_URI"]
    token = os.environ["MILVUS_TOKEN"]

    db = MilvusVectorDB(uri=uri, token=token)
    db.connect()
    db.ensure_collections()

    emb = DeterministicEmbedder()

    db.upsert_customer(
        [
            {
                "id": "custA-1",
                "customer_id": "customer_A",
                "text": "Customer A policy: retention and deletion timelines for personal data.",
                "embedding": emb.embed("retention deletion policy"),
            },
            {
                "id": "custB-1",
                "customer_id": "customer_B",
                "text": "Customer B requirements: security controls and access management.",
                "embedding": emb.embed("security access controls"),
            },
        ]
    )

    results = db.search(
        query_embedding=emb.embed("data deletion timelines"),
        collection_name=DEFAULT_CUSTOMER_COLLECTION,
        top_k=5,
        filter_expr='customer_id == "customer_A"',
    )

    assert len(results) >= 1
    assert all(r["customer_id"] == "customer_A" for r in results)
    assert all(r["id"].startswith("custA") for r in results)
