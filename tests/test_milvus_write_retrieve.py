import os
import pytest

from app.vector_db.milvus_client import MilvusVectorDB, DEFAULT_RAI_COLLECTION
from app.vector_db.embedder import DeterministicEmbedder


@pytest.mark.unit
def test_write_and_retrieve_rai():
    uri = os.environ["MILVUS_URI"]
    token = os.environ["MILVUS_TOKEN"]

    db = MilvusVectorDB(uri=uri, token=token)
    db.connect()
    db.ensure_collections()

    emb = DeterministicEmbedder()

    items = [
        {
            "id": "rai-1",
            "text": "EU AI Act: risk management, post-market monitoring, and governance obligations.",
            "source": "unit_test",
            "doc_id": "doc-eu-ai-act",
            "embedding": emb.embed("EU AI Act risk management"),
        },
        {
            "id": "rai-2",
            "text": "Fairness testing: demographic parity and disparate impact in lending decisions.",
            "source": "unit_test",
            "doc_id": "doc-fairness-lending",
            "embedding": emb.embed("fairness in lending"),
        },
    ]

    db.upsert_rai(items)

    results = db.search(
        query_embedding=emb.embed("fairness lending disparate impact"),
        collection_name=DEFAULT_RAI_COLLECTION,
        top_k=2,
    )

    assert len(results) >= 1
    ids = [r["id"] for r in results]
    assert "rai-2" in ids
    assert any(r.get("doc_id") for r in results)
