import os
import pytest

from app.vector_db.milvus_client import MilvusVectorDB


@pytest.mark.unit
def test_db_connect_and_collections_create():
    uri = os.environ["MILVUS_URI"]
    token = os.environ["MILVUS_TOKEN"]

    db = MilvusVectorDB(uri=uri, token=token)
    db.connect()
    db.ensure_collections()

    assert True
