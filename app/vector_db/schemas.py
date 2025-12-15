from __future__ import annotations

from pymilvus import CollectionSchema, FieldSchema, DataType

from .embedder import EMBED_DIM


def rai_schema() -> CollectionSchema:
    """
    Schema for Responsible AI documents (test collection).
    """
    return CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
        ],
        description="RAI documents (unit test
