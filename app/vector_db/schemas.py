from pathlib import Path

schemas_code = """from pymilvus import CollectionSchema, FieldSchema, DataType

from app.vector_db.embedder import EMBED_DIM


def rai_schema() -> CollectionSchema:
    return CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
        ],
        description="RAI documents (unit test schema)",
        enable_dynamic_field=True,
    )


def customer_schema() -> CollectionSchema:
    return CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="customer_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
        ],
        description="Customer context (unit test schema)",
        enable_dynamic_field=True,
    )
"""

Path("app/vector_db/schemas.py").write_text(schemas_code, encoding="utf-8")
print("schemas.py fixed")
