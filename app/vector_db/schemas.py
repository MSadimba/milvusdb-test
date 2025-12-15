from pathlib import Path

schemas_code = (
    "from pymilvus import CollectionSchema, FieldSchema, DataType\n\n"
    "from app.vector_db.embedder import EMBED_DIM\n\n\n"
    "def rai_schema() -> CollectionSchema:\n"
    "    return CollectionSchema(\n"
    "        fields=[\n"
    "            FieldSchema(name=\"id\", dtype=DataType.VARCHAR, is_primary=True, max_length=128),\n"
    "            FieldSchema(name=\"text\", dtype=DataType.VARCHAR, max_length=65535),\n"
    "            FieldSchema(name=\"source\", dtype=DataType.VARCHAR, max_length=256),\n"
    "            FieldSchema(name=\"doc_id\", dtype=DataType.VARCHAR, max_length=256),\n"
    "            FieldSchema(name=\"embedding\", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),\n"
    "        ],\n"
    "        description=\"RAI documents (unit test schema)\",\n"
    "        enable_dynamic_field=True,\n"
    "    )\n\n\n"
    "def customer_schema() -> CollectionSchema:\n"
    "    return CollectionSchema(\n"
    "        fields=[\n"
    "            FieldSchema(name=\"id\", dtype=DataType.VARCHAR, is_primary=True, max_length=128),\n"
    "            FieldSchema(name=\"customer_id\", dtype=DataType.VARCHAR, max_length=128),\n"
    "            FieldSchema(name=\"text\", dtype=DataType.VARCHAR, max_length=65535),\n"
    "            FieldSchema(name=\"embedding\", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),\n"
    "        ],\n"
    "        description=\"Customer context (unit test schema)\",\n"
    "        enable_dynamic_field=True,\n"
    "    )\n"
)

Path("app/vector_db/schemas.py").write_text(schemas_code, encoding="utf-8")
print("schemas.py overwritten cleanly")
