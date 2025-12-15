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
