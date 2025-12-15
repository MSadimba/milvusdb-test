from typing import List
import hashlib

EMBED_DIM = 384  # Must match Milvus schema dimension


class DeterministicEmbedder:
    """
    Deterministic embedder for unit tests.
    This replaces real LLM / HuggingFace embeddings so tests are stable,
    fast, and do not require network access.
    """

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        # Hash the text so the same input always gives the same vector
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = []

        # Expand the hash deterministically to EMBED_DIM floats
        for i in range(self.dim):
            b = digest[i % len(digest)]
            vector.append((b - 128) / 128.0)  # roughly in [-1, 1]

        return vector
