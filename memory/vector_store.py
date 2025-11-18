"""
Vector store adapter.

- If settings.vector_db == "chroma", this module should talk to chromadb (or its HTTP API).
- For Sprint-1 Part A we include a compact `SimpleVectorStore` JSON-backed fallback.
- API:
    - add_document(vector, metadata) -> id
    - similarity_search(vector, top_k=K) -> list of {id, score, metadata}
"""

import json
import logging
import math
import os
import uuid
from typing import Any, Dict, List

from configs.config import settings

logger = logging.getLogger(__name__)


class SimpleVectorStore:
    """
    Minimal JSON-backed vector store for development and CI.
    Not intended for production (no efficient NN search), but useful for unit tests.
    """

    def __init__(self, storage_path: str = "data/vector_store.json"):
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        if not os.path.exists(self.storage_path):
            with open(self.storage_path, "w") as f:
                json.dump({"vectors": []}, f)

    def _load(self) -> Dict[str, Any]:
        with open(self.storage_path, "r") as f:
            return json.load(f)

    def _save(self, data: Dict[str, Any]) -> None:
        with open(self.storage_path, "w") as f:
            json.dump(data, f)

    def add_document(self, vector: List[float], metadata: Dict[str, Any]) -> str:
        data = self._load()
        doc_id = str(uuid.uuid4())
        data["vectors"].append({"id": doc_id, "vector": vector, "metadata": metadata})
        self._save(data)
        return doc_id

    def similarity_search(self, vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        data = self._load()

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        scores = []
        for entry in data.get("vectors", []):
            score = cosine(vector, entry["vector"])
            scores.append((score, entry))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [
            {"score": s, "id": it["id"], "metadata": it["metadata"]} for s, it in scores[:top_k]
        ]
