"""
Embedding abstraction: provides `embed_texts` and `embed_text`.
Default behavior:
 - If `sentence_transformers` is installed, use it.
 - Else fallback to a deterministic shim (sha256-based vector) for CI/dev.
This lets tests run offline and gives a clean interface for later replacement.
"""

import hashlib
import logging
import math
from typing import List, Sequence

import torch

logger = logging.getLogger(__name__)

try:
    # optional heavy dep; used in production if installed
    from sentence_transformers import SentenceTransformer

    _ST_AVAILABLE = True
except Exception:
    _ST_AVAILABLE = False


class EmbeddingClient:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        if _ST_AVAILABLE:
            logger.info("Loading SentenceTransformer model: %s", model_name)
            self._model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if self._model:
            embeddings = self._model.encode(list(texts), convert_to_numpy=False)

            if isinstance(embeddings, torch.Tensor):
                return embeddings.detach().cpu().tolist()

            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()

            if isinstance(embeddings, list) and not isinstance(embeddings[0], list):
                return [e.detach().cpu().tolist() for e in embeddings]

            if isinstance(embeddings, list) and isinstance(embeddings[0], list):
                return embeddings

            return [list(map(float, e)) for e in embeddings]

        # fallback deterministic embedding
        return [self._shim_embed(t) for t in texts]

    @staticmethod
    def _shim_embed(text: str, dim: int = 384) -> List[float]:
        # Deterministic numeric vector based on sha256 bytes (normalized)
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = []
        i = 0
        while len(vec) < dim:
            vec.append(digest[i % len(digest)] / 255.0)
            i += 1
        # normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec
