# Path: sera/embeddings/embedder.py

"""
Embedder for SERA

Responsibilities
- Provide a single Embedder class that supports multiple backends:
    - sentence-transformers (local model)  -- default
    - openai (remote API)                  -- optional (if OPENAI_API_KEY set)
    - shim (deterministic fallback)        -- CI/offline safe
- Expose async-friendly method `embed_chunks(chunks)` which returns
  List[List[float]] aligned to the input chunks order.
- Batch embeddings for throughput and use `asyncio.to_thread` so CPU/GPU
  heavy operations don't block the event loop.
- Robust handling of torch/numpy outputs and device transfers (cpu,mps,cuda).
- Retry logic for transient failures and graceful fallback to shim.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

from configs.config import settings

logger = logging.getLogger(__name__)

# Optional heavy deps
try:
    from sentence_transformers import SentenceTransformer

    _HAS_ST = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _HAS_ST = False

try:
    import torch

    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False

try:
    import numpy as np  # used for safety conversions

    _HAS_NUMPY = True
except Exception:
    np = None  # type: ignore
    _HAS_NUMPY = False

# Optional OpenAI support
try:
    import openai

    _HAS_OPENAI = True
except Exception:
    openai = None  # type: ignore
    _HAS_OPENAI = False


@dataclass
class EmbedderConfig:
    backend: str = getattr(settings, "embed_backend", "sentence-transformers")
    model_name: str = getattr(settings, "embed_model_name", "all-MiniLM-L6-v2")
    batch_size: int = int(getattr(settings, "embed_batch_size", 64))
    device: str = getattr(settings, "embed_device", "auto")  # "auto" | "cpu" | "cuda" | "mps"
    retry_attempts: int = int(getattr(settings, "embed_retry_attempts", 3))
    retry_backoff: float = 1.0  # seconds
    openai_model: str = getattr(settings, "openai_embed_model", "text-embedding-3-small")


class Embedder:
    def __init__(self, cfg: Optional[EmbedderConfig] = None):
        self.cfg = cfg or EmbedderConfig()
        self._model = None
        self._openai_ready = _HAS_OPENAI and bool(
            getattr(settings, "openai_api_key", None) or getattr(settings, "openai_api_key", None)
        )
        if self.cfg.backend == "sentence-transformers" and _HAS_ST:
            self._init_sentence_transformer()
        elif self.cfg.backend == "openai" and self._openai_ready:
            # openai client uses env var or settings; we will rely on openai being configured externally
            logger.info(
                "Embedder configured to use OpenAI backend (model=%s)", self.cfg.openai_model
            )
        else:
            logger.warning("Embedder falling back to shim backend (deterministic).")
            self.cfg.backend = "shim"

    def _init_sentence_transformer(self):
        # Determine device
        device = self._resolve_device(self.cfg.device)
        try:
            logger.info(
                "Loading SentenceTransformer model '%s' on device '%s'", self.cfg.model_name, device
            )
            # SentenceTransformer handles device internally via .to(device)
            self._model = SentenceTransformer(self.cfg.model_name)

            if _HAS_TORCH and device != "cpu":
                try:
                    self._model.to(device)
                except Exception:
                    logger.debug(
                        "Could not .to(device) the ST model; continuing on default device."
                    )
        except Exception as exc:
            logger.exception("Failed to initialize SentenceTransformer: %s", exc)
            self._model = None
            self.cfg.backend = "shim"

    @staticmethod
    def _resolve_device(device_cfg: str) -> str:
        if device_cfg == "auto":
            if _HAS_TORCH and torch.cuda.is_available():
                return "cuda"
            if _HAS_TORCH and getattr(torch, "has_mps", False):  # mac mps
                try:
                    if torch.backends.mps.is_available():
                        return "mps"
                except Exception:
                    pass
            return "cpu"
        return device_cfg

    # ----------------------
    # Public API
    # ----------------------
    async def embed_chunks(self, texts: Iterable[str]) -> List[List[float]]:
        """
        Embed a sequence of text strings (chunks). Returns a list of embeddings
        aligned with the input order.

        This function is async and will use threads internally for heavy ops.
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        if self.cfg.backend == "sentence-transformers" and self._model is not None:
            return await self._embed_with_sentence_transformers(texts_list)
        if self.cfg.backend == "openai" and self._openai_ready:
            return await self._embed_with_openai(texts_list)
        # shim fallback (deterministic, fast)
        return [self._shim_embed(t) for t in texts_list]

    # ----------------------
    # Sentence-Transformers Backend
    # ----------------------
    async def _embed_with_sentence_transformers(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embed using SentenceTransformer.
        Handles both sync encode() and hypothetical future async variants.
        """

        results: List[List[float]] = []
        batch_size = max(1, int(self.cfg.batch_size))

        # 1) Define a synchronous encoding function for to_thread() usage
        def _encode_batch_sync(batch_texts: List[str]):
            try:
                emb = self._model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )

                # Normalize outputs
                if _HAS_NUMPY and isinstance(emb, np.ndarray):
                    return emb.tolist()

                if _HAS_TORCH and hasattr(emb, "tolist"):
                    return emb.tolist()

                return [list(map(float, e)) for e in emb]

            except Exception as exc:
                logger.exception("SentenceTransformer batch encode failed: %s", exc)
                raise

        idx = 0
        while idx < len(texts):
            batch = texts[idx : idx + batch_size]
            attempt = 0

            while attempt < self.cfg.retry_attempts:
                try:
                    batch_embs = await asyncio.to_thread(_encode_batch_sync, batch)
                    results.extend(batch_embs)
                    break

                except MemoryError as me:
                    logger.warning("MemoryError in batch size=%d: %s", len(batch), me)
                    if len(batch) > 1:
                        self.cfg.batch_size = max(1, self.cfg.batch_size // 2)
                        batch_size = self.cfg.batch_size
                        logger.info("Reduced batch_size to %d", batch_size)
                        continue
                    raise

                except Exception as exc:
                    attempt += 1
                    backoff = self.cfg.retry_backoff * (2 ** (attempt - 1))
                    logger.warning(
                        "Embedder batch attempt %d failed: %s — retrying in %.2fs",
                        attempt,
                        exc,
                        backoff,
                    )
                    await asyncio.sleep(backoff)

                    if attempt >= self.cfg.retry_attempts:
                        logger.error("Retries exhausted — falling back to shim.")
                        results.extend([self._shim_embed(t) for t in batch])
                        break

            idx += batch_size

        return results

    # ----------------------
    # OpenAI Backend (optional)
    # ----------------------
    async def _embed_with_openai(self, texts: List[str]) -> List[List[float]]:
        if not _HAS_OPENAI:
            logger.error("OpenAI client not available in environment; falling back to shim.")
            return [self._shim_embed(t) for t in texts]

        batch_size = max(1, int(self.cfg.batch_size))
        results: List[List[float]] = []
        idx = 0

        # openai library is sync; run in thread
        def _openai_call(batch_texts: List[str]) -> List[List[float]]:
            # openai requires API key configured in environment or openai.api_key set
            out_vectors = []
            for txt in batch_texts:
                resp = openai.Embedding.create(input=txt, model=self.cfg.openai_model)
                vec = resp["data"][0]["embedding"]
                out_vectors.append([float(x) for x in vec])
            return out_vectors

        while idx < len(texts):
            batch = texts[idx : idx + batch_size]
            attempt = 0
            while attempt < self.cfg.retry_attempts:
                try:
                    batch_embs = await asyncio.to_thread(_openai_call, batch)
                    results.extend(batch_embs)
                    break
                except Exception as exc:
                    attempt += 1
                    backoff = self.cfg.retry_backoff * (2 ** (attempt - 1))
                    logger.warning(
                        "OpenAI embed attempt %d failed: %s — retrying in %.2fs",
                        attempt,
                        exc,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    if attempt >= self.cfg.retry_attempts:
                        logger.error(
                            "OpenAI batch failed after retries; falling back to shim for this batch."
                        )
                        results.extend([self._shim_embed(t) for t in batch])
                        break
            idx += batch_size

        return results

    # ----------------------
    # Shim deterministic fallback embedding
    # ----------------------
    def _shim_embed(self, text: str, dim: int = 384) -> List[float]:
        """
        Deterministic shim: SHA256 bytes expanded to `dim`, normalized.
        Useful for CI/tests where heavy deps or API keys are unavailable.
        """
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = []
        i = 0
        while len(vec) < dim:
            vec.append(float(digest[i % len(digest)]) / 255.0)
            i += 1
        # normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec
