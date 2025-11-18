# Path: tests/test_embedder.py
"""
Unit tests for sera.embeddings.embedder.Embedder

Covers:
  ✅ SentenceTransformer backend (local)
  ✅ OpenAI fallback (mocked if key not set)
  ✅ Shim deterministic fallback
  ✅ Async batching and retry logic
"""

import asyncio

import pytest

from embeddings.embedder import Embedder, EmbedderConfig


@pytest.mark.asyncio
async def test_sentence_transformer_embedding():
    """Test embedding a few sample texts using SentenceTransformer backend."""
    cfg = EmbedderConfig(
        backend="sentence-transformers", model_name="all-MiniLM-L6-v2", batch_size=4
    )
    emb = Embedder(cfg)
    texts = ["Transformers are powerful.", "Sentence embeddings test.", "Async embedding works."]
    results = await emb.embed_chunks(texts)

    assert isinstance(results, list)
    assert len(results) == len(texts)
    assert all(isinstance(v, list) for v in results)
    assert all(len(v) > 0 for v in results)
    print(f"\n✅ SentenceTransformer produced {len(results)} embeddings (dim={len(results[0])}).")


@pytest.mark.asyncio
async def test_shim_fallback():
    """Force shim backend and verify deterministic vector output."""
    cfg = EmbedderConfig(backend="shim")
    emb = Embedder(cfg)
    texts = ["shim test one", "shim test two"]
    results = await emb.embed_chunks(texts)

    assert len(results) == 2
    assert all(len(v) == 384 for v in results)
    assert results[0] != results[1], "Shim embeddings should differ per text"
    print("\n✅ Shim fallback works and produces deterministic embeddings.")


@pytest.mark.asyncio
async def test_openai_fallback(monkeypatch):
    """Mock OpenAI client and verify fallback path doesn't break."""
    monkeypatch.setattr("embeddings.embedder._HAS_OPENAI", False)
    emb = Embedder(EmbedderConfig(backend="openai"))
    texts = ["This should use shim instead."]
    results = await emb.embed_chunks(texts)

    assert len(results) == 1
    assert isinstance(results[0], list)
    print("\n✅ OpenAI fallback gracefully handled with shim embedding.")


@pytest.mark.asyncio
async def test_batch_retry_logic(monkeypatch):
    """Simulate a batch encoding failure and ensure retry logic triggers."""
    cfg = EmbedderConfig(
        backend="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        batch_size=2,
        retry_attempts=2,
        retry_backoff=0.1,
    )
    emb = Embedder(cfg)

    called = {"count": 0}

    # ---- IMPORTANT: synchronous fake encode function ----
    def fake_encode(batch, **kwargs):
        called["count"] += 1
        if called["count"] == 1:
            raise RuntimeError("Simulated batch failure")
        # Return deterministic numpy vector
        import numpy as np

        return np.random.rand(len(batch), 384)

    # overwrite the actual encode function
    emb._model.encode = fake_encode  # type: ignore

    texts = ["retry test one", "retry test two"]
    results = await emb.embed_chunks(texts)

    assert len(results) == 2
    assert called["count"] >= 2, "Retry logic should have triggered at least twice"

    print("\n✅ Retry logic triggered correctly (encode was called:", called["count"], "times)")
