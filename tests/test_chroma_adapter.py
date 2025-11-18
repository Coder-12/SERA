# Path: tests/test_chroma_adapter.py

from unittest.mock import MagicMock

import pytest

from memory.chroma_adapter import ChromaAdapter
from memory.vector_store_interface import SearchResult


class FakeCollection:

    def __init__(self, supported_includes):
        """
        supported_includes: list of sets of includes this fake collection accepts
        Example:
            [{"embeddings", "metadatas", "distances"},
             {"embeddings", "documents"}]
        """
        self.supported_includes = supported_includes
        self.vectors = []
        self.ids = []
        self.metadatas = []
        self.distances = [[0.1, 0.2, 0.3]]

    def add(self, ids=None, embeddings=None, metadatas=None, **kwargs):
        self.ids.extend(ids)
        self.vectors.extend(embeddings)
        self.metadatas.extend(metadatas)

    def query(self, query_embeddings, n_results, include):
        inc_set = set(include)
        ok = any(inc_set.issubset(s) for s in self.supported_includes)
        if not ok:
            raise ValueError(f"Unsupported include fields: {include}")

        return {
            "ids": [self.ids],
            "embeddings": [self.vectors],
            "metadatas": [self.metadatas],
            "distances": self.distances,
        }


class FakeClient:

    def __init__(self, supported_includes):
        self.collection = FakeCollection(supported_includes)

    def list_collections(self):
        return ["fake"]

    def get_collection(self, name):
        return self.collection


@pytest.mark.parametrize(
    "supported_includes",
    [
        # Simulate old version
        [{"embeddings", "metadatas", "documents", "distances"}],
        # Simulate mid version
        [{"embeddings", "documents"}],
        # Simulate strict version
        [{"embeddings"}],
    ],
)
def test_chroma_include_detection(monkeypatch, supported_includes):
    """
    Ensures ChromaAdapter detects the correct include list across versions.
    """

    # Patch chroma client init
    fake_client = FakeClient(supported_includes)

    def fake_init_client(self):
        return fake_client

    monkeypatch.setattr(ChromaAdapter, "_init_client", fake_init_client)

    adapter = ChromaAdapter(collection_name="test")

    # Trigger include detection
    include = adapter._detect_query_include()

    assert "embeddings" in include
    # ensure detected include is subset of allowed includes
    assert any(set(include).issubset(s) for s in supported_includes)


def test_chroma_add_and_query(monkeypatch):
    """
    Ensures basic add -> query workflow always works.
    """

    supported = [{"embeddings", "metadatas", "distances", "documents"}]
    fake_client = FakeClient(supported)

    def fake_init_client(self):
        return fake_client

    monkeypatch.setattr(ChromaAdapter, "_init_client", fake_init_client)

    adapter = ChromaAdapter(collection_name="test")

    vectors = [[0.1, 0.2, 0.3, 0.4]]
    metadatas = [{"chunk_id": "c1"}]

    ids = adapter.add_documents(vectors, metadatas)
    assert len(ids) == 1

    results = adapter.similarity_search([0.1, 0.2, 0.3, 0.4], top_k=1)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].metadata["chunk_id"] == "c1"
    assert results[0].score > 0


def test_chroma_score_conversion(monkeypatch):
    """
    Ensures distance → similarity conversion is stable.
    """

    supported = [{"embeddings", "metadatas", "documents", "distances"}]
    fake_client = FakeClient(supported)
    fake_client.collection.distances = [[3.0]]  # L2 distance

    def fake_init_client(self):
        return fake_client

    monkeypatch.setattr(ChromaAdapter, "_init_client", fake_init_client)

    adapter = ChromaAdapter(collection_name="test")
    adapter.add_documents([[0, 0, 0, 1]], [{"x": 1}])

    res = adapter.similarity_search([0, 0, 0, 1], 1)
    score = res[0].score

    # L2->sim conversion: 1/(1+dist)
    assert 0 < score < 1


def test_chroma_metadata_passthrough(monkeypatch):
    supported = [{"embeddings"}]
    fake_client = FakeClient(supported)

    def fake_init_client(self):
        return fake_client

    monkeypatch.setattr(ChromaAdapter, "_init_client", fake_init_client)

    adapter = ChromaAdapter(collection_name="test")
    adapter.add_documents([[1, 2, 3, 4]], [{"hello": "world"}])

    res = adapter.similarity_search([1, 2, 3, 4], 1)
    assert res[0].metadata["hello"] == "world"
