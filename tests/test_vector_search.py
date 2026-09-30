from memory.simple_adapter import SimpleVectorAdapter


def test_vector_search(tmp_path, top_k: int = 5):
    """Persist vectors locally, reload them, and return the closest matching chunk first."""
    store_path = tmp_path / "vectors.json"
    store = SimpleVectorAdapter(store_path)
    store.add_documents(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
        [
            {"chunk_id": "chunk-1", "text_preview": "closest"},
            {"chunk_id": "chunk-2", "text_preview": "nearby"},
            {"chunk_id": "chunk-3", "text_preview": "distant"},
        ],
    )

    reloaded_store = SimpleVectorAdapter(store_path, autosave=False)
    results = reloaded_store.similarity_search([1.0, 0.0], top_k=top_k)

    assert len(results) == 3
    assert results[0].metadata["chunk_id"] == "chunk-1"
    assert results[0].score >= results[1].score >= results[2].score
    assert all(0.0 <= result.score <= 1.0 for result in results)
    assert all(result.metadata["text_preview"] for result in results)
