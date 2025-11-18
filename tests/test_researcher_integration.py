import asyncio

from agents.researcher_agent import ResearcherAgent
from agents.retriever_arxiv import ArxivRetriever
from memory.vector_store import SimpleVectorStore


def test_researcher_runs(tmp_path):
    vs_path = tmp_path / "vs.json"
    store = SimpleVectorStore(storage_path=str(vs_path))
    retriever = (
        ArxivRetriever()
    )  # Uses real arXiv by default (may be slow) — replace by mocks in CI
    agent = ResearcherAgent(retriever=retriever, vector_store=store)
    # run synchronously for test convenience
    results = asyncio.run(agent.run("machine learning", max_results=1))
    assert isinstance(results, dict)
