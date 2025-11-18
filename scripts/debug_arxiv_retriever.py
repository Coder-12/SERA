# Path: scripts/debug_arxiv_retriever.py
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import asyncio

from agents.retriever_arxiv_real import ArxivRetriever


async def main():
    retriever = ArxivRetriever(timeout_seconds=10, max_retries=1)
    query = "transformer"  # change freely: "machine learning", "LLM", "state space models"
    url = f"{retriever.base_url}?search_query={query}&max_results=2"
    print(f"🔎 Querying: {url}\n")

    try:
        text = await retriever._get_text_with_retry(url)
        print("✅ Raw response (first 800 chars):")
        print(text)
    except Exception as e:
        print(f"❌ HTTP error: {e}")
        return

    results = retriever._parse_atom_feed(text)
    print(f"\n📦 Parsed {len(results)} results.")
    for i, r in enumerate(results):
        print(f"[{i}] {r.title[:80]!r} | {r.pdf_url}")


if __name__ == "__main__":
    asyncio.run(main())
