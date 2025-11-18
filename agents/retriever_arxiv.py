"""
ArXiv retrieval wrapper.
- Provides an async `search` method that returns standardized metadata list.
- Uses httpx for async HTTP requests. If you prefer a dedicated library, swap internals.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel

from configs.config import settings

logger = logging.getLogger(__name__)


class PaperMeta(BaseModel):
    id: str
    title: str
    abstract: str
    authors: List[str] = []


class ArxivRetriever:
    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or settings.arxiv_api_base

    async def search(self, query: str, max_results: int = 5) -> List[PaperMeta]:
        """
        Query arXiv API using the OAI/RSS ATOM query endpoint, parse minimal fields.
        Returns a list of PaperMeta.
        """
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
        }
        url = f"{self.base_url}?{urlencode(params)}"
        logger.info("ArXiv query url=%s", url)
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            body = r.text

        # parse ATOM XML minimally (title, id, summary)
        root = ET.fromstring(body)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = []
        for entry in root.findall("atom:entry", ns):
            paper_id = (
                entry.find("atom:id", ns).text if entry.find("atom:id", ns) is not None else ""
            )
            title = entry.find("atom:title", ns).text or ""
            summary = entry.find("atom:summary", ns).text or ""
            authors = [
                a.find("atom:name", ns).text
                for a in entry.findall("atom:author", ns)
                if a.find("atom:name", ns) is not None
            ]
            entries.append(
                PaperMeta(
                    id=paper_id.strip(),
                    title=title.strip(),
                    abstract=summary.strip(),
                    authors=authors,
                )
            )
        return entries
