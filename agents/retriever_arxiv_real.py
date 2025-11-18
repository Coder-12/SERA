"""
Async ArXiv retriever (robust, dependency-light).

Provides:
  - ArxivRetriever.search(query, max_results) -> list[PaperMeta]
  - ArxivRetriever.fetch_metadata(paper_id) -> PaperMeta | None

Design goals:
  - Async-friendly (httpx.AsyncClient)
  - Simple retry/backoff (no extra deps required)
  - Defensive XML parsing of ATOM feed
  - Clear Pydantic model for returned metadata
  - Respectable defaults for timeouts and retries; configurable via constructor
"""

from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from typing import List, Optional
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PaperMeta(BaseModel):
    """
    Minimal paper metadata model returned by the retriever.
    Fields may be empty strings if not present.
    """

    paper_id: str
    title: str
    abstract: str
    authors: List[str] = []
    pdf_url: Optional[str] = None
    published: Optional[str] = None  # ISO date string if available


class ArxivRetriever:
    def __init__(
        self,
        base_url: str = "https://export.arxiv.org/api/query",
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        http_headers: Optional[dict] = None,
    ):
        """
        :param base_url: ArXiv API base ATOM query endpoint.
        :param timeout_seconds: Per-request timeout.
        :param max_retries: Number of attempts for transient failures.
        :param backoff_factor: Exponential backoff base in seconds.
        :param http_headers: Optional headers (e.g., {"User-Agent": "sera/0.1"}).
        """
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.http_headers = http_headers or {"User-Agent": "SERA-Arxiv-Retriever/1.0"}

    async def _get_text_with_retry(self, url: str) -> str:
        """
        Perform an HTTP GET with simple exponential backoff retry.
        Raises httpx.HTTPError on permanent failure.
        """
        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt < self.max_retries:
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout_seconds, headers=self.http_headers
                ) as client:
                    response = await client.get(url)
                    response.raise_for_status()  # raise if not 200 OK !
                    return response.text
            except Exception as exc:
                last_exc = exc
                wait = self.backoff_factor * (2**attempt)
                logger.warning(
                    "Request to %s failed (attempt %d/%d): %s - retrying in %.1fs",
                    url,
                    attempt + 1,
                    self.max_retries,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)
                attempt += 1

        # final attempt without catching to bubble up the last exception
        logger.error("All retries failed for URL: %s", url)
        raise last_exc  # type: ignore

    def _parse_atom_feed(self, body: str) -> List[PaperMeta]:
        """
        Parse a real arXiv Atom XML feed into a list of PaperMeta objects.

        This version exactly matches the structure:
          <entry>
            <id>http://arxiv.org/abs/...v1</id>
            <title>...</title>
            <summary>...</summary>
            <link href="https://arxiv.org/pdf/..."/>
            <author><name>...</name></author>...
            <published>...</published>
            <updated>...</updated>
          </entry>
        """
        try:
            root = ET.fromstring(body)
        except ET.ParseError as e:
            logger.exception("Failed to parse ATOM XML: %s", e)
            return []

        ns_atom = "{http://www.w3.org/2005/Atom}"
        ns_arxiv = "{http://arxiv.org/schemas/atom}"

        def safe_text(elem):
            return (elem.text or "").strip() if elem is not None and elem.text else ""

        results: List[PaperMeta] = []

        # Find all <entry> nodes under <feed>
        entries = root.findall(f"{ns_atom}entry") or root.findall(".//entry")

        for entry in entries:
            try:
                # --- Core metadata ---
                paper_id = safe_text(entry.find(f"{ns_atom}id"))
                title = safe_text(entry.find(f"{ns_atom}title"))
                abstract = safe_text(entry.find(f"{ns_atom}summary"))
                published = safe_text(entry.find(f"{ns_atom}published")) or safe_text(
                    entry.find(f"{ns_atom}updated")
                )

                # --- Authors ---
                authors = []
                for author_elem in entry.findall(f"{ns_atom}author"):
                    name = safe_text(author_elem.find(f"{ns_atom}name"))
                    if name:
                        authors.append(name)

                # --- PDF Link ---
                pdf_url = None

                # First try namespaced links (<atom:link> tags)
                links = entry.findall(f"{ns_atom}link") or entry.findall("link")

                for link in links:
                    href = link.attrib.get("href", "")
                    type_ = link.attrib.get("type", "")
                    rel = link.attrib.get("rel", "")
                    # Most arXiv entries don’t set type, so rely on URL suffix
                    if "pdf" in href or href.endswith(".pdf"):
                        pdf_url = href
                        break

                # Fallback: if no explicit PDF link, derive from paper_id
                if not pdf_url and paper_id and "arxiv.org/abs/" in paper_id:
                    pdf_url = paper_id.replace("/abs/", "/pdf/") + ".pdf"

                # --- Construct PaperMeta ---
                results.append(
                    PaperMeta(
                        paper_id=paper_id,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        pdf_url=pdf_url,
                        published=published,
                    )
                )

            except Exception as e:
                logger.exception("Failed to parse one <entry>; skipping. Error: %s", e)
                continue

        logger.info(f"✅ Parsed {len(results)} papers from arXiv feed.")
        return results

    async def search(self, query: str, max_results: int = 5) -> List[PaperMeta]:
        """
        Search arXiv for query and return up to max_results PaperMeta items.
        Note: query should follow arXiv query format, e.g. 'cat:cs.LG+AND+machine+learning'
        but simple keywords also work.
        """
        params = {"search_query": query, "start": 0, "max_results": max_results}
        url = f"{self.base_url}?{urlencode(params)}"
        logger.info("Arxiv search: %s", url)
        body = await self._get_text_with_retry(url)
        return self._parse_atom_feed(body)

    async def fetch_metadata(self, paper_id_or_url: str) -> Optional[PaperMeta]:
        """
        Fetch a single paper's metadata by its arXiv ID or full URL.
        Example accepted inputs:
            - '2104.11502v1'
            - 'https://arxiv.org/abs/2104.11502v1'
        """
        # Extract the actual ID
        import re

        match = re.search(r"(\d{4}\.\d{4,5}(v\d+)?)", paper_id_or_url)
        paper_id = match.group(1) if match else paper_id_or_url.strip()

        params = {"id_list": paper_id, "max_results": 1}
        url = f"{self.base_url}?{urlencode(params)}"

        try:
            body = await self._get_text_with_retry(url)
            items = self._parse_atom_feed(body)
            if items:
                return items[0]
            else:
                logger.warning(f"No metadata found for paper_id={paper_id}")
                return None
        except Exception as exc:
            logger.exception("fetch_metadata failed for %s: %s", paper_id_or_url, exc)
            return None
