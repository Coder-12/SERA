# Path: sera/services/pdf_downloader.py

"""
Async PDF Downloader for SERA

Responsibilities:
- Download PDFs via streaming (httpx.AsyncClient) into a configured download directory.
- Use a safe .part temporary file and atomic rename on success.
- Enforce max file size and simple PDF validity check (%PDF header).
- Retry transient network errors with exponential backoff (configurable).
- Provide a simple cache check by paper_id -> file path.
- Emit structured logs via the standard logging module.

Usage (example):
    from configs.config import settings
    from sera.services.pdf_downloader import PDFDownloader

    downloader = PDFDownloader(base_dir=settings.download_dir)
    path = await downloader.fetch(paper_id="arXiv:0001.00001", pdf_url="https://arxiv.org/pdf/0001.00001.pdf")
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

from configs.config import settings

logger = logging.getLogger(__name__)


class DownloadError(Exception):
    """Raised when a PDF cannot be downloaded or validated."""


@dataclass
class PDFDownloader:
    """
    Asynchronous PDF Downloader.

    Parameters
    ----------
    base_dir:
        Directory where PDFs will be stored. Files are saved as <paper_id>.pdf.
    max_size_mb:
        Maximum allowed PDF size in megabytes (prevents huge downloads).
    timeout_seconds:
        Per-request timeout for httpx requests.
    max_retries:
        Number of retry attempts for transient failures.
    backoff_factor:
        Base backoff seconds; actual wait uses exponential backoff (backoff_factor * 2**attempt).
    user_agent:
        HTTP User-Agent header to present to servers.
    """

    base_dir: Path | str = Path("data/papers")
    max_size_mb: int = int(getattr(settings, "max_pdf_size_mb", 25))
    timeout_seconds: int = int(getattr(settings, "download_timeout_seconds", 60))
    max_retries: int = int(getattr(settings, "max_download_retries", 3))
    backoff_factor: float = float(getattr(settings, "download_backoff_factor", 1.5))
    user_agent: str = "SERA-PDF-Downloader/1.0"

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # bytes limit
        self._max_bytes = int(self.max_size_mb) * 1024 * 1024
        # httpx client default headers
        self._headers = {"User-Agent": self.user_agent}
        # optional semaphore for concurrency control (caller may pass its own)
        self._download_sem: Optional[asyncio.Semaphore] = None

    def set_semaphore(self, sem: asyncio.Semaphore) -> None:
        """Optionally set a semaphore to bound concurrent downloads."""
        self._download_sem = sem

    def _target_path(self, paper_id: str) -> Path:
        # sanitize paper_id for filename safety: replace slashes and colons
        safe_name = paper_id.replace("/", "_").replace(":", "_")
        return self.base_dir / f"{safe_name}.pdf"

    def _temp_path(self, paper_id: str) -> Path:
        safe_name = paper_id.replace("/", "_").replace(":", "_")
        return self.base_dir / f"{safe_name}.pdf.part"

    def is_cached(self, paper_id: str) -> bool:
        """Return True if the PDF already exists and passes a minimal validity check."""
        target = self._target_path(paper_id)
        if not target.exists():
            return False
        try:
            with target.open("rb") as fh:
                header = fh.read(4)
                return header.startswith(b"%PDF")
        except Exception:
            logger.warning("Cached file exists but failed header check: %s", target)
            return False

    async def fetch(self, paper_id: str, pdf_url: str, force: bool = False) -> Path:
        """
        Download the PDF for `paper_id` from `pdf_url`.

        Returns the Path to the saved PDF.

        Raises:
            DownloadError on failure (after retries), or if validation fails.
        """
        if not pdf_url:
            raise DownloadError("No pdf_url provided")

        target = self._target_path(paper_id)
        temp = self._temp_path(paper_id)

        # If cached and not forced, return immediately
        if not force and self.is_cached(paper_id):
            logger.info("PDF cached, skipping download: %s", target)
            return target

        # Bound concurrency if sem provided
        sem = self._download_sem
        if sem is not None:
            async with sem:
                return await self._download_with_retries(pdf_url, temp, target)
        else:
            return await self._download_with_retries(pdf_url, temp, target)

    async def _download_with_retries(self, pdf_url: str, temp: Path, target: Path) -> Path:
        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt < self.max_retries:
            try:
                return await self._download_once(pdf_url, temp, target)
            except Exception as exc:
                last_exc = exc
                wait = self.backoff_factor * (2**attempt)
                logger.warning(
                    "PDF download failed (attempt %d/%d) for %s: %s — retrying in %.1fs",
                    attempt + 1,
                    self.max_retries,
                    pdf_url,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)
                attempt += 1

        logger.error("All download retries failed for URL: %s", pdf_url)
        raise DownloadError(f"Failed to download {pdf_url}") from last_exc

    async def _download_once(self, pdf_url: str, temp: Path, target: Path) -> Path:
        """
        Single attempt to download the PDF with streaming. Writes to .part then renames.
        """
        # remove stale temp file if exists
        try:
            if temp.exists():
                temp.unlink()
        except Exception:
            logger.debug("Could not remove stale temp file: %s", temp)

        timeout = httpx.Timeout(self.timeout_seconds)
        async with httpx.AsyncClient(
            timeout=timeout, headers=self._headers, follow_redirects=True
        ) as client:
            # Stream response to file to avoid OOM
            async with client.stream("GET", pdf_url) as resp:
                resp.raise_for_status()

                # If content-length header exists, enforce max size early
                content_length = resp.headers.get("Content-Length")
                if content_length is not None:
                    try:
                        cl = int(content_length)
                        if cl > self._max_bytes:
                            raise DownloadError(
                                f"Content-Length {cl} exceeds max allowed {self._max_bytes} bytes"
                            )
                    except ValueError:
                        pass  # non-integer header, ignore

                bytes_written = 0
                # Write to temp file
                with temp.open("wb") as fh:
                    async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        bytes_written += len(chunk)
                        if bytes_written > self._max_bytes:
                            # Abort and cleanup
                            fh.flush()
                            temp.unlink(missing_ok=True)
                            raise DownloadError(
                                f"Downloaded data exceeded {self.max_size_mb} MB limit"
                            )

        # Basic validation: must start with %PDF
        try:
            with temp.open("rb") as fh:
                header = fh.read(4)
                if not header.startswith(b"%PDF"):
                    temp.unlink(missing_ok=True)
                    raise DownloadError(
                        "Downloaded file does not look like a PDF (missing %PDF header)"
                    )
        except Exception as exc:
            raise DownloadError("Failed to validate downloaded PDF") from exc

        # Atomic rename to final path
        try:
            shutil.move(str(temp), str(target))
            logger.info("Downloaded and saved PDF: %s (bytes=%d)", target, bytes_written)
        except Exception as exc:
            # attempt to clean temp file
            temp.unlink(missing_ok=True)
            raise DownloadError("Failed to move temporary PDF to final location") from exc

        return target
