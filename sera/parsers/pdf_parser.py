# Path: sera/parsers/pdf_parser.py

"""
PDF Parser for SERA (PyMuPDF / fitz based)

Purpose
-------
- Extract clean, ordered text from a local PDF file (downloaded by PDFDownloader).
- Return a list of PageData objects (page index, text, basic stats) suitable for chunking.
- Be async-friendly: heavy CPU-bound work runs in threadpool via asyncio.to_thread.
- Fail gracefully with clear exceptions when PDF is corrupt or missing dependencies.

Notes
-----
- This implementation prefers PyMuPDF (`fitz`). Install with:
    pip install pymupdf

- If `fitz` is not available, the parser raises ImportError with an actionable message.
- For scanned/PDF-as-image files, OCR fallback (Tesseract) is not included here but can be added later.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class PDFParseError(Exception):
    """Raised when PDF parsing fails or file is invalid."""


try:
    import fitz  # PyMuPDF

    _HAS_FITZ = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_FITZ = False


@dataclass
class PageData:
    """Simple serializable container for per-page text data."""

    page_index: int
    text: str
    n_chars: int
    n_words: int


class PDFParser:
    """
    PDFParser extracts page-ordered text from PDFs.

    Usage (async):
        parser = PDFParser()
        pages = await parser.parse(path_to_pdf, max_pages=100)

    Returned value:
        List[PageData] in ascending page_index order.
    """

    def __init__(self, max_pages: Optional[int] = None):
        """
        :param max_pages: optionally limit pages parsed (None = parse all)
        """
        self.max_pages = max_pages

        if not _HAS_FITZ:
            logger.warning(
                "PyMuPDF (fitz) not available. PDF parsing will not work until installed."
            )

    async def parse(
        self, pdf_path: Path | str, *, max_pages: Optional[int] = None
    ) -> List[PageData]:
        """
        Parse the PDF at `pdf_path` and return a list of PageData.

        :param pdf_path: path-like or string to a local PDF file
        :param max_pages: optional override to limit number of pages parsed
        :raises PDFParseError: on invalid file, missing dependency, or parse failure
        """
        if not _HAS_FITZ:
            raise PDFParseError(
                "PyMuPDF (fitz) is required for PDF parsing. Install via `pip install pymupdf`."
            )

        p = Path(pdf_path)
        if not p.exists() or not p.is_file():
            raise PDFParseError(f"PDF file not found: {pdf_path}")

        effective_max = max_pages if max_pages is not None else self.max_pages

        # Run the synchronous parsing in a thread to avoid blocking the event loop
        try:
            pages = await asyncio.to_thread(self._parse_sync, p, effective_max)
            return pages
        except Exception as exc:
            logger.exception("PDF parsing failed for %s: %s", pdf_path, exc)
            raise PDFParseError(f"Failed to parse PDF {pdf_path}: {exc}") from exc

    def _parse_sync(self, pdf_path: Path, max_pages: Optional[int]) -> List[PageData]:
        """
        Synchronous parsing using PyMuPDF (runs in a thread).
        """
        # Defensive open: handle malformed files
        doc = None
        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            raise PDFParseError(f"Could not open PDF: {exc}") from exc

        try:
            total_pages = doc.page_count
            pages_to_read = total_pages if max_pages is None else min(total_pages, int(max_pages))

            results: List[PageData] = []
            for page_idx in range(pages_to_read):
                try:
                    page = doc.load_page(page_idx)
                    # Use "text" extraction which is generally robust (may include newlines)
                    text = page.get_text("text") or ""
                    # Post-process whitespace: collapse multiple spaces, strip
                    cleaned = " ".join(text.split()).strip()
                    n_chars = len(cleaned)
                    n_words = len(cleaned.split()) if cleaned else 0
                    results.append(
                        PageData(
                            page_index=page_idx, text=cleaned, n_chars=n_chars, n_words=n_words
                        )
                    )
                except Exception:
                    # Skip problematic pages but continue parsing rest, while logging
                    logger.exception(
                        "Failed to parse page %d of %s; skipping page.", page_idx, pdf_path
                    )
                    continue

            if not results:
                raise PDFParseError("No extractable text found in PDF.")

            return results
        finally:
            try:
                doc.close()
            except Exception:
                pass
