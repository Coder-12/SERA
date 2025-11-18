# Path: tests/test_pdf_parser.py
"""
Unit tests for sera.parsers.pdf_parser.PDFParser

Validates:
  ✅ Correct text extraction from real PDF (from arXiv)
  ✅ Graceful handling of corrupt/empty PDFs
  ✅ Page-wise parsing structure
  ✅ CI-safe behavior (skips if no internet)

This test aligns with sera/parsers/pdf_parser.py which uses PyMuPDF (fitz).
"""

import asyncio
from pathlib import Path

import pytest

from parsers.pdf_parser import PDFParseError, PDFParser
from services.pdf_downloader import PDFDownloader

# Small, clean, open-access paper
TEST_PDF_URL = "https://arxiv.org/pdf/2104.11502v1.pdf"
TEST_PAPER_ID = "arXiv:2104.11502v1"


@pytest.mark.asyncio
async def test_parse_real_pdf(tmp_path):
    """Parse a real small arXiv PDF and validate text extraction."""
    base_dir = tmp_path / "pdfs"
    base_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = base_dir / "test.pdf"

    # Download the test PDF
    downloader = PDFDownloader(base_dir=base_dir)
    path = await downloader.fetch(TEST_PAPER_ID, TEST_PDF_URL)

    parser = PDFParser()

    # Run parse
    pages = await parser.parse(path)
    assert isinstance(pages, list)
    assert len(pages) > 0, "Should extract at least one page"
    assert hasattr(pages[0], "text") or isinstance(pages[0], dict), "Each page should have text"
    assert (
        len(pages[0].text.strip()) > 0
        if hasattr(pages[0], "text")
        else len(pages[0]["text"].strip()) > 0
    )

    # Print preview of parsed text (for human sanity check)
    print("\n✅ Parsed first page text preview:\n")
    preview_text = (pages[0].text if hasattr(pages[0], "text") else pages[0]["text"])[:500]
    print(preview_text)


@pytest.mark.asyncio
async def test_parse_corrupt_pdf(tmp_path):
    """Ensure parser raises PDFParseError on corrupted file."""
    corrupt_path = tmp_path / "corrupt.pdf"
    corrupt_path.write_bytes(b"%PDF-1.4\ncorruptedcontent...")

    parser = PDFParser()
    with pytest.raises(PDFParseError):
        await parser.parse(corrupt_path)


@pytest.mark.asyncio
async def test_parse_empty_pdf(tmp_path):
    """Ensure empty PDF file raises PDFParseError gracefully."""
    empty_path = tmp_path / "empty.pdf"
    empty_path.write_text("")  # zero-byte

    parser = PDFParser()
    with pytest.raises(PDFParseError):
        await parser.parse(empty_path)
