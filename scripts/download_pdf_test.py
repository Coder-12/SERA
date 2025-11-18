import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import asyncio
from pathlib import Path

from services.pdf_downloader import PDFDownloader


async def main():
    base_dir = Path("data/test_papers")  # choose any folder
    downloader = PDFDownloader(base_dir=base_dir)
    pdf_path = await downloader.fetch(
        "arXiv:2104.11502v1", "https://arxiv.org/pdf/2104.11502v1.pdf"
    )
    print("Downloaded PDF:", pdf_path)


asyncio.run(main())
