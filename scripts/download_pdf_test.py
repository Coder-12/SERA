import asyncio
from pathlib import Path

from sera.services.pdf_downloader import PDFDownloader


async def main():
    base_dir = Path("data/test_papers")  # choose any folder
    downloader = PDFDownloader(base_dir=base_dir)
    pdf_path = await downloader.fetch(
        "arXiv:2104.11502v1", "https://arxiv.org/pdf/2104.11502v1.pdf"
    )
    print("Downloaded PDF:", pdf_path)


asyncio.run(main())
