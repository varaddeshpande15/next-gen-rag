import logging
import asyncio
from typing import List, Dict
from .sarvam_ocr import process_pdf_sarvam
from .paddle_ocr import process_pdf_paddle

logger = logging.getLogger(__name__)

async def process_pdf(file_path: str, language: str = "hi-IN") -> List[Dict]:
    try:
        return await asyncio.to_thread(process_pdf_sarvam, file_path, language)
    except Exception as e:
        logger.warning(f"Sarvam API failed: {e}. Falling back to PaddleOCR...")
        return await asyncio.to_thread(process_pdf_paddle, file_path)
