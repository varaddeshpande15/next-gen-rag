import pdf2image
import numpy as np
from paddleocr import PaddleOCR
from typing import List, Dict
from .preprocess import preprocess_for_ocr

ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')

def process_pdf_paddle(file_path: str) -> List[Dict]:
    images = pdf2image.convert_from_path(file_path)
    pages = []
    for i, img in enumerate(images):
        open_cv_image = np.array(img)[:, :, ::-1].copy()
        processed_img = preprocess_for_ocr(open_cv_image)
        result = ocr_engine.ocr(processed_img)
        
        text_content = []
        if result and len(result) > 0 and result[0] is not None:
            for line in result[0]:
                try:
                    text_content.append(str(line[1][0]))
                except (IndexError, TypeError, Exception):
                    continue
                
        text = "\n".join(text_content)
        if text.strip():
            pages.append({"page_num": i + 1, "text": text})
            
    return pages
