from paddleocr import PaddleOCR
import numpy as np

ocr = PaddleOCR(use_angle_cls=True, lang='en')
img = np.zeros((100, 100, 3), dtype=np.uint8) # Blank image
res = ocr.ocr(img)
print("Result for blank img:", res)

# Try with some random noise
img2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
res2 = ocr.ocr(img2)
print("Result for noise img:", res2)
