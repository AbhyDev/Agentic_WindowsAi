import pytesseract
import cv2
import numpy as np
from PIL import Image
from screenshot import capture_screen
from langchain.tools import tool
# If tesseract is not in your PATH, specify:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
@tool
def ocr(target: str, conf_thresh=60):
    """
    Run Tesseract OCR on the screenshot, look for 'target' (case-insensitive).
    Return the center (x,y) of the bounding box if found, else None.
    """
    screen_img = capture_screen()
    # Convert to OpenCV image
    cv_img = cv2.cvtColor(np.array(screen_img), cv2.COLOR_RGB2BGR)
    data = pytesseract.image_to_data(cv_img, output_type=pytesseract.Output.DICT)
    
    for i, text in enumerate(data['text']):
        if not text: continue
        if target.lower() in text.lower() and int(data['conf'][i]) >= conf_thresh:
            x = data['left'][i] + data['width'][i] // 2
            y = data['top'][i]  + data['height'][i] // 2
            return x, y
    return None
