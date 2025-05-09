import pyautogui
from PIL import Image

def capture_screen() -> Image.Image:
    """
    Take a full-screen screenshot and return a PIL Image.
    """
    
    return pyautogui.screenshot()
