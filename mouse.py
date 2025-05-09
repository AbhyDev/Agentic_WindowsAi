import pyautogui
from langchain.tools import tool

@tool
def point_mouse(x: int, y: int):
    """
    Move the mouse to the specified (x, y) coordinates.
    """
    pyautogui.moveTo(x, y, click=2)
