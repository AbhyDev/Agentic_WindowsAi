import pyautogui
from langchain.tools import tool

@tool
def clicked(x: int, y: int):
    """
    Move the mouse to the specified (x, y) coordinates.
    """
    pyautogui.moveTo(x, y, click=2)
