from langchain.tools import tool

@tool
def add(a,b):
    """Adds two numbers together.
    Args:
        a (int): The first number.
        b (int): The second number.
    Returns:
        int: The sum of the two numbers.
    """
    return a + b