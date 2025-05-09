import whisper
from langchain.tools import tool

@tool
def speech(audio):
    """Transcribe audio to text using Whisper model."""
    model = whisper.load_model("base")
    result = model.transcribe(audio)

    return result["text"]
