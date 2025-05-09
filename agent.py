import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from func import add
from tesseract import ocr
from mouse import point_mouse
from click import clicked
from speech2text import speech
from CLIP import analyze_screen_with_grid, locate_patch_coords
#Apis
os.environ["TAVILY_API_KEY"] = "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"
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = "AIzaSyB5QXOFYTukqGXFkB0K7GUCTtHs92uTsrA"

#Search
search = TavilySearchResults(max_results=2)
search_results = search.invoke("Curreent events in the world")

#Tools
tools = [search,speech, add, ocr, point_mouse, clicked,analyze_screen_with_grid,locate_patch_coords]

#Model
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")


memory = MemorySaver()

agent = create_react_agent(model, tools, checkpointer=memory)
config = {"configurable": {"thread_id":"abc123" } }

for step in agent.stream(
  {"messages": [HumanMessage(content="I wanna find 'VALORANT' on my screen, then click on it. to locate it, you tool: CLIP tool")]},config,
  stream_mode = "values"                                                   
):
  step["messages"][-1].pretty_print()