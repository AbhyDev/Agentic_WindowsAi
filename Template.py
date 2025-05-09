# export TAVILY_API_KEY="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"

import getpass
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

#Apis
os.environ["TAVILY_API_KEY"] = "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"
if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = "AIzaSyB5QXOFYTukqGXFkB0K7GUCTtHs92uTsrA"

#Search
search = TavilySearchResults(max_results=2)
search_results = search.invoke("Curreent events in the world")
# print(search_results)

#Tools
tools = [search]

#Model
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# response = model.invoke([HumanMessage(content="What is the current date?")])
# print(response.content)

model_tools = model.bind_tools(tools)
# response2 = model_tools.invoke([HumanMessage(content="yo bro, how is the weather in Bengaluru rn?")])
# print(response2.content)
# print(response2.tool_calls)

# agent = create_react_agent(model, tools)

# response3 = agent.invoke({"messages": [HumanMessage(content="hi! weather in Bengaluru rn?")]})
# print(response3["messages"])
# for step in agent.stream(
#   {"messages": [HumanMessage(content="what we talked about before")]},
#   stream_mode = "values"
# ):
#   step["messages"][-1].pretty_print()

memory = MemorySaver()

agent = create_react_agent(model, tools, checkpointer=memory)
config = {"configurable": {"thread_id":"abc123" } }
for chunk in agent.stream(
    {"messages": [HumanMessage(content="sum of 5 and 6?")]}, config
):
    print(chunk)
    print("----")   
