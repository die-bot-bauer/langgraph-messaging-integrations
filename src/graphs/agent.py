from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI


my_agent = create_react_agent(
    "openai:gpt-4o",
    tools=[
        create_manage_memory_tool("memories"),
        create_search_memory_tool("memories"),
    ],
)
