from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated

# --- Toolbelt ---
arxiv_tool = ArxivQueryRun()
wikipedia_api_wrapper = WikipediaAPIWrapper()
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)
tavily_tool = TavilySearch(max_results=5)
tool_belt = [arxiv_tool, wikipedia_tool, tavily_tool]

# --- Model ---
model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
model = model.bind_tools(tool_belt)

# --- State ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# --- Nodes ---
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tool_belt)

# --- Graph ---
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("action", tool_node)
graph.set_entry_point("agent")

def should_continue(state):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    return END

graph.add_conditional_edges("agent", should_continue)
graph.add_edge("action", "agent")
agent_graph = graph.compile()

# --- FastAPI ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data["question"]
    inputs = {"messages": [HumanMessage(content=question)]}
    result = await agent_graph.ainvoke(inputs)
    messages = result["messages"]
    # Collect tool names used
    tools_used = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                name = call.get("name")
                if name and name not in tools_used:
                    tools_used.append(name)
    return {
        "response": messages[-1].content,
        "tools_used": tools_used
    }

# Serve the static frontend (React build)
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")