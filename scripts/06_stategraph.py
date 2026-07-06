"""
Script 06 — StateGraph (Manual Agent)
Rebuilds the same agent from 04_memory.py using StateGraph directly,
instead of the create_agent shortcut. Compare both to see what
create_agent was doing behind the scenes.

Key concepts:
  - StateGraph: defines the graph structure (nodes + edges)
  - MessagesState: built-in state schema that tracks a list of messages
  - Nodes: functions the graph can run (call_model, call_tools)
  - Edges: decide which node runs next (conditional routing)
  - compile(): turns the builder into a runnable graph
"""

import os
import urllib.error
import urllib.request
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import SystemMessage
from langchain_core.globals import set_debug
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

set_debug(True)
load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

SYSTEM_PROMPT = """You are a literary data assistant.

## Capabilities

- `fetch_text_from_url`: loads document text from a URL into the conversation.
Do not guess line counts or positions—ground them in tool results from the saved file."""


# --- Tool ---
@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch the document from a URL."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"
    text = raw.decode("utf-8", errors="replace")
    return text


# --- Model ---
tools = [fetch_text_from_url]

model = init_chat_model(
    f"ollama:{MODEL}",
    base_url=BASE_URL,
    temperature=0.5,
    timeout=300,
    max_tokens=1024,
    streaming=True,
)

# bind_tools tells the model what tools are available so it can
# generate tool_calls in its response. create_agent does this for you.
model_with_tools = model.bind_tools(tools)


# --- Nodes ---
# Each node is a function that takes state and returns updated state.

def call_model(state: MessagesState):
    """Call the LLM with the system prompt + conversation messages."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


# ToolNode is a prebuilt node that executes tool calls from the model's response.
# It looks at the last AIMessage's tool_calls and runs the matching functions.
tool_node = ToolNode(tools)


# --- Routing ---
# After the model responds, we check: did it call a tool, or is it done?
# This is the "conditional edge" that creates the ReAct loop.

def should_continue(state: MessagesState):
    """Route to 'tools' if the model made tool calls, otherwise 'end'."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# --- Build the graph ---
# This is what create_agent does internally:
#   1. A "model" node that calls the LLM
#   2. A "tools" node that executes tool calls
#   3. An edge from START -> model (always start with the model)
#   4. A conditional edge from model -> tools OR end
#   5. An edge from tools -> model (loop back after tool execution)

builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("model", call_model)
builder.add_node("tools", tool_node)

# Add edges
builder.add_edge(START, "model")                    # always start with model
builder.add_conditional_edges("model", should_continue)  # model -> tools or end
builder.add_edge("tools", "model")                  # after tools, go back to model

# Compile with checkpointer for memory
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# --- Run ---
# Same 3-turn test as 04_memory.py to show identical behavior
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "session-1"}}

    # Turn 1: fetch and summarize
    print("=== Turn 1 ===")
    result1 = graph.invoke(
        {"messages": [{"role": "user", "content":
            "Fetch https://peps.python.org/pep-0020/ "
            "and summarize the main principles in 2-3 sentences."
        }]},
        config=config,
    )
    print(result1["messages"][-1].content)

    # Turn 2: follow-up — agent should remember what it fetched
    print("\n=== Turn 2 ===")
    result2 = graph.invoke(
        {"messages": [{"role": "user", "content":
            "Which principle do you think is most relevant to writing clean code?"
        }]},
        config=config,
    )
    print(result2["messages"][-1].content)

    # Turn 3: different thread — agent has NO memory of previous turns
    print("\n=== Turn 3 (new thread) ===")
    result3 = graph.invoke(
        {"messages": [{"role": "user", "content":
            "What did I ask you to fetch?"
        }]},
        config={"configurable": {"thread_id": "session-2"}},
    )
    print(result3["messages"][-1].content)
