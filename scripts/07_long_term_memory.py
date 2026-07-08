"""
Script 07 — Long-Term Memory (Store)
Adds InMemoryStore so the agent can persist data across threads.

Key concepts:
  - Checkpointer (short-term): conversation history scoped to a thread_id
  - Store (long-term): shared data accessible from any thread
  - Namespaces: organize stored data, e.g. ("user_123", "preferences")
  - store.put(): save data
  - store.search(): retrieve data

Builds on 06_stategraph.py — uses StateGraph directly.
"""

import os
import uuid
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langchain_core.globals import set_debug
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime

set_debug(True)
load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# --- Context ---
# Passed at invocation time to identify the user.
# The agent uses this to namespace stored memories per user.
@dataclass
class UserContext:
    user_id: str


# --- Model ---
model = init_chat_model(
    f"ollama:{MODEL}",
    base_url=BASE_URL,
    temperature=0.5,
    timeout=300,
    max_tokens=1024,
)


# --- Node ---
def call_model(state: MessagesState, runtime: Runtime[UserContext]):
    """Call the LLM with user memories injected into the system prompt."""
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")

    # Search the store for any saved memories about this user
    memories = runtime.store.search(namespace, query=state["messages"][-1].content, limit=5)
    memory_text = "\n".join([item.value["data"] for item in memories])

    system_msg = f"""You are a helpful assistant.

## User memories
{memory_text if memory_text else "No memories saved yet."}

## Instructions
- If the user asks you to remember something, confirm that you will.
- Be concise."""

    response = model.invoke(
        [SystemMessage(content=system_msg)] + state["messages"]
    )

    # If the user asks the model to remember something, store it
    last_message = state["messages"][-1].content.lower()
    if "remember" in last_message:
        # Extract what to remember (everything after "remember")
        memory_content = state["messages"][-1].content
        runtime.store.put(
            namespace,
            str(uuid.uuid4()),
            {"data": memory_content},
        )

    return {"messages": [response]}


# --- Build the graph ---
builder = StateGraph(MessagesState, context_schema=UserContext)
builder.add_node("model", call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)

checkpointer = InMemorySaver()
store = InMemoryStore()

graph = builder.compile(
    checkpointer=checkpointer,
    store=store,
)


# --- Run ---
if __name__ == "__main__":
    user_ctx = UserContext(user_id="user_alice")

    # Thread 1: user asks the agent to remember something
    print("=== Thread 1: Save a memory ===")
    result1 = graph.invoke(
        {"messages": [{"role": "user", "content":
            "Please remember that my favorite language is Python."
        }]},
        config={"configurable": {"thread_id": "thread-1"}},
        context=user_ctx,
    )
    print(result1["messages"][-1].content)

    # Thread 1: save another memory
    print("\n=== Thread 1: Save another memory ===")
    result2 = graph.invoke(
        {"messages": [{"role": "user", "content":
            "Also remember that I prefer concise answers."
        }]},
        config={"configurable": {"thread_id": "thread-1"}},
        context=user_ctx,
    )
    print(result2["messages"][-1].content)

    # Thread 2: completely new conversation — but memories persist!
    print("\n=== Thread 2: New thread, memories should be available ===")
    result3 = graph.invoke(
        {"messages": [{"role": "user", "content":
            "What do you know about my preferences?"
        }]},
        config={"configurable": {"thread_id": "thread-2"}},
        context=user_ctx,
    )
    print(result3["messages"][-1].content)

    # Different user: should NOT see Alice's memories
    print("\n=== Thread 3: Different user ===")
    result4 = graph.invoke(
        {"messages": [{"role": "user", "content":
            "What do you know about my preferences?"
        }]},
        config={"configurable": {"thread_id": "thread-3"}},
        context=UserContext(user_id="user_bob"),
    )
    print(result4["messages"][-1].content)

    # Peek at what's in the store
    print("\n=== Store contents for Alice ===")
    items = store.search(("user_alice", "memories"), query="", limit=10)
    for item in items:
        print(f"  - {item.value['data']}")

    print("\n=== Store contents for Bob ===")
    items = store.search(("user_bob", "memories"), query="", limit=10)
    for item in items:
        print(f"  - {item.value['data']}")
    if not items:
        print("  (empty)")
