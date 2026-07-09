"""
Script 09 — Delete Messages (Memory Management)
Permanently removes messages from graph state using RemoveMessage.

Key concepts:
  - RemoveMessage: marks a message for deletion from state
  - Unlike trim_messages (08), this actually deletes from the checkpointer
  - Messages are deleted by their id
  - Can delete specific messages or all messages (REMOVE_ALL_MESSAGES)
  - Must keep valid message history (e.g., tool results need their
    preceding AI message with tool_calls)

Builds on 06_stategraph.py — uses StateGraph with a delete node.
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import RemoveMessage
from langchain_core.messages import SystemMessage
from langchain_core.globals import set_debug
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver

set_debug(True)
load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

model = init_chat_model(
    f"ollama:{MODEL}",
    base_url=BASE_URL,
    temperature=0.5,
    timeout=300,
    max_tokens=1024,
)


def call_model(state: MessagesState):
    """Call the LLM."""
    response = model.invoke(
        [SystemMessage(content="You are a helpful assistant. Be concise.")]
        + state["messages"]
    )
    return {"messages": [response]}


def delete_old_messages(state: MessagesState):
    """Delete all but the last 2 messages from state.

    This runs after every model call. It keeps the conversation short
    by permanently removing older messages from the checkpointer.
    """
    messages = state["messages"]
    if len(messages) > 2:
        # Mark all but the last 2 for deletion
        to_delete = [RemoveMessage(id=m.id) for m in messages[:-2]]
        print(f"\n  [delete] Removing {len(to_delete)} old messages, "
              f"keeping last 2\n")
        return {"messages": to_delete}


# --- Build the graph ---
# Two nodes run in sequence: model responds, then old messages are deleted.
builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_node("delete", delete_old_messages)

builder.add_edge(START, "model")
builder.add_edge("model", "delete")    # after model, clean up old messages
builder.add_edge("delete", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# --- Run ---
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "delete-demo"}}

    conversations = [
        "Hi, my name is Alice.",
        "I work as a software engineer.",
        "My favorite language is Python.",
        "What do you know about me?",
    ]

    for i, msg in enumerate(conversations, 1):
        print(f"=== Turn {i}: {msg} ===")
        result = graph.invoke(
            {"messages": [{"role": "user", "content": msg}]},
            config=config,
        )
        print(result["messages"][-1].content)

        # Show what's actually in state after deletion
        state = graph.get_state(config)
        msg_count = len(state.values["messages"])
        print(f"  [state] Messages in checkpointer: {msg_count}")
        print()

    # By turn 4, earlier messages are gone — the agent can't remember Alice's name
    # because those messages were permanently deleted, not just trimmed.
    print("=== Compare with 08_trim_messages.py ===")
    print("trim_messages: model sees less, but full history stays in state")
    print("RemoveMessage: messages are permanently gone from state")
