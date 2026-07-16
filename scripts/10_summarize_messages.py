"""
Script 10 — Summarize Messages (Memory Management)
Compresses older messages into a summary to preserve context without
keeping the full history. Best of both worlds: saves tokens but
retains knowledge from earlier turns.

Key concepts:
  - Extended state: adds a "summary" field alongside messages
  - Summarization node: uses the LLM to compress older messages
  - RemoveMessage: deletes old messages after summarizing them
  - Conditional routing: only summarize when message count exceeds a threshold
  - Summary is prepended to the system prompt on each model call

Compare with:
  - 08_trim_messages.py: drops old messages, loses context
  - 09_delete_messages.py: permanently deletes, loses context
  - This script: summarizes first, then deletes — context preserved

Builds on 06_stategraph.py.
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import RemoveMessage
from langchain_core.messages import SystemMessage, HumanMessage
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


# --- State ---
# Extend MessagesState to include a summary field.
# This persists across turns via the checkpointer.
class State(MessagesState):
    summary: str


# --- Nodes ---

def call_model(state: State):
    """Call the LLM, injecting any existing summary into the system prompt."""
    summary = state.get("summary", "")

    system_content = "You are a helpful assistant. Be concise."
    if summary:
        system_content += f"\n\nSummary of earlier conversation:\n{summary}"

    response = model.invoke(
        [SystemMessage(content=system_content)] + state["messages"]
    )
    return {"messages": [response]}


def summarize_conversation(state: State):
    """Summarize all messages and delete all but the last 2.

    Uses the existing summary (if any) as context so information
    accumulates across multiple summarization rounds.
    """
    summary = state.get("summary", "")

    if summary:
        summary_prompt = (
            f"This is the summary of the conversation so far: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_prompt = "Create a brief summary of the conversation above:"

    # Ask the model to summarize
    messages = state["messages"] + [HumanMessage(content=summary_prompt)]
    response = model.invoke(messages)

    # Delete all but the last 2 messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

    print(f"\n  [summarize] Compressed {len(state['messages'])} messages into summary, "
          f"keeping last 2\n")

    return {
        "summary": response.content,
        "messages": delete_messages,
    }


# --- Routing ---

def should_summarize(state: State):
    """Only summarize when we have more than 4 messages."""
    if len(state["messages"]) > 4:
        return "summarize"
    return END


# --- Build the graph ---
#
# Flow:
#   START -> model -> (more than 4 messages?) -> summarize -> END
#                  -> (4 or fewer?)           -> END

builder = StateGraph(State)

builder.add_node("model", call_model)
builder.add_node("summarize", summarize_conversation)

builder.add_edge(START, "model")
builder.add_conditional_edges("model", should_summarize)
builder.add_edge("summarize", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# --- Run ---
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "summary-demo"}}

    conversations = [
        "Hi, my name is Alice.",
        "I work as a software engineer at a startup.",
        "My favorite language is Python and I use it daily.",
        "I also enjoy hiking on weekends.",
        "What do you know about me?",
    ]

    for i, msg in enumerate(conversations, 1):
        print(f"=== Turn {i}: {msg} ===")
        result = graph.invoke(
            {"messages": [{"role": "user", "content": msg}]},
            config=config,
        )
        print(result["messages"][-1].content)

        # Show state
        state = graph.get_state(config)
        msg_count = len(state.values["messages"])
        summary = state.values.get("summary", "")
        print(f"  [state] Messages: {msg_count}")
        if summary:
            print(f"  [state] Summary: {summary[:150]}...")
        print()

    # Turn 5 triggers summarization. The model should still know Alice's name
    # because it was captured in the summary — even though those messages
    # were deleted.
    print("=== Final state ===")
    state = graph.get_state(config)
    print(f"Messages in state: {len(state.values['messages'])}")
    print(f"Summary: {state.values.get('summary', '(none)')}")
