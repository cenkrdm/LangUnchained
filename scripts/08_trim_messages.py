"""
Script 08 — Trim Messages (Memory Management)
Trims older messages to keep the conversation within token limits.

Key concepts:
  - trim_messages(): removes messages from the start or end of the history
  - strategy="last": keep the most recent messages, drop older ones
  - token_counter: function to estimate token count
  - max_tokens: the budget — messages are trimmed to fit within this
  - start_on="human": ensure trimmed history starts with a human message
    (some LLMs require this)

Builds on 06_stategraph.py — uses StateGraph directly.
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
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
    """Call the LLM with trimmed messages to stay within token limits."""

    # Trim messages before sending to the model.
    # This does NOT delete them from state — it only filters what the model sees.
    # The full history is still in the checkpointer.
    trimmed = trim_messages(
        state["messages"],
        strategy="last",                          # keep the most recent messages
        token_counter=count_tokens_approximately,  # fast approximate token count
        max_tokens=256,                            # small budget to demonstrate trimming
        start_on="human",                          # trimmed result starts with a human message
        end_on=("human", "tool"),                  # trimmed result ends with human or tool message
    )

    print(f"\n  [trim] {len(state['messages'])} messages in state "
          f"-> {len(trimmed)} after trimming\n")

    response = model.invoke(
        [SystemMessage(content="You are a helpful assistant. Be concise.")] + trimmed
    )
    return {"messages": [response]}


# --- Build the graph ---
builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# --- Run ---
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "trim-demo"}}

    # Send several messages to build up history
    conversations = [
        "Hi, my name is Alice.",
        "I work as a software engineer.",
        "My favorite language is Python.",
        "I also like TypeScript for frontend work.",
        "What do you know about me?",
    ]

    for i, msg in enumerate(conversations, 1):
        print(f"=== Turn {i}: {msg} ===")
        result = graph.invoke(
            {"messages": [{"role": "user", "content": msg}]},
            config=config,
        )
        print(result["messages"][-1].content)
        print()

    # Check: the full history is still in the checkpointer
    state = graph.get_state(config)
    print(f"=== Total messages in state: {len(state.values['messages'])} ===")
    print("The model only saw trimmed messages, but state has everything.")
