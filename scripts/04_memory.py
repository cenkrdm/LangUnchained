"""
Script 04 — Memory (Quickstart Step 4)
Adds InMemorySaver checkpointer so the agent remembers previous messages.
Builds on 03_model_config.py concepts.
"""

import os
import urllib.error
import urllib.request
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.globals import set_debug

set_debug(True)
load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

SYSTEM_PROMPT = """You are a literary data assistant.

## Capabilities

- `fetch_text_from_url`: loads document text from a URL into the conversation.
Do not guess line counts or positions—ground them in tool results from the saved file."""


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


model = init_chat_model(
    f"ollama:{MODEL}",
    base_url=BASE_URL,
    temperature=0.5,
    timeout=300,
    max_tokens=1024,
    streaming=True,
)

# --- Memory ---
# InMemorySaver stores conversation state in RAM.
# Each thread_id gets its own conversation history.
# In production, you'd use a persistent checkpointer (e.g., database-backed).
checkpointer = InMemorySaver()

# --- Agent ---
# Pass checkpointer to create_agent to enable memory.
agent = create_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

# --- Run ---
if __name__ == "__main__":
    # thread_id groups messages into a conversation.
    # Same thread_id = agent remembers previous exchanges.
    config = {"configurable": {"thread_id": "session-1"}}

    # Turn 1: fetch and summarize
    print("=== Turn 1 ===")
    result1 = agent.invoke(
        {"messages": [{"role": "user", "content":
            "Fetch https://peps.python.org/pep-0020/ "
            "and summarize the main principles in 2-3 sentences."
        }]},
        config=config,
    )
    print(result1["messages"][-1].content)

    # Turn 2: follow-up — the agent should remember what it fetched
    print("\n=== Turn 2 ===")
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content":
            "Which principle do you think is most relevant to writing clean code?"
        }]},
        config=config,
    )
    print(result2["messages"][-1].content)

    # Turn 3: different thread — agent has NO memory of previous turns
    print("\n=== Turn 3 (new thread) ===")
    result3 = agent.invoke(
        {"messages": [{"role": "user", "content":
            "What did I ask you to fetch?"
        }]},
        config={"configurable": {"thread_id": "session-2"}},
    )
    print(result3["messages"][-1].content)
