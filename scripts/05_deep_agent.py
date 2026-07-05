"""
Script 05 — Deep Agent (Quickstart Step 5)
Uses create_deep_agent which comes with built-in planning, file system tools,
and subagent capabilities on top of what create_agent provides.
Builds on 04_memory.py concepts.
"""

import os
import urllib.error
import urllib.request
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from deepagents import create_deep_agent
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

checkpointer = InMemorySaver()

# --- Both agents side by side ---
# Regular agent: only has the tools you give it
agent = create_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

# Deep agent: same tools + built-in planning, filesystem, grep, subagents
deep_agent = create_deep_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

# --- Run ---
if __name__ == "__main__":
    # A task that requires counting — regular agent can't do this reliably,
    # deep agent can because it has grep/filesystem tools built in.
    content = (
        "Fetch https://peps.python.org/pep-0020/ and answer:\n"
        "1) How many lines contain the word 'better'?\n"
        "2) What is the first line that contains 'Simple'?\n"
        "3) A one-sentence summary of the document."
    )

    print("=== Regular Agent ===\n")
    result1 = agent.invoke(
        {"messages": [{"role": "user", "content": content}]},
        config={"configurable": {"thread_id": "regular-1"}},
    )
    print(result1["messages"][-1].content)

    print("\n\n=== Deep Agent ===\n")
    result2 = deep_agent.invoke(
        {"messages": [{"role": "user", "content": content}]},
        config={"configurable": {"thread_id": "deep-1"}},
    )
    print(result2["messages"][-1].content)
