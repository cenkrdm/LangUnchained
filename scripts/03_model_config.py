"""
Script 03 — Model Configuration (Quickstart Step 3)
Adds timeout, max_tokens, and streaming to the model setup.
Builds on 02_research_agent.py concepts.
"""

import os
import urllib.error
import urllib.request
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
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


# --- Model ---
# New parameters compared to 02:
#   timeout=300   — max seconds to wait for a response (prevents hanging on slow models)
#   max_tokens=1024 — caps response length (saves tokens, keeps output focused)
#   streaming=True  — tokens arrive incrementally instead of all at once
model = init_chat_model(
    f"ollama:{MODEL}",
    base_url=BASE_URL,
    temperature=0.5,
    timeout=300,
    max_tokens=1024,
    streaming=True,
)

# --- Agent ---
agent = create_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
)

# --- Run ---
if __name__ == "__main__":
    # stream() yields events as the agent processes — you see output as it happens
    # instead of waiting for the full response.
    print("=== Streaming agent output ===\n")
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content":
            "Fetch https://peps.python.org/pep-0020/ "
            "and summarize the main principles in 2-3 sentences."
        }]}
    ):
        # Each chunk is a dict with the node name as key
        for node_name, node_output in chunk.items():
            messages = node_output.get("messages", [])
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"[tool call] {tc['name']}({tc['args']})")
                elif hasattr(msg, "content") and msg.content:
                    print(f"[{node_name}] {msg.content[:200]}")
