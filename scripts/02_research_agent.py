"""
Script 02 — Research Agent (Quickstart Step 2)
Adds a detailed system prompt and a real tool (fetch_text_from_url).
Builds on 01_hello_chain.py concepts.
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

# --- System Prompt ---
# More detailed than 01. Describes the agent's role and available tools
# so the model knows what it can do and how to behave.
SYSTEM_PROMPT = """You are a literary data assistant.

## Capabilities

- `fetch_text_from_url`: loads document text from a URL into the conversation.
Do not guess line counts or positions—ground them in tool results from the saved file."""


# --- Tool ---
# A real tool that fetches content from the web.
# The docstring, name, and argument names all become part of the model's prompt.
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
model = init_chat_model(
    f"ollama:{MODEL}",
    base_url=BASE_URL,
    temperature=0.5,
)

# --- Agent ---
agent = create_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
)

# --- Run ---
if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content":
            "Fetch https://peps.python.org/pep-0020/ "
            "and summarize the main principles in 2-3 sentences."
        }]}
    )
    print(result["messages"][-1].content)
