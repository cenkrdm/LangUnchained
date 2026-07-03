"""
Script 01 — Hello Agent (Quickstart)
Follows the LangChain quickstart: init_chat_model + create_agent + a simple tool.
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool

from langchain_core.globals import set_debug
set_debug(True)

load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Tool ---
# The docstring becomes part of the model's prompt — keep it clear and specific.
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# --- Model ---
# init_chat_model uses a "provider:model" string, so swapping providers is one edit.
# For Ollama, base_url points to your local server.
model = init_chat_model(
    f"ollama:{MODEL}",
    base_url=BASE_URL,
    temperature=0.5,
)

# --- Agent ---
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant. Be concise.",
)

# --- Run ---
if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in Istanbul?"}]}
    )
    print(result["messages"][-1].content_blocks)
