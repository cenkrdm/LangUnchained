"""
Tests for Script 02 — Research Agent
Verifies the agent uses fetch_text_from_url and summarizes the content.

NOTE: These tests hit real Ollama + real URLs — integration tests.
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
import urllib.error
import urllib.request

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


def make_agent():
    model = init_chat_model(
        f"ollama:{MODEL}",
        base_url=BASE_URL,
        temperature=0,
    )
    return create_agent(
        model=model,
        tools=[fetch_text_from_url],
        system_prompt=SYSTEM_PROMPT,
    )


def test_agent_calls_fetch_tool():
    """The model should decide to call fetch_text_from_url when given a URL."""
    agent = make_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content":
            "Fetch https://peps.python.org/pep-0020/ "
            "and summarize the main principles in 2-3 sentences."
        }]}
    )
    messages = result["messages"]

    # Message 0: user input
    assert isinstance(messages[0], HumanMessage)

    # Message 1: AI decides to call the fetch tool
    ai_msg = messages[1]
    assert isinstance(ai_msg, AIMessage)
    assert len(ai_msg.tool_calls) > 0
    assert ai_msg.tool_calls[0]["name"] == "fetch_text_from_url"
    assert "pep-0020" in ai_msg.tool_calls[0]["args"]["url"]

    # Message 2: tool result contains actual content from the page
    tool_msg = messages[2]
    assert isinstance(tool_msg, ToolMessage)
    assert "Beautiful is better than ugly" in tool_msg.content

    # Final message: AI summarizes the content (no more tool calls)
    final_msg = messages[-1]
    assert isinstance(final_msg, AIMessage)
    assert len(final_msg.tool_calls) == 0
    assert len(final_msg.content) > 0


def test_tool_handles_bad_url():
    """The tool should return an error message for an invalid URL."""
    result = fetch_text_from_url.invoke({"url": "https://thisurldoesnotexist.invalid/page"})
    assert "Fetch failed" in result


if __name__ == "__main__":
    print("Running test_agent_calls_fetch_tool...")
    test_agent_calls_fetch_tool()
    print("PASSED\n")

    print("Running test_tool_handles_bad_url...")
    test_tool_handles_bad_url()
    print("PASSED\n")

    print("All tests passed!")
