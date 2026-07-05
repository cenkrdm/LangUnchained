"""
Tests for Script 01 — Hello Agent
Verifies the agent loop: model decides to call the tool, tool runs, model responds.

NOTE: These tests hit a real Ollama instance — they are integration tests, not unit tests.
Set temperature=0 for more deterministic results.
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def make_agent():
    model = init_chat_model(
        f"ollama:{MODEL}",
        base_url=BASE_URL,
        temperature=0,  # more deterministic for tests
    )
    return create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="You are a helpful assistant. Be concise.",
    )


def test_agent_calls_weather_tool():
    """The model should decide to call get_weather when asked about weather."""
    agent = make_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in Istanbul?"}]}
    )
    messages = result["messages"]

    # Message 0: user input
    assert isinstance(messages[0], HumanMessage)

    # Message 1: AI decides to call a tool (content is empty, tool_calls is populated)
    ai_msg = messages[1]
    assert isinstance(ai_msg, AIMessage)
    assert len(ai_msg.tool_calls) > 0
    assert ai_msg.tool_calls[0]["name"] == "get_weather"
    assert ai_msg.tool_calls[0]["args"]["city"] == "Istanbul"

    # Message 2: tool result
    tool_msg = messages[2]
    assert isinstance(tool_msg, ToolMessage)
    assert "Istanbul" in tool_msg.content

    # Message 3: final AI response incorporating the tool result
    final_msg = messages[-1]
    assert isinstance(final_msg, AIMessage)
    assert len(final_msg.tool_calls) == 0  # no more tool calls
    assert "Istanbul" in final_msg.content


def test_agent_uses_correct_city():
    """The model should extract the city name from the question."""
    agent = make_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]}
    )
    messages = result["messages"]

    tool_call = messages[1].tool_calls[0]
    assert tool_call["args"]["city"] == "Tokyo"

    tool_result = messages[2]
    assert "Tokyo" in tool_result.content


if __name__ == "__main__":
    print("Running test_agent_calls_weather_tool...")
    test_agent_calls_weather_tool()
    print("PASSED\n")

    print("Running test_agent_uses_correct_city...")
    test_agent_uses_correct_city()
    print("PASSED\n")

    print("All tests passed!")
