"""
Tests for Script 04 — Memory
Verifies that the agent remembers across turns within a thread
and forgets across different threads.

NOTE: Integration tests hitting real Ollama.
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

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
        temperature=0,
    )
    checkpointer = InMemorySaver()
    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="You are a helpful assistant. Be concise.",
        checkpointer=checkpointer,
    )
    return agent, checkpointer


def test_memory_persists_within_thread():
    """Agent should remember previous turns in the same thread."""
    agent, checkpointer = make_agent()
    config = {"configurable": {"thread_id": "test-thread-1"}}

    # Turn 1: ask about weather
    agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]},
        config=config,
    )

    # Turn 2: follow-up referencing turn 1
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What city did I just ask about?"}]},
        config=config,
    )
    final = result["messages"][-1]
    assert isinstance(final, AIMessage)
    assert "Tokyo" in final.content

    # Verify message history grew across turns
    state = agent.get_state(config)
    messages = state.values["messages"]
    human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    assert len(human_msgs) >= 2


def test_memory_isolated_between_threads():
    """Different thread_ids should have independent memory."""
    agent, checkpointer = make_agent()

    # Thread A: ask about Paris
    agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in Paris?"}]},
        config={"configurable": {"thread_id": "thread-A"}},
    )

    # Thread B: ask what was discussed — should NOT know about Paris
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What city did I ask about?"}]},
        config={"configurable": {"thread_id": "thread-B"}},
    )
    final = result["messages"][-1]
    assert isinstance(final, AIMessage)
    # Thread B has no history, so it should not mention Paris
    assert "Paris" not in final.content


if __name__ == "__main__":
    print("Running test_memory_persists_within_thread...")
    test_memory_persists_within_thread()
    print("PASSED\n")

    print("Running test_memory_isolated_between_threads...")
    test_memory_isolated_between_threads()
    print("PASSED\n")

    print("All tests passed!")
