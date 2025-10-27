"""
Tests for LinAlgAgent implementation.
"""

from typing import Any

import pytest

from linalg_zero.grpo.openpipe_art.base_types import RESPOND_ACTION_NAME
from linalg_zero.grpo.openpipe_art.data_types import RunConfig
from linalg_zero.grpo.openpipe_art.linalg_agent import LinAlgAgent, create_linalg_agent
from linalg_zero.grpo.openpipe_art.linalg_env import LinAlgEnvironment, create_sample_tasks


class TestLinAlgAgent:
    """Test cases for LinAlgAgent class."""

    def test_agent_initialization(self) -> None:
        """Test basic agent initialization."""
        tools_info = [{"function": {"name": "test_tool", "description": "A test tool", "parameters": {}}}]

        agent = LinAlgAgent(tools_info=tools_info, model="test-model", provider="test-provider", temperature=0.5)

        assert agent.model == "test-model"
        assert agent.provider == "test-provider"
        assert agent.temperature == 0.5
        assert len(agent.tools_info) == 1
        assert agent.get_available_tools() == ["test_tool"]

    def test_get_model_info(self) -> None:
        """Test model information retrieval."""
        tools_info: list[dict[str, Any]] = []
        agent = LinAlgAgent(tools_info=tools_info, model="gpt-4", provider="openai")

        info = agent.get_model_info()
        assert info["model"] == "gpt-4"
        assert info["provider"] == "openai"
        assert info["temperature"] == 0.0
        assert info["tools_count"] == 0

    def test_update_temperature(self) -> None:
        """Test temperature update functionality."""
        agent = LinAlgAgent(tools_info=[], model="test-model", provider="test-provider")

        assert agent.temperature == 0.0
        agent.update_temperature(0.8)
        assert agent.temperature == 0.8

    def test_add_tools(self) -> None:
        """Test adding new tools to agent."""
        agent = LinAlgAgent(tools_info=[], model="test-model", provider="test-provider")

        new_tools = [{"function": {"name": "new_tool", "description": "A new tool", "parameters": {}}}]

        assert len(agent.tools_info) == 0
        agent.add_tools(new_tools)
        assert len(agent.tools_info) == 1
        assert agent.get_available_tools() == ["new_tool"]

    def test_message_to_action_tool_call(self) -> None:
        """Test converting tool call message to action."""
        agent = LinAlgAgent(tools_info=[], model="test-model", provider="test-provider")

        message = {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_123", "function": {"name": "determinant", "arguments": '{"matrix": [[1, 2], [3, 4]]}'}}
            ],
        }

        action = agent._message_to_action(message)
        assert action.name == "determinant"
        assert action.kwargs == {"matrix": [[1, 2], [3, 4]]}

    def test_message_to_action_response(self) -> None:
        """Test converting response message to action."""
        agent = LinAlgAgent(tools_info=[], model="test-model", provider="test-provider")

        message = {"role": "assistant", "content": "The answer is 42"}

        action = agent._message_to_action(message)
        assert action.name == RESPOND_ACTION_NAME
        assert action.kwargs == {"content": "The answer is 42"}

    def test_get_tool_call_id(self) -> None:
        """Test extracting tool call ID from message."""
        agent = LinAlgAgent(tools_info=[], model="test-model", provider="test-provider")

        message_with_id = {"tool_calls": [{"id": "call_xyz", "function": {"name": "test"}}]}

        message_without_id = {"tool_calls": [{"function": {"name": "test"}}]}

        empty_message = {}

        assert agent._get_tool_call_id(message_with_id) == "call_xyz"
        assert agent._get_tool_call_id(message_without_id) == "call_001"
        assert agent._get_tool_call_id(empty_message) == "call_001"

    def test_generate_placeholder_action_tool_call(self):
        """Test placeholder action generation for tool calls."""
        tools_info = [{"function": {"name": "determinant", "description": "Calculate determinant", "parameters": {}}}]

        agent = LinAlgAgent(tools_info=tools_info, model="test-model", provider="test-provider")

        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Calculate determinant"},
        ]

        action, message, cost = agent._generate_placeholder_action(messages)

        assert action.name == "determinant"
        assert action.kwargs == {}
        assert cost == 0.0
        assert message["role"] == "assistant"
        assert "tool_calls" in message

    def test_generate_placeholder_action_response(self):
        """Test placeholder action generation for responses."""
        agent = LinAlgAgent(tools_info=[], model="test-model", provider="test-provider")

        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Question"},
            {"role": "tool", "content": "Tool result"},
        ]

        action, message, cost = agent._generate_placeholder_action(messages)

        assert action.name == RESPOND_ACTION_NAME
        assert "content" in action.kwargs
        assert cost == 0.0
        assert message["role"] == "assistant"


class TestCreateLinAlgAgent:
    """Test cases for create_linalg_agent function."""

    def test_create_agent_from_environment(self):
        """Test creating agent from environment."""
        # Create a mock environment with tools
        config = RunConfig(model_provider="test", user_model_provider="test", model="test-model")

        tasks = create_sample_tasks()

        # Import here to avoid circular imports during testing
        from linalg_zero.grpo.openpipe_art.linalg_tools import get_linalg_tools

        tools = get_linalg_tools()

        env = LinAlgEnvironment(tools=tools, tasks=tasks, config=config)

        agent = create_linalg_agent(env=env, model="gpt-4", provider="openai", temperature=0.3)

        assert isinstance(agent, LinAlgAgent)
        assert agent.model == "gpt-4"
        assert agent.provider == "openai"
        assert agent.temperature == 0.3
        assert len(agent.tools_info) > 0


if __name__ == "__main__":
    pytest.main([__file__])
