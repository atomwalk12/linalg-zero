"""
Linear Algebra Agent for GRPO training.

This module implements an agent that can solve linear algebra problems
using tool calling capabilities and model inference.
"""

import json
from typing import Any

from linalg_zero.shared.utils import get_logger

from .base_types import RESPOND_ACTION_NAME, Action, SolveResult
from .linalg_env import LinAlgEnvironment

logger = get_logger(__name__)


class Agent:
    """Abstract base class for agents - self-contained interface."""

    def solve(self, env: LinAlgEnvironment, task_index: int | None = None, max_num_steps: int = 30) -> SolveResult:
        """Solve a task in the given environment."""
        raise NotImplementedError


class LinAlgAgent(Agent):
    """
    Linear algebra agent that solves mathematical problems using tool calling.

    This agent extends the self-contained Agent interface and integrates with
    the existing model infrastructure to solve linear algebra problems step-by-step.
    """

    def __init__(
        self,
        tools_info: list[dict[str, Any]],
        model: str,
        provider: str,
        temperature: float = 0.0,
        max_retries: int = 3,
        system_prompt: str | None = None,
    ):
        """
        Initialize the LinAlgAgent.

        Args:
            tools_info: List of available tool information dictionaries
            model: Model name/identifier for inference
            provider: Model provider (e.g., "openai", "anthropic", "local")
            temperature: Sampling temperature for model generation
            max_retries: Maximum number of retries for failed model calls
            system_prompt: Optional system prompt override
        """
        self.tools_info = tools_info
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_retries = max_retries
        self.system_prompt = system_prompt or self._get_default_system_prompt()

        # Initialize model client based on provider
        self._init_model_client()

        logger.info(f"Initialized LinAlgAgent with model={model}, provider={provider}")

    def _init_model_client(self) -> None:
        """Initialize the model client based on the provider."""
        # This will be implemented when integrating with existing model infrastructure
        # For now, we'll use a placeholder that can be extended
        self.model_client = None
        logger.debug(f"Model client initialization for provider '{self.provider}' - placeholder")

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for linear algebra problem solving."""
        return """You are a mathematical assistant specialized in linear algebra.
You have access to various mathematical tools for matrix operations.

When solving problems:
1. Use the available tools to perform calculations
2. Show your reasoning step by step
3. Provide your final answer clearly

Available tools will be provided in the function calling interface.
Use them to solve linear algebra problems accurately."""

    def solve(self, env: LinAlgEnvironment, task_index: int | None = None, max_num_steps: int = 30) -> SolveResult:
        """
        Solve a linear algebra task in the given environment.

        Args:
            env: LinAlgEnvironment instance
            task_index: Index of task to solve, random if None
            max_num_steps: Maximum number of steps to take

        Returns:
            SolveResult with reward, messages, and metadata
        """
        logger.info(f"Starting solve with task_index={task_index}, max_steps={max_num_steps}")

        # Initialize tracking variables
        total_cost = 0.0
        messages: list[dict[str, Any]] = []

        # Reset environment and get initial observation
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0

        # Initialize conversation with system prompt and initial observation
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": obs},
        ]

        logger.debug(f"Initial observation: {obs}")

        # Main solving loop
        for step in range(max_num_steps):
            logger.debug(f"Step {step + 1}/{max_num_steps}")

            try:
                # Generate next action from model
                action, next_message, step_cost = self.generate_next_action(messages)
                total_cost += step_cost

                # Execute action in environment
                env_response = env.step(action)
                reward = env_response.reward
                info = {**info, **env_response.info.model_dump()}

                # Update conversation based on action type
                if action.name != RESPOND_ACTION_NAME:
                    # Tool call - add tool response to conversation
                    messages.extend([
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": self._get_tool_call_id(next_message),
                            "name": action.name,
                            "content": env_response.observation,
                        },
                    ])
                    logger.debug(f"Tool call: {action.name} -> {env_response.observation}")
                else:
                    # Response action - add user feedback
                    messages.extend([
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ])
                    logger.debug(f"Agent response: {action.kwargs.get('content', '')}")

                # Check if episode is done
                if env_response.done:
                    logger.info(f"Episode completed after {step + 1} steps with reward {reward}")
                    break

            except Exception as e:
                logger.exception(f"Error in step {step + 1}")
                # Add error message to conversation and continue
                messages.append({
                    "role": "user",
                    "content": f"An error occurred: {e}. Please try a different approach.",
                })
                continue

        else:
            logger.warning(f"Episode reached maximum steps ({max_num_steps}) without completion")

        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )

    def generate_next_action(self, messages: list[dict[str, Any]]) -> tuple[Action, dict[str, Any], float]:
        """
        Generate the next action based on conversation history.

        Args:
            messages: Conversation history

        Returns:
            Tuple of (Action, message_dict, cost)
        """
        # For now, implement a placeholder that will be replaced with actual model inference
        # This follows the pattern from tau-bench but will be adapted for our model infrastructure

        if self.model_client is None:
            # Placeholder implementation for testing
            return self._generate_placeholder_action(messages)

        # TODO: Implement actual model inference using existing infrastructure
        # This will be integrated with the art.Model or similar infrastructure
        try:
            response = self._call_model(messages)
            next_message = response["message"]
            cost = response.get("cost", 0.0)
            action = self._message_to_action(next_message)
        except Exception:
            logger.exception("Model call failed")
            # Fallback to placeholder
            return self._generate_placeholder_action(messages)
        else:
            return action, next_message, cost

    def _call_model(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Call the model with the given messages.

        This method will be implemented to integrate with the existing model infrastructure.

        TODO: CRITICAL - Implement art.Model integration for rollout function
        This is the key integration point between LinAlgAgent and the GRPO training system.

        Expected implementation:
        1. Convert messages to format expected by art.Model
        2. Call model.generate() or similar method
        3. Parse response and extract tool calls or text responses
        4. Return in expected format: {"message": {...}, "cost": float}

        Integration points:
        - art.Model passed to rollout function needs to be accessible here
        - May need to modify LinAlgAgent constructor to accept art.Model directly
        - Consider using provider="art" to route to art.Model integration
        """
        # TODO: Integrate with existing model infrastructure (art.Model, etc.)
        # For now, raise NotImplementedError to indicate this needs implementation
        raise NotImplementedError("Model integration not yet implemented - see TODO above")

    def _generate_placeholder_action(self, messages: list[dict[str, Any]]) -> tuple[Action, dict[str, Any], float]:
        """
        Generate a placeholder action for testing purposes.

        This method provides basic functionality for testing the agent structure
        before full model integration.
        """
        # Simple heuristic: if we haven't used any tools yet, try a basic tool call
        tool_calls_made = sum(1 for msg in messages if msg.get("role") == "tool")

        if tool_calls_made == 0 and self.tools_info:
            # Make a simple tool call
            tool_info = self.tools_info[0]  # Use first available tool
            tool_name = tool_info["function"]["name"]

            # Create a simple tool call message
            next_message = {
                "role": "assistant",
                "content": f"I'll use the {tool_name} tool to help solve this problem.",
                "tool_calls": [
                    {
                        "id": "call_001",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": "{}",  # Empty arguments for placeholder
                        },
                    }
                ],
            }

            action = Action(name=tool_name, kwargs={})
            return action, next_message, 0.0
        else:
            # Provide a response action
            next_message = {
                "role": "assistant",
                "content": "Based on the available information, I believe this completes the solution.",
            }

            action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": next_message["content"]})
            return action, next_message, 0.0

    def _message_to_action(self, message: dict[str, Any]) -> Action:
        """
        Convert a model message to an Action.

        Args:
            message: Message dictionary from model response

        Returns:
            Action object
        """
        # Check if message contains tool calls
        if (
            "tool_calls" in message
            and message["tool_calls"] is not None
            and len(message["tool_calls"]) > 0
            and message["tool_calls"][0]["function"] is not None
        ):
            tool_call = message["tool_calls"][0]
            try:
                kwargs = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool arguments: {tool_call['function']['arguments']}")
                kwargs = {}

            return Action(
                name=tool_call["function"]["name"],
                kwargs=kwargs,
            )
        else:
            # Regular response action
            content = message.get("content", "")
            return Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})

    def _get_tool_call_id(self, message: dict[str, Any]) -> str:
        """
        Extract tool call ID from message.

        Args:
            message: Message containing tool call

        Returns:
            Tool call ID string
        """
        if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0:
            return message["tool_calls"][0].get("id", "call_001")
        return "call_001"

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the current model configuration.

        Returns:
            Dictionary with model information
        """
        return {
            "model": self.model,
            "provider": self.provider,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "tools_count": len(self.tools_info),
        }

    def update_temperature(self, temperature: float) -> None:
        """
        Update the sampling temperature.

        Args:
            temperature: New temperature value
        """
        self.temperature = temperature
        logger.debug(f"Updated temperature to {temperature}")

    def add_tools(self, new_tools_info: list[dict[str, Any]]) -> None:
        """
        Add new tools to the agent's available tools.

        Args:
            new_tools_info: List of new tool information dictionaries
        """
        self.tools_info.extend(new_tools_info)
        logger.info(f"Added {len(new_tools_info)} new tools. Total tools: {len(self.tools_info)}")

    def get_available_tools(self) -> list[str]:
        """
        Get list of available tool names.

        Returns:
            List of tool names
        """
        return [tool["function"]["name"] for tool in self.tools_info]


def create_linalg_agent(
    env: LinAlgEnvironment,
    model: str = "gpt-4",
    provider: str = "openai",
    temperature: float = 0.0,
    **kwargs: dict[str, Any],
) -> LinAlgAgent:
    """
    Create a LinAlgAgent with tools from the environment.

    Args:
        env: LinAlgEnvironment instance to get tools from
        model: Model name/identifier
        provider: Model provider
        temperature: Sampling temperature
        **kwargs: Additional arguments for LinAlgAgent

    Returns:
        Configured LinAlgAgent instance
    """
    tools_info = env.get_available_tools()

    return LinAlgAgent(tools_info=tools_info, model=model, provider=provider, temperature=temperature, **kwargs)
