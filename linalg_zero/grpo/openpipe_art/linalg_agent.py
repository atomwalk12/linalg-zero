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
        art_model: Any = None,
    ):
        """
        Initialize the LinAlgAgent.

        Args:
            tools_info: List of available tool information dictionaries
            model: Model name/identifier for inference
            provider: Model provider (e.g., "openai", "anthropic", "local", "art")
            temperature: Sampling temperature for model generation
            max_retries: Maximum number of retries for failed model calls
            system_prompt: Optional system prompt override
            art_model: art.Model instance for art provider integration
        """
        self.tools_info = tools_info
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_retries = max_retries
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.art_model = art_model

        # Initialize model client based on provider
        self._init_model_client()

        logger.info(f"Initialized LinAlgAgent with model={model}, provider={provider}")

    def _init_model_client(self) -> None:
        """Initialize the model client based on the provider."""
        if self.provider == "art":
            # art.Model will be injected during rollout - no initialization needed here
            self.model_client = None
            logger.debug("Art model client - will be injected during rollout")
        elif self.provider == "openai":
            self._init_openai_client()
        elif self.provider == "anthropic":
            self._init_anthropic_client()
        elif self.provider == "local":
            self._init_local_client()
        else:
            logger.warning(f"Unknown provider '{self.provider}', using placeholder client")
            self.model_client = None

    def _init_openai_client(self) -> None:
        """Initialize OpenAI client."""
        try:
            import openai

            self.model_client = openai.OpenAI()
            logger.debug("Initialized OpenAI client")
        except ImportError:
            logger.exception("OpenAI package not available. Install with: pip install openai")
            self.model_client = None
        except Exception:
            logger.exception("Failed to initialize OpenAI client")
            self.model_client = None

    def _init_anthropic_client(self) -> None:
        """Initialize Anthropic client."""
        try:
            import anthropic  # type: ignore[reportMissingImports]

            self.model_client = anthropic.Anthropic()
            logger.debug("Initialized Anthropic client")
        except ImportError:
            logger.exception("Anthropic package not available. Install with: pip install anthropic")
            self.model_client = None
        except Exception:
            logger.exception("Failed to initialize Anthropic client")
            self.model_client = None

    def _init_local_client(self) -> None:
        """Initialize local model client (e.g., vLLM, transformers)."""
        # TODO: Implement local model client integration
        # This could integrate with existing unsloth/transformers infrastructure
        logger.debug("Local model client initialization - placeholder")
        self.model_client = None

    def _ensure_openai_client(self) -> None:
        """Ensure OpenAI client is initialized, raise if not."""
        if self.model_client is None:
            raise RuntimeError("OpenAI client not initialized")

    def _ensure_anthropic_client(self) -> None:
        """Ensure Anthropic client is initialized, raise if not."""
        if self.model_client is None:
            raise RuntimeError("Anthropic client not initialized")

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for linear algebra problem solving."""
        # Try to use the existing system prompt from shared module
        try:
            from linalg_zero.shared.system_prompts import get_system_prompt

            return get_system_prompt()
        except ImportError:
            logger.warning("Could not import system prompt from shared module, using fallback")

        # Fallback system prompt
        tools_description = self._format_tools_for_prompt()
        return f"""You are a mathematical assistant specialized in linear algebra.
You have access to various mathematical tools for matrix operations.

When solving problems:
1. Think step by step about the problem
2. Use the available tools to perform calculations
3. Format your response using XML tags:
   - Use <think>...</think> for your reasoning (no calculations inside)
   - Use <tool_call>{{"name": "tool_name", "arguments": {{"param": "value"}}}}</tool_call> for tool calls
   - Use <answer>result</answer> for your final numerical answer
4. Never include both <tool_call> and <answer> in the same response
5. Show your work clearly and provide accurate results

Available tools:
{tools_description}

Use these tools to solve linear algebra problems accurately."""

    def _format_tools_for_prompt(self) -> str:
        """Format available tools for inclusion in system prompt."""
        if not self.tools_info:
            return "No tools available."

        tool_descriptions = []
        for tool in self.tools_info:
            func_info = tool.get("function", {})
            name = func_info.get("name", "unknown")
            description = func_info.get("description", "No description available")
            tool_descriptions.append(f"- {name}: {description}")

        return "\n".join(tool_descriptions)

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
        # Attempt model inference with retry logic
        for attempt in range(self.max_retries):
            try:
                if self.provider == "art" and self.art_model is None:
                    logger.warning("Art model not set, using placeholder")
                    return self._generate_placeholder_action(messages)

                if self.model_client is None and self.provider != "art":
                    logger.warning(f"Model client not initialized for provider '{self.provider}', using placeholder")
                    return self._generate_placeholder_action(messages)

                # Call the appropriate model
                response = self._call_model(messages)
                next_message = response["message"]
                cost = response.get("cost", 0.0)
                action = self._message_to_action(next_message)

                logger.debug(f"Generated action: {action.name}")
                return action, next_message, cost  # noqa: TRY300

            except Exception as e:
                logger.warning(f"Model call attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt == self.max_retries - 1:
                    # Final attempt failed, use placeholder
                    logger.exception("All model call attempts failed, using placeholder action")
                    return self._generate_placeholder_action(messages)

                # Wait before retry (exponential backoff)
                import time

                time.sleep(2**attempt)

        # Should not reach here, but fallback just in case
        return self._generate_placeholder_action(messages)

    def _call_model(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Call the model with the given messages.

        This method integrates with different model providers including art.Model.

        Args:
            messages: Conversation history in OpenAI format

        Returns:
            Dictionary with "message" and "cost" keys
        """
        if self.provider == "art" and self.art_model is not None:
            return self._call_art_model(messages)
        elif self.provider == "openai" and self.model_client is not None:
            return self._call_openai_model(messages)
        elif self.provider == "anthropic" and self.model_client is not None:
            return self._call_anthropic_model(messages)
        elif self.provider == "local" and self.model_client is not None:
            return self._call_local_model(messages)
        else:
            logger.exception(f"No valid model client for provider '{self.provider}'")
            raise RuntimeError(f"Model client not available for provider '{self.provider}'")

    def _call_art_model(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Call art.Model for inference.

        Args:
            messages: Conversation history

        Returns:
            Dictionary with message and cost
        """
        try:
            # TODO: Implement actual art.Model API call
            # This is a placeholder implementation that needs to be completed
            # based on the actual art.Model interface

            # For now, simulate a model response
            logger.debug("Calling art.Model (placeholder implementation)")

            # Extract the last user message for context
            user_message = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break

            # Simple heuristic for demonstration
            if "matrix" in user_message.lower() or "determinant" in user_message.lower():
                # Simulate a tool call response
                response_message = {
                    "role": "assistant",
                    "content": "I'll help you solve this linear algebra problem using the available tools.",
                    "tool_calls": [
                        {
                            "id": "call_001",
                            "type": "function",
                            "function": {
                                "name": "matrix_determinant",
                                "arguments": '{"matrix": "A"}',
                            },
                        }
                    ],
                }
            else:
                # Simulate a regular response
                response_message = {
                    "role": "assistant",
                    "content": "Based on the calculations, I can provide the final answer.",
                }

            return {"message": response_message, "cost": 0.0}  # noqa: TRY300

        except Exception:
            logger.exception("Art model call failed")
            raise

    def _call_openai_model(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Call OpenAI API for inference.

        Args:
            messages: Conversation history

        Returns:
            Dictionary with message and cost
        """
        try:
            self._ensure_openai_client()

            response = self.model_client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                tools=[{"type": "function", "function": tool["function"]} for tool in self.tools_info],
                tool_choice="auto",
            )

            message = response.choices[0].message.model_dump()
            cost = 0.0  # TODO: Calculate actual cost based on usage

            return {"message": message, "cost": cost}  # noqa: TRY300

        except Exception:
            logger.exception("OpenAI model call failed")
            raise

    def _call_anthropic_model(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Call Anthropic API for inference.

        Args:
            messages: Conversation history

        Returns:
            Dictionary with message and cost
        """
        try:
            self._ensure_anthropic_client()

            # Convert OpenAI format to Anthropic format
            system_message = ""
            anthropic_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append(msg)

            response = self.model_client.messages.create(  # type: ignore[attr-defined]
                model=self.model,
                max_tokens=1024,
                temperature=self.temperature,
                system=system_message,
                messages=anthropic_messages,
            )

            # Convert back to OpenAI format
            message = {
                "role": "assistant",
                "content": response.content[0].text if response.content else "",
            }
            cost = 0.0  # TODO: Calculate actual cost

            return {"message": message, "cost": cost}  # noqa: TRY300

        except Exception:
            logger.exception("Anthropic model call failed")
            raise

    def _call_local_model(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Call local model for inference.

        Args:
            messages: Conversation history

        Returns:
            Dictionary with message and cost
        """
        try:
            # TODO: Implement local model integration
            # This could use the existing unsloth/transformers infrastructure
            logger.debug("Local model call - placeholder implementation")

            # For now, return a placeholder response
            message = {
                "role": "assistant",
                "content": "Local model response placeholder",
            }

            return {"message": message, "cost": 0.0}  # noqa: TRY300

        except Exception:
            logger.exception("Local model call failed")
            raise

    def set_art_model(self, art_model: Any) -> None:
        """
        Set the art.Model instance for inference.

        This method allows injecting the art.Model after agent initialization,
        which is useful for the rollout function integration.

        Args:
            art_model: art.Model instance
        """
        self.art_model = art_model
        logger.debug("Art model instance set for agent")

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

    def validate_configuration(self) -> bool:
        """
        Validate the agent configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        if self.provider == "art":
            if self.art_model is None:
                logger.error("Art provider requires art_model to be set")
                return False
        elif self.provider in ["openai", "anthropic", "local"]:
            if self.model_client is None:
                logger.error(f"Provider '{self.provider}' requires model_client to be initialized")
                return False
        else:
            logger.warning(f"Unknown provider '{self.provider}'")
            return False

        if not self.tools_info:
            logger.warning("No tools available for agent")

        if self.temperature < 0.0 or self.temperature > 2.0:
            logger.warning(f"Temperature {self.temperature} may be outside normal range [0.0, 2.0]")

        return True

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
            "has_art_model": self.art_model is not None,
            "has_model_client": self.model_client is not None,
            "configuration_valid": self.validate_configuration(),
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

        # Update system prompt to include new tools
        self.system_prompt = self._get_default_system_prompt()

    def get_available_tools(self) -> list[str]:
        """
        Get list of available tool names.

        Returns:
            List of tool names
        """
        return [tool["function"]["name"] for tool in self.tools_info]

    def update_model_config(self, model: str | None = None, temperature: float | None = None) -> None:
        """
        Update model configuration parameters.

        Args:
            model: New model name/identifier
            temperature: New temperature value
        """
        if model is not None:
            self.model = model
            logger.debug(f"Updated model to {model}")

        if temperature is not None:
            self.temperature = temperature
            logger.debug(f"Updated temperature to {temperature}")

    def get_provider_status(self) -> dict[str, Any]:
        """
        Get detailed status information about the model provider.

        Returns:
            Dictionary with provider status information
        """
        status = {
            "provider": self.provider,
            "model": self.model,
            "client_initialized": self.model_client is not None,
            "art_model_available": self.art_model is not None,
        }

        if self.provider == "art":
            status["ready"] = self.art_model is not None
        else:
            status["ready"] = self.model_client is not None

        return status


def create_linalg_agent(
    env: LinAlgEnvironment,
    model: str = "gpt-4",
    provider: str = "openai",
    temperature: float = 0.0,
    art_model: Any = None,
    **kwargs: dict[str, Any],
) -> LinAlgAgent:
    """
    Create a LinAlgAgent with tools from the environment.

    Args:
        env: LinAlgEnvironment instance to get tools from
        model: Model name/identifier
        provider: Model provider
        temperature: Sampling temperature
        art_model: art.Model instance for art provider
        **kwargs: Additional arguments for LinAlgAgent

    Returns:
        Configured LinAlgAgent instance
    """
    tools_info = env.get_available_tools()

    return LinAlgAgent(
        tools_info=tools_info,
        model=model,
        provider=provider,
        temperature=temperature,
        art_model=art_model,
        **kwargs,
    )
