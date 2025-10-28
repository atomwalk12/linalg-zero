"""
Linear Algebra Agent for GRPO training.

This module implements an agent that can solve linear algebra problems
using tool calling capabilities following the tau-bench pattern.
"""

import json
from typing import Any

from litellm import acompletion
from litellm.types.utils import ModelResponse

from linalg_zero.shared.system_prompts import get_math_system_prompt
from linalg_zero.shared.utils import get_logger

from .base_types import RESPOND_ACTION_NAME, Action, SolveResult
from .linalg_env import LinAlgEnvironment

logger = get_logger(__name__)


class Agent:
    """Abstract base class for agents - self-contained interface."""

    async def solve(
        self, env: LinAlgEnvironment, task_index: int | None = None, max_num_steps: int = 30
    ) -> SolveResult:
        """Solve a task in the given environment."""
        raise NotImplementedError


class LinAlgAgent(Agent):
    """
    Linear algebra agent that solves mathematical problems using tool calling.

    This agent follows the tau-bench pattern for model integration and tool calling.
    """

    def __init__(
        self,
        tools_info: list[dict[str, Any]],
        model: str,
        provider: str,
        system_prompt: str,
        temperature: float = 0.0,
        art_model: Any = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize the LinAlgAgent following tau-bench patterns.

        Args:
            tools_info: List of available tool information dictionaries
            model: Model name/identifier for inference
            provider: Model provider (e.g., "openai", "anthropic", "local", "art")
            system_prompt: System prompt for the agent
            temperature: Sampling temperature for model generation
            art_model: art.Model instance for art provider integration
        """
        self.tools_info = tools_info
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.art_model = art_model
        self.base_url = base_url
        self.api_key = api_key

        logger.info(f"Initialized LinAlgAgent with model={model}, provider={provider}")

    async def llm_completion(self, messages: list[dict[str, Any]]) -> ModelResponse:
        completion_obj = await acompletion(
            messages=messages,
            model=self.model,
            custom_llm_provider=self.provider,
            base_url=self.base_url,
            api_key=self.api_key,
            tools=self.tools_info,
            temperature=self.temperature,
        )
        assert isinstance(completion_obj, ModelResponse)  # noqa: S101
        return completion_obj

    async def solve(
        self, env: LinAlgEnvironment, task_index: int | None = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0

        # Async environment reset
        env_reset_res = await env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": obs},
        ]

        # Track metrics like tau-bench
        final_prompt_tokens = 0
        avg_completion_tokens = 0
        max_completion_tokens = 0
        forced_stop = True
        _curr_step_number = 0

        for _curr_step_number in range(max_num_steps):
            # Async LLM call
            if self.provider == "art" and self.art_model is not None:
                openai_client = self.art_model.openai_client()
                res = await openai_client.chat.completions.create(
                    model=self.art_model.name,
                    messages=messages,
                    tools=self.tools_info,
                    temperature=self.temperature,
                )
            else:
                res = await self.llm_completion(messages)

            # Track token usage
            final_prompt_tokens = res.usage.prompt_tokens
            avg_completion_tokens += res.usage.completion_tokens
            max_completion_tokens = max(max_completion_tokens, res.usage.completion_tokens)

            next_message = res.choices[0].message.model_dump()

            # Limit tool calls before appending
            if (
                "tool_calls" in next_message
                and next_message["tool_calls"] is not None
                and len(next_message["tool_calls"]) > 0
            ):
                next_message["tool_calls"] = next_message["tool_calls"][:1]

            messages.append(next_message)

            total_cost += res._hidden_params.get("response_cost") or 0.0
            action = self._message_to_action(next_message)

            # Async environment step
            env_response = await env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}

            # Update messages based on action type
            if action.name != RESPOND_ACTION_NAME:
                messages.append({
                    "role": "tool",
                    "tool_call_id": next_message["tool_calls"][0]["id"],
                    "name": next_message["tool_calls"][0]["function"]["name"],
                    "content": env_response.observation,
                })
            else:
                messages.append({
                    "role": "user",
                    "content": env_response.observation,
                })

            if env_response.done:
                forced_stop = False
                break

            # Check for forced stop conditions
            if final_prompt_tokens > 20000 or res.choices[0].finish_reason == "length":
                break

        # Add metrics to info
        info["total_steps"] = _curr_step_number + 1
        info["avg_completion_tokens"] = avg_completion_tokens / info["total_steps"]
        info["max_completion_tokens"] = max_completion_tokens
        info["final_prompt_tokens"] = final_prompt_tokens
        info["forced_stop"] = 1 if forced_stop else 0

        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )

    def set_art_model(self, art_model: Any) -> None:
        """
        Set the art.Model instance for inference.

        Args:
            art_model: art.Model instance
        """
        self.art_model = art_model
        logger.debug("Art model instance set for agent")

        # Validate that the art model has the expected interface
        if not hasattr(art_model, "openai_client"):
            logger.warning("Art model does not have 'openai_client' method - may cause inference issues")
        if not hasattr(art_model, "get_inference_name"):
            logger.warning("Art model does not have 'get_inference_name' method - may cause inference issues")

    def _message_to_action(self, message: dict[str, Any]) -> Action:
        """
        Convert a model message to an Action following tau-bench pattern.

        Args:
            message: Message dictionary from model response

        Returns:
            Action object
        """
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
        system_prompt=get_math_system_prompt(include_examples=False),
        provider=provider,
        temperature=temperature,
        art_model=art_model,
        **kwargs,
    )
