"""
Base environment and tool classes for the linear algebra system.
Self-contained without external dependencies.
"""

import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from hashlib import sha256
from typing import Any

from .base_types import (
    RESPOND_ACTION_NAME,
    Action,
    EnvInfo,
    EnvResetResponse,
    EnvResponse,
    RewardResult,
    Task,
)

ToHashable = str | int | float | dict[str, "ToHashable"] | list["ToHashable"] | set["ToHashable"]
Hashable = str | int | float | tuple["Hashable"] | tuple[tuple[str, "Hashable"]]


def to_hashable(item: ToHashable) -> Hashable:
    """Convert item to hashable format."""
    if isinstance(item, dict):
        return tuple((key, to_hashable(value)) for key, value in sorted(item.items()))
    elif isinstance(item, list):
        return tuple(to_hashable(element) for element in item)
    elif isinstance(item, set):
        return tuple(sorted(to_hashable(element) for element in item))
    else:
        return item


def consistent_hash(value: Hashable) -> str:
    """Create consistent hash of a value."""
    return sha256(str(value).encode("utf-8")).hexdigest()


class Tool(ABC):
    """Abstract base class for tools."""

    @staticmethod
    @abstractmethod
    def invoke(data: dict[str, Any], **kwargs) -> str:
        """Invoke the tool with given data and arguments."""
        pass

    @staticmethod
    @abstractmethod
    def get_info() -> dict[str, Any]:
        """Get tool information including function schema."""
        pass


class UserStrategy(ABC):
    """Abstract base class for user simulation strategies."""

    @abstractmethod
    def reset(self, instruction: str) -> str:
        """Reset user with new instruction."""
        pass

    @abstractmethod
    def step(self, content: str) -> str:
        """Process agent response and return user response."""
        pass

    @abstractmethod
    def get_total_cost(self) -> float:
        """Get total cost of user interactions."""
        pass


class SimpleUserStrategy(UserStrategy):
    """Simple user strategy that provides basic responses."""

    def __init__(self):
        self.total_cost = 0.0
        self.current_instruction = ""

    def reset(self, instruction: str) -> str:
        """Reset with new instruction."""
        self.current_instruction = instruction
        return f"Please solve: {instruction}"

    def step(self, content: str) -> str:
        """Process agent response."""
        # Simple response - in real implementation this would be more sophisticated
        if any(keyword in content.lower() for keyword in ["answer", "result", "solution"]):
            return "Thank you for the solution. ###STOP###"
        else:
            return "Please continue with your solution."

    def get_total_cost(self) -> float:
        """Get total cost."""
        return self.total_cost


class Env:
    """Base environment class."""

    def __init__(
        self,
        data_load_func: Callable[[], dict[str, Any]],
        tools: list[type[Tool]],
        tasks: list[Task],
        wiki: str = "",
        rules: list[str] | None = None,
        user_strategy: UserStrategy | None = None,
        task_index: int | None = None,
    ) -> None:
        super().__init__()
        self.data_load_func = data_load_func
        self.data = data_load_func()
        self.tools_map: dict[str, type[Tool]] = {tool.get_info()["function"]["name"]: tool for tool in tools}
        self.tools_info = [tool.get_info() for tool in tools]
        self.terminate_tools = []
        self.tasks = tasks
        if task_index is not None:
            self.task_index = task_index
        else:
            self.task_index = random.randint(0, len(tasks) - 1) if tasks else 0
        self.task = tasks[self.task_index] if tasks else None
        self.wiki = wiki
        self.rules = rules or []
        self.user = user_strategy or SimpleUserStrategy()
        self.actions: list[Action] = []

    def reset(self, task_index: int | None = None) -> EnvResetResponse:
        """Reset environment with new task."""
        if task_index is None:
            task_index = random.randint(0, len(self.tasks) - 1) if self.tasks else 0
        self.task_index = task_index
        self.data = self.data_load_func()
        self.task = self.tasks[task_index] if self.tasks else None
        self.actions = []

        if self.task:
            initial_observation = self.user.reset(instruction=self.task.instruction)
            return EnvResetResponse(observation=initial_observation, info=EnvInfo(task=self.task, source="user"))
        else:
            return EnvResetResponse(
                observation="No tasks available",
                info=EnvInfo(task=Task(user_id="", actions=[], instruction="", outputs=[]), source="system"),
            )

    def step(self, action: Action) -> EnvResponse:
        """Process action and return response."""
        self.actions.append(action)

        info = EnvInfo(task=self.task)
        reward = 0
        done = False

        if action.name == RESPOND_ACTION_NAME:
            observation = self.user.step(action.kwargs["content"])
            info.source = "user"
            done = "###STOP###" in observation
        elif action.name in self.tools_map:
            try:
                observation = self.tools_map[action.name].invoke(data=self.data, **action.kwargs)
            except Exception as e:
                observation = f"Error: {e}"
            info.source = action.name
            if action.name in self.terminate_tools:
                done = True
        else:
            observation = f"Unknown action {action.name}"
            info.source = action.name

        if done:
            reward_res = self.calculate_reward()
            reward = reward_res.reward
            info.reward_info = reward_res
            info.user_cost = self.user.get_total_cost()

        return EnvResponse(observation=observation, reward=reward, done=done, info=info)

    def get_data_hash(self) -> str:
        """Get hash of current data state."""
        return consistent_hash(to_hashable(self.data))

    def calculate_reward(self) -> RewardResult:
        """Calculate reward for current episode."""
        # Basic reward calculation - can be overridden by subclasses
        reward = 1.0
        actions = [action for action in self.task.actions if action.name != RESPOND_ACTION_NAME] if self.task else []

        from .base_types import RewardActionInfo

        info = RewardActionInfo(r_actions=1.0, gt_data_hash=self.get_data_hash())

        return RewardResult(reward=reward, info=info, actions=actions)
