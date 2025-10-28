"""
Base environment and tool classes for the linear algebra system.
Self-contained without external dependencies.
"""

import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from hashlib import sha256
from typing import Any

from .base_types import (
    Action,
    EnvResetResponse,
    EnvResponse,
    RewardResult,
    Task,
)

ToHashable = str | int | float | dict[str, "ToHashable"] | list["ToHashable"] | set["ToHashable"]
Hashable = str | int | float | tuple["Hashable", ...] | tuple[tuple[str, "Hashable"], ...]


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
    def invoke(data: dict[str, Any], **kwargs: Any) -> str:
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

    def __init__(self) -> None:
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


class Env(ABC):
    """Base environment class."""

    def __init__(
        self,
        tools: Sequence[type[Tool]],
        tasks: Sequence[Task],
        wiki: str = "",
        rules: list[str] | None = None,
        user_strategy: UserStrategy | None = None,
        task_index: int | None = None,
    ) -> None:
        super().__init__()
        self.tools_map: dict[str, type[Tool]] = {tool.get_info()["function"]["name"]: tool for tool in tools}
        self.tools_info = [tool.get_info() for tool in tools]
        self.terminate_tools: list[str] = []
        self.tasks = list(tasks)  # Convert to list for internal use
        if task_index is not None:
            self.task_index = task_index
        else:
            self.task_index = random.randint(0, len(tasks) - 1) if tasks else 0
        self.task = tasks[self.task_index] if tasks else None
        self.wiki = wiki
        self.rules = rules or []
        self.user = user_strategy or SimpleUserStrategy()
        self.actions: list[Action] = []

    @abstractmethod
    async def reset(self, task_index: int | None = None) -> EnvResetResponse:
        """Reset environment with new task."""
        pass

    @abstractmethod
    async def step(self, action: Action) -> EnvResponse:
        """Process action and return response."""
        pass

    def get_data_hash(self) -> str:
        """Get hash of current data state."""
        # Return empty hash since data is passed directly to tools
        return consistent_hash(to_hashable({}))

    @abstractmethod
    def calculate_reward(self) -> RewardResult:
        """Calculate reward for current episode."""
        pass
