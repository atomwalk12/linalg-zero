# Copyright Sierra

from functools import cache

from tau_bench.envs.base import Env
from tau_bench.envs.linear_algebra.tasks import load_tasks
from tau_bench.envs.linear_algebra.tools import ALL_TOOLS
from tau_bench.envs.user import UserStrategy
from tau_bench.types import Task

from linalg_zero.shared.system_prompts import get_math_system_prompt


@cache
def _load_tasks_cached(hf_path: str, split: str, dev: bool = False) -> tuple[Task, ...]:
    """Cache tasks loading to avoid repeated HuggingFace calls."""
    return tuple(load_tasks(hf_path, split=split, dev=dev))


class LinearAlgebraEnv(Env):
    """Linear algebra environment for mathematical reasoning tasks."""

    def __init__(
        self,
        hf_path: str = "atomwalk12/linalgzero-grpo",
        user_strategy: str | UserStrategy = UserStrategy.LLM,
        user_model: str = "gpt-4o",
        user_provider: str | None = None,
        task_split: str = "test",
        task_index: int | None = None,
    ):
        # Load tasks based on split
        tasks = self._get_tasks(hf_path, task_split)

        super().__init__(
            data_load_func=lambda: {},  # Linear algebra doesn't need external data
            tools=ALL_TOOLS,
            tasks=list(tasks),
            wiki=get_math_system_prompt(include_examples=False),  # Linear algebra uses system prompts instead
            rules=[],  # Rules embedded in system prompts
            user_strategy=user_strategy,
            user_model=user_model,
            user_provider=user_provider,
            task_index=task_index,
        )
        self.terminate_tools = []

    def _get_tasks(self, hf_path: str, task_split: str) -> tuple[Task, ...]:
        """Get tasks for the specified split."""
        split_mapping = {
            "test": ("test", False),
            "train": ("train", False),
            "valid": ("validation", False),
            "dev": ("train", True),  # Development mode uses limited train set
        }

        if task_split not in split_mapping:
            raise ValueError(f"Unknown task split: {task_split}. Valid splits: {list(split_mapping.keys())}")

        split, dev = split_mapping[task_split]
        return _load_tasks_cached(hf_path, split, dev)
