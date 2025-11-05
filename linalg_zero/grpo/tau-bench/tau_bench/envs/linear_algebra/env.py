# Copyright Sierra

from functools import cache

from tau_bench.envs.base import Env
from tau_bench.envs.linear_algebra.tasks import load_tasks
from tau_bench.envs.linear_algebra.tools import ALL_TOOLS
from tau_bench.envs.user import UserStrategy
from tau_bench.types import (
    RESPOND_ACTION_NAME,
    RewardOutputInfo,
    RewardResult,
    Task,
)

from linalg_zero.grpo.compute_score import calculate_reward
from linalg_zero.grpo.reward_funcs import (
    answer_correct,
    think_correct,
    validate_answer,
)
from linalg_zero.grpo.verifiers.xml_parser import XMLParser
from linalg_zero.shared.system_prompts import get_math_system_prompt


@cache
def _load_tasks_cached(hf_path: str, split: str) -> tuple[Task, ...]:
    """Cache tasks loading to avoid repeated HuggingFace calls."""
    return tuple(load_tasks(hf_path, split=split))


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
            wiki=get_math_system_prompt(include_examples=False),
            rules=[],
            user_strategy=user_strategy,
            user_model=user_model,
            user_provider=user_provider,
            task_index=task_index,
            parser=XMLParser(),
        )
        self.terminate_tools = []

    def _get_tasks(self, hf_path: str, task_split: str) -> tuple[Task, ...]:
        """Get tasks for the specified split."""
        split_mapping = {
            "test": "test",
            "train": "train",
            "val": "validation",
        }

        if task_split not in split_mapping:
            raise ValueError(f"Unknown task split: {task_split}. Valid splits: {list(split_mapping.keys())}")

        split = split_mapping[task_split]
        return _load_tasks_cached(hf_path, split)

    async def calculate_reward(self) -> RewardResult:
        assert self.parser is not None, "Parser cannot be None"

        # Extract the produced tool calls and answer.
        tool_calls = self.actions[:-1]
        answer = self.actions[-1]

        if len(tool_calls) == 0:
            return RewardResult(
                reward=0.0,
                info=RewardOutputInfo(
                    r_outputs=0.0,
                    outputs={"structural_error": "no_tool_calls", "answer_found": False},
                ),
                actions=tool_calls,
            )

        if answer.name != RESPOND_ACTION_NAME:
            return RewardResult(
                reward=0.0,
                info=RewardOutputInfo(
                    r_outputs=0.0,
                    outputs={"structural_error": "no_respond_action", "answer_found": False},
                ),
                actions=tool_calls,
            )

        # Calculate answer reward (1.0 for correctness + 0.2 for format).
        answer_rewards = [(validate_answer, 1.0), (think_correct, 0.1), (answer_correct, 0.2)]
        answer_reward, meta = calculate_reward(
            self.parser,
            ground_truth=self.task.outputs[0],
            completion=answer.content,
            reward_funcs_with_weights=answer_rewards,
        )
        answer_found = meta["validate_answer"]

        # Now, calculate local penalties for intermediate tool calls.
        tool_rewards = [
            (think_correct, 0.1),
            # Tool call reward is implicit. If we reach this phase, the outcome
            # may result in a non-zero reward, otherwise if tool calls are
            # malformed, reward is implicitly 0.
            # (tool_call_correct, 0.2)
        ]
        penalty = 0.0
        for action in tool_calls:
            _, metadata = calculate_reward(
                self.parser, completion=action.content, reward_funcs_with_weights=tool_rewards
            )
            if not metadata["think_correct"]:
                penalty += 0.1

        # By substracting we ensures the task is solved in the least
        # amount of tool calls possible.
        reward = max(0, answer_reward - penalty)

        # NOTE: it is possible to extract the reward configuration from info.
        return RewardResult(
            reward=reward,
            info=RewardOutputInfo(r_outputs=answer_reward, outputs={"answer_found": answer_found}),
            actions=tool_calls,
        )
