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

from linalg_zero.grpo.verifiers.xml_parser import XMLParser
from linalg_zero.shared.system_prompts import get_sft_system_prompt

from .compute_reward import answer_correct, think_correct, validate_answer


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
            wiki=get_sft_system_prompt(),
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

    def format_reward(self, actions: list) -> float:
        """
        Reward proper formatting with higher weight for final answer.

        Intermediate turns (tool calls):
        - Check for <think> tag in content
        - Check for valid tool_call (not None)
        - Weight: 1.0 per turn

        Final turn (answer):
        - Check for <think> tag in content
        - Check for <answer> tag in content
        - Weight: 2.0 (higher importance)

        Returns:
            Float between 0.0 and 1.0 (weighted average of correct formats)
        """
        if not actions:
            return 0.0

        correct_formats = 0.0
        total_weight = 0.0

        # Check intermediate turns (all except last)
        for action in actions[:-1]:
            has_think = think_correct(completion=action.content)

            correct_formats += 0.5
            if has_think:
                correct_formats += 0.5
            total_weight += 1.0

        # Check final turn (weight = 2.0 for answer importance)
        final_action = actions[-1]
        has_think = think_correct(completion=final_action.content)
        has_answer = answer_correct(completion=final_action.content)

        if has_think and has_answer:
            correct_formats += 2.0
        total_weight += 2.0

        return correct_formats / total_weight if total_weight > 0 else 0.0

    async def calculate_reward(self, format_weight: float = 0.1) -> RewardResult:
        """
        Calculate reward using weighted combination of correctness, format, and efficiency.

        reward = (1-w)*correctness + w*format + efficiency_penalty

        Components:
        - correctness: 1.0 (correct answer) or -1.0 (wrong answer)
        - format: 0.0 to 1.0 (proportional to correct formatting)
        - efficiency_penalty: -0.1 * |num_turns - expected| / expected

        Returns:
            RewardResult with final reward and detailed breakdown in outputs
        """
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

        # Calculate correctness (-1.0 or 1.0)
        correctness_binary = validate_answer(
            ground_truth=self.task.outputs[0],
            completion=answer.content,
        )
        correctness_score = 1.0 if correctness_binary else -1.0

        # Calculate format (0.0 to 1.0)
        format_score = self.format_reward(self.actions)

        # Calculate efficiency penalty
        expected_turns = len(self.task.actions)
        num_turns = len(tool_calls)
        efficiency_penalty = 0.0
        if expected_turns > 0 and num_turns != expected_turns:
            efficiency_penalty = max(-0.5, -0.1 * abs(num_turns - expected_turns) / expected_turns)

        # Weighted combination
        final_reward = (1.0 - format_weight) * correctness_score + format_weight * format_score + efficiency_penalty

        # Return with detailed breakdown
        return RewardResult(
            reward=final_reward,
            info=RewardOutputInfo(
                r_outputs=final_reward,
                outputs={
                    "answer_found": correctness_binary,
                    "correctness_score": correctness_score,
                    "format_score": format_score,
                    "efficiency_penalty": efficiency_penalty,
                    "num_turns": num_turns,
                    "expected_turns": expected_turns,
                },
            ),
            actions=tool_calls,
        )
