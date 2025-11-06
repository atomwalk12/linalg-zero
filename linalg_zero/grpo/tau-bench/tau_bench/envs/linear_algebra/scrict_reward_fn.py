import ast

from tau_bench.types import (
    RESPOND_ACTION_NAME,
    RewardOutputInfo,
    RewardResult,
)

from linalg_zero.grpo.compute_score import calculate_reward
from linalg_zero.grpo.reward_funcs import (
    answer_correct,
    reward_tool_output,
    think_correct,
    validate_answer,
)


def _coerce_to_python(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def _validate_tool_calls(tool_calls, expected_actions, tools_map, data):
    validation: dict[str, bool | str] = {}

    if len(tool_calls) != len(expected_actions):
        validation["structural_error"] = "tool_count_mismatch"
        return False, validation

    for idx, (observed_action, expected_action) in enumerate(zip(tool_calls, expected_actions, strict=True)):
        step_key = f"{observed_action.name}_step_{idx}"

        if observed_action.name != expected_action.name:
            validation[step_key] = "wrong_tool"
            return False, validation

        tool_cls = tools_map.get(observed_action.name)
        if tool_cls is None:
            validation[step_key] = "unknown_tool"
            return False, validation

        try:
            raw_output = tool_cls.invoke(data=data, **observed_action.kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            validation[step_key] = f"error:{exc}"
            return False, validation

        actual_output = _coerce_to_python(raw_output)
        if isinstance(actual_output, str) and actual_output.lower().startswith("error"):
            validation[step_key] = actual_output
            return False, validation

        expected_values = list(expected_action.kwargs.values())
        expected_output = _coerce_to_python(expected_values[0]) if expected_values else None

        try:
            match_score = reward_tool_output(
                ground_truth=expected_output,
                tool_output=actual_output,
            )
        except Exception as exc:  # pragma: no cover - defensive
            validation[step_key] = f"comparison_error:{exc}"
            return False, validation

        if match_score < 1.0:
            validation[step_key] = False
            return False, validation

        validation[step_key] = True

    return True, validation


async def calculate_reward_strict(self) -> RewardResult:
    """Reward function with additional intermediate tool-output validation."""

    assert self.parser is not None, "Parser cannot be None"

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

    answer_rewards = [(validate_answer, 1.0), (think_correct, 0.1), (answer_correct, 0.2)]
    answer_reward, meta = calculate_reward(
        self.parser,
        ground_truth=self.task.outputs[0],
        completion=answer.content,
        reward_funcs_with_weights=answer_rewards,
    )
    answer_found = meta["validate_answer"]

    tool_rewards = [
        (think_correct, 0.1),
    ]
    penalty = 0.0
    for action in tool_calls:
        _, metadata = calculate_reward(self.parser, completion=action.content, reward_funcs_with_weights=tool_rewards)
        if not metadata["think_correct"]:
            penalty += 0.1

    reward = max(0, answer_reward - penalty)

    expected_actions = [action for action in self.task.actions if action.name != RESPOND_ACTION_NAME]
    tools_valid, tool_validation = _validate_tool_calls(
        tool_calls,
        expected_actions,
        self.tools_map,
        self.data,
    )
    if not tools_valid:
        reward = 0.0

    outputs: dict[str, bool | str] = {"answer_found": answer_found}
    if tool_validation:
        outputs.update(tool_validation)

    return RewardResult(
        reward=reward,
        info=RewardOutputInfo(r_outputs=reward, outputs=outputs),
        actions=tool_calls,
    )
