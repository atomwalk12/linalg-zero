import json
from collections.abc import Callable

from linalg_zero.grpo.reward_funcs import (
    reward_final_answer,
    reward_response_format,
    reward_tool_output,
)
from linalg_zero.grpo.verifiers.xml_parser import XMLParser
from linalg_zero.shared.lib import LibTypes


def get_tool_reward(ground_truth: LibTypes, tool_output: LibTypes) -> tuple[float, dict]:
    """Computes the reward for a single tool call."""
    reward_funcs_with_weights: list[tuple[Callable[[LibTypes, LibTypes], float], float]] = [
        (reward_tool_output, 1.0),
    ]

    reward = 0.0
    metadata: dict[str, str | bool] = {}
    for reward_func, weight in reward_funcs_with_weights:
        try:
            score = reward_func(tool_output, ground_truth)
            reward += score * weight
            metadata[reward_func.__name__] = True
        except Exception as e:
            # If reward function fails, contribute 0
            metadata[reward_func.__name__] = False
            metadata[f"{reward_func.__name__}_error"] = str(e)

    return reward, metadata


def get_interaction_reward(
    parser: XMLParser, completion: list[dict] | str, ground_truth: LibTypes
) -> tuple[float, dict]:
    """
    Computes the reward for a single tool_call->tool_response turn. It simulates
    a user that provides feedback based on the tool response.
    """
    reward_funcs_with_weights: list[tuple[Callable[[XMLParser, list[dict] | str, LibTypes], float], float]] = [
        (reward_final_answer, 1.0),
        (reward_response_format, 0.2),
    ]

    reward = 0.0
    metadata: dict[str, str | bool] = {}
    for reward_func, weight in reward_funcs_with_weights:
        try:
            score = reward_func(parser, completion, ground_truth)
            reward += score * weight
            metadata[reward_func.__name__] = True
        except Exception as e:
            # If reward function fails, contribute 0
            metadata[reward_func.__name__] = False
            metadata[f"{reward_func.__name__}_error"] = str(e)

    return reward, metadata


def calc_reward(solution_str: str, ground_truth: str, **kwargs: dict) -> float:
    """
    Calculates the reward for the complete trajectory. It is the solution retrieved
    after all interactions (computed by the get_interaction_answer) finish.
    """
    parser = XMLParser()
    parsed: LibTypes = json.loads(ground_truth)
    reward, _ = get_interaction_reward(parser, solution_str, parsed)
    return reward
