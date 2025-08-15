from linalg_zero.grpo.reward_funcs import (
    reward_final_answer,
    reward_response_format,
    reward_tool_output,
)
from linalg_zero.grpo.verifiers.xml_parser import XMLParser
from linalg_zero.shared.lib import LibTypes


def get_tool_reward(ground_truth: LibTypes, tool_output: LibTypes) -> tuple[float, dict]:
    """Computes the reward for a single tool call."""
    reward_funcs_with_weights = [
        (reward_tool_output, 1.0),
    ]

    total_score = 0.0
    metadata = {}
    for reward_func, weight in reward_funcs_with_weights:
        try:
            score = reward_func(tool_output, ground_truth)
            total_score += score * weight
            metadata[reward_func.__name__] = True
        except Exception:
            # If reward function fails, contribute 0
            metadata[reward_func.__name__] = False

    return total_score, metadata


def get_interaction_reward(parser: XMLParser, completion: list[dict], ground_truth: str) -> tuple[float, dict]:
    """Computes the reward for a single tool_call->response interaction."""
    reward_funcs_with_weights = [(reward_final_answer, 1.0), (reward_response_format, 0.2)]

    total_score = 0.0
    metadata = {}
    for reward_func, weight in reward_funcs_with_weights:
        try:
            score = reward_func(parser, completion, ground_truth)
            total_score += score * weight
            metadata[reward_func.__name__] = True
        except Exception:
            # If reward function fails, contribute 0
            metadata[reward_func.__name__] = False

    return total_score, metadata


def calc_reward(solution_str, ground_truth, **kwargs) -> float:
    """Calculates the reward for a single tool_call->response interaction."""
    parser = XMLParser()
    reward, _ = get_interaction_reward(parser, solution_str, ground_truth)
    return reward
