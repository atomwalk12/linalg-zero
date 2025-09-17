from linalg_zero.grpo.verifiers.xml_parser import XMLParser
from linalg_zero.grpo.verify import parse_string, verify_answers
from linalg_zero.shared.types import LibTypes


def reward_tool_output(*, ground_truth: LibTypes, tool_output: LibTypes) -> float:
    """Reward function that checks if the tool output matches the ground truth."""
    return 1.0 if verify_answers(ground_truth, tool_output) else 0.0


def reward_response_format(parser: XMLParser, *, ground_truth: LibTypes, completion: list[dict] | str) -> float:
    if isinstance(completion, list):
        # Extract the last assistant message to score during a user-assistant interaction
        # Malformed tool are not penalized here, but during execution in `LinalgZeroTool`
        assistant_messages = parser.get_messages(completion, role="assistant")
        if not assistant_messages:
            return 0.0

        # The response format is starts with <think> tags and ends with <answer> tags
        message = assistant_messages[-1]["content"]
    else:
        message = completion

    analysis = parser.analyze_message(message)
    return 1.0 if analysis["is_valid_think_then_answer"] else 0.0


def reward_final_answer(parser: XMLParser, *, ground_truth: LibTypes, completion: list[dict] | str) -> float:
    """Reward function that checks if the completion answer matches the ground truth."""
    if isinstance(completion, list):
        assistant_messages = parser.get_messages(completion, role="assistant")
        if not assistant_messages:
            return 0.0

        message = assistant_messages[-1]["content"]
    else:
        message = completion

    analysis = parser.analyze_message(message)
    answer = analysis["answer"]
    target = parse_string(answer) if answer else None
    if target is None:
        return 0.0
    return 1.0 if verify_answers(ground_truth, target) else 0.0


def reward_num_tool_calls(parser: XMLParser, completion: list[dict]) -> float:
    """Count the number of tool calls in the completion."""
    num_tool_calls = len(parser.get_messages(completion, role="tool"))
    return float(num_tool_calls)


def reward_num_tool_errors(parser: XMLParser, completion: list[dict]) -> float:
    """Count the number of errors in tool messages."""
    num_errors = sum([
        1.0
        for msg in parser.get_messages(completion, role="tool")
        if (content := msg.get("content")) and "error" in content.lower()
    ])
    return num_errors


def reward_execution_success_rate(parser: XMLParser, completion: list[dict]) -> float:
    """Combines num_tool_calls and num_errors to return execution success rate."""
    tool_messages = parser.get_messages(completion, role="tool")
    num_tool_calls = len(tool_messages)

    if num_tool_calls == 0:
        return 0.0

    num_errors = sum(1.0 for msg in tool_messages if (content := msg.get("content")) and "error" in content.lower())

    # This returns the success rate, as opposed to the error rate
    return 1.0 - (num_errors / num_tool_calls)
