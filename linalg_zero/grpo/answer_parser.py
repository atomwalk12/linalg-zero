from math_verify import parse, verify


def verify_answers(completion_answer: str, ground_truth_answer: str, timeout_seconds: int = 5) -> bool:
    """
    Compare completion answer against ground truth using math_verify's \\boxed{} notation.

    Args:
        completion_answer: Model's answer text from within the <answer> tags
        ground_truth_answer: Expected answer as given during dataset generation
        timeout_seconds: Verification timeout

    Returns:
        True if answers match, False otherwise
    """
    completion_parsed = parse(f"\\boxed{{{completion_answer}}}", parsing_timeout=timeout_seconds)
    ground_truth_parsed = parse(f"\\boxed{{{ground_truth_answer}}}", parsing_timeout=timeout_seconds)

    return verify(completion_parsed, ground_truth_parsed, timeout_seconds=timeout_seconds)
