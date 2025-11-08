from linalg_zero.grpo.verifiers.xml_parser import XMLParser
from linalg_zero.grpo.verify import parse_string, verify_answers
from linalg_zero.shared.types import LibTypes

parser = XMLParser()


def validate_answer(ground_truth: LibTypes, completion: str) -> float:
    """Reward function that checks if the completion answer matches the ground truth."""
    answer = parser.extract_last_answer(completion)
    target = parse_string(answer) if answer else None
    if target is None:
        return 0.0
    return 1.0 if verify_answers(ground_truth, target) else 0.0


def think_correct(completion: str) -> float:
    has_think = parser.extract_last_thought(completion)
    return 1.0 if has_think else 0.0


def answer_correct(completion: str) -> float:
    has_answer = parser.extract_last_answer(completion)
    return 1.0 if has_answer else 0.0
