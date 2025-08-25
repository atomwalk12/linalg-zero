import ast

from math_verify import verify
from sympy import Float, Integer, Matrix

from linalg_zero.shared.types import LibTypes


def parse_string(s: str) -> LibTypes | None:
    """
    Parse string to the most appropriate library type, or None if unsuccessful.
    """
    s = s.strip()

    if not s:
        return None

    try:
        parsed = ast.literal_eval(s)

        if isinstance(parsed, (int, float, list, tuple)):
            return list(parsed) if isinstance(parsed, tuple) else parsed

    except (ValueError, SyntaxError):
        pass

    return None


def verify_answers(ground_truth: LibTypes, target_answer: LibTypes, timeout: int = 5) -> bool:
    """Verify if the target answer matches the ground truth using math_verify."""

    if isinstance(ground_truth, list):
        gt = Matrix(ground_truth)
        target = Matrix(target_answer)
    elif isinstance(ground_truth, float):
        gt = Float(ground_truth)
        target = Float(target_answer)
    elif isinstance(ground_truth, int):
        gt = Integer(ground_truth)
        target = Integer(target_answer)
    else:
        raise TypeError(f"Unsupported answer type: {type(ground_truth)}")

    return verify(gt, target, timeout_seconds=timeout)
