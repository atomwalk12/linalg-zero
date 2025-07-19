"""Function calling (fc) methods used to build the SFT dataset."""

from collections.abc import Callable
from typing import Any


def permutation_count(n: int, k: int) -> int:
    """Calculates the number of permutations of k elements from a set of n elements.

    Args:
        n: The total number of elements in the set.
        k: The number of elements to choose for the permutation.

    Returns:
        The number of permutations.
    """
    # Tool:
    # {"name": "permutation_count", "description": "Calculates the number of permutations of k elements from a set of n elements.", "parameters": {"n": {"description": "The total number of elements in the set.", "type": "int"}, "k": {"description": "The number of elements to choose for the permutation.", "type": "int"}}}
    # Answer:
    # {"name": "permutation_count", "arguments": {"n": 10, "k": 3}}
    import math

    return int(math.factorial(n) / math.factorial(n - k))


def get_division(dividend: int, divisor: int) -> float:
    """Divides two numbers by making an API call to a division service.

    Args:
        dividend: The dividend in the division operation.
        divisor: The divisor in the division operation.

    Returns:
        Division of the 2 numbers.
    """
    # Tool:
    # {"name": "get_division", "description": "Divides two numbers by making an API call to a division service.", "parameters": {"divisor": {"description": "The divisor in the division operation.", "type": "int", "default": ""}, "dividend": {"description": "The dividend in the division operation.", "type": "int", "default": ""}}}
    # Answer:
    # {"name": "get_division", "arguments": {"divisor": 25, "dividend": 100}}
    return dividend / divisor


def get_multiplication(a: int, b: int) -> int:
    """Performs multiplication of a and b then returns the result.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        Multiplication of the 2 numbers.
    """
    # Tool:
    # {"name": "get_multiplication", "description": "Performs multiplication of a and b then returns the result.", "parameters": {"a": {"description": "The first number.", "type": "int"}, "b": {"description": "The second number.", "type": "int"}}}
    # Answer:
    # {"name": "get_multiplication", "arguments": {"a": 15, "b": 7}}
    return a * b


def get_lib() -> dict[str, Callable[..., Any]]:
    return {
        "permutation_count": permutation_count,
        "get_division": get_division,
        "get_multiplication": get_multiplication,
    }


def get_tools() -> dict[str, dict[str, Any]]:
    """Returns the tool representation of the functions in the library."""
    from transformers.utils.chat_template_utils import get_json_schema

    return {name: get_json_schema(func) for name, func in get_lib().items()}
