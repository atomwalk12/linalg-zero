"""Linear algebra functions library for demonstrating function calling verification."""

import math
from collections.abc import Callable
from typing import Any

from transformers.utils.chat_template_utils import get_json_schema


def add_numbers(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The sum of the two numbers.
    """
    return float(a) + float(b)


def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The product of the two numbers.
    """
    return float(a) * float(b)


def divide_numbers(dividend: float, divisor: float) -> float:
    """Divide two numbers.

    Args:
        dividend: The number to be divided.
        divisor: The number to divide by.

    Returns:
        The result of the division.
    """
    if divisor == 0:
        raise ValueError("Cannot divide by zero")
    return float(dividend) / float(divisor)


def multiply_matrices(matrix_a: list[list[float]], matrix_b: list[list[float]]) -> list[list[float]]:
    """Multiply two matrices.

    Args:
        matrix_a: The first matrix as a list of lists.
        matrix_b: The second matrix as a list of lists.

    Returns:
        The product matrix as a list of lists.
    """
    rows_a, cols_a = len(matrix_a), len(matrix_a[0])
    rows_b, cols_b = len(matrix_b), len(matrix_b[0])

    if cols_a != rows_b:
        raise ValueError("Matrix dimensions incompatible for multiplication")

    result = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result


def transpose_matrix(matrix: list[list[float]]) -> list[list[float]]:
    """Transpose a matrix.

    Args:
        matrix: The matrix to transpose as a list of lists.

    Returns:
        The transposed matrix as a list of lists.
    """
    if not matrix or not matrix[0]:
        return []

    rows, cols = len(matrix), len(matrix[0])
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]


def determinant(matrix: list[list[float]]) -> float:
    """Calculate the determinant of a square matrix.

    Args:
        matrix: The square matrix as a list of lists.

    Returns:
        The determinant of the matrix.
    """
    n = len(matrix)

    if n != len(matrix[0]):
        raise ValueError("Matrix must be square")

    if n == 1:
        return matrix[0][0]
    elif n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        det = 0
        for j in range(n):
            minor = [[matrix[i][k] for k in range(n) if k != j] for i in range(1, n)]
            det += ((-1) ** j) * matrix[0][j] * determinant(minor)
        return det


def frobenius_norm(matrix: list[list[float]]) -> float:
    """Calculate the Frobenius norm of a matrix.

    Args:
        matrix: The matrix as a list of lists.

    Returns:
        The Frobenius norm of the matrix.
    """
    total = 0.0
    for row in matrix:
        for element in row:
            total += element * element
    return math.sqrt(total)


def matrix_trace(matrix: list[list[float]]) -> float:
    """Calculate the trace (sum of diagonal elements) of a square matrix.

    Args:
        matrix: The square matrix as a list of lists.

    Returns:
        The trace of the matrix.
    """
    if len(matrix) != len(matrix[0]):
        raise ValueError("Matrix must be square")

    return sum(matrix[i][i] for i in range(len(matrix)))


def permutation_count(n: int, k: int) -> int:
    """Calculate the number of permutations of k elements from a set of n elements.

    Args:
        n: The total number of elements in the set.
        k: The number of elements to choose for the permutation.

    Returns:
        The number of permutations.
    """
    if k > n or k < 0:
        return 0
    return math.factorial(n) // math.factorial(n - k)


def vector_dot_product(vector_a: list[float], vector_b: list[float]) -> float:
    """Calculate the dot product of two vectors.

    Args:
        vector_a: The first vector as a list of numbers.
        vector_b: The second vector as a list of numbers.

    Returns:
        The dot product of the two vectors.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same length")

    return sum(a * b for a, b in zip(vector_a, vector_b, strict=False))


def get_division(dividend: int, divisor: int) -> float:
    """Divides two numbers by making an API call to a division service.

    Args:
        dividend: The dividend in the division operation.
        divisor: The divisor in the division operation.

    Returns:
        Division of the 2 numbers.
    """
    return dividend / divisor


def get_multiplication(a: int, b: int) -> int:
    """Performs multiplication of a and b then returns the result.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        Multiplication of the 2 numbers.
    """
    return a * b


def get_lib() -> dict[str, Callable[..., Any]]:
    """Return the library of available functions."""
    return {
        "add_numbers": add_numbers,
        "multiply_numbers": multiply_numbers,
        "divide_numbers": divide_numbers,
        "multiply_matrices": multiply_matrices,
        "transpose_matrix": transpose_matrix,
        "determinant": determinant,
        "frobenius_norm": frobenius_norm,
        "matrix_trace": matrix_trace,
        "permutation_count": permutation_count,
        "vector_dot_product": vector_dot_product,
        "get_division": get_division,
        "get_multiplication": get_multiplication,
    }


def get_tools() -> list[dict[str, Any]]:
    """Returns the tool representation of the functions in the library."""
    return [get_json_schema(func) for func in get_lib().values()]
