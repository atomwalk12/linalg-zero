import math
from collections.abc import Callable
from typing import Any

from sympy import Matrix, ShapeError
from transformers.utils.chat_template_utils import get_json_schema

from linalg_zero.generator import Precision
from linalg_zero.generator.sympy.templates import MathFormatter
from linalg_zero.shared.types import assert_lib_returns


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


def multiply_matrices(matrix_a: list[list[float | int]], matrix_b: list[list[float | int]]) -> list[list[float | int]]:
    """Multiply two matrices using SymPy.

    Examples:
        >>> multiply_matrices([[1, 2], [3, 4]], [[2, 0], [1, 3]])
        [[4, 6], [10, 12]]
        >>> multiply_matrices([[1, 0], [0, 1]], [[5, 6], [7, 8]])  # Identity matrix
        [[5, 6], [7, 8]]

    Args:
        matrix_a: The first matrix as a list of lists.
        matrix_b: The second matrix as a list of lists.

    Returns:
        The product matrix as a list of lists.
    """
    try:
        sym_a = Matrix(matrix_a)
        sym_b = Matrix(matrix_b)
        result_matrix: Matrix = sym_a * sym_b
        result = MathFormatter.sympy_to_primitive(result_matrix, precision=Precision.MULTIPLY_MATRICES)

        if isinstance(result, list) and all(isinstance(row, list) for row in result):
            return result
        else:
            raise TypeError(f"Expected list of lists, got {type(result)}")
    except ShapeError as e:
        raise ValueError(f"Matrix dimensions incompatible for multiplication: {e}") from e


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

    Examples:
        >>> frobenius_norm([[1, 0], [0, 1]])  # Identity matrix
        1.4142135623730951
        >>> frobenius_norm([[3, 4]])  # Single row
        5.0
        >>> frobenius_norm([[1], [2], [3]])  # Single column
        3.7416573867739413

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


def solve_linear_system(matrix_a: list[list[float | int]], vector_b: list[float | int]) -> list[float | int]:
    """Solve the linear system Ax = b for x using SymPy.

    Examples:
        >>> solve_linear_system([[2, 1], [1, 3]], [7, 8])
        [2.0, 3.0]
        >>> solve_linear_system([[1, 0], [0, 1]], [5, 3])  # Identity matrix
        [5.0, 3.0]

    Args:
        matrix_a: The coefficient matrix as a list of lists.
        vector_b: The right-hand side vector as a list.

    Returns:
        The solution vector x as a list.
    """
    try:
        sym_a = Matrix(matrix_a)
        sym_b = Matrix(vector_b)

        solution_matrix = sym_a.LUsolve(sym_b)

        result = MathFormatter.sympy_to_primitive(solution_matrix, precision=Precision.SOLVE_LINEAR_SYSTEM)

        if isinstance(result, list):
            return result

    except Exception as e:
        raise ValueError(f"Cannot solve linear system: {e}") from e

    raise TypeError(f"Expected list, got {type(result)}")


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
        "divide_numbers": divide_numbers,
        "multiply_matrices": multiply_matrices,
        "solve_linear_system": solve_linear_system,
        "frobenius_norm": frobenius_norm,
    }


def get_tools() -> list[dict[str, Any]]:
    """Returns the tool representation of the functions in the library."""
    return [get_json_schema(func) for func in get_lib().values()]


def get_lib_types_list() -> list[type]:
    """
    Get the list of library return types.
    This is a check to ensure grpo training uses well-tested types in math-verify.
    This only influences the reward functions, and will likely work with other types
    as well. Make sure the types defined below coincide by using this function.
    """
    return assert_lib_returns({float, int, list}, get_lib())
