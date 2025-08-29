from collections.abc import Callable
from typing import Any

from sympy import Matrix, ShapeError
from sympy.matrices.exceptions import NonSquareMatrixError
from sympy.solvers.solvers import NonInvertibleMatrixError
from transformers.utils.chat_template_utils import get_json_schema

from linalg_zero.generator.difficulty_config import Precision
from linalg_zero.generator.sympy.templates import MathFormatter
from linalg_zero.shared.types import assert_lib_returns


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
        result = MathFormatter.sympy_to_primitive(result_matrix, precision=Precision.MATRIX_VECTOR_MULTIPLICATION)

        if isinstance(result, list) and all(isinstance(row, list) for row in result):
            return result
        else:
            raise TypeError(f"Expected list of lists, got {type(result)}")
    except ShapeError as e:
        raise ValueError(f"Matrix dimensions incompatible for multiplication: {e}") from e


def determinant(matrix: list[list[float]]) -> float:
    """Calculate the determinant of a square matrix using SymPy.

    Examples:
        >>> determinant([[1, 2], [3, 4]])
        -2.0
        >>> determinant([[2, 0], [0, 3]])  # Diagonal matrix
        6.0
        >>> determinant([[1]])  # 1x1 matrix
        1.0

    Args:
        matrix: The square matrix as a list of lists.

    Returns:
        The determinant of the matrix.
    """
    try:
        sym_matrix = Matrix(matrix)

        det_result = sym_matrix.det()
        result = MathFormatter.sympy_to_primitive(det_result, precision=Precision.DETERMINANT)

        if isinstance(result, (int, float)):
            return float(result)

    except NonSquareMatrixError as e:
        raise ValueError("Matrix must be square") from e
    except Exception as e:
        raise ValueError(f"Cannot calculate determinant: {e}") from e

    raise TypeError(f"Expected numeric result, got {type(result)}")


def solve_linear_system(matrix_a: list[list[float | int]], vector_b: list[float | int]) -> list[list[float | int]]:
    """Solve the linear system Ax = b for x using SymPy.

    Examples:
        >>> solve_linear_system([[2, 1], [1, 3]], [7, 8])
        [[2.0], [3.0]]
        >>> solve_linear_system([[1, 0], [0, 1]], [5, 3])  # Identity matrix
        [[5.0], [3.0]]

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

        result = MathFormatter.sympy_to_primitive(solution_matrix, precision=Precision.LINEAR_SYSTEM_SOLVER)

        if isinstance(result, list):
            return result

    except NonInvertibleMatrixError as e:
        raise NonInvertibleMatrixError(f"Cannot solve linear system: {e}") from e
    except ShapeError as e:
        raise ShapeError(f"Matrix dimensions incompatible for solving linear system: {e}") from e

    raise TypeError(f"Expected list, got {type(result)}")


def get_lib() -> dict[str, Callable[..., Any]]:
    """Return the library of available functions."""
    return {
        "multiply_matrices": multiply_matrices,
        "solve_linear_system": solve_linear_system,
        "determinant": determinant,
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
