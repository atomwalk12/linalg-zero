from collections.abc import Callable
from typing import Any

from sympy import Matrix, ShapeError
from sympy.matrices.exceptions import NonSquareMatrixError
from sympy.solvers.solvers import NonInvertibleMatrixError
from transformers.utils.chat_template_utils import get_json_schema

from linalg_zero.generator.difficulty_config import Precision
from linalg_zero.generator.sympy.templates import MathFormatter
from linalg_zero.shared.types import assert_lib_returns


def multiply_matrices(matrix_a: list[list[float | int]], vector_b: list[list[float | int]]) -> list[list[float | int]]:
    """Multiply two matrices or a matrix and a column vector using SymPy.

    This function supports:
    - matrix x matrix
    - matrix x vector (the vector must be provided as a list of lists, i.e.,
      a column vector like [[v1], [v2], ...]). In all cases, inputs must be
      list-of-lists.

    Examples:
        >>> multiply_matrices([[1, 2], [3, 4]], [[2, 0], [1, 3]])
        [[4, 6], [10, 12]]
        >>> multiply_matrices([[1, 2], [3, 4]], [[5], [6]])  # matrix x column vector
        [[17], [39]]
        >>> multiply_matrices([[1, 0], [0, 1]], [[5, 6], [7, 8]])  # Identity x matrix
        [[5, 6], [7, 8]]

    Args:
        matrix_a: The first matrix as a list of lists.
        vector_b: The second operand as a list of lists (matrix or column vector).

    Returns:
        The product as a list of lists.
    """
    try:
        sym_a = Matrix(matrix_a)
        sym_b = Matrix(vector_b)
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


def frobenius_norm(matrix: list[list[float | int]]) -> float:
    """Calculate the Frobenius norm of a matrix using SymPy.

    The Frobenius norm is the square root of the sum of the absolute squares
    of all elements in the matrix: ||A||_F = sqrt(sum(|a_ij|^2)).

    Examples:
        >>> frobenius_norm([[1, 2], [3, 4]])
        5.477226
        >>> frobenius_norm([[3, 4]])  # Single row
        5.0
        >>> frobenius_norm([[0, 0], [0, 0]])  # Zero matrix
        0.0

    Args:
        matrix: The matrix as a list of lists.

    Returns:
        The Frobenius norm of the matrix.
    """
    try:
        sym_matrix = Matrix(matrix)

        # Calculate Frobenius norm: sqrt(sum of squared elements)
        norm_result = sym_matrix.norm()
        result = MathFormatter.sympy_to_primitive(norm_result, precision=Precision.FROBENIUS_NORM)

        if isinstance(result, (int, float)):
            return float(result)

    except Exception as e:
        raise ValueError(f"Cannot calculate Frobenius norm: {e}") from e

    raise TypeError(f"Expected numeric result, got {type(result)}")


def matrix_rank(matrix: list[list[float | int]]) -> int:
    """Calculate the rank of a matrix using SymPy.

    The rank of a matrix is the dimension of the vector space spanned by its
    rows or columns - the number of linearly independent rows or columns.

    Examples:
        >>> matrix_rank([[1, 2], [3, 4]])
        2
        >>> matrix_rank([[1, 2], [2, 4]])  # Linearly dependent rows
        1
        >>> matrix_rank([[0, 0], [0, 0]])  # Zero matrix
        0

    Args:
        matrix: The matrix as a list of lists.

    Returns:
        The rank of the matrix.
    """
    try:
        sym_matrix = Matrix(matrix)
        rank_result = sym_matrix.rank()

        if isinstance(rank_result, int):
            return rank_result

    except Exception as e:
        raise ValueError(f"Cannot calculate matrix rank: {e}") from e

    raise TypeError(f"Expected integer result, got {type(rank_result)}")


def matrix_transpose(matrix: list[list[float | int]]) -> list[list[float | int]]:
    """Calculate the transpose of a matrix using SymPy.

    The transpose of a matrix A is obtained by reflecting the matrix over its main diagonal,
    switching the row and column indices of the matrix.

    Examples:
        >>> matrix_transpose([[1, 2, 3], [4, 5, 6]])
        [[1, 4], [2, 5], [3, 6]]
        >>> matrix_transpose([[1, 2], [3, 4]])
        [[1, 3], [2, 4]]
        >>> matrix_transpose([[1]])  # 1x1 matrix
        [[1]]

    Args:
        matrix: The matrix as a list of lists.

    Returns:
        The transpose of the matrix as a list of lists.
    """
    try:
        sym_matrix = Matrix(matrix)
        transpose_result = sym_matrix.T

        result = MathFormatter.sympy_to_primitive(transpose_result, precision=Precision.MATRIX_TRANSPOSE)

        if isinstance(result, list) and all(isinstance(row, list) for row in result):
            return result

    except Exception as e:
        raise ValueError(f"Cannot calculate matrix transpose: {e}") from e

    raise TypeError(f"Expected list of lists, got {type(result)}")


def get_lib() -> dict[str, Callable[..., Any]]:
    """Return the library of available functions."""
    return {
        "multiply_matrices": multiply_matrices,
        "solve_linear_system": solve_linear_system,
        "determinant": determinant,
        "frobenius_norm": frobenius_norm,
        "matrix_rank": matrix_rank,
        "matrix_transpose": matrix_transpose,
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
