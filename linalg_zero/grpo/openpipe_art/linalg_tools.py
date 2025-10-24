"""
Linear algebra tool wrappers.

This module provides tool wrappers that adapt the mathematical functions
from linalg_zero.shared.lib to our tool interface.
"""

from typing import Any

from transformers.utils.chat_template_utils import get_json_schema

from linalg_zero.shared.lib import (
    determinant,
    frobenius_norm,
    matrix_cofactor,
    matrix_rank,
    matrix_trace,
    matrix_transpose,
)

from .base_env import Tool


class LinAlgTool(Tool):
    """Base class for linear algebra tools."""

    @staticmethod
    def invoke(data: dict[str, Any], **kwargs: dict[str, Any]) -> str:
        """
        Invoke the tool with given data and arguments.

        Args:
            data: Environment data dictionary
            **kwargs: Tool-specific arguments

        Returns:
            String representation of the result
        """
        raise NotImplementedError("Subclasses must implement invoke method")

    @staticmethod
    def get_info() -> dict[str, Any]:
        """
        Get tool information including function schema.

        Returns:
            Dictionary with tool information
        """
        raise NotImplementedError("Subclasses must implement get_info method")


class MatrixTransposeTool(LinAlgTool):
    """Tool wrapper for matrix transpose operation."""

    @staticmethod
    def invoke(data: dict[str, Any], **kwargs: dict[str, Any]) -> str:
        """Calculate matrix transpose."""
        matrix = kwargs.get("matrix")
        if matrix is None:
            raise ValueError("Matrix parameter is required")

        result = matrix_transpose(matrix)
        return str(result)

    @staticmethod
    def get_info() -> dict[str, Any]:
        """Get matrix transpose tool information using automatic schema generation."""
        return get_json_schema(matrix_transpose)


class DeterminantTool(LinAlgTool):
    """Tool wrapper for determinant calculation."""

    @staticmethod
    def invoke(data: dict[str, Any], **kwargs: dict[str, Any]) -> str:
        """Calculate matrix determinant."""
        matrix = kwargs.get("matrix")
        if matrix is None:
            raise ValueError("Matrix parameter is required")

        result = determinant(matrix)
        return str(result)

    @staticmethod
    def get_info() -> dict[str, Any]:
        """Get determinant tool information using automatic schema generation."""
        return get_json_schema(determinant)


class MatrixCofactorTool(LinAlgTool):
    """Tool wrapper for matrix cofactor calculation."""

    @staticmethod
    def invoke(data: dict[str, Any], **kwargs: dict[str, Any]) -> str:
        """Calculate matrix cofactor."""
        matrix = kwargs.get("matrix")
        if matrix is None:
            raise ValueError("Matrix parameter is required")

        result = matrix_cofactor(matrix)
        return str(result)

    @staticmethod
    def get_info() -> dict[str, Any]:
        """Get matrix cofactor tool information using automatic schema generation."""
        return get_json_schema(matrix_cofactor)


class FrobeniusNormTool(LinAlgTool):
    """Tool wrapper for Frobenius norm calculation."""

    @staticmethod
    def invoke(data: dict[str, Any], **kwargs: dict[str, Any]) -> str:
        """Calculate Frobenius norm."""
        matrix = kwargs.get("matrix")
        if matrix is None:
            raise ValueError("Matrix parameter is required")

        result = frobenius_norm(matrix)
        return str(result)

    @staticmethod
    def get_info() -> dict[str, Any]:
        """Get Frobenius norm tool information using automatic schema generation."""
        return get_json_schema(frobenius_norm)


class MatrixRankTool(LinAlgTool):
    """Tool wrapper for matrix rank calculation."""

    @staticmethod
    def invoke(data: dict[str, Any], **kwargs: Any) -> str:
        """Calculate matrix rank."""
        matrix = kwargs.get("matrix")
        if matrix is None:
            raise ValueError("Matrix parameter is required")

        result = matrix_rank(matrix)
        return str(result)

    @staticmethod
    def get_info() -> dict[str, Any]:
        """Get matrix rank tool information using automatic schema generation."""
        return get_json_schema(matrix_rank)


class MatrixTraceTool(LinAlgTool):
    """Tool wrapper for matrix trace calculation."""

    @staticmethod
    def invoke(data: dict[str, Any], **kwargs: Any) -> str:
        """Calculate matrix trace."""
        matrix = kwargs.get("matrix")
        if matrix is None:
            raise ValueError("Matrix parameter is required")

        result = matrix_trace(matrix)
        return str(result)

    @staticmethod
    def get_info() -> dict[str, Any]:
        """Get matrix trace tool information using automatic schema generation."""
        return get_json_schema(matrix_trace)


def get_linalg_tools() -> list[type[LinAlgTool]]:
    """
    Get all available linear algebra tools.

    Returns:
        List of tool classes
    """
    return [
        MatrixTransposeTool,
        DeterminantTool,
        MatrixCofactorTool,
        FrobeniusNormTool,
        MatrixRankTool,
        MatrixTraceTool,
    ]


def get_linalg_tools_info() -> list[dict[str, Any]]:
    """
    Get information for all linear algebra tools.

    Returns:
        List of tool information dictionaries
    """
    return [tool.get_info() for tool in get_linalg_tools()]
