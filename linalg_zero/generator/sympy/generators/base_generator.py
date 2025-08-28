import random
from typing import Any

from sympy import Integer, Matrix, Rational

from linalg_zero.generator.difficulty_config import (
    EntropyController,
)
from linalg_zero.generator.models import DifficultyCategory
from linalg_zero.generator.sympy.base import SympyProblemGenerator
from linalg_zero.generator.sympy.templates import TemplateEngine


class MatrixVectorBaseGenerator(SympyProblemGenerator):
    """Base class for matrix-vector problem generators."""

    def __init__(self, entropy: float, difficulty_level: DifficultyCategory, **kwargs: Any) -> None:
        super().__init__(entropy, difficulty_level=difficulty_level, **kwargs)
        self.template_engine = TemplateEngine()

    def _generate_matrix(self, rows: int, cols: int, entropy: float, controller: EntropyController) -> Matrix:
        """Generate a matrix consisting of integers or rationals."""
        matrix_elements = []
        for _ in range(rows):
            row = []
            for _ in range(cols):
                if self.config.allow_rationals and random.random() < 0.3:
                    # 30% chance of rational numbers when allowed
                    numerator = controller.generate_integer(entropy)
                    denominator = controller.generate_integer(entropy)

                    # Avoid zero denominators and ensure non-zero numerators for variety
                    if numerator == 0:
                        numerator = random.choice([1, -1])
                    element = Rational(numerator, denominator)
                else:
                    number = controller.generate_integer(entropy)

                    # Avoid too many zeros to keep problems interesting
                    if number == 0 and random.random() < 0.7:
                        number = random.choice([1, -1])
                    element = Integer(number)

                row.append(element)
            matrix_elements.append(row)

        return Matrix(matrix_elements)

    def _generate_vector(self, size: int, entropy: float, controller: EntropyController) -> Matrix:
        """Generate a vector consisting of integers or rationals."""
        vector_elements = []

        for i in range(size):
            if self.config.allow_rationals and random.random() < 0.2:
                # 20% chance of rational numbers for vectors
                numerator = controller.generate_integer(entropy)
                denominator = controller.generate_integer(entropy)
                if numerator == 0:
                    numerator = random.choice([1, -1])
                element = Rational(numerator, denominator)
            else:
                number = controller.generate_integer(entropy)
                # Ensure first element is non-zero to avoid zero vectors
                if i == 0 and number == 0:
                    number = random.choice([1, -1])
                element = Integer(number)

            vector_elements.append(element)

        return Matrix(vector_elements)

    def _generate_invertible_matrix(self, size: int, entropy: float, controller: EntropyController) -> Matrix:
        """Generate an invertible matrix with retry logic."""
        max_attempts = 1000

        for _ in range(max_attempts):
            matrix = self._generate_matrix(size, size, entropy, controller)

            try:
                det = matrix.det()
                if det != 0:
                    return matrix
            except Exception:  # noqa: S112
                continue

        raise ValueError(f"Failed to generate invertible matrix after {max_attempts} attempts")
