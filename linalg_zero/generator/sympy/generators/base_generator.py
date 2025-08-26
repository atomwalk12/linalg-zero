import random
from typing import Any

from sympy import Integer, Matrix

from linalg_zero.generator.models import DifficultyCategory
from linalg_zero.generator.sympy.base import (
    SympyProblemGenerator,
)
from linalg_zero.generator.sympy.number_generator import EntropyController
from linalg_zero.generator.sympy.templates import TemplateEngine


class MatrixVectorBaseGenerator(SympyProblemGenerator):
    """Base class for matrix-vector problem generators."""

    def __init__(
        self, entropy: float, difficulty_level: DifficultyCategory, integers_only: bool, **kwargs: Any
    ) -> None:
        super().__init__(entropy, difficulty_level, **kwargs)
        self.integers_only = integers_only

        # TODO: the entropy threshold calculation should be adjusted
        # Scale dimension with entropy using floor division
        self.max_dimension = min(4, max(2, int(entropy // 1.5) + 1))
        self.template_engine = TemplateEngine()

    def _generate_matrix(self, rows: int, cols: int, entropy: float, controller: EntropyController) -> Matrix:
        """Generate a matrix with entropy-controlled complexity."""

        # Ensure minimum entropy per element to get reasonable numbers
        min_entropy_per_element = 1.0
        num_elements = rows * cols
        entropy_per_element = max(min_entropy_per_element, entropy / max(3, num_elements))

        matrix_elements = []
        for _ in range(rows):
            row = []
            for _ in range(cols):
                if self.integers_only:
                    element = controller.generate_integer(entropy_per_element)
                elif entropy_per_element < 1.5:
                    # Mostly integers for small entropy
                    if random.random() < 0.9:
                        element = controller.generate_integer(entropy_per_element)
                    else:
                        element = controller.generate_rational(entropy_per_element)
                else:
                    # Allow more rational numbers for higher entropy
                    if random.random() < 0.7:
                        element = controller.generate_integer(entropy_per_element)
                    else:
                        element = controller.generate_rational(entropy_per_element)

                # Avoid zero elements to maintain interesting problems
                if element == 0:
                    element = Integer(1) if random.random() < 0.5 else Integer(-1)

                row.append(element)
            matrix_elements.append(row)

        return Matrix(matrix_elements)

    def _generate_vector(self, size: int, entropy: float, controller: EntropyController) -> Matrix:
        """Generate a vector with entropy-controlled complexity."""

        # Ensure minimum entropy per element to get reasonable numbers
        min_entropy_per_element = 1.0
        entropy_per_element = max(min_entropy_per_element, entropy / max(2, size))

        vector_elements = []
        for i in range(size):
            if self.integers_only:
                element = controller.generate_integer(entropy_per_element)
            elif entropy_per_element < 1.5:
                # Mostly integers for small entropy
                if random.random() < 0.9:
                    element = controller.generate_integer(entropy_per_element)
                else:
                    element = controller.generate_rational(entropy_per_element)
            else:
                # Allow more rational numbers for higher entropy
                if random.random() < 0.8:
                    element = controller.generate_integer(entropy_per_element)
                else:
                    element = controller.generate_rational(entropy_per_element)

            # Avoid all-zero vectors
            if i == 0 and element == 0:
                element = Integer(1)

            vector_elements.append(element)

        return Matrix(vector_elements)
