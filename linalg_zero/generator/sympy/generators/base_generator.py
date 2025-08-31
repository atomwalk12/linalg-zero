import random
from typing import Any

from sympy import Integer, Matrix, Rational

from linalg_zero.generator.entropy_control import EntropyController, SampleArgs
from linalg_zero.generator.models import DifficultyCategory
from linalg_zero.generator.sympy.base import ProblemContext, SympyProblemGenerator


class MatrixVectorBaseGenerator(SympyProblemGenerator):
    """Base class for matrix-vector problem generators."""

    def __init__(self, difficulty_level: DifficultyCategory, **kwargs: Any) -> None:
        super().__init__(difficulty_level=difficulty_level, **kwargs)
        self.constraints = kwargs.get("constraints", {})
        self.gen_constraints = kwargs.get("gen_constraints", {})
        self.entropy_controller = EntropyController()

    def _generate_matrix(self, rows: int, cols: int, entropy: float) -> Matrix:
        """Generate a matrix consisting of integers or rationals."""
        matrix_elements = []
        for _ in range(rows):
            row = []
            for _ in range(cols):
                if self.config.allow_rationals and random.random() < 0.3:
                    # 30% chance of rational numbers when allowed
                    numerator = self.entropy_controller.generate_integer(entropy)
                    denominator = self.entropy_controller.generate_integer(entropy)

                    # Avoid zero denominators and ensure non-zero numerators for variety
                    if numerator == 0:
                        numerator = random.choice([1, -1])
                    if denominator == 0:
                        denominator = random.choice([1, -1])
                    element = Rational(numerator, denominator)
                else:
                    number = self.entropy_controller.generate_integer(entropy)

                    # Avoid too many zeros to keep problems interesting
                    if number == 0 and random.random() < 0.7:
                        number = random.choice([1, -1])
                    element = Integer(number)

                row.append(element)
            matrix_elements.append(row)

        return Matrix(matrix_elements)

    def _generate_vector(self, size: int, entropy: float) -> Matrix:
        """Generate a vector consisting of integers or rationals."""
        vector_elements = []

        for i in range(size):
            if self.config.allow_rationals and random.random() < 0.2:
                # 20% chance of rational numbers for vectors
                numerator = self.entropy_controller.generate_integer(entropy)
                denominator = self.entropy_controller.generate_integer(entropy)

                if numerator == 0:
                    numerator = random.choice([1, -1])
                if denominator == 0:
                    denominator = random.choice([1, -1])
                element = Rational(numerator, denominator)
            else:
                number = self.entropy_controller.generate_integer(entropy)
                # Ensure first element is non-zero to avoid zero vectors
                if i == 0 and number == 0:
                    number = random.choice([1, -1])
                element = Integer(number)

            vector_elements.append(element)

        return Matrix(vector_elements)

    def _generate_invertible_matrix(self, size: int, entropy: float) -> Matrix:
        """Generate an invertible matrix with retry logic."""
        max_attempts = 1000

        for _ in range(max_attempts):
            matrix = self._generate_matrix(size, size, entropy)

            try:
                det = matrix.det()
                if det != 0:
                    return matrix
            except Exception:  # noqa: S112
                continue

        raise ValueError(f"Failed to generate invertible matrix after {max_attempts} attempts")

    def _get_matrix_with_constraints(
        self, context: ProblemContext, added_constraints: dict[str, Any] | None = None
    ) -> Matrix:
        """Generate matrix based on constructor constraints.

        Uses self.gen_constraints set by wrapper components to determine:
        - Dimensions (rows/cols, square, size)
        - Special properties (invertible)
        - Entropy allocation

        Args:
            context: Problem generation context

        Returns:
            Generated matrix matching the constraints
        """
        user_provided = self.gen_constraints
        additional = added_constraints or {}
        constraints = {**user_provided, **additional}

        valid_keys = {"rows", "cols", "square", "size", "invertible", "entropy"}
        invalid_keys = set(constraints.keys()) - valid_keys
        if invalid_keys:
            raise ValueError(f"Invalid constraint keys: {invalid_keys}. Valid keys are: {valid_keys}")

        # Determine dimensions
        if "rows" in constraints and "cols" in constraints:
            rows, cols = constraints["rows"], constraints["cols"]
        elif constraints.get("square", False):
            size = constraints.get("size", self.config.get_random_matrix_size())
            rows = cols = size
        else:
            rows = self.config.get_random_matrix_size()
            cols = self.config.get_random_matrix_size()

        # Determine entropy allocation
        if "entropy" in constraints:
            matrix_entropy = constraints["entropy"]
        else:
            sample_args = SampleArgs(num_modules=1, entropy=context.entropy)
            matrix_entropy = sample_args.entropy

        # Generate matrix based on special properties
        if constraints.get("invertible", False):
            if rows != cols:
                raise ValueError("Invertible matrices must be square")
            matrix_A = self._generate_invertible_matrix(rows, matrix_entropy)
        else:
            matrix_A = self._generate_matrix(rows, cols, matrix_entropy)

        context.record_entropy_usage(matrix_entropy)
        return matrix_A
