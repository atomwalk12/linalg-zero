import random
from typing import Any

from sympy import Matrix
from typing_extensions import override

from linalg_zero.generator import Precision
from linalg_zero.generator.models import DifficultyCategory
from linalg_zero.generator.sympy.base import (
    ProblemContext,
    ProblemTemplate,
)
from linalg_zero.generator.sympy.generators.base_generator import MatrixVectorBaseGenerator
from linalg_zero.generator.sympy.number_generator import EntropyController
from linalg_zero.generator.sympy.templates import MathFormatter
from linalg_zero.grpo.verify import verify_answers
from linalg_zero.shared.lib import multiply_matrices
from linalg_zero.shared.types import LibTypes


class MatrixVectorMultiplicationGenerator(MatrixVectorBaseGenerator):
    def __init__(self, entropy: float, difficulty_level: DifficultyCategory, **kwargs: Any) -> None:
        """Initialize matrix-vector multiplication generator."""
        super().__init__(entropy, difficulty_level, integers_only=True, **kwargs)
        self.precision = Precision.MULTIPLY_MATRICES
        self.math_formatter = MathFormatter()

    def generate_mathematical_content(self, context: ProblemContext) -> ProblemTemplate:
        """Generate matrix-vector multiplication problem content."""

        entropy_controller = EntropyController(context.entropy)

        # Allocate entropy between matrix and vector generation
        matrix_entropy = context.entropy * 1.0
        vector_entropy = context.entropy * 0.9

        # Determine matrix dimensions based on difficulty category
        difficulty_category = context.difficulty_level
        if difficulty_category == DifficultyCategory.EASY:
            rows, cols = 2, 2
        elif difficulty_category == DifficultyCategory.MEDIUM:
            rows, cols = 3, 3
        elif difficulty_category == DifficultyCategory.HARD:
            rows = random.randint(3, self.max_dimension)
            cols = random.randint(3, self.max_dimension)
        else:
            raise ValueError(f"Invalid difficulty category: {difficulty_category}")

        # Generate matrix A and vector x
        matrix_A = self._generate_matrix(rows, cols, matrix_entropy, entropy_controller)
        context.record_entropy_usage(matrix_entropy)

        vector_x = self._generate_vector(cols, vector_entropy, entropy_controller)
        context.record_entropy_usage(vector_entropy)
        sympy_sol, lib_result = self._multiply_matrices_sympy(matrix_A, vector_x)
        _ = context.record_tool_call("multiply_matrices", lib_result, is_final=True)

        problem_expression = matrix_A * vector_x
        problem_type = "compute_product"

        # Generate question templates
        question_templates = self.template_engine.create_default_templates(problem_type, self.difficulty_level)

        return ProblemTemplate(
            expression=problem_expression,
            variables=[],
            sympy_solution=sympy_sol,
            lib_result=lib_result,
            question_templates=[t.template_string for t in question_templates],
            context_info={
                "matrix_dimensions": (rows, cols),
                "problem_type": problem_type,
                "matrix": matrix_A,
                "vector": vector_x,
                "entropy_used": context.entropy,
            },
            difficulty_markers={
                "matrix_complexity": matrix_entropy,
                "vector_complexity": vector_entropy,
                "dimension_size": max(rows, cols),
            },
            difficulty=self.difficulty_level,
        )

    def format_question(self, template: ProblemTemplate) -> str:
        """Formats the problem as a natural language question (the user's query)."""

        # Use the template engine for consistent question formatting
        matrix = template.context_info["matrix"]
        vector = template.context_info["vector"]

        # Get templates for matrix-vector multiplication
        templates = self.template_engine.create_default_templates("compute_product", self.difficulty_level)
        if templates:
            selected_template = self.template_engine.select_template(
                templates, "compute_product", self.difficulty_level
            )
            question_text = self.template_engine.generate_question(
                template=selected_template, variables={"matrix": matrix, "vector": vector}, precision=self.precision
            )
        else:
            raise ValueError("No templates available for matrix-vector multiplication")

        return question_text

    @override
    def format_solution(self, template: ProblemTemplate) -> str:
        """The solution string used as the ground truth in the final dataset entry."""
        solution = template.sympy_solution

        if not isinstance(solution, Matrix):
            raise TypeError(f"The solution should be a vector: {solution}")

        return self.template_engine.format_answer(solution, precision=self.precision)

    @override
    def verify_problem(self, template: ProblemTemplate) -> bool:
        """
        Verify the mathematical correctness using end-to-end math_verify verification.
        This is the single point where we ensure sympy and lib.py results match.
        """
        lib_result = template.lib_result
        ground_truth = self.math_formatter.sympy_to_primitive(template.sympy_solution, precision=self.precision)
        assert isinstance(ground_truth, LibTypes)  # noqa: S101

        if not verify_answers(ground_truth, lib_result):
            raise ValueError(f"Verification failed: sympy={ground_truth} vs lib={lib_result}")

        return True

    def _multiply_matrices_sympy(self, matrix_a: Matrix, matrix_b: Matrix) -> tuple[Matrix, list[list[float]]]:
        """Multiply two sympy matrices using lib.py function."""

        a_list = self.math_formatter.sympy_to_primitive(matrix_a)
        b_list = self.math_formatter.sympy_to_primitive(matrix_b)
        assert isinstance(a_list, list) and isinstance(b_list, list)  # noqa: S101

        lib_result = multiply_matrices(a_list, b_list)
        sympy_result = matrix_a * matrix_b

        return sympy_result, lib_result
