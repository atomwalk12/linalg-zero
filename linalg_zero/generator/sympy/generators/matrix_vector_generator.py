import random
from collections.abc import Callable
from typing import Any

from sympy import Integer, Matrix
from typing_extensions import override

from linalg_zero.generator.models import Question
from linalg_zero.generator.sympy.base import (
    ProblemContext,
    ProblemTemplate,
    SympyProblemGenerator,
    create_sympy_factory,
)
from linalg_zero.generator.sympy.entropy import EntropyController
from linalg_zero.generator.sympy.templates import MathFormatter, TemplateEngine
from linalg_zero.generator.utils.difficulty import DifficultyCategory
from linalg_zero.grpo.verify import verify_answers
from linalg_zero.shared.lib import multiply_matrices


class MatrixVectorBaseGenerator(SympyProblemGenerator):
    """Base class for matrix-vector problem generators."""

    def __init__(self, entropy: float, difficulty_level: DifficultyCategory, **kwargs: Any) -> None:
        super().__init__(entropy, difficulty_level, **kwargs)

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
                if entropy_per_element < 1.5:
                    # Mostly integers for small entropy
                    if random.random() < 0.9:
                        element = controller.generate_integer(entropy_per_element, signed=True)
                    else:
                        element = controller.generate_rational(entropy_per_element, signed=True)
                else:
                    # Allow more rational numbers for higher entropy
                    if random.random() < 0.7:
                        element = controller.generate_integer(entropy_per_element, signed=True)
                    else:
                        element = controller.generate_rational(entropy_per_element, signed=True)

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
            if entropy_per_element < 1.5:
                # Mostly integers for small entropy
                if random.random() < 0.9:
                    element = controller.generate_integer(entropy_per_element, signed=True)
                else:
                    element = controller.generate_rational(entropy_per_element, signed=True)
            else:
                # Allow more rational numbers for higher entropy
                if random.random() < 0.8:
                    element = controller.generate_integer(entropy_per_element, signed=True)
                else:
                    element = controller.generate_rational(entropy_per_element, signed=True)

            # Avoid all-zero vectors
            if i == 0 and element == 0:
                element = Integer(1)

            vector_elements.append(element)

        return Matrix(vector_elements)


class MatrixVectorMultiplicationGenerator(MatrixVectorBaseGenerator):
    def __init__(self, entropy: float, difficulty_level: DifficultyCategory, **kwargs: Any) -> None:
        """Initialize matrix-vector multiplication generator."""
        super().__init__(entropy, difficulty_level, **kwargs)
        self.precision = -1
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
        sympy_solution = template.sympy_solution

        sympy_primitive = self.math_formatter.sympy_to_primitive(sympy_solution, precision=self.precision)

        # Use the same verification function for both generation and training
        if not verify_answers(str(sympy_primitive), str(lib_result)):
            raise ValueError(f"Verification failed: sympy={sympy_primitive} vs lib={lib_result}")

        return True

    def _multiply_matrices_sympy(self, matrix_a: Matrix, matrix_b: Matrix) -> tuple[Matrix, list[list[float]]]:
        """Multiply two sympy matrices using lib.py function."""
        # Prepare library input
        a_list = self.math_formatter.sympy_to_primitive(matrix_a)
        b_list = self.math_formatter.sympy_to_primitive(matrix_b)
        assert isinstance(a_list, list) and isinstance(b_list, list)  # noqa: S101

        lib_result = multiply_matrices(a_list, b_list, precision=self.precision)

        # Compute sympy reference with same precision
        sympy_result = matrix_a * matrix_b

        return sympy_result, lib_result


def create_matrix_vector_multiplication_factory(
    entropy: float, difficulty: DifficultyCategory
) -> Callable[[], Question]:
    """Helper to create matrix-vector multiplication factory with specific parameters."""
    return create_sympy_factory(
        MatrixVectorMultiplicationGenerator,
        entropy=entropy,
        difficulty_level=difficulty,
        problem_type="matrix_vector_multiplication",
        topic="linear_algebra",
    )
