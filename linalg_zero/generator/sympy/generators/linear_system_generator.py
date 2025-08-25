import random
from collections.abc import Callable
from typing import Any

import sympy
from typing_extensions import override

from linalg_zero.generator import Precision
from linalg_zero.generator.models import Question
from linalg_zero.generator.sympy.base import ProblemContext, ProblemTemplate, create_sympy_factory
from linalg_zero.generator.sympy.entropy import EntropyController
from linalg_zero.generator.sympy.generators.matrix_vector_generator import MatrixVectorBaseGenerator
from linalg_zero.generator.sympy.templates import MathFormatter
from linalg_zero.generator.utils.difficulty import DifficultyCategory
from linalg_zero.grpo.verify import verify_answers
from linalg_zero.shared.lib import solve_linear_system
from linalg_zero.shared.types import LibTypes


class LinearSystemGenerator(MatrixVectorBaseGenerator):
    """
    Generator for inverse matrix-vector equation solving problems.

    This generator creates "Solve Ax = b for x" problems using backwards construction:
    generate matrix A and solution vector x first, then compute b = Ax, and present
    the equation Ax = b asking to solve for x.

    Mathematical Process:
        1. Generate matrix A and solution vector x using entropy allocation
        2. Compute b = A * x (backwards construction gives us target)
        3. Present problem as "Solve Ax = b for x"
        4. Verify that A * (solution_x) = b using math_verify
    """

    def __init__(self, entropy: float, difficulty_level: DifficultyCategory, **kwargs: Any) -> None:
        """Initialize inverse matrix-vector equation solver generator."""
        super().__init__(entropy, difficulty_level, integers_only=True, **kwargs)
        self.precision = Precision.SOLVE_LINEAR_SYSTEM
        self.math_formatter = MathFormatter()

    def _generate_invertible_matrix(
        self, rows: int, cols: int, entropy: float, controller: EntropyController, max_attempts: int = 10
    ) -> sympy.Matrix:
        """Generate an invertible matrix with retry logic."""
        for _ in range(max_attempts):
            matrix_A = self._generate_matrix(rows, cols, entropy, controller)

            try:
                det = matrix_A.det()
                if det != 0:
                    return matrix_A
            except Exception:  # noqa: S112
                continue

        raise ValueError("Could not generate invertible matrix after maximum attempts")

    def generate_mathematical_content(self, context: ProblemContext) -> ProblemTemplate:
        """
        Generate inverse matrix-vector equation solving problem content.

        This method creates "Solve Ax = b for x" problems using backwards construction:
        generate A and x, compute b = Ax, then ask to solve for x.
        """
        # Set constraint for matrix invertibility
        context.constraints["matrix_invertible"] = True

        entropy_controller = EntropyController(context.entropy)

        # Allocate entropy between matrix and vector generation
        # Use a more conservative split to ensure reasonable numbers
        matrix_entropy = context.entropy * 1.0  # Full entropy for matrix
        vector_entropy = context.entropy * 0.9  # Slightly less for vector

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

        # Generate matrix A
        matrix_A = self._generate_invertible_matrix(rows, cols, matrix_entropy, entropy_controller)
        context.record_entropy_usage(matrix_entropy)

        # Generate solution vector x (this is what we want them to find)
        solution_x = self._generate_vector(cols, vector_entropy, entropy_controller)
        context.record_entropy_usage(vector_entropy)

        # Compute b = A*x (backwards construction - this becomes our target)
        vector_b = matrix_A * solution_x

        sympy_sol, lib_result = self._solve_linear_system_sympy(matrix_A, vector_b)
        context.record_tool_call("solve_linear_system", lib_result, is_final=True)

        # Create symbolic variables for rendering the equation
        x_symbols = sympy.Matrix([sympy.Symbol(f"x_{i + 1}") for i in range(cols)])

        # Inverse problem: "Solve Ax = b for x"
        problem_expression = sympy.Eq(matrix_A * x_symbols, vector_b)
        problem_type = "solve_for_vector"

        question_templates = self.template_engine.create_default_templates("solve", self.difficulty_level)

        return ProblemTemplate(
            expression=problem_expression,
            variables=[sympy.Symbol(f"x_{i + 1}") for i in range(cols)],
            sympy_solution=sympy_sol,
            lib_result=lib_result,
            question_templates=[t.template_string for t in question_templates],
            context_info={
                "matrix_dimensions": (rows, cols),
                "problem_type": problem_type,
                "entropy_used": context.entropy,
                "matrix_A": matrix_A,
                "x_symbols": x_symbols,
                "target_b": vector_b,
            },
            difficulty_markers={
                "matrix_complexity": matrix_entropy,
                "vector_complexity": vector_entropy,
                "dimension_size": max(rows, cols),
            },
            difficulty=self.difficulty_level,
        )

    def format_question(self, template: ProblemTemplate) -> str:
        """Format matrix-vector equation solving problem as natural language question."""
        # Use the template engine for consistent question formatting
        matrix_a = template.context_info["matrix_A"]
        target_b = template.context_info["target_b"]
        x_symbols = template.context_info["x_symbols"]

        # Get templates for solving equations
        templates = self.template_engine.create_default_templates("solve", self.difficulty_level)
        if templates:
            selected_template = self.template_engine.select_template(templates, "solve", self.difficulty_level)
            question_text = self.template_engine.generate_question(
                template=selected_template,
                variables={"matrix": matrix_a, "x_symbols": x_symbols, "target_b": target_b},
                precision=self.precision,
            )
        else:
            raise ValueError("No templates available for linear system solving")

        return question_text

    @override
    def format_solution(self, template: ProblemTemplate) -> str:
        """The solution string used as the ground truth in the final dataset entry."""
        solution = template.sympy_solution

        if not isinstance(solution, sympy.Matrix):
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

        ground_truth = self.math_formatter.sympy_to_primitive(sympy_solution, precision=self.precision)
        assert isinstance(ground_truth, LibTypes)  # noqa: S101

        # Use the same verification function for both generation and training
        if not verify_answers(ground_truth, lib_result):
            raise ValueError(f"Verification failed: sympy={ground_truth} vs lib={lib_result}")

        return True

    def _solve_linear_system_sympy(
        self, matrix_a: sympy.Matrix, vector_b: sympy.Matrix
    ) -> tuple[sympy.Matrix, list[float | int]]:
        """Solve linear system Ax = b using lib.py function."""
        matrix_a_sympy = MathFormatter.sympy_to_primitive(matrix_a, precision=self.precision)
        vector_b_sympy = MathFormatter.sympy_to_primitive(vector_b, precision=self.precision)
        assert isinstance(matrix_a_sympy, list) and isinstance(vector_b_sympy, list)  # noqa: S101

        # Prepare library input
        lib_result = solve_linear_system(matrix_a_sympy, vector_b_sympy)

        # Compute sympy reference solution
        sympy_result = matrix_a.LUsolve(vector_b)

        return sympy_result, lib_result


def create_matrix_vector_equation_solver_factory(
    entropy: float, difficulty: DifficultyCategory
) -> Callable[[], Question]:
    """Helper to create matrix-vector equation solver factory with specific parameters."""
    return create_sympy_factory(
        LinearSystemGenerator,
        entropy=entropy,
        difficulty_level=difficulty,
        problem_type="matrix_vector_equation_solving",
        topic="linear_algebra",
    )
