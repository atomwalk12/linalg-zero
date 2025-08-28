from typing import Any

import sympy
from typing_extensions import override

from linalg_zero.generator.difficulty_config import (
    Precision,
    validate_tool_calls,
)
from linalg_zero.generator.entropy_control import EntropyController, SampleArgs
from linalg_zero.generator.models import DifficultyCategory, Task
from linalg_zero.generator.sympy.base import ProblemContext, ProblemTemplate
from linalg_zero.generator.sympy.generators.base_generator import MatrixVectorBaseGenerator
from linalg_zero.generator.sympy.templates import MathFormatter
from linalg_zero.shared.lib import solve_linear_system


class LinearSystemGenerator(MatrixVectorBaseGenerator):
    """
    Generator for linear system solving problems.

    This generator creates "Solve Ax = b for x" problems using backwards construction:
    generate matrix A and solution vector x first, then compute b = Ax, and present
    the equation Ax = b asking to solve for x.
    """

    def __init__(self, entropy: float, difficulty_level: DifficultyCategory, **kwargs: Any) -> None:
        """Initialize linear system solver generator."""
        super().__init__(entropy, difficulty_level, **kwargs)
        assert self.problem_type == Task.LINEAR_SYSTEM_SOLVER  # noqa: S101

        # Validate that this problem type uses exactly 1 tool call
        validate_tool_calls(expected=self.config.target_tool_calls, actual=1, problem_type=self.problem_type)

    @property
    def precision(self) -> Precision:
        """The precision of the problem."""
        return Precision.LINEAR_SYSTEM_SOLVER

    @override
    def generate_mathematical_content(self, context: ProblemContext) -> ProblemTemplate:
        """
        Generate linear system solving problem content.

        This method creates "Solve Ax = b for x" problems using backwards construction:
        generate A and x, compute b = Ax, then ask to solve for x.
        """
        # Set constraint for matrix invertibility
        context.constraints["matrix_invertible"] = True

        # Split entropy between matrix and vector generation
        sample_args = SampleArgs(num_modules=2, entropy=context.entropy)

        component_args = sample_args.split(count=2)
        matrix_sample_args, vector_sample_args = component_args

        matrix_entropy = matrix_sample_args.entropy
        vector_entropy = vector_sample_args.entropy

        entropy_controller = EntropyController(context.entropy)

        # Generate problem dimension, invertible matrix and solution vector
        size = self.config.get_random_matrix_size()
        matrix_A = self._generate_invertible_matrix(size, matrix_entropy, entropy_controller)
        context.record_entropy_usage(matrix_entropy)

        solution_x = self._generate_vector(size, vector_entropy, entropy_controller)
        context.record_entropy_usage(vector_entropy)

        # Compute b = A*x (backwards construction - this becomes our target)
        vector_b = matrix_A * solution_x

        sympy_sol, lib_result = self._solve_linear_system_sympy(matrix_A, vector_b)
        context.record_tool_call(solve_linear_system.__name__, lib_result, is_final=True)

        # Create symbolic variables for rendering the equation
        x_symbols = sympy.Matrix([sympy.Symbol(f"x_{i + 1}") for i in range(size)])

        # Problem: "Solve Ax = b for x"
        problem_expression = sympy.Eq(matrix_A * x_symbols, vector_b)

        question_templates = self.template_engine.create_default_templates(self.problem_type, self.difficulty_level)

        return ProblemTemplate(
            expression=problem_expression,
            variables=[sympy.Symbol(f"x_{i + 1}") for i in range(size)],
            sympy_solution=sympy_sol,
            lib_result=lib_result,
            question_templates=[t.template_string for t in question_templates],
            context_info={
                "matrix_dimensions": (size, size),
                "problem_type": self.problem_type,
                "matrix_A": matrix_A,
                "x_symbols": x_symbols,
                "target_b": vector_b,
            },
            difficulty_markers={
                "entropy_used": context.used_entropy,
                "matrix_size": (size, size),
                "vector_size": size,
                "tool_calls_used": 1,
            },
            difficulty=self.difficulty_level,
        )

    @override
    def get_template_variables(self, template: ProblemTemplate) -> dict[str, Any]:
        """Return the variables dictionary to pass to the template engine."""
        matrix_a = template.context_info["matrix_A"]
        target_b = template.context_info["target_b"]
        x_symbols = template.context_info["x_symbols"]
        return {"matrix": matrix_a, "x_symbols": x_symbols, "target_b": target_b}

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
