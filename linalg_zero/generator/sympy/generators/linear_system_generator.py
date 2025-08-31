from typing import Any

import sympy
from sympy.matrices import Matrix
from typing_extensions import override

from linalg_zero.generator.difficulty_config import (
    Precision,
    validate_tool_calls,
)
from linalg_zero.generator.entropy_control import SampleArgs
from linalg_zero.generator.generation_constraints import GenerationConstraints
from linalg_zero.generator.models import DifficultyCategory, Task
from linalg_zero.generator.sympy.base import ProblemContext, ProblemTemplate
from linalg_zero.generator.sympy.generators.base_generator import MatrixVectorBaseGenerator
from linalg_zero.generator.sympy.templates import MathFormatter
from linalg_zero.shared.lib import solve_linear_system


class LinearSystemGenerator(MatrixVectorBaseGenerator):
    """
    Generator for linear system solving problems (independent variant).

    Creates "Solve Ax = b for x" problems using backwards construction:
    generate matrix A and solution vector x first, then compute b = Ax.
    """

    def __init__(
        self,
        difficulty_level: DifficultyCategory,
        **kwargs: Any,
    ) -> None:
        """Initialize independent linear system solver generator.

        Args:
            difficulty_level: The difficulty category for the problem
            **kwargs: Additional keyword arguments
        """
        super().__init__(difficulty_level=difficulty_level, **kwargs)
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

        Independent variant: generate A and x, compute b = Ax, then ask to solve for x.
        """
        # Set constraint for matrix invertibility
        context.constraints["matrix_invertible"] = True

        matrix_entropy, vector_entropy = self._split_entropy(context)

        size = self._determine_size(context)

        matrix_A = self._generate_matrix_A(size, matrix_entropy, context)
        vector_b = self._generate_vector_b(matrix_A, size, vector_entropy, context)

        sympy_sol, lib_result = self._solve_linear_system_sympy(matrix_A, vector_b)

        # Record tool call with input data
        input_data = self._prepare_tool_call_input_data(matrix_a=matrix_A, vector_b=vector_b)
        context.record_tool_call(solve_linear_system.__name__, lib_result, input_data, is_final=True)

        # Create symbolic variables for rendering the equation
        x_symbols = sympy.Matrix([sympy.Symbol(f"x_{i + 1}") for i in range(size)])

        # Problem: "Solve Ax = b for x"
        problem_expression = sympy.Eq(matrix_A * x_symbols, vector_b)

        context_info = self._prepare_context_info(matrix_A, x_symbols, vector_b, size)

        question_templates = self._question_templates(context_info)

        return ProblemTemplate(
            expression=problem_expression,
            variables=[sympy.Symbol(f"x_{i + 1}") for i in range(size)],
            sympy_solution=sympy_sol,
            lib_result=lib_result,
            question_templates=question_templates,
            context_info=context_info,
            difficulty_markers=self.build_difficulty_markers(context),
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
    ) -> tuple[sympy.Matrix, list[list[float | int]]]:
        """Solve linear system Ax = b using lib.py function."""
        # Convert to primitives (this applies precision constraints)
        matrix_a_sympy = MathFormatter.sympy_to_primitive(matrix_a, precision=self.precision)
        vector_b_sympy = MathFormatter.sympy_to_primitive(vector_b, precision=self.precision)
        assert isinstance(matrix_a_sympy, list) and isinstance(vector_b_sympy, list)  # noqa: S101

        # Calculate using lib.py
        lib_result = solve_linear_system(matrix_a_sympy, vector_b_sympy)

        # This ensures there is no precision loss during verification
        matrix_a_precision_matched = Matrix(matrix_a_sympy)
        vector_b_precision_matched = Matrix(vector_b_sympy)
        sympy_result = matrix_a_precision_matched.LUsolve(vector_b_precision_matched)

        return sympy_result, lib_result

    def _split_entropy(self, context: ProblemContext) -> tuple[float, float]:
        """Split entropy between matrix and vector generation."""
        sample_args = SampleArgs(num_modules=2, entropy=context.entropy)
        matrix_sample_args, vector_sample_args = sample_args.split(count=2)
        return matrix_sample_args.entropy, vector_sample_args.entropy

    def _determine_size(self, context: ProblemContext) -> int:
        """Determine problem dimension (independent: random from config)."""
        return self.config.get_random_matrix_size()

    def _generate_matrix_A(
        self,
        size: int,
        matrix_entropy: float,
        context: ProblemContext,
    ) -> sympy.Matrix:
        # Use constraint-based generation for square invertible matrix
        # Temporarily set constraints for this specific call
        additional = GenerationConstraints(square=True, invertible=True, size=size, entropy=matrix_entropy)

        matrix_A = self._get_matrix_with_constraints(context, added_constraints=additional)
        return matrix_A

    def _generate_vector_b(
        self,
        matrix_A: sympy.Matrix,
        size: int,
        vector_entropy: float,
        context: ProblemContext,
    ) -> sympy.Matrix:
        solution_x = self._generate_vector(size, vector_entropy)
        context.record_entropy_usage(vector_entropy)
        return matrix_A * solution_x

    def _prepare_context_info(
        self,
        matrix_A: sympy.Matrix,
        x_symbols: sympy.Matrix,
        vector_b: sympy.Matrix,
        size: int,
    ) -> dict[str, Any]:
        return {
            "matrix_dimensions": (size, size),
            "problem_type": self.problem_type,
            "matrix_A": matrix_A,
            "x_symbols": x_symbols,
            "target_b": vector_b,
        }

    def _question_templates(self, context_info: dict[str, Any]) -> list[str] | None:
        question_templates = self.template_engine.create_default_templates(self.problem_type, self.difficulty_level)
        return [t.template_string for t in question_templates]


class LinearSystemGeneratorDependent(LinearSystemGenerator):
    """Dependent variant: uses provided b vector from previous component."""

    def __init__(
        self,
        difficulty_level: DifficultyCategory,
        input_vector_b: sympy.Matrix,
        input_vector_b_index: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(difficulty_level=difficulty_level, is_independent=False, **kwargs)
        assert self.problem_type == Task.LINEAR_SYSTEM_SOLVER  # noqa: S101
        self.input_vector_b = input_vector_b
        self.input_index = input_vector_b_index

    def _split_entropy(self, context: ProblemContext) -> tuple[float, float]:
        sample_args = SampleArgs(num_modules=1, entropy=context.entropy)
        return sample_args.entropy, 0.0

    def _determine_size(self, context: ProblemContext) -> int:
        return int(self.input_vector_b.shape[0])

    def _generate_vector_b(
        self,
        matrix_A: sympy.Matrix,
        size: int,
        vector_entropy: float,
        context: ProblemContext,
    ) -> sympy.Matrix:
        # No entropy usage for provided vector b
        return self.input_vector_b

    def _prepare_context_info(
        self,
        matrix_A: sympy.Matrix,
        x_symbols: sympy.Matrix,
        vector_b: sympy.Matrix,
        size: int,
    ) -> dict[str, Any]:
        context_info = super()._prepare_context_info(matrix_A, x_symbols, vector_b, size)
        context_info["input_variable_name"] = "b"
        context_info["input_indices"] = self.input_index
        return context_info

    def _question_templates(self, context_info: dict[str, Any]) -> list[str] | None:
        # Defer to composition-aware question formatting downstream
        return None

    def _prepare_tool_call_input_data(self, **kwargs: dict[str, Any]) -> dict[str, Any]:
        """Prepare input data for dependent generator including dependency info."""
        base_data = super()._prepare_tool_call_input_data(**kwargs)
        assert self.input_vector_b == kwargs["vector_b"]  # noqa: S101
        base_data.update({
            "dependent_on": {"input_vector_b": self.input_index},
            "input_vector_b": MathFormatter.sympy_to_primitive(self.input_vector_b, precision=self.precision),
        })
        return base_data

    @override
    def get_template_variables(self, template: ProblemTemplate) -> dict[str, Any]:
        base_vars = super().get_template_variables(template)

        # The indices start from 0, so we add 1 to make it more readable
        base_vars["target_b"] = f"the result from step {self.input_index + 1}"
        input_var = template.context_info["input_variable_name"]
        base_vars["input_variable_name"] = input_var
        return base_vars
