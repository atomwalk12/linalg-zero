from typing import Any

import sympy
from sympy import Matrix
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
from linalg_zero.shared.lib import frobenius_norm


class FrobeniusNormGenerator(MatrixVectorBaseGenerator):
    """
    This generator creates problems asking to compute the Frobenius norm of a matrix.
    Independent variant: always generates its own input matrix.
    """

    def __init__(
        self,
        difficulty_level: DifficultyCategory,
        **kwargs: Any,
    ) -> None:
        """Initialize independent frobenius norm generator.

        Args:
            difficulty_level: The difficulty category for the problem
            **kwargs: Additional keyword arguments
        """
        super().__init__(difficulty_level=difficulty_level, **kwargs)
        assert self.problem_type == Task.FROBENIUS_NORM  # noqa: S101

        validate_tool_calls(expected=self.config.target_tool_calls, actual=1, problem_type=self.problem_type)

    @property
    def precision(self) -> Precision:
        """The precision of the problem."""
        return Precision.FROBENIUS_NORM

    @override
    def generate_mathematical_content(self, context: ProblemContext) -> ProblemTemplate:
        """
        Generate Frobenius norm calculation problem content.

        This method creates problems asking to compute ||A||_F for a matrix A.
        Uses difficulty configuration to determine matrix size and complexity.
        """
        matrix_A = self._get_matrix(context)

        sympy_norm, lib_result = self._calculate_frobenius_norm_sympy(matrix_A)

        # Record tool call with input data
        input_data = self._prepare_tool_call_input_data(matrix=matrix_A)
        context.record_tool_call(frobenius_norm.__name__, lib_result, input_data, is_final=True)

        # Generate question templates
        problem_expression = matrix_A

        question_templates = self.template_engine.create_default_templates(self.problem_type, self.difficulty_level)

        return ProblemTemplate(
            expression=problem_expression,
            variables=[],
            sympy_solution=sympy_norm,
            lib_result=lib_result,
            question_templates=[t.template_string for t in question_templates],
            context_info={
                "matrix": matrix_A,
            },
            difficulty_markers=self.build_difficulty_markers(context, matrix_size=(matrix_A.rows, matrix_A.cols)),
            difficulty=self.difficulty_level,
        )

    @override
    def get_template_variables(self, template: ProblemTemplate) -> dict[str, Any]:
        """Return the variables dictionary to pass to the template engine."""
        matrix = template.context_info["matrix"]
        return {"matrix": matrix}

    def _calculate_frobenius_norm_sympy(self, matrix_a: Matrix) -> tuple[Any, float]:
        """Calculate Frobenius norm using both SymPy and lib.py function."""
        # Convert to primitives (this applies precision constraints)
        matrix_a_primitive = MathFormatter.sympy_to_primitive(matrix_a, precision=self.precision)
        assert isinstance(matrix_a_primitive, list)  # noqa: S101

        # Calculate using lib.py with the primitives
        lib_result = frobenius_norm(matrix_a_primitive)

        # Convert primitives back to SymPy Matrix at the same precision level
        # This ensures both calculations work with the same precision
        matrix_a_precision_matched = Matrix(matrix_a_primitive)
        sympy_result = matrix_a_precision_matched.norm()

        return sympy_result, lib_result

    def _get_matrix(self, context: ProblemContext) -> Matrix:
        """Generate and return a matrix, recording entropy usage.

        Independent variant: selects size, consumes entropy, and generates the matrix.
        """
        # Get matrix size and entropy
        matrix_rows = self.config.get_random_matrix_size()
        matrix_cols = self.config.get_random_matrix_size()

        sample_args = SampleArgs(num_modules=1, entropy=context.entropy)
        matrix_entropy = sample_args.entropy

        # Generate matrix A and record entropy usage
        entropy_controller = EntropyController(context.entropy)
        matrix_A = self._generate_matrix(matrix_rows, matrix_cols, matrix_entropy, entropy_controller)
        context.record_entropy_usage(matrix_entropy)
        return matrix_A


class FrobeniusNormGeneratorDependent(FrobeniusNormGenerator):
    """Dependent variant: consumes a provided input matrix and does not use entropy."""

    def __init__(
        self,
        difficulty_level: DifficultyCategory,
        input_matrix: sympy.Matrix,
        input_index: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(difficulty_level=difficulty_level, **kwargs)
        assert self.problem_type == Task.FROBENIUS_NORM  # noqa: S101
        self.input_matrix = input_matrix
        self.input_index = input_index

    def _get_matrix(self, context: ProblemContext) -> Matrix:
        # No entropy usage for provided matrix
        return self.input_matrix

    def _prepare_tool_call_input_data(self, **kwargs: dict[str, Any]) -> dict[str, Any]:
        """Prepare input data for dependent generator including dependency info."""
        base_data = super()._prepare_tool_call_input_data(**kwargs)
        assert self.input_matrix == kwargs["matrix"]  # noqa: S101
        base_data.update({
            "dependent_on": self.input_index,
            "input_matrix": MathFormatter.sympy_to_primitive(self.input_matrix, precision=self.precision),
        })
        return base_data
