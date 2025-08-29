from typing import Any

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
    """

    def __init__(self, difficulty_level: DifficultyCategory, **kwargs: Any) -> None:
        """Initialize frobenius norm generator."""
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
        # Get matrix size and entropy
        matrix_rows = self.config.get_random_matrix_size()
        matrix_cols = self.config.get_random_matrix_size()

        sample_args = SampleArgs(num_modules=1, entropy=context.entropy)
        matrix_entropy = sample_args.entropy

        # Generate matrix A and find result
        entropy_controller = EntropyController(context.entropy)
        matrix_A = self._generate_matrix(matrix_rows, matrix_cols, matrix_entropy, entropy_controller)
        context.record_entropy_usage(matrix_entropy)

        sympy_norm, lib_result = self._calculate_frobenius_norm_sympy(matrix_A)
        context.record_tool_call(frobenius_norm.__name__, lib_result, is_final=True)

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
            difficulty_markers={
                "entropy_used": context.used_entropy,
                "matrix_size": (matrix_rows, matrix_cols),
                "target_tool_calls": self.config.target_tool_calls,
            },
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
