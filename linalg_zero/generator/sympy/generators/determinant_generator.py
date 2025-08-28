from typing import Any

from sympy import Float, Integer, Matrix, Rational
from typing_extensions import override

from linalg_zero.generator import Precision
from linalg_zero.generator.difficulty_config import (
    validate_tool_calls,
)
from linalg_zero.generator.entropy_control import EntropyController, SampleArgs
from linalg_zero.generator.models import DifficultyCategory, Task
from linalg_zero.generator.sympy.base import ProblemContext, ProblemTemplate
from linalg_zero.generator.sympy.generators.base_generator import MatrixVectorBaseGenerator
from linalg_zero.generator.sympy.templates import MathFormatter
from linalg_zero.shared.lib import determinant


class DeterminantGenerator(MatrixVectorBaseGenerator):
    """
    This generator creates problems asking to compute the determinant of a square matrix.
    """

    def __init__(self, entropy: float, difficulty_level: DifficultyCategory, **kwargs: Any) -> None:
        """Initialize determinant generator."""
        super().__init__(entropy, difficulty_level, **kwargs)
        self.precision = Precision.DETERMINANT
        assert self.problem_type == Task.DETERMINANT  # noqa: S101

        validate_tool_calls(expected=self.config.target_tool_calls, actual=1, problem_type=self.problem_type)

    @override
    def generate_mathematical_content(self, context: ProblemContext) -> ProblemTemplate:
        """
        Generate determinant calculation problem content.

        This method creates problems asking to compute det(A) for a square matrix A.
        Uses difficulty configuration to determine matrix size and complexity.
        """
        # Get matrix size and entropy
        size = self.config.get_random_matrix_size()

        sample_args = SampleArgs(num_modules=1, entropy=context.entropy)
        matrix_entropy = sample_args.entropy

        # Generate square matrix A and find result
        entropy_controller = EntropyController(context.entropy)
        matrix_A = self._generate_matrix(size, size, matrix_entropy, entropy_controller)
        context.record_entropy_usage(matrix_entropy)

        sympy_det, lib_result = self._calculate_determinant_sympy(matrix_A)
        context.record_tool_call(self.problem_type, lib_result, is_final=True)

        # Generate question templates
        problem_expression = matrix_A

        question_templates = self.template_engine.create_default_templates(self.problem_type, self.difficulty_level)

        return ProblemTemplate(
            expression=problem_expression,
            variables=[],
            sympy_solution=sympy_det,
            lib_result=lib_result,
            question_templates=[t.template_string for t in question_templates],
            context_info={
                "matrix": matrix_A,
            },
            difficulty_markers={
                "entropy_used": context.used_entropy,
                "matrix_size": (size, size),
                "target_tool_calls": self.config.target_tool_calls,
            },
            difficulty=self.difficulty_level,
        )

    @override
    def get_template_variables(self, template: ProblemTemplate) -> dict[str, Any]:
        """Return the variables dictionary to pass to the template engine."""
        matrix = template.context_info["matrix"]
        return {"matrix": matrix}

    def _calculate_determinant_sympy(self, matrix_a: Matrix) -> tuple[Float | Integer | Rational, float]:
        """Calculate determinant using both SymPy and lib.py function."""
        matrix_a_primitive = MathFormatter.sympy_to_primitive(matrix_a, precision=self.precision)
        assert isinstance(matrix_a_primitive, list)  # noqa: S101

        lib_result = determinant(matrix_a_primitive)

        sympy_result = matrix_a.det()
        assert isinstance(sympy_result, (Float, Integer, Rational))  # noqa: S101

        return sympy_result, lib_result
