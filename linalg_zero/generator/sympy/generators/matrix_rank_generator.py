from typing import Any

from sympy import Integer, Matrix
from typing_extensions import override

from linalg_zero.generator.difficulty_config import (
    Precision,
    validate_tool_calls,
)
from linalg_zero.generator.entropy_control import EntropyController, SampleArgs
from linalg_zero.generator.models import DifficultyCategory, Task
from linalg_zero.generator.sympy.base import ProblemContext, ProblemTemplate
from linalg_zero.generator.sympy.generators.base_generator import MatrixVectorBaseGenerator
from linalg_zero.shared.lib import matrix_rank


class MatrixRankGenerator(MatrixVectorBaseGenerator):
    """
    This generator creates problems asking to compute the rank of a matrix.
    """

    def __init__(self, difficulty_level: DifficultyCategory, **kwargs: Any) -> None:
        """Initialize matrix rank generator."""
        super().__init__(difficulty_level=difficulty_level, **kwargs)
        assert self.problem_type == Task.MATRIX_RANK  # noqa: S101

        validate_tool_calls(expected=self.config.target_tool_calls, actual=1, problem_type=self.problem_type)

    @property
    def precision(self) -> Precision:
        """The precision of the problem."""
        return Precision.MATRIX_RANK

    @override
    def generate_mathematical_content(self, context: ProblemContext) -> ProblemTemplate:
        """
        Generate matrix rank calculation problem content.

        This method creates problems asking to compute rank(A) for a matrix A.
        Uses difficulty configuration to determine matrix size and complexity.
        """
        # Get matrix dimensions and entropy
        rows = self.config.get_random_matrix_size()
        cols = self.config.get_random_matrix_size()

        sample_args = SampleArgs(num_modules=1, entropy=context.entropy)
        matrix_entropy = sample_args.entropy

        # Generate matrix A and find result
        entropy_controller = EntropyController(context.entropy)
        matrix_A = self._generate_matrix(rows, cols, matrix_entropy, entropy_controller)
        context.record_entropy_usage(matrix_entropy)

        sympy_rank, lib_result = self._calculate_rank_sympy(matrix_A)

        # Record tool call with input data
        input_data = self._prepare_tool_call_input_data(matrix=matrix_A)
        context.record_tool_call(matrix_rank.__name__, lib_result, input_data, is_final=True)

        # Generate question templates
        problem_expression = matrix_A

        question_templates = self.template_engine.create_default_templates(self.problem_type, self.difficulty_level)

        return ProblemTemplate(
            expression=problem_expression,
            variables=[],
            sympy_solution=sympy_rank,
            lib_result=lib_result,
            question_templates=[t.template_string for t in question_templates],
            context_info={
                "matrix": matrix_A,
            },
            difficulty_markers=self.build_difficulty_markers(context),
            difficulty=self.difficulty_level,
        )

    @override
    def get_template_variables(self, template: ProblemTemplate) -> dict[str, Any]:
        """Return the variables dictionary to pass to the template engine."""
        matrix = template.context_info["matrix"]
        return {"matrix": matrix}

    def _calculate_rank_sympy(self, matrix_a: Matrix) -> tuple[Integer, int]:
        """Calculate matrix rank using both SymPy and lib.py function."""
        # Convert to primitives for lib.py calculation
        from linalg_zero.generator.sympy.templates import MathFormatter

        matrix_a_primitive = MathFormatter.sympy_to_primitive(matrix_a, precision=self.precision)
        assert isinstance(matrix_a_primitive, list)  # noqa: S101

        # Calculate using lib.py with the primitives
        lib_result = matrix_rank(matrix_a_primitive)

        # Calculate using SymPy
        sympy_result = matrix_a.rank()
        assert isinstance(sympy_result, int)  # noqa: S101

        return Integer(sympy_result), lib_result


class MatrixRankGeneratorDependent(MatrixRankGenerator):
    """Dependent variant: reports dependency index in difficulty markers."""

    def __init__(self, difficulty_level: DifficultyCategory, input_index: int, **kwargs: Any) -> None:
        super().__init__(difficulty_level=difficulty_level, **kwargs)
        assert self.problem_type == Task.MATRIX_RANK  # noqa: S101
        self.input_index = input_index

    def _prepare_tool_call_input_data(self, **kwargs: dict[str, Any]) -> dict[str, Any]:
        """Prepare input data for dependent generator including dependency info."""
        base_data = super()._prepare_tool_call_input_data(**kwargs)
        base_data.update({"dependent_on": self.input_index})
        return base_data
