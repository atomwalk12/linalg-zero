from typing import Any

from sympy import Matrix
from typing_extensions import override

from linalg_zero.generator.difficulty_config import (
    Precision,
    validate_tool_calls,
)
from linalg_zero.generator.generation_constraints import GenerationConstraints
from linalg_zero.generator.models import DifficultyCategory, Task
from linalg_zero.generator.sympy.base import ProblemContext, ProblemTemplate
from linalg_zero.generator.sympy.generators.base_generator import MatrixVectorBaseGenerator
from linalg_zero.generator.sympy.templates import MathFormatter
from linalg_zero.shared.lib import matrix_cofactor


class MatrixCofactorGenerator(MatrixVectorBaseGenerator):
    """
    This generator creates problems asking to compute the cofactor matrix of a square matrix.
    """

    def __init__(self, difficulty_level: DifficultyCategory, **kwargs: Any) -> None:
        """Initialize matrix cofactor generator."""
        super().__init__(difficulty_level=difficulty_level, **kwargs)
        assert self.problem_type == Task.MATRIX_COFACTOR  # noqa: S101

        validate_tool_calls(expected=self.config.target_tool_calls, actual=1, problem_type=self.problem_type)

    @property
    def precision(self) -> Precision:
        """The precision of the problem."""
        return Precision.MATRIX_COFACTOR

    def _get_matrix(self, context: ProblemContext) -> Matrix:
        """Generate or retrieve the matrix for cofactor calculation."""
        return self._get_matrix_with_constraints(context, added_constraints=GenerationConstraints(square=True))

    @override
    def generate_mathematical_content(self, context: ProblemContext) -> ProblemTemplate:
        """
        Generate matrix cofactor calculation problem content.

        This method creates problems asking to compute the cofactor matrix for a square matrix A.
        Uses difficulty configuration to determine matrix size and complexity.
        """
        # Get matrix using overrideable method
        matrix_A = self._get_matrix(context)

        sympy_cofactor, lib_result = self._calculate_cofactor_sympy(matrix_A)

        # Record tool call with input data
        input_data = self._prepare_tool_call_input_data(matrix=matrix_A)
        context.record_tool_call(matrix_cofactor.__name__, lib_result, input_data, is_final=True)

        # Generate question templates
        problem_expression = matrix_A

        question_templates = self.template_engine.create_default_templates(self.problem_type, self.difficulty_level)

        return ProblemTemplate(
            expression=problem_expression,
            variables={"matrix": matrix_A},
            sympy_solution=sympy_cofactor,
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

    def _calculate_cofactor_sympy(self, matrix_a: Matrix) -> tuple[Matrix, list[list[float | int]]]:
        """Calculate matrix cofactor using both SymPy and lib.py function."""
        # Convert to primitives for lib.py calculation
        matrix_a_primitive = MathFormatter.sympy_to_primitive(matrix_a, precision=self.precision)
        assert isinstance(matrix_a_primitive, list)  # noqa: S101

        # Calculate using lib.py with the primitives
        lib_result = matrix_cofactor(matrix_a_primitive)

        # Convert primitives back to SymPy Matrix at the same precision level
        # This ensures both calculations work with the same precision
        matrix_a_precision_matched = Matrix(matrix_a_primitive)
        sympy_result = matrix_a_precision_matched.cofactor_matrix()

        return sympy_result, lib_result


class MatrixCofactorGeneratorDependent(MatrixCofactorGenerator):
    """Dependent variant: uses provided input matrix and reports dependency index in difficulty markers."""

    def __init__(
        self,
        difficulty_level: DifficultyCategory,
        input_matrix: Matrix,
        input_matrix_index: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(difficulty_level=difficulty_level, **kwargs)
        assert self.problem_type == Task.MATRIX_COFACTOR  # noqa: S101
        self.input_matrix = input_matrix
        self.input_index = input_matrix_index

    def _get_matrix(self, context: ProblemContext) -> Matrix:
        """Return the provided input matrix without consuming entropy."""
        return self.input_matrix

    def _prepare_tool_call_input_data(self, **kwargs: dict[str, Any]) -> dict[str, Any]:
        """Prepare input data for dependent generator including dependency info."""
        base_data = super()._prepare_tool_call_input_data(**kwargs)
        assert self.input_matrix == kwargs["matrix"]  # noqa: S101
        base_data.update({
            "dependent_on": {"input_matrix": self.input_index},
            "input_matrix": MathFormatter.sympy_to_primitive(self.input_matrix, precision=self.precision),
        })
        return base_data

    @override
    def get_template_variables(self, template: ProblemTemplate) -> dict[str, Any]:
        """Return template variables for dependent matrix cofactor generator."""
        return {"matrix": f"step {self.input_index + 1}"}
