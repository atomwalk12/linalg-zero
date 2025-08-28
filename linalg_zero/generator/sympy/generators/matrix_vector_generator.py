from typing import Any

from sympy import Matrix
from typing_extensions import override

from linalg_zero.generator import Precision
from linalg_zero.generator.difficulty_config import (
    validate_tool_calls,
)
from linalg_zero.generator.entropy_control import EntropyController, SampleArgs
from linalg_zero.generator.models import DifficultyCategory, Task
from linalg_zero.generator.sympy.base import (
    ProblemContext,
    ProblemTemplate,
)
from linalg_zero.generator.sympy.generators.base_generator import MatrixVectorBaseGenerator
from linalg_zero.shared.lib import multiply_matrices


class MatrixVectorMultiplicationGenerator(MatrixVectorBaseGenerator):
    def __init__(self, entropy: float, difficulty_level: DifficultyCategory, **kwargs: Any) -> None:
        """Initialize matrix-vector multiplication generator."""
        super().__init__(entropy, difficulty_level, **kwargs)
        assert self.problem_type == Task.MATRIX_VECTOR_MULTIPLICATION  # noqa: S101

        # Validate that this problem type uses exactly 1 tool call
        validate_tool_calls(expected=self.config.target_tool_calls, actual=1, problem_type=self.problem_type)

    @property
    def precision(self) -> Precision:
        return Precision.MULTIPLY_MATRICES

    @override
    def generate_mathematical_content(self, context: ProblemContext) -> ProblemTemplate:
        """Generate matrix-vector multiplication problem content."""

        # Create SampleArgs for entropy distribution
        sample_args = SampleArgs(num_modules=2, entropy=context.entropy)

        # Split entropy between matrix and vector generation using Dirichlet distribution
        component_args = sample_args.split(count=2)
        matrix_sample_args, vector_sample_args = component_args

        matrix_entropy = matrix_sample_args.entropy
        vector_entropy = vector_sample_args.entropy

        # Get matrix dimensions from difficulty configuration
        rows = self.config.get_random_matrix_size()
        cols = self.config.get_random_matrix_size()

        # Generate matrix A and vector x
        entropy_controller = EntropyController(context.entropy)

        matrix_A = self._generate_matrix(rows, cols, matrix_entropy, entropy_controller)
        context.record_entropy_usage(matrix_entropy)

        vector_x = self._generate_vector(cols, vector_entropy, entropy_controller)
        context.record_entropy_usage(vector_entropy)
        sympy_sol, lib_result = self._multiply_matrices_sympy(matrix_A, vector_x)
        context.record_tool_call(multiply_matrices.__name__, lib_result, is_final=True)

        problem_expression = matrix_A * vector_x

        # Generate question templates
        question_templates = self.template_engine.create_default_templates(self.problem_type, self.difficulty_level)

        return ProblemTemplate(
            expression=problem_expression,
            variables=[],
            sympy_solution=sympy_sol,
            lib_result=lib_result,
            question_templates=[t.template_string for t in question_templates],
            context_info={
                "matrix_dimensions": (rows, cols),
                "problem_type": self.problem_type,
                "matrix": matrix_A,
                "vector": vector_x,
            },
            difficulty_markers={
                "entropy_used": context.used_entropy,
                "matrix_size": (rows, cols),
                "vector_size": cols,
                "target_tool_calls": self.config.target_tool_calls,
            },
            difficulty=self.difficulty_level,
        )

    @override
    def get_template_variables(self, template: ProblemTemplate) -> dict[str, Any]:
        """Return the variables dictionary to pass to the template engine."""
        matrix = template.context_info["matrix"]
        vector = template.context_info["vector"]
        return {"matrix": matrix, "vector": vector}

    def _multiply_matrices_sympy(self, matrix_a: Matrix, matrix_b: Matrix) -> tuple[Matrix, list[list[float]]]:
        """Multiply two sympy matrices using lib.py function."""

        a_list = self.formatter.sympy_to_primitive(matrix_a, precision=self.precision)
        b_list = self.formatter.sympy_to_primitive(matrix_b, precision=self.precision)
        assert isinstance(a_list, list) and isinstance(b_list, list)  # noqa: S101

        lib_result = multiply_matrices(a_list, b_list)
        sympy_result = matrix_a * matrix_b

        return sympy_result, lib_result
