from typing import Any

import sympy
from sympy import Matrix
from typing_extensions import override

from linalg_zero.generator.difficulty_config import (
    Precision,
    validate_tool_calls,
)
from linalg_zero.generator.entropy_control import SampleArgs
from linalg_zero.generator.generation_constraints import GenerationConstraints
from linalg_zero.generator.models import DifficultyCategory, Task
from linalg_zero.generator.sympy.base import (
    ProblemContext,
    ProblemTemplate,
)
from linalg_zero.generator.sympy.generators.base_generator import MatrixVectorBaseGenerator
from linalg_zero.generator.sympy.templates import MathFormatter
from linalg_zero.shared.lib import multiply_matrices


class MatrixVectorMultiplicationGenerator(MatrixVectorBaseGenerator):
    def __init__(
        self,
        difficulty_level: DifficultyCategory,
        **kwargs: Any,
    ) -> None:
        """Initialize independent matrix-vector multiplication generator."""
        super().__init__(difficulty_level=difficulty_level, **kwargs)
        assert self.problem_type == Task.MATRIX_VECTOR_MULTIPLICATION  # noqa: S101

        # Validate that this problem type uses exactly 1 tool call
        validate_tool_calls(expected=self.config.target_tool_calls, actual=1, problem_type=self.problem_type)

    @property
    def precision(self) -> Precision:
        return Precision.MATRIX_VECTOR_MULTIPLICATION

    @override
    def generate_mathematical_content(self, context: ProblemContext) -> ProblemTemplate:
        """Generate matrix-vector multiplication problem content (independent)."""

        matrix_entropy, vector_entropy = self._split_entropy(context)
        rows, cols = self._determine_dimensions(context)

        matrix_A = self._generate_matrix_A(rows, cols, matrix_entropy, context)
        vector_b = self._generate_vector_b(cols, vector_entropy, context)
        sympy_sol, lib_result = self._multiply_matrices_sympy(matrix_a=matrix_A, vector_b=vector_b)

        # Record tool call with input data
        input_data = self._prepare_tool_call_input_data(matrix_a=matrix_A, vector_b=vector_b)
        context.record_tool_call(multiply_matrices.__name__, lib_result, input_data, is_final=True)

        problem_expression = matrix_A * vector_b

        context_info = {
            "matrix_dimensions": (rows, cols),
            "problem_type": self.problem_type,
            "matrix": matrix_A,
            "vector": vector_b,
        }

        question_templates = self._question_templates(context_info)

        return ProblemTemplate(
            expression=problem_expression,
            variables={"matrix": matrix_A, "vector": vector_b},
            sympy_solution=sympy_sol,
            lib_result=lib_result,
            question_templates=question_templates,
            context_info={**context_info},
            difficulty_markers=self.build_difficulty_markers(
                context, matrix_size=(matrix_A.rows, matrix_A.cols), vector_size=vector_b.rows
            ),
            difficulty=self.difficulty_level,
        )

    @override
    def get_template_variables(self, template: ProblemTemplate) -> dict[str, Any]:
        """Return the variables dictionary to pass to the template engine."""
        matrix = template.context_info["matrix"]
        vector = template.context_info["vector"]
        return {"matrix": matrix, "vector": vector}

    def _multiply_matrices_sympy(self, matrix_a: Matrix, vector_b: Matrix) -> tuple[Matrix, list[list[float]]]:
        """Multiply two sympy matrices using lib.py function."""
        # Convert to primitives (this applies precision constraints)
        a_list = self.formatter.sympy_to_primitive(matrix_a, precision=self.precision)
        b_list = self.formatter.sympy_to_primitive(vector_b, precision=self.precision)
        assert isinstance(a_list, list) and isinstance(b_list, list)  # noqa: S101

        # Calculate using lib.py with the primitives
        lib_result = self.lib["multiply_matrices"](a_list, b_list)

        # Convert primitives back to SymPy matrices at the same precision level
        # This ensures both calculations work with the same precision
        matrix_a_precision_matched = Matrix(a_list)
        vector_b_precision_matched = Matrix(b_list)
        sympy_result = matrix_a_precision_matched * vector_b_precision_matched

        return sympy_result, lib_result

    def _split_entropy(self, context: ProblemContext) -> tuple[float, float]:
        """Split entropy between matrix and vector generation (independent)."""
        sample_args = SampleArgs(num_modules=2, entropy=context.entropy)
        matrix_sample_args, vector_sample_args = sample_args.split(count=2)
        return matrix_sample_args.entropy, vector_sample_args.entropy

    def _determine_dimensions(self, context: ProblemContext) -> tuple[int, int]:
        """Select matrix dimensions (independent): both from config."""
        rows = self.config.get_random_matrix_size()
        cols = self.config.get_random_matrix_size()
        return rows, cols

    def _generate_matrix_A(
        self,
        rows: int,
        cols: int,
        matrix_entropy: float,
        context: ProblemContext,
    ) -> Matrix:
        # Use constraint-based generation with specific dimensions
        # Temporarily set constraints for this specific call
        mandatory = GenerationConstraints(rows=rows, cols=cols, entropy=matrix_entropy)

        matrix_A = self._get_matrix_with_constraints(context, added_constraints=mandatory)

        return matrix_A

    def _generate_vector_b(
        self,
        cols: int,
        vector_entropy: float,
        context: ProblemContext,
    ) -> Matrix:
        # Use centralized vector generation. Provide fixed entropy via constraints
        # so the allocator records exactly this amount.
        constraints = GenerationConstraints(entropy=vector_entropy)
        return self._get_vector_with_constraints(context, size=cols, added_constraints=constraints)

    def _question_templates(self, context_info: dict[str, Any]) -> list[str] | None:
        question_templates = self.template_engine.create_default_templates(self.problem_type, self.difficulty_level)
        return [t.template_string for t in question_templates]


class MatrixVectorMultiplicationGeneratorDependent(MatrixVectorMultiplicationGenerator):
    def __init__(
        self,
        difficulty_level: DifficultyCategory,
        input_vector_b: sympy.Matrix,
        input_vector_b_index: int,
        **kwargs: Any,
    ) -> None:
        # Force is_independent=False for template selection if needed downstream
        super().__init__(difficulty_level=difficulty_level, is_independent=False, **kwargs)
        assert self.problem_type == Task.MATRIX_VECTOR_MULTIPLICATION  # noqa: S101
        self.input_vector_b = input_vector_b
        self.input_index = input_vector_b_index

    def _split_entropy(self, context: ProblemContext) -> tuple[float, float]:
        """Allocate some entropy to matrix generation even when vector is provided."""
        # Reserve all available context entropy for matrix generation; vector consumes none.
        return context.entropy, 0.0

    def _determine_dimensions(self, context: ProblemContext) -> tuple[int, int]:
        req_cols = int(self.input_vector_b.rows)
        rows = self.config.get_random_matrix_size()
        return rows, req_cols

    def _generate_vector_b(
        self,
        cols: int,
        vector_entropy: float,
        context: ProblemContext,
    ) -> Matrix:
        # No entropy usage for provided vector
        return self.input_vector_b

    def _question_templates(self, context_info: dict[str, Any]) -> list[str] | None:
        # Composition will handle question formatting
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
        """Return template variables for dependent matrix-vector multiplication generator."""
        matrix = template.context_info["matrix"]
        return {"matrix": matrix, "vector": f"step {self.input_index + 1}"}


class MatrixMatrixMultiplicationGeneratorDependent(MatrixVectorMultiplicationGenerator):
    def __init__(
        self,
        difficulty_level: DifficultyCategory,
        input_matrix_A: sympy.Matrix,
        input_matrix_B: sympy.Matrix,
        input_matrix_A_index: int,
        input_matrix_B_index: int,
        sources: dict[str, str],
        **kwargs: Any,
    ) -> None:
        super().__init__(difficulty_level=difficulty_level, is_independent=False, **kwargs)
        assert self.problem_type == Task.MATRIX_VECTOR_MULTIPLICATION  # noqa: S101
        self.input_matrix_A = input_matrix_A
        self.input_matrix_B = input_matrix_B
        self.input_index_matrix_A = input_matrix_A_index
        self.input_index_matrix_B = input_matrix_B_index
        self.sources = sources

    @override
    def get_template_variables(self, template: ProblemTemplate) -> dict[str, Any]:
        base_vars = {}

        # Use sources to determine how to display each matrix reference
        source_A = self.sources.get("input_matrix_A", "result")
        source_B = self.sources.get("input_matrix_B", "result")

        # Matrix A reference
        if source_A == "result":
            base_vars["matrix_A_ref"] = f"step {self.input_index_matrix_A + 1}"
        else:
            matrix_a = template.context_info["matrix"]
            assert self.input_matrix_A == matrix_a  # noqa: S101
            base_vars["matrix_A_ref"] = self.input_matrix_A

        # Matrix B reference
        if source_B == "result":
            base_vars["matrix_B_ref"] = f"step {self.input_index_matrix_B + 1}"
        else:
            matrix_b = template.context_info["vector"]
            assert self.input_matrix_B == matrix_b  # noqa: S101
            base_vars["matrix_B_ref"] = self.input_matrix_B

        return base_vars

    def _split_entropy(self, context: ProblemContext) -> tuple[float, float]:
        # Both inputs provided; do not consume additional entropy
        return 0.0, 0.0

    def _determine_dimensions(self, context: ProblemContext) -> tuple[int, int]:
        # Validate matrix multiplication compatibility: A.cols must equal B.rows
        a_rows, a_cols = self.input_matrix_A.rows, self.input_matrix_A.cols
        b_rows, b_cols = self.input_matrix_B.rows, self.input_matrix_B.cols

        if a_cols != b_rows:
            raise ValueError(
                f"Matrix multiplication incompatible: A({a_rows}x{a_cols}) * B({b_rows}x{b_cols}) - A.cols({a_cols}) ≠ B.rows({b_rows})"
            )

        # Result matrix dimensions: A.rows x B.cols
        return int(a_rows), int(b_cols)

    def _generate_vector_b(
        self,
        cols: int,
        vector_entropy: float,
        context: ProblemContext,
    ) -> Matrix:
        # No entropy usage for provided vector
        return self.input_matrix_B

    def _generate_matrix_A(
        self,
        rows: int,
        cols: int,
        matrix_entropy: float,
        context: ProblemContext,
    ) -> Matrix:
        return self.input_matrix_A

    def _question_templates(self, context_info: dict[str, Any]) -> list[str] | None:
        # Composition will handle question formatting
        return None

    def _prepare_tool_call_input_data(self, **kwargs: dict[str, Any]) -> dict[str, Any]:
        """Prepare input data for dependent generator including dependency info."""
        base_data = super()._prepare_tool_call_input_data(**kwargs)
        assert self.input_matrix_A == kwargs["matrix_a"]  # noqa: S101
        assert self.input_matrix_B == kwargs["vector_b"]  # noqa: S101
        base_data.update({
            "dependent_on": {
                "input_matrix_A": self.input_index_matrix_A,
                "input_matrix_B": self.input_index_matrix_B,
            },
            "input_matrix_A": MathFormatter.sympy_to_primitive(self.input_matrix_A, precision=self.precision),
            "input_matrix_B": MathFormatter.sympy_to_primitive(self.input_matrix_B, precision=self.precision),
        })

        # Remove the inputs that are not assigned to the result of the previous step
        for key, value in self.sources.items():
            if value != "result":
                base_data.pop(key)
                base_data["dependent_on"].pop(key)

        return base_data
