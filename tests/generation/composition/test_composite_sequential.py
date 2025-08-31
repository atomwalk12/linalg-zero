import pytest

from linalg_zero.generator.composition.components import (
    DeterminantWrapperComponent,
    FrobeniusNormWrapperComponent,
    LinearSystemSolverWrapperComponent,
    MatrixVectorMultiplicationWrapperComponent,
    RankWrapperComponent,
    TraceWrapperComponent,
    TransposeWrapperComponent,
)
from linalg_zero.generator.composition.composition import (
    CompositeProblem,
    SequentialComposition,
)
from linalg_zero.generator.difficulty_config import get_problem_config
from linalg_zero.generator.models import DifficultyCategory, Task, Topic
from linalg_zero.generator.sympy.generators.determinant_generator import (
    DeterminantGenerator,
    DeterminantGeneratorDependent,
)
from linalg_zero.generator.sympy.generators.frobenius_norm_generator import (
    FrobeniusNormGenerator,
    FrobeniusNormGeneratorDependent,
)
from linalg_zero.generator.sympy.generators.linear_system_generator import (
    LinearSystemGenerator,
    LinearSystemGeneratorDependent,
)
from linalg_zero.generator.sympy.generators.matrix_rank_generator import (
    MatrixRankGenerator,
    MatrixRankGeneratorDependent,
)
from linalg_zero.generator.sympy.generators.matrix_trace_generator import (
    MatrixTraceGenerator,
    MatrixTraceGeneratorDependent,
)
from linalg_zero.generator.sympy.generators.matrix_transpose_generator import (
    MatrixTransposeGenerator,
    MatrixTransposeGeneratorDependent,
)
from linalg_zero.generator.sympy.generators.matrix_vector_generator import (
    MatrixVectorMultiplicationGenerator,
    MatrixVectorMultiplicationGeneratorDependent,
)


def make_composite(components: list, difficulty: DifficultyCategory = DifficultyCategory.MEDIUM) -> CompositeProblem:
    config = get_problem_config(difficulty, Topic.LINEAR_ALGEBRA, Task.COMPOSITE_SEQUENTIAL)
    sample_args = config.create_sample_args_for_composition(num_components=len(components))
    return CompositeProblem(
        components=components,
        composition_strategy=SequentialComposition(),
        sample_args=sample_args,
        difficulty_level=difficulty,
        problem_type=Task.COMPOSITE_SEQUENTIAL,
        topic=Topic.LINEAR_ALGEBRA,
    )


class TestSequential_MVM_then_LinearSystem:
    """Sequential composition: MatrixVectorMultiplication -> LinearSystemSolver."""

    def test_mvm_then_linear_system_end_to_end(self):
        composite = make_composite([
            MatrixVectorMultiplicationWrapperComponent(
                name=Task.MATRIX_VECTOR_MULTIPLICATION, constraints={"is_independent": True}
            ),
            LinearSystemSolverWrapperComponent(name=Task.LINEAR_SYSTEM_SOLVER, constraints={"is_independent": True}),
        ])

        q = composite.generate()

        # Strict checks
        assert q.is_valid
        assert q.tool_calls_required == 2

        # Ordering of steps must be preserved
        tool_names = [step["tool"] for step in q.stepwise]
        assert tool_names == ["multiply_matrices", "solve_linear_system"]

        # Question contains both steps in order
        text = q.question
        assert text.startswith("First, ") and "Then, " in text

        # Answer should contain both tools in JSON format
        import json

        answer_data = json.loads(q.answer)
        assert "tool_1" in answer_data and "tool_2" in answer_data

        # Parse both parts and assert shapes/types
        p1 = answer_data["tool_1"]
        p2 = answer_data["tool_2"]

        # Part 1 is matrix-vector product: column vector [[..],[..]]
        assert isinstance(p1, list) and len(p1) >= 1 and all(isinstance(r, list) and len(r) == 1 for r in p1)
        # Part 2 is linear system solution: list [..,..] or column vector [[..],[..]] depending on formatting
        assert isinstance(p2, list)

    def test_mvm_then_linear_system_stability(self):
        composite = make_composite([
            MatrixVectorMultiplicationWrapperComponent(
                name=Task.MATRIX_VECTOR_MULTIPLICATION, constraints={"is_independent": True}
            ),
            LinearSystemSolverWrapperComponent(name=Task.LINEAR_SYSTEM_SOLVER, constraints={"is_independent": True}),
        ])

        for _ in range(5):
            q = composite.generate()
            assert q.is_valid
            assert q.tool_calls_required == 2
            tool_names = [step["tool"] for step in q.stepwise]
            assert tool_names == ["multiply_matrices", "solve_linear_system"]


class TestSequential_LinearSystem_then_MVM:
    """Sequential composition: LinearSystemSolver -> MatrixVectorMultiplication."""

    def test_linear_system_then_mvm_end_to_end(self):
        composite = make_composite([
            LinearSystemSolverWrapperComponent(name=Task.LINEAR_SYSTEM_SOLVER, constraints={"is_independent": True}),
            MatrixVectorMultiplicationWrapperComponent(
                name=Task.MATRIX_VECTOR_MULTIPLICATION, constraints={"is_independent": True}
            ),
        ])

        q = composite.generate()

        # Strict checks
        assert q.is_valid
        assert q.tool_calls_required == 2

        # Ordering of steps must be preserved
        tool_names = [step["tool"] for step in q.stepwise]
        assert tool_names == ["solve_linear_system", "multiply_matrices"]

        # Question contains both steps in order
        text = q.question
        assert text.startswith("First, ") and "Then, " in text

        # Answer has two parts that can be parsed as JSON
        import json

        answer_data = json.loads(q.answer)
        assert "tool_1" in answer_data and "tool_2" in answer_data
        assert isinstance(answer_data["tool_1"], list)
        assert isinstance(answer_data["tool_2"], list)

    def test_mvm_then_linear_system_dependent_second(self):
        """Second component (LinearSystem) depends on first (MVM) output."""
        composite = make_composite([
            MatrixVectorMultiplicationWrapperComponent(
                name=Task.MATRIX_VECTOR_MULTIPLICATION, constraints={"is_independent": True}
            ),
            LinearSystemSolverWrapperComponent(
                name=Task.LINEAR_SYSTEM_SOLVER, constraints={"is_independent": False, "input_index": 0}
            ),
        ])

        q = composite.generate()

        assert q.is_valid
        assert q.tool_calls_required == 2
        tool_names = [step["tool"] for step in q.stepwise]
        assert tool_names == ["multiply_matrices", "solve_linear_system"]

        # Verify dependency metadata for the second step
        second = q.stepwise[1]
        assert second["tool"] == "solve_linear_system"
        assert "verification" in second and isinstance(second["verification"], dict)
        verification = second["verification"]
        assert verification.get("dependent_on") == 0
        assert "input_vector_b" in verification

    def test_linear_system_then_mvm_dependent_second(self):
        """Second component (MVM) depends on first (LinearSystem) output."""
        composite = make_composite([
            LinearSystemSolverWrapperComponent(name=Task.LINEAR_SYSTEM_SOLVER, constraints={"is_independent": True}),
            MatrixVectorMultiplicationWrapperComponent(
                name=Task.MATRIX_VECTOR_MULTIPLICATION, constraints={"is_independent": False, "input_index": 0}
            ),
        ])

        q = composite.generate()

        assert q.is_valid
        assert q.tool_calls_required == 2
        tool_names = [step["tool"] for step in q.stepwise]
        assert tool_names == ["solve_linear_system", "multiply_matrices"]

        # Verify dependency metadata for the second step
        second = q.stepwise[1]
        assert second["tool"] == "multiply_matrices"
        assert "verification" in second and isinstance(second["verification"], dict)
        verification = second["verification"]
        assert verification.get("dependent_on") == 0
        assert "input_vector_b" in verification or "input" in verification


class TestWrapperComponentGeneratorSelectionComprehensive:
    """Comprehensive tests across all wrapper components to ensure consistency."""

    @pytest.mark.parametrize(
        "wrapper_class,task,independent_generator",
        [
            (DeterminantWrapperComponent, Task.DETERMINANT, DeterminantGenerator),
            (FrobeniusNormWrapperComponent, Task.FROBENIUS_NORM, FrobeniusNormGenerator),
            (LinearSystemSolverWrapperComponent, Task.LINEAR_SYSTEM_SOLVER, LinearSystemGenerator),
            (RankWrapperComponent, Task.MATRIX_RANK, MatrixRankGenerator),
            (TraceWrapperComponent, Task.MATRIX_TRACE, MatrixTraceGenerator),
            (TransposeWrapperComponent, Task.MATRIX_TRANSPOSE, MatrixTransposeGenerator),
            (
                MatrixVectorMultiplicationWrapperComponent,
                Task.MATRIX_VECTOR_MULTIPLICATION,
                MatrixVectorMultiplicationGenerator,
            ),
        ],
    )
    def test_all_wrappers_independent_case(self, wrapper_class, task, independent_generator):
        """Test that all wrapper components correctly select independent generator when is_independent=True."""
        component = wrapper_class(name=task, constraints={"is_independent": True})

        assert component.generator_class is independent_generator
        assert component.is_independent is True
        assert component.name == task

    @pytest.mark.parametrize(
        "wrapper_class,task,dependent_generator",
        [
            (DeterminantWrapperComponent, Task.DETERMINANT, DeterminantGeneratorDependent),
            (FrobeniusNormWrapperComponent, Task.FROBENIUS_NORM, FrobeniusNormGeneratorDependent),
            (LinearSystemSolverWrapperComponent, Task.LINEAR_SYSTEM_SOLVER, LinearSystemGeneratorDependent),
            (RankWrapperComponent, Task.MATRIX_RANK, MatrixRankGeneratorDependent),
            (TraceWrapperComponent, Task.MATRIX_TRACE, MatrixTraceGeneratorDependent),
            (TransposeWrapperComponent, Task.MATRIX_TRANSPOSE, MatrixTransposeGeneratorDependent),
            (
                MatrixVectorMultiplicationWrapperComponent,
                Task.MATRIX_VECTOR_MULTIPLICATION,
                MatrixVectorMultiplicationGeneratorDependent,
            ),
        ],
    )
    def test_all_wrappers_dependent_case(self, wrapper_class, task, dependent_generator):
        """Test that all wrapper components correctly select dependent generator when is_independent=False."""
        component = wrapper_class(name=task, constraints={"is_independent": False, "input_index": 0})

        assert component.generator_class is dependent_generator
        assert component.is_independent is False
        assert component.name == task
