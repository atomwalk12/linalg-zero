from linalg_zero.generator.composition.components import (
    LinearSystemSolverWrapperComponent,
    MatrixVectorMultiplicationWrapperComponent,
)
from linalg_zero.generator.composition.composition import (
    CompositeProblem,
    SequentialComposition,
)
from linalg_zero.generator.difficulty_config import get_problem_config
from linalg_zero.generator.models import DifficultyCategory, Task, Topic


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
            MatrixVectorMultiplicationWrapperComponent(name=Task.MATRIX_VECTOR_MULTIPLICATION, constraints={}),
            LinearSystemSolverWrapperComponent(name=Task.LINEAR_SYSTEM_SOLVER, constraints={}),
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
            MatrixVectorMultiplicationWrapperComponent(name=Task.MATRIX_VECTOR_MULTIPLICATION, constraints={}),
            LinearSystemSolverWrapperComponent(name=Task.LINEAR_SYSTEM_SOLVER, constraints={}),
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
            LinearSystemSolverWrapperComponent(name=Task.LINEAR_SYSTEM_SOLVER, constraints={}),
            MatrixVectorMultiplicationWrapperComponent(name=Task.MATRIX_VECTOR_MULTIPLICATION, constraints={}),
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
