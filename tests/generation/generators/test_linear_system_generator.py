import json

import pytest
from sympy import ShapeError
from sympy.solvers.solvers import NonInvertibleMatrixError

from linalg_zero.generator.difficulty_config import get_problem_config
from linalg_zero.generator.models import DifficultyCategory, Question, Task, Topic
from linalg_zero.generator.sympy.generators.linear_system_generator import (
    LinearSystemGenerator,
)
from linalg_zero.shared.lib import solve_linear_system


class TestLinearSystemGenerator:
    """Focused end-to-end tests for LinearSystemGenerator."""

    config = get_problem_config(DifficultyCategory.TWO_TOOL_CALLS, Topic.LINEAR_ALGEBRA, Task.LINEAR_SYSTEM_SOLVER)

    def _make_generator(self, difficulty: DifficultyCategory) -> LinearSystemGenerator:
        return LinearSystemGenerator(
            entropy=self.config.sample_entropy,
            difficulty_level=difficulty,
            problem_type=Task.LINEAR_SYSTEM_SOLVER,
            topic=Topic.LINEAR_ALGEBRA,
        )

    def test_basic_generation_easy(self):
        generator = self._make_generator(DifficultyCategory.ONE_TOOL_CALL)
        q = generator.generate()

        assert isinstance(q, Question)
        assert q.is_valid
        assert q.topic == Topic.LINEAR_ALGEBRA
        assert q.difficulty == DifficultyCategory.ONE_TOOL_CALL
        assert q.tool_calls_required == 1
        assert len(q.question) > 0
        assert len(q.answer) > 0

        # Answer should be a JSON list of lists (column vector for x)
        parsed = json.loads(q.answer)
        assert isinstance(parsed, list)
        assert len(parsed) >= 1
        assert all(isinstance(row, list) and len(row) == 1 for row in parsed)
        assert all(isinstance(row[0], (int, float)) for row in parsed)

    def test_medium_and_hard_generation(self):
        for difficulty in (DifficultyCategory.TWO_TOOL_CALLS, DifficultyCategory.THREE_TOOL_CALLS):
            generator = self._make_generator(difficulty)
            q = generator.generate()

            assert q.is_valid
            assert q.difficulty == difficulty
            assert q.tool_calls_required == 1

            parsed = json.loads(q.answer)
            assert isinstance(parsed, list)
            assert len(parsed) >= 1
            assert all(isinstance(row, list) and len(row) == 1 for row in parsed)
            assert all(isinstance(row[0], (int, float)) for row in parsed)

    def test_question_contains_solve_language(self):
        generator = self._make_generator(DifficultyCategory.TWO_TOOL_CALLS)
        q = generator.generate()

        text = q.question.lower()
        # Expect language around solving Ax = b
        assert any(kw in text for kw in ["solve", "what is", " = ", "for x", "find", "determine", "compute"])

    def test_multiple_generations_stability(self):
        generator = self._make_generator(DifficultyCategory.TWO_TOOL_CALLS)

        for _ in range(10):
            q = generator.generate()
            assert q.is_valid
            parsed = json.loads(q.answer)
            assert isinstance(parsed, list)
            assert len(parsed) >= 1
            assert all(isinstance(row, list) and len(row) == 1 for row in parsed)
            assert all(isinstance(row[0], (int, float)) for row in parsed)

    def test_solve_linear_system_tool_function_examples(self):
        # Identity matrix case: x == b
        result = solve_linear_system([[1, 0], [0, 1]], [5, 3])
        assert len(result) == 2
        assert result[0][0] == pytest.approx(5.0, rel=1e-9)
        assert result[1][0] == pytest.approx(3.0, rel=1e-9)

        # Another solvable system
        result = solve_linear_system([[2, 1], [1, 3]], [7, 8])

        # Expected solution is [[13/5],[9/5]] (column vector)
        assert len(result) == 2
        assert result[0][0] == pytest.approx(13.0 / 5.0, rel=1e-9)
        assert result[1][0] == pytest.approx(9.0 / 5.0, rel=1e-9)

    def test_solve_linear_system_tool_function_error_handling(self):
        # Matrix with no inverse (determinant is 0) should raise NonInvertibleMatrixError
        with pytest.raises(NonInvertibleMatrixError):
            solve_linear_system([[1, 2], [2, 4]], [3, 6])

    def test_solve_linear_system_tool_function_error_handling_shape_error(self):
        # Singular system (multiple solutions) should raise NonInvertibleMatrixError
        with pytest.raises(ShapeError):
            solve_linear_system([[1, 2], [2, 4]], [3, 3, 6])
