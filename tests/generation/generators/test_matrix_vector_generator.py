import json

from linalg_zero.generator.difficulty_config import get_problem_config
from linalg_zero.generator.models import DifficultyCategory, Question, Task, Topic
from linalg_zero.generator.sympy.generators.matrix_vector_generator import (
    MatrixVectorMultiplicationGenerator,
)


class TestMatrixVectorMultiplicationGenerator:
    """Focused end-to-end tests for MatrixVectorMultiplicationGenerator."""

    config = get_problem_config(
        DifficultyCategory.TWO_TOOL_CALLS, Topic.LINEAR_ALGEBRA, Task.MATRIX_VECTOR_MULTIPLICATION
    )

    def _make_generator(self, difficulty: DifficultyCategory) -> MatrixVectorMultiplicationGenerator:
        return MatrixVectorMultiplicationGenerator(
            entropy=self.config.sample_entropy,
            difficulty_level=difficulty,
            problem_type=Task.MATRIX_VECTOR_MULTIPLICATION,
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

        # Answer should be a JSON list of lists (column vector)
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

    def test_question_contains_multiplication_language(self):
        generator = self._make_generator(DifficultyCategory.TWO_TOOL_CALLS)
        q = generator.generate()

        text = q.question.lower()
        # Check for indicative keywords/templates
        assert any(kw in text for kw in [" * ", "product", "compute", "calculate", "find", "what is"])

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
