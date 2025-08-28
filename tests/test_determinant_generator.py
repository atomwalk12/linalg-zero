import pytest

from linalg_zero.generator.difficulty_config import get_problem_config
from linalg_zero.generator.models import DifficultyCategory, Question, Task, Topic
from linalg_zero.generator.sympy.generators.determinant_generator import DeterminantGenerator
from linalg_zero.shared.lib import determinant


class TestDeterminantGenerator:
    """Test suite for DeterminantGenerator."""

    config = get_problem_config(DifficultyCategory.MEDIUM, Topic.LINEAR_ALGEBRA, Task.DETERMINANT)

    def test_determinant_generator_easy_difficulty(self):
        """Test determinant generator with easy difficulty (2x2 matrices)."""
        generator = DeterminantGenerator(
            entropy=self.config.sample_entropy,
            difficulty_level=DifficultyCategory.EASY,
            problem_type=Task.DETERMINANT,
            topic=Topic.LINEAR_ALGEBRA,
        )
        question = generator.generate()

        # Basic question validation
        assert isinstance(question, Question)
        assert question.is_valid
        assert len(question.question) > 0
        assert len(question.answer) > 0
        assert question.topic == Topic.LINEAR_ALGEBRA
        assert question.difficulty == DifficultyCategory.EASY

        # Check that answer is a valid number (can be float)
        answer_float = float(question.answer)
        assert isinstance(answer_float, float)

        # Check tool call usage (should be exactly 1 for determinant)
        assert question.tool_calls_required == 1

    def test_determinant_generator_medium_difficulty(self):
        """Test determinant generator with medium difficulty (3x3 matrices)."""
        generator = DeterminantGenerator(
            entropy=self.config.sample_entropy,
            difficulty_level=DifficultyCategory.MEDIUM,
            problem_type=Task.DETERMINANT,
            topic=Topic.LINEAR_ALGEBRA,
        )
        question = generator.generate()

        # Basic question validation
        assert isinstance(question, Question)
        assert question.is_valid
        assert question.difficulty == DifficultyCategory.MEDIUM

        # Answer should be float
        answer_float = float(question.answer)
        assert isinstance(answer_float, float)

    def test_determinant_generator_hard_difficulty(self):
        """Test determinant generator with hard difficulty."""
        generator = DeterminantGenerator(
            entropy=self.config.sample_entropy,
            difficulty_level=DifficultyCategory.HARD,
            problem_type=Task.DETERMINANT,
            topic=Topic.LINEAR_ALGEBRA,
        )
        question = generator.generate()

        # Basic question validation
        assert isinstance(question, Question)
        assert question.is_valid
        assert question.difficulty == DifficultyCategory.HARD

        # Answer should be float
        answer_float = float(question.answer)
        assert isinstance(answer_float, float)

    def test_determinant_generator_question_formatting(self):
        """Test that questions are properly formatted with determinant language."""
        generator = DeterminantGenerator(
            entropy=self.config.sample_entropy,
            difficulty_level=DifficultyCategory.EASY,
            problem_type=Task.DETERMINANT,
            topic=Topic.LINEAR_ALGEBRA,
        )
        question = generator.generate()

        # Question should contain determinant-related keywords
        question_lower = question.question.lower()
        assert any(
            keyword in question_lower for keyword in ["determinant", "det(", "det ", "calculate", "compute", "find"]
        )

    def test_determinant_generator_consistency(self):
        """Test that the same generator produces consistent results."""
        generator = DeterminantGenerator(
            entropy=self.config.sample_entropy,
            difficulty_level=DifficultyCategory.EASY,
            problem_type=Task.DETERMINANT,
            topic=Topic.LINEAR_ALGEBRA,
        )

        # Generate multiple questions
        questions = [generator.generate() for _ in range(5)]

        # All should be valid
        for question in questions:
            assert question.is_valid
            assert isinstance(float(question.answer), float)
            assert question.topic == Topic.LINEAR_ALGEBRA

    def test_determinant_tool_function_examples(self):
        """Test the determinant tool function with known examples."""
        # Test 2x2 matrix: [[1, 2], [3, 4]] should have determinant -2
        result = determinant([[1, 2], [3, 4]])
        assert result == -2
        assert isinstance(result, float)

        # Test diagonal matrix: [[2, 0], [0, 3]] should have determinant 6
        result = determinant([[2, 0], [0, 3]])
        assert result == 6

        # Test identity matrix: [[1, 0], [0, 1]] should have determinant 1
        result = determinant([[1, 0], [0, 1]])
        assert result == 1

        # Test 1x1 matrix
        result = determinant([[5]])
        assert result == 5

    def test_determinant_tool_function_error_handling(self):
        """Test that determinant function properly handles errors."""
        # Non-square matrix should raise ValueError
        with pytest.raises(ValueError, match="Matrix must be square"):
            determinant([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix

    def test_determinant_generator_verification(self):
        """Test that generator properly verifies problems using both SymPy and lib.py."""
        generator = DeterminantGenerator(
            entropy=self.config.sample_entropy,
            difficulty_level=DifficultyCategory.EASY,
            problem_type=Task.DETERMINANT,
            topic=Topic.LINEAR_ALGEBRA,
        )

        # Generate and manually verify a problem
        question = generator.generate()

        # The question should have been verified during generation
        assert question.is_valid

        # We can't directly access the internal matrix, but we know it passed verification
        # This means SymPy det() and lib.py determinant() agreed on the result

    def test_determinant_generator_different_difficulty_levels(self):
        """Test that different difficulty levels produce valid results with appropriate complexity."""
        difficulties = [DifficultyCategory.EASY, DifficultyCategory.MEDIUM, DifficultyCategory.HARD]

        for difficulty in difficulties:
            generator = DeterminantGenerator(
                entropy=self.config.sample_entropy,
                difficulty_level=difficulty,
                problem_type=Task.DETERMINANT,
                topic=Topic.LINEAR_ALGEBRA,
            )
            question = generator.generate()

            assert question.is_valid
            assert question.tool_calls_required == 1  # Always exactly 1 tool call
            assert question.difficulty == difficulty

    def test_multiple_generations_no_errors(self):
        """Test that multiple generations don't cause errors or crashes."""
        generator = DeterminantGenerator(
            entropy=self.config.sample_entropy,
            difficulty_level=DifficultyCategory.MEDIUM,
            problem_type=Task.DETERMINANT,
            topic=Topic.LINEAR_ALGEBRA,
        )

        # Generate many questions to test stability
        for _ in range(20):
            question = generator.generate()
            assert question.is_valid
            # Ensure answer can be parsed as float
            float(question.answer)

    def test_determinant_generator_problem_type(self):
        """Test that the problem type is correctly set."""
        generator = DeterminantGenerator(
            entropy=self.config.sample_entropy,
            difficulty_level=DifficultyCategory.EASY,
            problem_type=Task.DETERMINANT,
            topic=Topic.LINEAR_ALGEBRA,
        )
        question = generator.generate()

        # Problem type should be set (though we don't expose it directly in Question)
        # We verify indirectly through question format and validation
        assert question.is_valid
        assert "determinant" in question.question.lower() or "det(" in question.question.lower()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
