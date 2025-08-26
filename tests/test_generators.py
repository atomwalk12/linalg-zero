from linalg_zero.generator.core import DatasetGenerator, QuestionGenerator
from linalg_zero.generator.models import DifficultyCategory, Question


def test_question_generator_factory_pattern() -> None:
    """Test QuestionGenerator uses injected factory correctly (core pattern test)."""

    def simple_factory() -> Question:
        return Question(question="Test question", answer="42", topic="test", difficulty=DifficultyCategory.EASY)

    generator = QuestionGenerator(question_factory=simple_factory)
    question = generator.generate()

    assert question.question == "Test question"
    assert question.answer == "42"
    assert question.topic == "test"
    assert question.is_valid is True  # Should be validated


def test_dataset_generation() -> None:
    """Test that DatasetGenerator creates datasets with expected properties."""
    generator = DatasetGenerator(topic="linear_algebra")
    questions = generator.generate_dataset(num_questions=3)

    # Check we got the right number of questions
    assert len(questions) == 3

    # Check all questions are valid Question objects
    for question in questions:
        assert isinstance(question, Question)
        assert question.is_valid is True  # Only valid questions should be returned
        assert len(question.question) > 0
        assert len(question.answer) > 0
        assert question.difficulty in [DifficultyCategory.EASY, DifficultyCategory.MEDIUM, DifficultyCategory.HARD]
