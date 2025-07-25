"""Tests for question generation functionality."""

from linalg_zero.generator.arithmetic import arithmetic_addition_factory
from linalg_zero.generator.core import DatasetGenerator, QuestionGenerator
from linalg_zero.generator.linalg import matrix_addition_factory
from linalg_zero.generator.models import Question
from linalg_zero.generator.utils.difficulty import DIFFICULTY_LEVELS, make_difficulty_booster, make_difficulty_reducer


def test_factory_functions_basic() -> None:
    """Test that factory functions produce valid Question objects."""
    # Test arithmetic factory
    arith_question = arithmetic_addition_factory()
    assert isinstance(arith_question, Question)
    assert arith_question.topic == "arithmetic"
    assert arith_question.difficulty == "easy"
    assert len(arith_question.text) > 0
    assert len(arith_question.answer) > 0

    # Test linear algebra factory
    linalg_question = matrix_addition_factory()
    assert isinstance(linalg_question, Question)
    assert linalg_question.topic == "linear_algebra"
    assert linalg_question.difficulty == "medium"
    assert len(linalg_question.text) > 0
    assert len(linalg_question.answer) > 0


def test_question_generator_factory_pattern() -> None:
    """Test QuestionGenerator uses injected factory correctly (core pattern test)."""

    def simple_factory() -> Question:
        return Question(text="Test question", answer="42", topic="test")

    generator = QuestionGenerator(question_factory=simple_factory)
    question = generator.generate()

    assert question.text == "Test question"
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
        assert len(question.text) > 0
        assert len(question.answer) > 0
        assert question.difficulty in ["easy", "medium", "hard"]


def test_difficulty_system() -> None:
    """Test the simplified difficulty system."""
    # Test booster function
    boost_one = make_difficulty_booster(1)
    assert boost_one("easy") == "medium"
    assert boost_one("medium") == "hard"
    assert boost_one("hard") == "hard"  # Can't go higher

    # Test reducer function
    reduce_one = make_difficulty_reducer(1)
    assert reduce_one("hard") == "medium"
    assert reduce_one("medium") == "easy"
    assert reduce_one("easy") == "easy"  # Can't go lower

    # Test available levels
    assert len(DIFFICULTY_LEVELS) == 3
    assert DIFFICULTY_LEVELS == ["easy", "medium", "hard"]
