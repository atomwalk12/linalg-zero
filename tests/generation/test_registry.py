"""Tests for factory registry functionality."""

import pytest

from linalg_zero.generator.models import DifficultyCategory, Question, Task, Topic
from linalg_zero.generator.registry import FactoryRegistry, create_default_registry


def simple_test_factory() -> Question:
    """Simple factory for testing."""
    return Question(
        question="Test", answer="42", topic="test", difficulty=DifficultyCategory.EASY, problem_type=Task.DETERMINANT
    )


def another_test_factory() -> Question:  # pragma: no cover
    """Another simple factory for testing."""
    return Question(
        question="Another test",
        answer="24",
        topic="test",
        difficulty=DifficultyCategory.EASY,
        problem_type=Task.DETERMINANT,
    )


def test_factory_registry_registration() -> None:
    """Test basic factory registration and retrieval."""
    registry = FactoryRegistry()

    # Register a factory
    registry.register_factory("test_topic", "test_problem", simple_test_factory)

    # Retrieve and test
    factory = registry.get_factory("test_topic", "test_problem")
    question = factory()

    assert question.question == "Test"
    assert question.answer == "42"


def test_factory_registry_list_topics() -> None:
    """Test listing topics."""
    registry = FactoryRegistry()

    registry.register_factory("math", "addition", simple_test_factory)
    registry.register_factory("science", "physics", another_test_factory)

    topics = registry.list_topics()
    assert "math" in topics
    assert "science" in topics
    assert len(topics) == 2


def test_factory_registry_list_problem_types() -> None:
    """Test listing problem types for a topic."""
    registry = FactoryRegistry()

    registry.register_factory("math", "addition", simple_test_factory)
    registry.register_factory("math", "subtraction", another_test_factory)

    problem_types = registry.list_problem_types("math")
    assert "addition" in problem_types
    assert "subtraction" in problem_types
    assert len(problem_types) == 2


def test_factory_registry_unknown_topic() -> None:
    """Test error handling for unknown topic."""
    registry = FactoryRegistry()

    with pytest.raises(ValueError, match="Unknown topic"):
        registry.get_factory("non-existent", "problem")


def test_factory_registry_unknown_problem_type() -> None:
    """Test error handling for unknown problem type."""
    registry = FactoryRegistry()
    registry.register_factory("math", "addition", simple_test_factory)

    with pytest.raises(ValueError, match="Unknown problem type"):
        registry.get_factory("math", "non-existent")


def test_create_default_registry() -> None:
    """Test that default registry is created with expected factories."""
    registry = create_default_registry()

    topics = registry.list_topics()
    assert Topic.LINEAR_ALGEBRA in topics

    linalg_problems = registry.list_problem_types(Topic.LINEAR_ALGEBRA)
    assert Task.MATRIX_VECTOR_MULTIPLICATION in linalg_problems


def test_default_registry_factories_work() -> None:
    """Test that factories in default registry actually work."""
    registry = create_default_registry()

    mv_factory = registry.get_factory(Topic.LINEAR_ALGEBRA, Task.MATRIX_VECTOR_MULTIPLICATION)
    question = mv_factory()
    assert question.topic == Topic.LINEAR_ALGEBRA
    assert len(question.answer) > 0


def test_random_factory_selection() -> None:
    """Test that random factory selection works."""
    registry = create_default_registry()

    random_factory = registry.get_random_factory(Topic.LINEAR_ALGEBRA)
    question = random_factory()
    assert question.topic == Topic.LINEAR_ALGEBRA
