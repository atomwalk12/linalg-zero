"""Tests for default registry setup and integration."""

from linalg_zero.generator.registry import create_default_registry


def test_create_default_registry() -> None:
    """Test that default registry is created with expected factories."""
    registry = create_default_registry()

    # Check topics are registered
    topics = registry.list_topics()
    assert "arithmetic" in topics
    assert "linear_algebra" in topics

    # Check arithmetic problem types
    arithmetic_problems = registry.list_problem_types("arithmetic")
    assert "addition" in arithmetic_problems

    # Check linear algebra problem types
    linalg_problems = registry.list_problem_types("linear_algebra")
    assert "matrix_determinant_2x2" in linalg_problems
    assert "vector_dot_product" in linalg_problems
    assert "matrix_addition" in linalg_problems


def test_default_registry_factories_work() -> None:
    """Test that factories in default registry actually work."""
    registry = create_default_registry()

    # Test arithmetic addition
    addition_factory = registry.get_factory("arithmetic", "addition")
    question = addition_factory()
    assert question.topic == "arithmetic"
    assert len(question.answer) > 0

    # Test matrix determinant
    det_factory = registry.get_factory("linear_algebra", "matrix_determinant_2x2")
    question = det_factory()
    assert question.topic == "linear_algebra"
    assert "determinant" in question.text.lower()


def test_random_factory_selection() -> None:
    """Test that random factory selection works."""
    registry = create_default_registry()

    # Get random arithmetic factory
    random_factory = registry.get_random_factory("arithmetic")
    question = random_factory()
    assert question.topic == "arithmetic"

    # Get random linear algebra factory
    random_factory = registry.get_random_factory("linear_algebra")
    question = random_factory()
    assert question.topic == "linear_algebra"
