from linalg_zero.generator.registry import create_default_registry


def test_create_default_registry() -> None:
    """Test that default registry is created with expected factories."""
    registry = create_default_registry()

    # Check topics are registered
    topics = registry.list_topics()
    assert "linear_algebra" in topics

    # Check linear algebra problem types
    linalg_problems = registry.list_problem_types("linear_algebra")
    assert "matrix_vector_multiplication" in linalg_problems


def test_default_registry_factories_work() -> None:
    """Test that factories in default registry actually work."""
    registry = create_default_registry()

    # Test matrix vector multiplication
    mv_factory = registry.get_factory("linear_algebra", "matrix_vector_multiplication")
    question = mv_factory()
    assert question.topic == "linear_algebra"
    assert len(question.answer) > 0


def test_random_factory_selection() -> None:
    """Test that random factory selection works."""
    registry = create_default_registry()

    # Get random linear algebra factory
    random_factory = registry.get_random_factory("linear_algebra")
    question = random_factory()
    assert question.topic == "linear_algebra"
