from linalg_zero.generator.models import Task, Topic
from linalg_zero.generator.registry import create_default_registry


def test_create_default_registry() -> None:
    """Test that default registry is created with expected factories."""
    registry = create_default_registry()

    # Check topics are registered
    topics = registry.list_topics()
    assert Topic.LINEAR_ALGEBRA in topics

    # Check linear algebra problem types
    linalg_problems = registry.list_problem_types(Topic.LINEAR_ALGEBRA)
    assert Task.MATRIX_VECTOR_MULTIPLICATION in linalg_problems


def test_default_registry_factories_work() -> None:
    """Test that factories in default registry actually work."""
    registry = create_default_registry()

    # Test matrix vector multiplication
    mv_factory = registry.get_factory(Topic.LINEAR_ALGEBRA, Task.MATRIX_VECTOR_MULTIPLICATION)
    question = mv_factory()
    assert question.topic == Topic.LINEAR_ALGEBRA
    assert len(question.answer) > 0


def test_random_factory_selection() -> None:
    """Test that random factory selection works."""
    registry = create_default_registry()

    # Get random linear algebra factory
    random_factory = registry.get_random_factory(Topic.LINEAR_ALGEBRA)
    question = random_factory()
    assert question.topic == Topic.LINEAR_ALGEBRA
