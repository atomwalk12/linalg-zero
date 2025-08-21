"""Factory registration and initialization."""

import random
from collections.abc import Callable

from linalg_zero.generator.arithmetic import arithmetic_addition_factory
from linalg_zero.generator.models import Question
from linalg_zero.generator.sympy.generators.algebra import (
    polynomial_roots_factory,
)


class FactoryRegistry:
    """Registry for managing different question factories."""

    def __init__(self) -> None:
        self._factories: dict[str, dict[str, Callable[[], Question]]] = {}

    def register_factory(self, topic: str, problem_type: str, factory: Callable[[], Question]) -> None:
        """Register a factory function."""
        if topic not in self._factories:
            self._factories[topic] = {}
        self._factories[topic][problem_type] = factory

    def get_factory(self, topic: str, problem_type: str) -> Callable[[], Question]:
        """Get a specific factory by topic and problem type."""
        if topic not in self._factories:
            raise ValueError(f"Unknown topic: {topic}")
        if problem_type not in self._factories[topic]:
            raise ValueError(f"Unknown problem type: {problem_type}")
        return self._factories[topic][problem_type]

    def get_random_factory(self, topic: str) -> Callable[[], Question]:
        """Get a random factory from the specified topic."""
        if topic not in self._factories:
            raise ValueError(f"Unknown topic: {topic}")
        problem_types = list(self._factories[topic].keys())
        random_type = random.choice(problem_types)
        return self._factories[topic][random_type]

    def list_topics(self) -> list[str]:
        """List all available topics."""
        return list(self._factories.keys())

    def list_problem_types(self, topic: str) -> list[str]:
        """List all problem types for a given topic."""
        if topic not in self._factories:
            raise ValueError(f"Unknown topic: {topic}")
        return list(self._factories[topic].keys())


def create_default_registry() -> FactoryRegistry:
    """Create and populate the default factory registry."""
    registry = FactoryRegistry()

    # Register linear algebra factories
    registry.register_factory("linear_algebra", "polynomial_roots", polynomial_roots_factory)

    # Register topic generators not strictly related to linear algebra
    registry.register_factory("arithmetic", "addition", arithmetic_addition_factory)

    return registry
