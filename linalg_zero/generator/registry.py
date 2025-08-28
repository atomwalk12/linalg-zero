"""Factory registration and initialization."""

import random
from collections.abc import Callable

from linalg_zero.generator.composition.components import (
    LinearSystemSolverWrapperComponent,
    MatrixVectorMultiplicationWrapperComponent,
)
from linalg_zero.generator.composition.composition import (
    SequentialComposition,
)
from linalg_zero.generator.difficulty_config import (
    get_problem_config,
)
from linalg_zero.generator.generator_factories import (
    create_composite_factory,
    create_determinant_factory,
    create_matrix_vector_equation_solver_factory,
    create_matrix_vector_multiplication_factory,
)
from linalg_zero.generator.models import DifficultyCategory, Question
from linalg_zero.generator.sympy.base import (
    CompositionStrategy,
    ProblemComponent,
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

    def register_composite_factory(
        self,
        topic: str,
        problem_type: str,
        components: list[ProblemComponent],
        composition_strategy: CompositionStrategy,
        difficulty_level: DifficultyCategory,
    ) -> None:
        """Register a composite factory"""
        config = get_problem_config(difficulty_level, problem_type, topic)
        num_components = len(components)

        factory = create_composite_factory(
            components=components,
            composition_strategy=composition_strategy,
            sample_args=config.create_sample_args_for_composition(num_components),
            difficulty_level=difficulty_level,
            problem_type=problem_type,
            topic=topic,
        )

        self.register_factory(topic, problem_type, factory)


def create_default_registry() -> FactoryRegistry:
    """Create and populate the default factory registry."""
    registry = FactoryRegistry()
    default_config = get_problem_config(DifficultyCategory.MEDIUM, "composite_linear_algebra")

    registry.register_factory(
        "linear_algebra",
        "determinant",
        create_determinant_factory(entropy=default_config.sample_entropy, difficulty=DifficultyCategory.EASY),
    )

    # Register linear algebra generators
    registry.register_factory(
        "linear_algebra",
        "matrix_vector_multiplication",
        create_matrix_vector_multiplication_factory(
            entropy=default_config.sample_entropy, difficulty=DifficultyCategory.MEDIUM
        ),
    )
    registry.register_factory(
        "linear_algebra",
        "matrix_vector_inverse_solver",
        create_matrix_vector_equation_solver_factory(
            entropy=default_config.sample_entropy, difficulty=DifficultyCategory.MEDIUM
        ),
    )

    # Sequential composition
    registry.register_composite_factory(
        topic="linear_algebra",
        problem_type="composite_sequential",
        components=[
            MatrixVectorMultiplicationWrapperComponent("mult_component"),
            LinearSystemSolverWrapperComponent("solve_component"),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.MEDIUM,
    )

    return registry
