import random
from collections.abc import Callable

from linalg_zero.generator.composition.components import (
    FrobeniusNormWrapperComponent,
    LinearSystemSolverWrapperComponent,
    MatrixVectorMultiplicationWrapperComponent,
)
from linalg_zero.generator.composition.composition import (
    SequentialComposition,
)
from linalg_zero.generator.difficulty_config import get_problem_config
from linalg_zero.generator.generator_factories import (
    create_composite_factory,
    create_determinant_factory,
    create_frobenius_norm_factory,
    create_linear_system_generator,
    create_matrix_rank_factory,
    create_matrix_transpose_factory,
    create_matrix_vector_multiplication_factory,
)
from linalg_zero.generator.models import DifficultyCategory, Question, Task, Topic
from linalg_zero.generator.sympy.base import (
    CompositionStrategy,
    ProblemComponent,
)


class FactoryRegistry:
    """Registry for managing different question factories."""

    def __init__(self) -> None:
        self._factories: dict[Topic, dict[Task, Callable[[], Question]]] = {}

    def register_factory(self, topic: Topic, problem_type: Task, factory: Callable[[], Question]) -> None:
        """Register a factory function."""
        if topic not in self._factories:
            self._factories[topic] = {}
        self._factories[topic][problem_type] = factory

    def get_factory(self, topic: Topic, problem_type: Task) -> Callable[[], Question]:
        """Get a specific factory by topic and problem type."""
        if topic not in self._factories:
            raise ValueError(f"Unknown topic: {topic}")
        if problem_type not in self._factories[topic]:
            raise ValueError(f"Unknown problem type: {problem_type}")
        return self._factories[topic][problem_type]

    def get_random_factory(self, topic: Topic) -> Callable[[], Question]:
        """Get a random factory from the specified topic."""
        if topic not in self._factories:
            raise ValueError(f"Unknown topic: {topic}")
        problem_types = list(self._factories[topic].keys())
        random_type = random.choice(problem_types)
        return self._factories[topic][random_type]

    def list_topics(self) -> list[Topic]:
        """List all available topics."""
        return list(self._factories.keys())

    def list_problem_types(self, topic: Topic) -> list[Task]:
        """List all problem types for a given topic."""
        if topic not in self._factories:
            raise ValueError(f"Unknown topic: {topic}")
        return list(self._factories[topic].keys())

    def register_composite_factory(
        self,
        topic: Topic,
        problem_type: Task,
        components: list[ProblemComponent],
        composition_strategy: CompositionStrategy,
        difficulty_level: DifficultyCategory,
    ) -> None:
        """Register a composite factory"""
        config = get_problem_config(difficulty_level, topic, problem_type)
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

    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.DETERMINANT,
        create_determinant_factory(difficulty=DifficultyCategory.EASY),
    )

    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.MATRIX_VECTOR_MULTIPLICATION,
        create_matrix_vector_multiplication_factory(difficulty=DifficultyCategory.MEDIUM),
    )
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.LINEAR_SYSTEM_SOLVER,
        create_linear_system_generator(difficulty=DifficultyCategory.MEDIUM),
    )
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.FROBENIUS_NORM,
        create_frobenius_norm_factory(difficulty=DifficultyCategory.MEDIUM),
    )
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.MATRIX_RANK,
        create_matrix_rank_factory(difficulty=DifficultyCategory.MEDIUM),
    )
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.MATRIX_TRANSPOSE,
        create_matrix_transpose_factory(difficulty=DifficultyCategory.MEDIUM),
    )

    # Sequential composition
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.COMPOSITE_SEQUENTIAL,
        components=[
            MatrixVectorMultiplicationWrapperComponent(
                name=Task.MATRIX_VECTOR_MULTIPLICATION,
                constraints={},
            ),
            LinearSystemSolverWrapperComponent(
                name=Task.LINEAR_SYSTEM_SOLVER,
                constraints={"input_index": 0},
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.MEDIUM,
    )

    # # Linear System Dependency: solve_linear_system → multiply_matrices → frobenius_norm
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.COMPOSITE_LINEAR_SYSTEM_DEPENDENCY,
        components=[
            LinearSystemSolverWrapperComponent(
                name=Task.LINEAR_SYSTEM_SOLVER,
                constraints={"is_independent": True},
            ),
            MatrixVectorMultiplicationWrapperComponent(
                name=Task.MATRIX_VECTOR_MULTIPLICATION,
                constraints={"is_independent": False, "input_index": 0},
            ),
            FrobeniusNormWrapperComponent(
                name=Task.FROBENIUS_NORM,
                constraints={"is_independent": False, "input_index": 1},
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.MEDIUM,
    )

    return registry
