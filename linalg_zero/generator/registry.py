import random
from collections.abc import Callable

from linalg_zero.generator.composition.components import (
    DeterminantWrapperComponent,
    FrobeniusNormWrapperComponent,
    LinearSystemSolverWrapperComponent,
    MatrixCofactorWrapperComponent,
    MatrixMatrixMultiplicationWrapperComponent,
    RankWrapperComponent,
    TransposeWrapperComponent,
)
from linalg_zero.generator.composition.composition import (
    SequentialComposition,
)
from linalg_zero.generator.difficulty_config import ProblemConfig, get_problem_config
from linalg_zero.generator.generation_constraints import EntropyConstraints, GenerationConstraints
from linalg_zero.generator.generator_factories import (
    create_composite_factory,
    create_determinant_factory,
    create_frobenius_norm_factory,
    create_linear_system_generator,
    create_matrix_cofactor_factory,
    create_matrix_matrix_multiplication_factory,
    create_matrix_rank_factory,
    create_matrix_transpose_factory,
)
from linalg_zero.generator.models import DifficultyCategory, Question, Task, Topic
from linalg_zero.generator.sympy.base import (
    CompositionStrategy,
    ProblemComponent,
)


def register_determinant_factory(registry: "FactoryRegistry") -> None:
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.DETERMINANT,
        create_determinant_factory(difficulty=DifficultyCategory.ONE_TOOL_CALL),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )


def register_matrix_matrix_multiplication_factory(registry: "FactoryRegistry") -> None:
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.MATRIX_MATRIX_MULTIPLICATION,
        create_matrix_matrix_multiplication_factory(difficulty=DifficultyCategory.ONE_TOOL_CALL),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )


def register_linear_system_solver_factory(registry: "FactoryRegistry") -> None:
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.LINEAR_SYSTEM_SOLVER,
        create_linear_system_generator(difficulty=DifficultyCategory.ONE_TOOL_CALL),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )


def register_frobenius_norm_factory(registry: "FactoryRegistry") -> None:
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.FROBENIUS_NORM,
        create_frobenius_norm_factory(difficulty=DifficultyCategory.ONE_TOOL_CALL),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )


def register_matrix_rank_factory(registry: "FactoryRegistry") -> None:
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.MATRIX_RANK,
        create_matrix_rank_factory(difficulty=DifficultyCategory.ONE_TOOL_CALL),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )


def register_matrix_transpose_factory(registry: "FactoryRegistry") -> None:
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.MATRIX_TRANSPOSE,
        create_matrix_transpose_factory(difficulty=DifficultyCategory.ONE_TOOL_CALL),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )


def register_matrix_cofactor_factory(registry: "FactoryRegistry") -> None:
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.MATRIX_COFACTOR,
        create_matrix_cofactor_factory(difficulty=DifficultyCategory.ONE_TOOL_CALL),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )


def register_composite_triple_transpose_determinant(
    registry: "FactoryRegistry", custom_config: ProblemConfig, entropy_ranges: dict[Task, float]
) -> None:
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.THREE_TRANSPOSE_MATRIXMULT_DETERMINANT,
        components=[
            TransposeWrapperComponent(
                name=Task.MATRIX_TRANSPOSE,
                constraints={"is_independent": True},
                gen_constraints=GenerationConstraints(square=True),
                entropy_constraints=EntropyConstraints(entropy=entropy_ranges[Task.MATRIX_TRANSPOSE]),
            ),
            MatrixMatrixMultiplicationWrapperComponent(
                name=Task.MATRIX_MATRIX_MULTIPLICATION,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix_A": 0, "input_matrix_B": 0},
                    "sources": {"input_matrix_A": "matrix", "input_matrix_B": "result"},
                },
                entropy_constraints=EntropyConstraints(entropy=entropy_ranges[Task.MATRIX_MATRIX_MULTIPLICATION]),
            ),
            DeterminantWrapperComponent(
                name=Task.DETERMINANT,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 1},
                    "sources": {"input_matrix": "result"},
                },
                entropy_constraints=EntropyConstraints(entropy=entropy_ranges[Task.DETERMINANT]),
            ),
        ],
        composition_strategy=SequentialComposition(config=custom_config),
        difficulty_level=DifficultyCategory.THREE_TOOL_CALLS,
        custom_config=custom_config,
    )


def register_composite_triple_system_frobenius(
    registry: "FactoryRegistry", custom_config: ProblemConfig, entropy_ranges: dict[Task, float]
) -> None:
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.THREE_SYSTEM_MATRIXMULT_FROBENIUS,
        components=[
            LinearSystemSolverWrapperComponent(
                name=Task.LINEAR_SYSTEM_SOLVER,
                constraints={"is_independent": True},
                gen_constraints=GenerationConstraints(square=True, invertible=True),
                entropy_constraints=EntropyConstraints(entropy=entropy_ranges[Task.LINEAR_SYSTEM_SOLVER]),
            ),
            MatrixMatrixMultiplicationWrapperComponent(
                name=Task.MATRIX_MATRIX_MULTIPLICATION,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix_A": 0},
                    "sources": {"input_matrix_A": "result"},
                },
                entropy_constraints=EntropyConstraints(entropy=entropy_ranges[Task.MATRIX_MATRIX_MULTIPLICATION]),
            ),
            FrobeniusNormWrapperComponent(
                name=Task.FROBENIUS_NORM,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 1},
                    "sources": {"input_matrix": "result"},
                },
                entropy_constraints=EntropyConstraints(entropy=entropy_ranges[Task.FROBENIUS_NORM]),
            ),
        ],
        composition_strategy=SequentialComposition(config=custom_config),
        difficulty_level=DifficultyCategory.THREE_TOOL_CALLS,
        custom_config=custom_config,
    )


def register_composite_triple_inverse_rank(
    registry: "FactoryRegistry", custom_config: ProblemConfig, entropy_ranges: dict[Task, float]
) -> None:
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.THREE_COFACTOR_MATRIXMULT_RANK,
        components=[
            MatrixCofactorWrapperComponent(
                name=Task.MATRIX_COFACTOR,
                constraints={"is_independent": True},
                gen_constraints=GenerationConstraints(square=True),
                entropy_constraints=EntropyConstraints(entropy=entropy_ranges[Task.MATRIX_COFACTOR]),
            ),
            MatrixMatrixMultiplicationWrapperComponent(
                name=Task.MATRIX_MATRIX_MULTIPLICATION,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix_A": 0, "input_matrix_B": 0},
                    "sources": {"input_matrix_A": "matrix", "input_matrix_B": "result"},
                },
                entropy_constraints=EntropyConstraints(entropy=entropy_ranges[Task.MATRIX_MATRIX_MULTIPLICATION]),
            ),
            RankWrapperComponent(
                name=Task.MATRIX_RANK,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 1},
                    "sources": {"input_matrix": "result"},
                },
                entropy_constraints=EntropyConstraints(entropy=entropy_ranges[Task.MATRIX_RANK]),
            ),
        ],
        composition_strategy=SequentialComposition(config=custom_config),
        difficulty_level=DifficultyCategory.THREE_TOOL_CALLS,
        custom_config=custom_config,
    )


def register_transpose_determinant(
    registry: "FactoryRegistry", custom_config: ProblemConfig, entropy_ranges: dict[Task, float]
) -> None:
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.TWO_TRANSPOSE_DETERMINANT,
        components=[
            TransposeWrapperComponent(
                name=Task.MATRIX_TRANSPOSE,
                constraints={"is_independent": True},
                gen_constraints=GenerationConstraints(square=True),
                entropy_constraints=EntropyConstraints(entropy=entropy_ranges[Task.MATRIX_TRANSPOSE]),
            ),
            DeterminantWrapperComponent(
                name=Task.DETERMINANT,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 0},
                    "sources": {"input_matrix": "result"},
                },
                entropy_constraints=EntropyConstraints(entropy=entropy_ranges[Task.DETERMINANT]),
            ),
        ],
        composition_strategy=SequentialComposition(config=custom_config),
        difficulty_level=DifficultyCategory.TWO_TOOL_CALLS,
        custom_config=custom_config,
    )


def register_cofactor_frobenius(
    registry: "FactoryRegistry", custom_config: ProblemConfig, entropy_ranges: dict[Task, float]
) -> None:
    """Register cofactor + frobenius_norm composition."""
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.TWO_COFACTOR_FROBENIUS,
        components=[
            MatrixCofactorWrapperComponent(
                name=Task.MATRIX_COFACTOR,
                constraints={"is_independent": True},
                gen_constraints=GenerationConstraints(square=True),
                entropy_constraints=EntropyConstraints(entropy=entropy_ranges[Task.MATRIX_COFACTOR]),
            ),
            FrobeniusNormWrapperComponent(
                name=Task.FROBENIUS_NORM,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 0},
                    "sources": {"input_matrix": "result"},
                },
                entropy_constraints=EntropyConstraints(entropy=entropy_ranges[Task.FROBENIUS_NORM]),
            ),
        ],
        composition_strategy=SequentialComposition(config=custom_config),
        difficulty_level=DifficultyCategory.TWO_TOOL_CALLS,
        custom_config=custom_config,
    )


def register_inverse_rank(
    registry: "FactoryRegistry", custom_config: ProblemConfig, entropy_ranges: dict[Task, float]
) -> None:
    """Register cofactor + matrix_rank composition."""
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.TWO_COFACTOR_RANK,
        components=[
            MatrixCofactorWrapperComponent(
                name=Task.MATRIX_COFACTOR,
                constraints={"is_independent": True},
                gen_constraints=GenerationConstraints(square=True),
                entropy_constraints=EntropyConstraints(entropy=entropy_ranges[Task.MATRIX_COFACTOR]),
            ),
            RankWrapperComponent(
                name=Task.MATRIX_RANK,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 0},
                    "sources": {"input_matrix": "result"},
                },
                entropy_constraints=EntropyConstraints(entropy=entropy_ranges[Task.MATRIX_RANK]),
            ),
        ],
        composition_strategy=SequentialComposition(config=custom_config),
        difficulty_level=DifficultyCategory.TWO_TOOL_CALLS,
        custom_config=custom_config,
    )


def register_transpose_frobenius(
    registry: "FactoryRegistry", custom_config: ProblemConfig, entropy_ranges: dict[Task, float]
) -> None:
    """Register transpose + frobenius_norm composition."""
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.TWO_TRANSPOSE_FROBENIUS,
        components=[
            TransposeWrapperComponent(
                name=Task.MATRIX_TRANSPOSE,
                constraints={"is_independent": True},
                entropy_constraints=EntropyConstraints(entropy=entropy_ranges[Task.MATRIX_TRANSPOSE]),
            ),
            FrobeniusNormWrapperComponent(
                name=Task.FROBENIUS_NORM,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 0},
                    "sources": {"input_matrix": "result"},
                },
                entropy_constraints=EntropyConstraints(entropy=entropy_ranges[Task.FROBENIUS_NORM]),
            ),
        ],
        composition_strategy=SequentialComposition(config=custom_config),
        difficulty_level=DifficultyCategory.TWO_TOOL_CALLS,
        custom_config=custom_config,
    )


class FactoryRegistry:
    """Registry for managing different question factories."""

    def __init__(self) -> None:
        self._factories: dict[Topic, dict[Task, Callable[[], Question]]] = {}
        self._factory_difficulties: dict[Topic, dict[Task, DifficultyCategory]] = {}
        self._composite_components: dict[Topic, dict[Task, list[tuple[Task, bool]]]] = {}

    def register_factory(
        self,
        topic: Topic,
        problem_type: Task,
        factory: Callable[[], Question],
        difficulty: DifficultyCategory | None = None,
    ) -> None:
        """Register a factory function."""
        if topic not in self._factories:
            self._factories[topic] = {}
            self._factory_difficulties[topic] = {}
        self._factories[topic][problem_type] = factory
        if difficulty is not None:
            self._factory_difficulties[topic][problem_type] = difficulty

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

    def get_factories_by_difficulty(
        self, topic: Topic, difficulty: DifficultyCategory
    ) -> list[Callable[[], Question]]:
        """Return all factories for the given topic and difficulty category."""
        if topic not in self._factories:
            raise ValueError(f"Unknown topic: {topic}")
        if topic not in self._factory_difficulties:
            return []
        factories: list[Callable[[], Question]] = []
        for task, task_difficulty in self._factory_difficulties[topic].items():
            if task_difficulty == difficulty:
                factories.append(self._factories[topic][task])
        return factories

    def register_composite_factory(
        self,
        topic: Topic,
        problem_type: Task,
        components: list[ProblemComponent],
        composition_strategy: CompositionStrategy,
        difficulty_level: DifficultyCategory,
        custom_config: ProblemConfig | None = None,
    ) -> None:
        """Register a composite factory"""
        # A provided config can be used to override the default entropy range.
        config = custom_config or get_problem_config(difficulty_level)
        num_components = len(components)

        factory = create_composite_factory(
            components=components,
            composition_strategy=composition_strategy,
            sample_args=config.create_sample_args_for_composition(num_components),
            difficulty_level=difficulty_level,
            problem_type=problem_type,
            topic=topic,
        )

        # Store difficulty for the composite factory
        self.register_factory(topic, problem_type, factory, difficulty=difficulty_level)

        # Store composite component metadata used in tests
        if topic not in self._composite_components:
            self._composite_components[topic] = {}
        self._composite_components[topic][problem_type] = [(c.name, c.is_independent) for c in components]

    def get_composite_components(self, topic: Topic, problem_type: Task) -> list[tuple[Task, bool]]:
        """Return (component task, is_independent) for a registered composite problem, or empty list."""
        return self._composite_components.get(topic, {}).get(problem_type, [])


def create_default_registry() -> FactoryRegistry:
    """Create and populate the default factory registry."""
    registry = FactoryRegistry()

    # ===================
    # 1-STEP COMPOSITIONS
    # ===================
    register_determinant_factory(registry)
    register_matrix_matrix_multiplication_factory(registry)
    register_linear_system_solver_factory(registry)
    register_frobenius_norm_factory(registry)
    register_matrix_rank_factory(registry)
    register_matrix_transpose_factory(registry)
    register_matrix_cofactor_factory(registry)

    # ===================
    # 2-STEP COMPOSITIONS
    # ===================
    custom_config = get_problem_config(DifficultyCategory.TWO_TOOL_CALLS)
    entropy_ranges: dict[Task, dict[Task, float]] = {
        Task.TWO_TRANSPOSE_DETERMINANT: {
            Task.MATRIX_TRANSPOSE: 0.8,
            Task.DETERMINANT: 0.8,
        },
        Task.TWO_COFACTOR_FROBENIUS: {
            Task.MATRIX_COFACTOR: 0.8,
            Task.FROBENIUS_NORM: 0.8,
        },
        Task.TWO_COFACTOR_RANK: {
            Task.MATRIX_COFACTOR: 0.8,
            Task.MATRIX_RANK: 0.8,
        },
        Task.TWO_TRANSPOSE_FROBENIUS: {
            Task.MATRIX_TRANSPOSE: 0.8,
            Task.FROBENIUS_NORM: 0.8,
        },
    }

    register_transpose_determinant(
        registry, custom_config=custom_config, entropy_ranges=entropy_ranges[Task.TWO_TRANSPOSE_DETERMINANT]
    )
    register_cofactor_frobenius(
        registry, custom_config=custom_config, entropy_ranges=entropy_ranges[Task.TWO_COFACTOR_FROBENIUS]
    )
    register_inverse_rank(registry, custom_config=custom_config, entropy_ranges=entropy_ranges[Task.TWO_COFACTOR_RANK])
    register_transpose_frobenius(
        registry, custom_config=custom_config, entropy_ranges=entropy_ranges[Task.TWO_TRANSPOSE_FROBENIUS]
    )

    # ===================
    # 3-STEP COMPOSITIONS
    # ===================
    custom_config = get_problem_config(DifficultyCategory.THREE_TOOL_CALLS)
    entropy_ranges = {
        Task.THREE_TRANSPOSE_MATRIXMULT_DETERMINANT: {
            Task.MATRIX_TRANSPOSE: 0.8,
            Task.MATRIX_MATRIX_MULTIPLICATION: 0.8,
            Task.DETERMINANT: 0.8,
        },
        Task.THREE_COFACTOR_MATRIXMULT_RANK: {
            Task.MATRIX_COFACTOR: 0.8,
            Task.MATRIX_MATRIX_MULTIPLICATION: 0.8,
            Task.MATRIX_RANK: 0.8,
        },
        Task.THREE_SYSTEM_MATRIXMULT_FROBENIUS: {
            Task.LINEAR_SYSTEM_SOLVER: 0.8,
            Task.MATRIX_MATRIX_MULTIPLICATION: 0.8,
            Task.FROBENIUS_NORM: 0.8,
        },
    }

    register_composite_triple_transpose_determinant(
        registry,
        custom_config=custom_config,
        entropy_ranges=entropy_ranges[Task.THREE_TRANSPOSE_MATRIXMULT_DETERMINANT],
    )
    register_composite_triple_inverse_rank(
        registry, custom_config=custom_config, entropy_ranges=entropy_ranges[Task.THREE_COFACTOR_MATRIXMULT_RANK]
    )
    register_composite_triple_system_frobenius(
        registry, custom_config=custom_config, entropy_ranges=entropy_ranges[Task.THREE_SYSTEM_MATRIXMULT_FROBENIUS]
    )

    return registry
