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
from linalg_zero.generator.entropy_control import EntropyConstraints
from linalg_zero.generator.generation_constraints import GenerationConstraints
from linalg_zero.generator.generator_factories import (
    create_composite_factory,
    create_sympy_factory,
)
from linalg_zero.generator.models import DifficultyCategory, Question, Task, Topic
from linalg_zero.generator.sympy.base import (
    CompositionStrategy,
    ProblemComponent,
)
from linalg_zero.generator.sympy.generators.determinant_generator import DeterminantGenerator
from linalg_zero.generator.sympy.generators.frobenius_norm_generator import FrobeniusNormGenerator
from linalg_zero.generator.sympy.generators.linear_system_generator import LinearSystemGenerator
from linalg_zero.generator.sympy.generators.matrix_cofactor_generator import MatrixCofactorGenerator
from linalg_zero.generator.sympy.generators.matrix_matrix_generator import (
    MatrixMatrixMultiplicationGenerator,
)
from linalg_zero.generator.sympy.generators.matrix_rank_generator import MatrixRankGenerator
from linalg_zero.generator.sympy.generators.matrix_transpose_generator import (
    MatrixTransposeGenerator,
)


def register_one_determinant_factory(registry: "FactoryRegistry", entropy: tuple[float, float] | float) -> None:
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.ONE_DETERMINANT,
        create_sympy_factory(
            generator_class=DeterminantGenerator,
            topic=Topic.LINEAR_ALGEBRA,
            problem_type=Task.ONE_DETERMINANT,
            difficulty_level=DifficultyCategory.ONE_TOOL_CALL,
            entropy=EntropyConstraints(entropy),
        ),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )


def register_one_matrix_matrix_multiplication_factory(
    registry: "FactoryRegistry", entropy: tuple[float, float] | float
) -> None:
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.ONE_MATRIX_MATRIX_MULTIPLICATION,
        create_sympy_factory(
            generator_class=MatrixMatrixMultiplicationGenerator,
            topic=Topic.LINEAR_ALGEBRA,
            problem_type=Task.ONE_MATRIX_MATRIX_MULTIPLICATION,
            difficulty_level=DifficultyCategory.ONE_TOOL_CALL,
            entropy=EntropyConstraints(entropy),
        ),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )


def register_one_linear_system_solver_factory(
    registry: "FactoryRegistry", entropy: tuple[float, float] | float
) -> None:
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.ONE_LINEAR_SYSTEM_SOLVER,
        create_sympy_factory(
            generator_class=LinearSystemGenerator,
            topic=Topic.LINEAR_ALGEBRA,
            problem_type=Task.ONE_LINEAR_SYSTEM_SOLVER,
            difficulty_level=DifficultyCategory.ONE_TOOL_CALL,
            entropy=EntropyConstraints(entropy),
        ),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )


def register_one_frobenius_norm_factory(registry: "FactoryRegistry", entropy: tuple[float, float] | float) -> None:
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.ONE_FROBENIUS_NORM,
        create_sympy_factory(
            generator_class=FrobeniusNormGenerator,
            topic=Topic.LINEAR_ALGEBRA,
            problem_type=Task.ONE_FROBENIUS_NORM,
            difficulty_level=DifficultyCategory.ONE_TOOL_CALL,
            entropy=EntropyConstraints(entropy),
        ),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )


def register_one_matrix_rank_factory(registry: "FactoryRegistry", entropy: tuple[float, float] | float) -> None:
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.ONE_RANK,
        create_sympy_factory(
            generator_class=MatrixRankGenerator,
            topic=Topic.LINEAR_ALGEBRA,
            problem_type=Task.ONE_RANK,
            difficulty_level=DifficultyCategory.ONE_TOOL_CALL,
            entropy=EntropyConstraints(entropy),
        ),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )


def register_one_matrix_transpose_factory(registry: "FactoryRegistry", entropy: tuple[float, float] | float) -> None:
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.ONE_TRANSPOSE,
        create_sympy_factory(
            generator_class=MatrixTransposeGenerator,
            topic=Topic.LINEAR_ALGEBRA,
            problem_type=Task.ONE_TRANSPOSE,
            difficulty_level=DifficultyCategory.ONE_TOOL_CALL,
            entropy=EntropyConstraints(entropy),
        ),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )


def register_one_matrix_cofactor_factory(registry: "FactoryRegistry", entropy: tuple[float, float] | float) -> None:
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.ONE_COFACTOR,
        create_sympy_factory(
            generator_class=MatrixCofactorGenerator,
            topic=Topic.LINEAR_ALGEBRA,
            problem_type=Task.ONE_COFACTOR,
            difficulty_level=DifficultyCategory.ONE_TOOL_CALL,
            entropy=EntropyConstraints(entropy),
        ),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )


def register_three_transpose_matrixmult_determinant(
    registry: "FactoryRegistry", entropy: dict[Task, tuple[float, float] | float]
) -> None:
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.THREE_TRANSPOSE_MATRIXMULT_DETERMINANT,
        components=[
            TransposeWrapperComponent(
                name=Task.ONE_TRANSPOSE,
                constraints={"is_independent": True},
                gen_constraints=GenerationConstraints(square=True),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_TRANSPOSE]),
            ),
            MatrixMatrixMultiplicationWrapperComponent(
                name=Task.ONE_MATRIX_MATRIX_MULTIPLICATION,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix_A": 0, "input_matrix_B": 0},
                    "sources": {"input_matrix_A": "matrix", "input_matrix_B": "result"},
                },
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_MATRIX_MATRIX_MULTIPLICATION]),
            ),
            DeterminantWrapperComponent(
                name=Task.ONE_DETERMINANT,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 1},
                    "sources": {"input_matrix": "result"},
                },
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_DETERMINANT]),
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.THREE_TOOL_CALLS,
    )


def register_three_system_matrixmult_frobenius(
    registry: "FactoryRegistry", entropy: dict[Task, tuple[float, float] | float]
) -> None:
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.THREE_SYSTEM_MATRIXMULT_FROBENIUS,
        components=[
            LinearSystemSolverWrapperComponent(
                name=Task.ONE_LINEAR_SYSTEM_SOLVER,
                constraints={"is_independent": True},
                gen_constraints=GenerationConstraints(square=True, invertible=True),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_LINEAR_SYSTEM_SOLVER]),
            ),
            MatrixMatrixMultiplicationWrapperComponent(
                name=Task.ONE_MATRIX_MATRIX_MULTIPLICATION,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix_A": 0},
                    "sources": {"input_matrix_A": "result"},
                },
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_MATRIX_MATRIX_MULTIPLICATION]),
            ),
            FrobeniusNormWrapperComponent(
                name=Task.ONE_FROBENIUS_NORM,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 1},
                    "sources": {"input_matrix": "result"},
                },
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_FROBENIUS_NORM]),
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.THREE_TOOL_CALLS,
    )


def register_three_cofactor_matrixmult_rank(
    registry: "FactoryRegistry", entropy: dict[Task, tuple[float, float] | float]
) -> None:
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.THREE_COFACTOR_MATRIXMULT_RANK,
        components=[
            MatrixCofactorWrapperComponent(
                name=Task.ONE_COFACTOR,
                constraints={"is_independent": True},
                gen_constraints=GenerationConstraints(square=True),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_COFACTOR]),
            ),
            MatrixMatrixMultiplicationWrapperComponent(
                name=Task.ONE_MATRIX_MATRIX_MULTIPLICATION,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix_A": 0, "input_matrix_B": 0},
                    "sources": {"input_matrix_A": "matrix", "input_matrix_B": "result"},
                },
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_MATRIX_MATRIX_MULTIPLICATION]),
            ),
            RankWrapperComponent(
                name=Task.ONE_RANK,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 1},
                    "sources": {"input_matrix": "result"},
                },
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_RANK]),
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.THREE_TOOL_CALLS,
    )


def register_two_transpose_determinant(
    registry: "FactoryRegistry", entropy: dict[Task, tuple[float, float] | float]
) -> None:
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.TWO_TRANSPOSE_DETERMINANT,
        components=[
            TransposeWrapperComponent(
                name=Task.ONE_TRANSPOSE,
                constraints={"is_independent": True},
                gen_constraints=GenerationConstraints(square=True),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_TRANSPOSE]),
            ),
            DeterminantWrapperComponent(
                name=Task.ONE_DETERMINANT,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 0},
                    "sources": {"input_matrix": "result"},
                },
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_DETERMINANT]),
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.TWO_TOOL_CALLS,
    )


def register_two_cofactor_frobenius(
    registry: "FactoryRegistry", entropy: dict[Task, tuple[float, float] | float]
) -> None:
    """Register cofactor + frobenius_norm composition."""
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.TWO_COFACTOR_FROBENIUS,
        components=[
            MatrixCofactorWrapperComponent(
                name=Task.ONE_COFACTOR,
                constraints={"is_independent": True},
                gen_constraints=GenerationConstraints(square=True),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_COFACTOR]),
            ),
            FrobeniusNormWrapperComponent(
                name=Task.ONE_FROBENIUS_NORM,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 0},
                    "sources": {"input_matrix": "result"},
                },
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_FROBENIUS_NORM]),
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.TWO_TOOL_CALLS,
    )


def register_two_cofactor_rank(registry: "FactoryRegistry", entropy: dict[Task, tuple[float, float] | float]) -> None:
    """Register cofactor + matrix_rank composition."""
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.TWO_COFACTOR_RANK,
        components=[
            MatrixCofactorWrapperComponent(
                name=Task.ONE_COFACTOR,
                constraints={"is_independent": True},
                gen_constraints=GenerationConstraints(square=True),
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_COFACTOR]),
            ),
            RankWrapperComponent(
                name=Task.ONE_RANK,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 0},
                    "sources": {"input_matrix": "result"},
                },
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_RANK]),
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.TWO_TOOL_CALLS,
    )


def register_two_transpose_frobenius(
    registry: "FactoryRegistry", entropy: dict[Task, tuple[float, float] | float]
) -> None:
    """Register transpose + frobenius_norm composition."""
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.TWO_TRANSPOSE_FROBENIUS,
        components=[
            TransposeWrapperComponent(
                name=Task.ONE_TRANSPOSE,
                constraints={"is_independent": True},
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_TRANSPOSE]),
            ),
            FrobeniusNormWrapperComponent(
                name=Task.ONE_FROBENIUS_NORM,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix": 0},
                    "sources": {"input_matrix": "result"},
                },
                entropy_constraints=EntropyConstraints(entropy[Task.ONE_FROBENIUS_NORM]),
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.TWO_TOOL_CALLS,
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
    ) -> None:
        """Register a composite factory"""
        factory = create_composite_factory(
            components=components,
            composition_strategy=composition_strategy,
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
    register_one_determinant_factory(registry, entropy=(0.8, 0.8))
    register_one_matrix_matrix_multiplication_factory(registry, entropy=(0.8, 0.8))
    register_one_linear_system_solver_factory(registry, entropy=(0.8, 0.8))
    register_one_frobenius_norm_factory(registry, entropy=(0.8, 0.8))
    register_one_matrix_rank_factory(registry, entropy=(0.8, 0.8))
    register_one_matrix_transpose_factory(registry, entropy=(0.8, 0.8))
    register_one_matrix_cofactor_factory(registry, entropy=(0.8, 0.8))

    # ===================
    # 2-STEP COMPOSITIONS
    # ===================
    entropy_ranges: dict[Task, dict[Task, tuple[float, float] | float]] = {
        Task.TWO_TRANSPOSE_DETERMINANT: {
            Task.ONE_TRANSPOSE: (0.8, 0.8),
            Task.ONE_DETERMINANT: (0.8, 0.8),
        },
        Task.TWO_COFACTOR_FROBENIUS: {
            Task.ONE_COFACTOR: (0.8, 0.8),
            Task.ONE_FROBENIUS_NORM: (0.8, 0.8),
        },
        Task.TWO_COFACTOR_RANK: {
            Task.ONE_COFACTOR: (0.8, 0.8),
            Task.ONE_RANK: (0.8, 0.8),
        },
        Task.TWO_TRANSPOSE_FROBENIUS: {
            Task.ONE_TRANSPOSE: (0.8, 0.8),
            Task.ONE_FROBENIUS_NORM: (0.8, 0.8),
        },
    }

    register_two_transpose_determinant(registry, entropy=entropy_ranges[Task.TWO_TRANSPOSE_DETERMINANT])
    register_two_cofactor_frobenius(registry, entropy=entropy_ranges[Task.TWO_COFACTOR_FROBENIUS])
    register_two_cofactor_rank(registry, entropy=entropy_ranges[Task.TWO_COFACTOR_RANK])
    register_two_transpose_frobenius(registry, entropy=entropy_ranges[Task.TWO_TRANSPOSE_FROBENIUS])

    # ===================
    # 3-STEP COMPOSITIONS
    # ===================
    entropy_ranges = {
        Task.THREE_TRANSPOSE_MATRIXMULT_DETERMINANT: {
            Task.ONE_TRANSPOSE: (0.8, 0.8),
            Task.ONE_MATRIX_MATRIX_MULTIPLICATION: (0.8, 0.8),
            Task.ONE_DETERMINANT: (0.8, 0.8),
        },
        Task.THREE_COFACTOR_MATRIXMULT_RANK: {
            Task.ONE_COFACTOR: (0.8, 0.8),
            Task.ONE_MATRIX_MATRIX_MULTIPLICATION: (0.8, 0.8),
            Task.ONE_RANK: (0.8, 0.8),
        },
        Task.THREE_SYSTEM_MATRIXMULT_FROBENIUS: {
            Task.ONE_LINEAR_SYSTEM_SOLVER: (0.8, 0.8),
            Task.ONE_MATRIX_MATRIX_MULTIPLICATION: (0.8, 0.8),
            Task.ONE_FROBENIUS_NORM: (0.8, 0.8),
        },
    }

    register_three_transpose_matrixmult_determinant(
        registry,
        entropy=entropy_ranges[Task.THREE_TRANSPOSE_MATRIXMULT_DETERMINANT],
    )
    register_three_cofactor_matrixmult_rank(registry, entropy=entropy_ranges[Task.THREE_COFACTOR_MATRIXMULT_RANK])
    register_three_system_matrixmult_frobenius(
        registry, entropy=entropy_ranges[Task.THREE_SYSTEM_MATRIXMULT_FROBENIUS]
    )

    return registry
