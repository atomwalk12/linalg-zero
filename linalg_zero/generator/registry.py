import random
from collections.abc import Callable

from linalg_zero.generator.composition.components import (
    DeterminantWrapperComponent,
    FrobeniusNormWrapperComponent,
    LinearSystemSolverWrapperComponent,
    MatrixInverseWrapperComponent,
    MatrixMatrixMultiplicationWrapperComponent,
    MatrixVectorMultiplicationWrapperComponent,
    RankWrapperComponent,
    TransposeWrapperComponent,
)
from linalg_zero.generator.composition.composition import (
    SequentialComposition,
)
from linalg_zero.generator.difficulty_config import get_problem_config
from linalg_zero.generator.generation_constraints import GenerationConstraints
from linalg_zero.generator.generator_factories import (
    create_composite_factory,
    create_determinant_factory,
    create_frobenius_norm_factory,
    create_linear_system_generator,
    create_matrix_inverse_factory,
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
        self._factory_difficulties: dict[Topic, dict[Task, DifficultyCategory]] = {}

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

        # Store difficulty for the composite factory
        self.register_factory(topic, problem_type, factory, difficulty=difficulty_level)


def create_default_registry() -> FactoryRegistry:
    """Create and populate the default factory registry."""
    registry = FactoryRegistry()

    # ===================
    # 1-STEP COMPOSITIONS
    # ===================

    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.DETERMINANT,
        create_determinant_factory(difficulty=DifficultyCategory.ONE_TOOL_CALL),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )

    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.MATRIX_VECTOR_MULTIPLICATION,
        create_matrix_vector_multiplication_factory(difficulty=DifficultyCategory.ONE_TOOL_CALL),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.LINEAR_SYSTEM_SOLVER,
        create_linear_system_generator(difficulty=DifficultyCategory.ONE_TOOL_CALL),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.FROBENIUS_NORM,
        create_frobenius_norm_factory(difficulty=DifficultyCategory.ONE_TOOL_CALL),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.MATRIX_RANK,
        create_matrix_rank_factory(difficulty=DifficultyCategory.ONE_TOOL_CALL),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.MATRIX_TRANSPOSE,
        create_matrix_transpose_factory(difficulty=DifficultyCategory.ONE_TOOL_CALL),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )
    registry.register_factory(
        Topic.LINEAR_ALGEBRA,
        Task.MATRIX_INVERSE,
        create_matrix_inverse_factory(difficulty=DifficultyCategory.ONE_TOOL_CALL),
        difficulty=DifficultyCategory.ONE_TOOL_CALL,
    )

    # ===================
    # 2-STEP COMPOSITIONS
    # ===================
    # Problem 1: A → A^T → det(A^T) (transpose + determinant)
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.COMPOSITE_TRANSPOSE_DETERMINANT_BALANCED,
        components=[
            TransposeWrapperComponent(
                name=Task.MATRIX_TRANSPOSE,
                constraints={"is_independent": True},
                gen_constraints=GenerationConstraints(square=True),
            ),
            DeterminantWrapperComponent(
                name=Task.DETERMINANT,
                constraints={"is_independent": False, "input_indices": {"input_matrix": 0}},
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.TWO_TOOL_CALLS,
    )

    # Problem 2: A → A^(-1) → ||A^(-1)||_F (inverse + frobenius_norm)
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.COMPOSITE_INVERSE_FROBENIUS,
        components=[
            MatrixInverseWrapperComponent(
                name=Task.MATRIX_INVERSE,
                constraints={"is_independent": True},
                gen_constraints=GenerationConstraints(square=True, invertible=True),
            ),
            FrobeniusNormWrapperComponent(
                name=Task.FROBENIUS_NORM,
                constraints={"is_independent": False, "input_indices": {"input_matrix": 0}},
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.TWO_TOOL_CALLS,
    )

    # Problem 3: A → A^(-1) → rank(A^(-1)) (inverse + matrix_rank)
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.COMPOSITE_INVERSE_RANK,
        components=[
            MatrixInverseWrapperComponent(
                name=Task.MATRIX_INVERSE,
                constraints={"is_independent": True},
                gen_constraints=GenerationConstraints(square=True, invertible=True),
            ),
            RankWrapperComponent(
                name=Task.MATRIX_RANK,
                constraints={"is_independent": False, "input_indices": {"input_matrix": 0}},
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.TWO_TOOL_CALLS,
    )

    # Problem 4: A → A^T → ||A^T||_F (transpose + frobenius_norm)
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.COMPOSITE_TRANSPOSE_FROBENIUS,
        components=[
            TransposeWrapperComponent(
                name=Task.MATRIX_TRANSPOSE,
                constraints={"is_independent": True},
            ),
            FrobeniusNormWrapperComponent(
                name=Task.FROBENIUS_NORM,
                constraints={"is_independent": False, "input_indices": {"input_matrix": 0}},
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.TWO_TOOL_CALLS,
    )

    # ===================
    # 3-STEP COMPOSITIONS
    # ===================

    # Problem 1: A → A^T → A^T*A → det(A^T*A) (transpose + multiply + determinant)
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.COMPOSITE_TRIPLE_TRANSPOSE_DETERMINANT,
        components=[
            TransposeWrapperComponent(
                name=Task.MATRIX_TRANSPOSE,
                constraints={"is_independent": True},
                gen_constraints=GenerationConstraints(square=True),
            ),
            MatrixMatrixMultiplicationWrapperComponent(
                name=Task.MATRIX_VECTOR_MULTIPLICATION,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix_A": 0, "input_matrix_B": 0},
                    "sources": {"input_matrix_A": "matrix", "input_matrix_B": "result"},
                },
            ),
            DeterminantWrapperComponent(
                name=Task.DETERMINANT,
                constraints={"is_independent": False, "input_indices": {"input_matrix": 1}},
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.THREE_TOOL_CALLS,
    )

    # Problem 2: A → A^(-1) → A^(-1)*B → rank(result) (inverse + multiply + matrix_rank)
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.COMPOSITE_TRIPLE_INVERSE_RANK,
        components=[
            MatrixInverseWrapperComponent(
                name=Task.MATRIX_INVERSE,
                constraints={"is_independent": True},
                gen_constraints=GenerationConstraints(square=True, invertible=True),
            ),
            MatrixMatrixMultiplicationWrapperComponent(
                name=Task.MATRIX_VECTOR_MULTIPLICATION,
                constraints={
                    "is_independent": False,
                    "input_indices": {"input_matrix_A": 0, "input_matrix_B": 0},
                    "sources": {"input_matrix_A": "matrix", "input_matrix_B": "result"},
                },
            ),
            RankWrapperComponent(
                name=Task.MATRIX_RANK,
                constraints={"is_independent": False, "input_indices": {"input_matrix": 1}},
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.THREE_TOOL_CALLS,
    )

    # Problem 3: solve(Ax=b) → C*x → ||C*x||_F (solve + multiply + frobenius_norm)
    registry.register_composite_factory(
        topic=Topic.LINEAR_ALGEBRA,
        problem_type=Task.COMPOSITE_TRIPLE_SYSTEM_FROBENIUS,
        components=[
            LinearSystemSolverWrapperComponent(
                name=Task.LINEAR_SYSTEM_SOLVER,
                constraints={"is_independent": True},
                gen_constraints=GenerationConstraints(square=True, invertible=True),
            ),
            MatrixVectorMultiplicationWrapperComponent(
                name=Task.MATRIX_VECTOR_MULTIPLICATION,
                constraints={"is_independent": False, "input_indices": {"input_vector_b": 0}},
            ),
            FrobeniusNormWrapperComponent(
                name=Task.FROBENIUS_NORM,
                constraints={"is_independent": False, "input_indices": {"input_matrix": 1}},
            ),
        ],
        composition_strategy=SequentialComposition(),
        difficulty_level=DifficultyCategory.THREE_TOOL_CALLS,
    )

    return registry
