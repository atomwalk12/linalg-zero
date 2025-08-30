from collections.abc import Callable
from typing import Any

from linalg_zero.generator.composition.composition import (
    CompositeProblem,
    CompositionStrategy,
    ProblemComponent,
)
from linalg_zero.generator.difficulty_config import SampleArgs
from linalg_zero.generator.models import DifficultyCategory, Question, Task, Topic
from linalg_zero.generator.sympy.base import SympyProblemGenerator
from linalg_zero.generator.sympy.generators.determinant_generator import DeterminantGenerator
from linalg_zero.generator.sympy.generators.frobenius_norm_generator import FrobeniusNormGenerator
from linalg_zero.generator.sympy.generators.linear_system_generator import LinearSystemGenerator
from linalg_zero.generator.sympy.generators.matrix_rank_generator import MatrixRankGenerator
from linalg_zero.generator.sympy.generators.matrix_transpose_generator import MatrixTransposeGenerator
from linalg_zero.generator.sympy.generators.matrix_vector_generator import (
    MatrixVectorMultiplicationGenerator,
)


def create_frobenius_norm_factory(difficulty: DifficultyCategory) -> Callable[[], Question]:
    """Helper to create frobenius norm generator with specific parameters."""
    return create_sympy_factory(
        FrobeniusNormGenerator,
        difficulty_level=difficulty,
        problem_type=Task.FROBENIUS_NORM,
        topic=Topic.LINEAR_ALGEBRA,
    )


def create_linear_system_generator(difficulty: DifficultyCategory) -> Callable[[], Question]:
    """Helper to create linear system generator with specific parameters."""
    return create_sympy_factory(
        LinearSystemGenerator,
        difficulty_level=difficulty,
        problem_type=Task.LINEAR_SYSTEM_SOLVER,
        topic=Topic.LINEAR_ALGEBRA,
    )


def create_matrix_vector_multiplication_factory(difficulty: DifficultyCategory) -> Callable[[], Question]:
    """Helper to create matrix-vector multiplication factory with specific parameters."""
    return create_sympy_factory(
        MatrixVectorMultiplicationGenerator,
        difficulty_level=difficulty,
        problem_type=Task.MATRIX_VECTOR_MULTIPLICATION,
        topic=Topic.LINEAR_ALGEBRA,
    )


def create_determinant_factory(difficulty: DifficultyCategory) -> Callable[[], Question]:
    """Helper to create determinant factory with specific parameters."""
    return create_sympy_factory(
        DeterminantGenerator,
        difficulty_level=difficulty,
        problem_type=Task.DETERMINANT,
        topic=Topic.LINEAR_ALGEBRA,
    )


def create_matrix_rank_factory(difficulty: DifficultyCategory) -> Callable[[], Question]:
    """Helper to create matrix rank factory with specific parameters."""
    return create_sympy_factory(
        MatrixRankGenerator,
        difficulty_level=difficulty,
        problem_type=Task.MATRIX_RANK,
        topic=Topic.LINEAR_ALGEBRA,
    )


def create_matrix_transpose_factory(difficulty: DifficultyCategory) -> Callable[[], Question]:
    """Helper to create matrix transpose factory with specific parameters."""
    return create_sympy_factory(
        MatrixTransposeGenerator,
        difficulty_level=difficulty,
        problem_type=Task.MATRIX_TRANSPOSE,
        topic=Topic.LINEAR_ALGEBRA,
    )


def create_composite_factory(
    components: list[ProblemComponent],
    composition_strategy: CompositionStrategy,
    sample_args: SampleArgs,
    difficulty_level: DifficultyCategory,
    problem_type: Task,
    topic: Topic,
) -> Callable[[], Question]:
    """
    Factory function for creating composite problem generators.
    """

    def factory() -> Question:
        generator = CompositeProblem(
            components=components,
            composition_strategy=composition_strategy,
            sample_args=sample_args,
            difficulty_level=difficulty_level,
            problem_type=problem_type,
            topic=topic,
        )
        return generator.generate()

    return factory


def create_sympy_factory(
    generator_class: type,
    difficulty_level: DifficultyCategory,
    problem_type: Task,
    topic: Topic,
    **kwargs: Any,
) -> Callable[[], Question]:
    """
    Convenience function for generating a factory function for registry registration.
    """

    def factory() -> Question:
        generator: SympyProblemGenerator = generator_class(
            difficulty_level=difficulty_level,
            problem_type=problem_type,
            topic=topic,
            **kwargs,
        )
        return generator.generate()

    return factory
