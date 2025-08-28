from collections.abc import Callable
from typing import Any

from linalg_zero.generator.composition.composition import (
    CompositeProblem,
    CompositionStrategy,
    ProblemComponent,
)
from linalg_zero.generator.difficulty_config import SampleArgs
from linalg_zero.generator.models import DifficultyCategory, Question
from linalg_zero.generator.sympy.base import SympyProblemGenerator
from linalg_zero.generator.sympy.generators.determinant_generator import DeterminantGenerator
from linalg_zero.generator.sympy.generators.linear_system_generator import LinearSystemGenerator
from linalg_zero.generator.sympy.generators.matrix_vector_generator import (
    MatrixVectorMultiplicationGenerator,
)


def create_matrix_vector_equation_solver_factory(
    entropy: float, difficulty: DifficultyCategory
) -> Callable[[], Question]:
    """Helper to create matrix-vector equation solver factory with specific parameters."""
    return create_sympy_factory(
        LinearSystemGenerator,
        entropy=entropy,
        difficulty_level=difficulty,
        problem_type="matrix_vector_equation_solving",
        topic="linear_algebra",
    )


def create_matrix_vector_multiplication_factory(
    entropy: float, difficulty: DifficultyCategory
) -> Callable[[], Question]:
    """Helper to create matrix-vector multiplication factory with specific parameters."""
    return create_sympy_factory(
        MatrixVectorMultiplicationGenerator,
        entropy=entropy,
        difficulty_level=difficulty,
        problem_type="matrix_vector_multiplication",
        topic="linear_algebra",
    )


def create_determinant_factory(entropy: float, difficulty: DifficultyCategory) -> Callable[[], Question]:
    """Helper to create determinant factory with specific parameters."""
    return create_sympy_factory(
        DeterminantGenerator,
        entropy=entropy,
        difficulty_level=difficulty,
        problem_type="calculate_determinant",
        topic="linear_algebra",
    )


def create_composite_factory(
    components: list[ProblemComponent],
    composition_strategy: CompositionStrategy,
    sample_args: SampleArgs,
    difficulty_level: DifficultyCategory,
    problem_type: str,
    topic: str,
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
    entropy: float,
    difficulty_level: DifficultyCategory,
    problem_type: str = "unknown",
    topic: str = "linear_algebra",
    **kwargs: Any,
) -> Callable[[], Question]:
    """
    Convenience function for generating a factory function for registry registration.
    """

    def factory() -> Question:
        generator: SympyProblemGenerator = generator_class(
            entropy=entropy,
            difficulty_level=difficulty_level,
            problem_type=problem_type,
            topic=topic,
            **kwargs,
        )
        return generator.generate()

    return factory
