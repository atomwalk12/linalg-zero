from collections.abc import Callable
from typing import Any

from linalg_zero.generator.composition.composition import (
    CompositeProblem,
    CompositionStrategy,
    ProblemComponent,
)
from linalg_zero.generator.entropy_control import EntropyConstraints
from linalg_zero.generator.generation_constraints import GenerationConstraints
from linalg_zero.generator.models import DifficultyCategory, Question, Task, Topic
from linalg_zero.generator.sympy.base import SympyProblemGenerator
from linalg_zero.generator.sympy.template_engine import TemplateEngine


def create_composite_factory(
    components: list[ProblemComponent],
    composition_strategy: CompositionStrategy,
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
            template_engine=TemplateEngine(),
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
    entropy: EntropyConstraints,
    gen_constraints: GenerationConstraints | None = None,
    **kwargs: Any,
) -> Callable[[], Question]:
    """
    Convenience function for generating a factory function for registry registration.
    """
    value = entropy.sample_entropy()

    def factory() -> Question:
        generator: SympyProblemGenerator = generator_class(
            difficulty_level=difficulty_level,
            problem_type=problem_type,
            topic=topic,
            template_engine=TemplateEngine(),
            entropy=value,
            local_index=0,
            gen_constraints=gen_constraints,
            constraints={},
            **kwargs,
        )
        return generator.generate()

    return factory
