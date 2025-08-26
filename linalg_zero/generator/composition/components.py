from typing import Any

from linalg_zero.generator.composition.composition import (
    ComponentResult,
    CompositionContext,
    ProblemComponent,
)
from linalg_zero.generator.sympy.base import ProblemTemplate, SympyProblemGenerator


class SympyGeneratorWrapperComponent(ProblemComponent):
    """Generic base class for wrapping sympy generators in the composition system."""

    def __init__(
        self,
        name: str,
        generator_class: type,
        component_type: str,
        context_update_mapping: dict[str, str],
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.generator_class = generator_class
        self.component_type = component_type
        self.context_update_mapping = context_update_mapping

    def generate(self, context: CompositionContext) -> ComponentResult:
        from linalg_zero.generator.sympy.base import ProblemContext

        # This context is used for communication and state tracking
        problem_context = ProblemContext(entropy=context.entropy, difficulty_level=context.difficulty_level)

        # Now, we perform the 3 key steps involved in component generation
        generator: SympyProblemGenerator = self.generator_class(
            entropy=problem_context.entropy, difficulty_level=context.difficulty_level
        )
        template: ProblemTemplate = generator.generate_mathematical_content(problem_context)
        formatted_question = generator.format_question(template)

        # Transfer the state of the problem context to the new problem template
        formatted_template = ProblemTemplate(
            expression=formatted_question,
            variables=template.variables,
            sympy_solution=template.sympy_solution,
            lib_result=template.lib_result,
            question_templates=template.question_templates,
            context_info={
                **template.context_info,
            },
            difficulty_markers=template.difficulty_markers,
            difficulty=template.difficulty,
        )

        # Build context updates using the mapping
        context_updates = {}
        for update_key, template_key in self.context_update_mapping.items():
            if template_key == "sympy_solution":
                context_updates[f"{self.name}_{update_key}"] = template.sympy_solution
            else:
                context_updates[f"{self.name}_{update_key}"] = template.context_info.get(template_key)

        return ComponentResult(
            template=formatted_template,
            context_updates=context_updates,
            entropy_consumed=problem_context.used_entropy,
            tool_calls_used=problem_context.tool_calls_count,
            generator=generator,
        )


class MatrixVectorMultiplicationWrapperComponent(SympyGeneratorWrapperComponent):
    """Wrapper for the MatrixVectorMultiplicationGenerator."""

    def __init__(self, name: str = "matrix_vector_mult", **kwargs: Any) -> None:
        from linalg_zero.generator.sympy.generators.matrix_vector_generator import (
            MatrixVectorMultiplicationGenerator,
        )

        super().__init__(
            name=name,
            generator_class=MatrixVectorMultiplicationGenerator,
            component_type="matrix_vector_multiplication",
            context_update_mapping={"matrix": "matrix", "vector": "vector", "result": "sympy_solution"},
            **kwargs,
        )


class LinearSystemSolverWrapperComponent(SympyGeneratorWrapperComponent):
    """Wrapper for the LinearSystemGenerator."""

    def __init__(self, name: str = "linear_system_solver", **kwargs: Any) -> None:
        from linalg_zero.generator.sympy.generators.linear_system_generator import (
            LinearSystemGenerator,
        )

        super().__init__(
            name=name,
            generator_class=LinearSystemGenerator,
            component_type="linear_system",
            context_update_mapping={
                "matrix_A": "matrix_A",
                "x_symbols": "x_symbols",
                "target_b": "target_b",
                "solution": "sympy_solution",
            },
            **kwargs,
        )
