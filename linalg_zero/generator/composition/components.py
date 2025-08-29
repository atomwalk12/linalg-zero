from typing import Any

from linalg_zero.generator.composition.composition import (
    ComponentResult,
    CompositionContext,
    ProblemComponent,
)
from linalg_zero.generator.models import Task, Topic
from linalg_zero.generator.sympy.base import ProblemContext, ProblemTemplate, SympyProblemGenerator
from linalg_zero.generator.sympy.generators.linear_system_generator import (
    LinearSystemGenerator,
)
from linalg_zero.generator.sympy.generators.matrix_vector_generator import (
    MatrixVectorMultiplicationGenerator,
)


class SympyGeneratorWrapperComponent(ProblemComponent):
    """Generic base class for wrapping sympy generators in the composition system."""

    def __init__(
        self,
        name: Task,
        generator_class: type[SympyProblemGenerator],
        component_type: Task,
        topic: Topic,
        context_update_mapping: dict[str, str],
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.generator_class = generator_class
        self.component_type = component_type
        self.topic = topic
        self.context_update_mapping = context_update_mapping

    def get_generator_params(self, context: CompositionContext) -> dict[str, Any]:
        """Override this method to provide additional parameters to the generator based on context."""
        return {}

    def generate(self, context: CompositionContext) -> ComponentResult:
        # This context is used for communication and state tracking
        problem_context = ProblemContext(
            entropy=context.entropy, difficulty_level=context.difficulty_level, step_counter=context._step_counter
        )

        # Get any additional parameters for parameterized generation
        additional_params = self.get_generator_params(context)

        # Now, we perform the 3 key steps involved in component generation
        generator: SympyProblemGenerator = self.generator_class(
            difficulty_level=context.difficulty_level,
            problem_type=self.component_type,
            topic=self.topic,
            entropy=problem_context.entropy,
            composite=True,
            **additional_params,
        )
        template: ProblemTemplate = generator.generate_mathematical_content(problem_context)
        generator.verify_problem(template)

        # Transfer the state of the problem context to the new problem template
        formatted_template = ProblemTemplate(
            expression=template.expression,
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

        context.stepwise_results.extend(problem_context.stepwise_results)
        context.golden_result.update(problem_context.golden_result)
        context._step_counter = problem_context._step_counter

        return ComponentResult(
            template=formatted_template,
            context_updates=context_updates,
            entropy_consumed=problem_context.used_entropy,
            tool_calls_used=problem_context.tool_calls_count,
            generator=generator,
        )


class MatrixVectorMultiplicationWrapperComponent(SympyGeneratorWrapperComponent):
    """Wrapper for the MatrixVectorMultiplicationGenerator."""

    def __init__(self, name: Task, **kwargs: Any) -> None:
        super().__init__(
            name=name,
            generator_class=MatrixVectorMultiplicationGenerator,
            component_type=Task.MATRIX_VECTOR_MULTIPLICATION,
            topic=Topic.LINEAR_ALGEBRA,
            context_update_mapping={"matrix": "matrix", "vector": "vector", "result": "sympy_solution"},
            **kwargs,
        )

    def get_generator_params(self, context: CompositionContext) -> dict[str, Any]:
        """Set composite flag to True."""
        return {}


class LinearSystemSolverWrapperComponent(SympyGeneratorWrapperComponent):
    """Wrapper for the LinearSystemGenerator."""

    def __init__(self, name: Task, constraints: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(
            name=name,
            generator_class=LinearSystemGenerator,
            component_type=Task.LINEAR_SYSTEM_SOLVER,
            topic=Topic.LINEAR_ALGEBRA,
            context_update_mapping={
                "matrix_A": "matrix_A",
                "x_symbols": "x_symbols",
                "target_b": "target_b",
                "solution": "sympy_solution",
            },
            **kwargs,
        )
        self.constraints = constraints

    def get_generator_params(self, context: CompositionContext) -> dict[str, Any]:
        """Extract previous component result to use as input_vector_b."""
        if context.component_results:
            previous_result = context.component_results[-1]
            input_index = self.constraints["input_index"]
            if not hasattr(previous_result.template, "sympy_solution"):
                raise ValueError(f"Previous component result has no sympy_solution: {previous_result}")

            vector_b = previous_result.template.sympy_solution
            if not hasattr(vector_b, "shape"):
                raise ValueError(f"Previous component result is not a Matrix: {type(vector_b)}")

            return {"input_vector_b": vector_b, "input_index": input_index}
        return {}
