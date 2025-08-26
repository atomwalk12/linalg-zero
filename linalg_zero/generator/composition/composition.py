from collections.abc import Callable

from linalg_zero.generator.composition.sample_args import SampleArgs
from linalg_zero.generator.context import CompositionContext
from linalg_zero.generator.models import ComponentResult, CompositeResultBuilder, DifficultyCategory
from linalg_zero.generator.sympy.base import (
    CompositionStrategy,
    ProblemComponent,
    ProblemContext,
    ProblemTemplate,
    SympyProblemGenerator,
)
from linalg_zero.generator.sympy.templates import MathFormatter, Precision


class SequentialComposition(CompositionStrategy):
    """
    Sequential composition strategy.

    Executes components in order, where each component can use results
    from previous components. Useful for multi-step problems.
    """

    def compose(
        self, components: list[ProblemComponent], sample_args: SampleArgs, base_context: CompositionContext
    ) -> list[ComponentResult]:
        """Execute components using DeepMind-style entropy distribution."""
        results = []

        # Split entropy among components using Dirichlet distribution
        component_sample_args = sample_args.split(len(components))

        for component, comp_sample_args in zip(components, component_sample_args, strict=True):
            if not component.can_execute(base_context):
                continue

            # Create a context copy with the allocated entropy for this component
            component_context = CompositionContext(comp_sample_args.entropy, base_context.difficulty_level)
            component_context.constraints = base_context.constraints.copy()
            component_context.shared_state = base_context.shared_state.copy()
            component_context.global_variables = base_context.global_variables.copy()

            result = component.generate(component_context)
            base_context.record_component_result(result)
            results.append(result)

        return results


class CompositeProblem(SympyProblemGenerator):
    """
    Generator for composite mathematical problems.

    Combines multiple ProblemComponent instances using a CompositionStrategy
    to create complex, multi-part mathematical problems.
    """

    def __init__(
        self,
        components: list[ProblemComponent],
        composition_strategy: CompositionStrategy,
        sample_args: SampleArgs,
        difficulty_level: DifficultyCategory,
        problem_type: str = "composite",
        topic: str = "mixed",
        template_engine: Callable | None = None,
        verification_engine: Callable | None = None,
    ):
        # Use concrete sample args directly
        total_entropy = sample_args.entropy

        super().__init__(
            entropy=total_entropy, difficulty_level=difficulty_level, problem_type=problem_type, topic=topic
        )

        self.components = components
        self.composition_strategy = composition_strategy
        self.sample_args = sample_args
        self.formatter = MathFormatter()

    def generate_mathematical_content(self, context: ProblemContext) -> ProblemTemplate:
        """Generate composed mathematical content."""
        # Convert to CompositionContext
        comp_context = CompositionContext(self.sample_args.entropy, context.difficulty_level)
        comp_context.constraints = context.constraints.copy()

        # Execute all components and store their results
        component_results = self.composition_strategy.compose(self.components, self.sample_args, comp_context)

        if not component_results:
            raise ValueError("No components could be executed")

        # This is a helper class to aggregate component results and create a composite template
        builder = CompositeResultBuilder(self.composition_strategy)
        for result in component_results:
            builder.add_component_result(result)

        template = builder.build_template(comp_context, component_results)

        # Transfer state back to original context
        self._transfer_context_state(comp_context, context)

        return template

    def _transfer_context_state(self, comp_context: CompositionContext, original_context: ProblemContext) -> None:
        """Transfer entropy and tool call tracking back to original context."""
        original_context.used_entropy = comp_context.used_entropy
        original_context.tool_calls_count = comp_context.tool_calls_count
        original_context.stepwise_results = comp_context.stepwise_results
        original_context.golden_result = comp_context.golden_result
        original_context._step_counter = comp_context._step_counter

    def format_question(self, template: ProblemTemplate) -> str:
        """Format composite problem as natural language multi-step question."""
        context_info = template.context_info
        composition_type = context_info.get("composition_type", "Unknown")

        if isinstance(template.expression, list) and len(template.expression) > 1:
            if composition_type == "SequentialComposition":
                return self._format_sequential_question(template)
            else:
                raise ValueError(f"Unknown composition type: {composition_type}")
        else:
            # Single expression
            expr = template.expression[0] if isinstance(template.expression, list) else template.expression
            return f"Solve: {expr}"

    def _format_sequential_question(self, template: ProblemTemplate) -> str:
        """Format sequential composition by delegating to individual generators."""
        component_results: list[ComponentResult] = template.context_info.get("component_results", [])

        if not component_results:
            raise ValueError("Sequential composition requires component results with generators")

        step_descriptions = []
        for i, result in enumerate(component_results, 1):
            formatted_question = result.generator.format_question(result.template)

            if i == 1:
                step_descriptions.append(f"First, {formatted_question.lower()}")
            else:
                step_descriptions.append(f"Then, {formatted_question.lower()}")

        return "\n\n".join(step_descriptions)

    def format_solution(self, template: ProblemTemplate) -> str:
        """Format composite problem solution using MathFormatter for clean output."""

        if not isinstance(template.sympy_solution, list):
            raise TypeError("The sympy solution should be a list because the number of provided components is a list.")

        if len(template.sympy_solution) == 1:
            solution = template.sympy_solution[0]
            formatted = self.formatter.sympy_to_primitive(solution, precision=Precision.FULL)
            return str(formatted)
        else:
            formatted_parts = []
            for i, sol in enumerate(template.sympy_solution, 1):
                formatted_sol = self.formatter.sympy_to_primitive(sol, precision=Precision.FULL)
                formatted_parts.append(f"Part {i}: {formatted_sol}")
            return "; ".join(formatted_parts)

    def verify_problem(self, template: ProblemTemplate) -> bool:
        """Verify the problem is mathematically correct."""
        # There is nothing to check since the problems are verified within individual components
        return True
