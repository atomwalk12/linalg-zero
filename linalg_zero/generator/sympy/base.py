from abc import ABC, abstractmethod

from linalg_zero.generator.composition.sample_args import SampleArgs
from linalg_zero.generator.context import CompositionContext, ProblemContext
from linalg_zero.generator.models import (
    ComponentResult,
    DifficultyCategory,
    ProblemTemplate,
    Question,
)


class ProblemComponent(ABC):
    """
    Abstract base class for composable problem components.

    A component represents an atomic piece of a mathematical problem that can
    be combined with other components to create more complex problems.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate(self, context: CompositionContext) -> ComponentResult:
        """
        Generate the component's mathematical content.
        """
        pass

    def can_execute(self, context: CompositionContext) -> bool:
        """
        Check if this component can execute given the current context.
        """
        return True


class CompositionStrategy(ABC):
    """Abstract base class for problem composition strategies."""

    @abstractmethod
    def compose(
        self, components: list[ProblemComponent], sample_args: SampleArgs, base_context: CompositionContext
    ) -> list[ComponentResult]:
        """
        Execute composition strategy on the given components.
        """
        pass


class SympyProblemGenerator(ABC):
    """
    Abstract base class for SymPy-based mathematical problem generators.

    It orchestrates the interactions around the problem resolution process,
    including content generation, query/answer formatting and verification.
    """

    def __init__(
        self,
        entropy: float,
        difficulty_level: DifficultyCategory,
        problem_type: str = "unknown",
        topic: str = "linear_algebra",
    ):
        self.entropy = entropy
        self.difficulty_level = difficulty_level
        self.problem_type = problem_type
        self.topic = topic

    @abstractmethod
    def generate_mathematical_content(self, context: ProblemContext) -> ProblemTemplate:
        """
        Generate the core mathematical content for the problem.
        """
        pass

    @abstractmethod
    def format_question(self, template: ProblemTemplate) -> str:
        """
        Convert SymPy content into a natural language query.
        """
        pass

    @abstractmethod
    def format_solution(self, template: ProblemTemplate) -> str:
        """
        Format the solution in a format recognisable by math-verify.
        """
        pass

    @abstractmethod
    def verify_problem(self, template: ProblemTemplate) -> bool:
        """Safety net to ensure the input/output is valid. Should always return True."""
        pass

    def generate(self) -> Question:
        """
        Orchestrates the problem generation process by generating a SymPy
        problem template, formatting it and verifying it.
        """
        with ProblemContext(self.entropy, self.difficulty_level) as context:
            # Generate mathematical content
            template = self.generate_mathematical_content(context)

            # Format natural language components
            question_text = self.format_question(template)
            answer_text = self.format_solution(template)

            # Verify correctness
            is_valid = self.verify_problem(template)

            return Question(
                question=question_text,
                answer=answer_text,
                difficulty=self.difficulty_level,
                topic=self.topic,
                problem_type=self.problem_type,
                is_valid=is_valid,
                entropy_used=context.used_entropy,
                tool_calls_required=context.tool_calls_count,
                stepwise=context.stepwise_results,
                golden=context.golden_result,
            )
