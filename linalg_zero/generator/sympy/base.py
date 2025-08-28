from abc import ABC, abstractmethod
from typing import Any

from linalg_zero.generator import Precision
from linalg_zero.generator.context import CompositionContext, ProblemContext
from linalg_zero.generator.difficulty_config import SampleArgs, get_problem_config
from linalg_zero.generator.models import (
    ComponentResult,
    DifficultyCategory,
    ProblemTemplate,
    Question,
    Task,
    Topic,
)
from linalg_zero.generator.sympy.templates import MathFormatter, TemplateEngine
from linalg_zero.grpo.verify import verify_answers
from linalg_zero.shared.types import LibTypes


class ProblemComponent(ABC):
    """
    Abstract base class for composable problem components.

    A component represents an atomic piece of a mathematical problem that can
    be combined with other components to create more complex problems.
    """

    def __init__(self, name: Task):
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
        problem_type: Task,
        topic: Topic,
    ):
        self.entropy = entropy
        self.difficulty_level = difficulty_level
        self.problem_type = problem_type
        self.topic = topic
        self.config = get_problem_config(difficulty_level, topic, problem_type)
        self.template_engine = TemplateEngine()
        self.formatter = MathFormatter()

    @property
    def precision(self) -> Precision:
        """The precision of the problem."""
        raise NotImplementedError("Implemented by subclasses.")

    @abstractmethod
    def generate_mathematical_content(self, context: ProblemContext) -> ProblemTemplate:
        """
        Generate the core mathematical content for the problem.
        """
        pass

    @abstractmethod
    def get_template_variables(self, template: ProblemTemplate) -> dict[str, Any]:
        """Return the variables dictionary to pass to the template engine."""
        pass

    def format_question(self, template: ProblemTemplate) -> str:
        """
        Convert SymPy content into a natural language query.
        """
        # Get problem-specific information from subclass
        problem_type = self.problem_type
        variables = self.get_template_variables(template)

        # Get templates for the problem type
        templates = self.template_engine.create_default_templates(problem_type, self.difficulty_level)
        if templates:
            selected_template = self.template_engine.select_template(templates, problem_type, self.difficulty_level)
            question_text = self.template_engine.generate_question(
                template=selected_template, variables=variables, precision=self.precision
            )
        else:
            raise ValueError(f"No templates available for {problem_type}")

        return question_text

    def format_solution(self, template: ProblemTemplate) -> str:
        """The solution string used as the ground truth in the final dataset entry."""
        return self.template_engine.format_answer(template.sympy_solution, precision=self.precision)

    def verify_problem(self, template: ProblemTemplate) -> bool:
        """
        Verify the mathematical correctness using end-to-end verification.
        This ensures sympy and lib.py results match.
        """
        lib_result = template.lib_result
        sympy_solution = template.sympy_solution

        ground_truth = self.formatter.sympy_to_primitive(sympy_solution, precision=self.precision)
        assert isinstance(ground_truth, LibTypes)  # noqa: S101

        if not verify_answers(ground_truth, lib_result):
            raise ValueError(f"Verification failed: sympy={ground_truth} vs lib={lib_result}")

        return True

    def generate(self) -> Question:
        """
        Orchestrates the problem generation process by generating a SymPy
        problem template, formatting it and verifying it.
        """
        with ProblemContext(self.entropy, self.difficulty_level, 0) as context:
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
