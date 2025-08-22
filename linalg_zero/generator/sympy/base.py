from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import sympy

from linalg_zero.generator.models import Question


@dataclass
class ProblemTemplate:
    """
    Data class with the main components for a problem.
    """

    expression: sympy.Expr
    variables: list[sympy.Symbol]
    solution: sympy.Expr | list[sympy.Expr] | str
    question_templates: list[str]
    context_info: dict[str, Any]
    difficulty_markers: dict[str, float]
    difficulty: int


class ProblemContext:
    """
    Context manager for state information around the resolution process.
    """

    def __init__(self, entropy: float, difficulty_level: int = 2):
        self.entropy = entropy
        self.difficulty_level = difficulty_level
        self.used_entropy = 0.0
        self.tool_calls_count = 0
        self.stepwise_results: list[dict[str, str]] = []
        self.golden_result: dict[str, str] = {}
        self._step_counter = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def record_entropy_usage(self, amount: float) -> None:
        """
        Record entropy usage for tracking problem complexity.
        """
        self.used_entropy += amount

    def record_tool_call(
        self,
        operation_type: str = "general",
        result: str = "",
        is_final: bool = False,
        depends_on: list[str] | None = None,
    ) -> str:
        """
        Record a tool call with its result. It tracks the dependencies between
        steps which will later be used to verify correctness during GRPO.
        """

        self.tool_calls_count += 1
        self._step_counter += 1
        step_id = str(self._step_counter)

        if result:
            step_data = {"tool": operation_type, "result": result, "step_id": step_id}

            if depends_on:
                step_data["depends_on"] = depends_on

            if is_final:
                self.golden_result = {"final_answer": result, "from_step_id": step_id}

            self.stepwise_results.append(step_data)

        return step_id


class SympyProblemGenerator(ABC):
    """
    Abstract base class for SymPy-based mathematical problem generators.

    It orchestrates the interactions around the problem resolution process,
    including content generation, query/answer formatting and verification.
    """

    def __init__(
        self,
        entropy: float = 3.0,
        difficulty_level: int = 2,
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
                text=question_text,
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


def create_sympy_factory(
    generator_class: type,
    entropy: float = 3.0,
    difficulty_level: int = 2,
    problem_type: str = "unknown",
    topic: str = "linear_algebra",
    **kwargs,
) -> Callable[[], Question]:
    """
    Convenience function for generating a factory function for registry registration.
    """

    def factory() -> Question:
        generator = generator_class(
            entropy=entropy, difficulty_level=difficulty_level, problem_type=problem_type, topic=topic, **kwargs
        )
        return generator.generate()

    return factory
