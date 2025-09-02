from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from sympy import Expr

from linalg_zero.shared.types import LibTypes

if TYPE_CHECKING:
    from linalg_zero.generator.context import CompositionContext
    from linalg_zero.generator.sympy.base import CompositionStrategy, SympyProblemGenerator


class Topic(Enum):
    """Enum for topics used in problem generation."""

    LINEAR_ALGEBRA = "linear_algebra"


class Task(Enum):
    """Enum for problem types used in problem generation."""

    DETERMINANT = "determinant"
    LINEAR_SYSTEM_SOLVER = "linear_system_solver"
    MATRIX_VECTOR_MULTIPLICATION = "matrix_vector_multiplication"
    MATRIX_MATRIX_MULTIPLICATION = "matrix_matrix_multiplication"
    FROBENIUS_NORM = "frobenius_norm"
    MATRIX_RANK = "matrix_rank"
    MATRIX_TRANSPOSE = "matrix_transpose"
    MATRIX_INVERSE = "matrix_inverse"
    MATRIX_TRACE = "matrix_trace"
    MATRIX_COFACTOR = "matrix_cofactor"
    COMPOSITE_SEQUENTIAL = "sequential"
    COMPOSITE_SYSTEM_DEPENDENCY = "system_dependency"
    COMPOSITE_TRANSPOSE_DETERMINANT = "transpose_determinant"
    COMPOSITE_TRANSPOSE_TRACE = "transpose_trace"
    COMPOSITE_SYSTEM_NORM = "system_norm"
    COMPOSITE_TRANSPOSE_DETERMINANT_BALANCED = "transpose_determinant_balanced"
    COMPOSITE_INVERSE_FROBENIUS = "inverse_frobenius"
    COMPOSITE_INVERSE_RANK = "inverse_rank"
    COMPOSITE_TRANSPOSE_FROBENIUS = "transpose_frobenius"
    COMPOSITE_TRIPLE_TRANSPOSE_DETERMINANT = "triple_transpose_determinant"
    COMPOSITE_TRIPLE_INVERSE_RANK = "triple_inverse_rank"
    COMPOSITE_TRIPLE_SYSTEM_FROBENIUS = "triple_system_frobenius"


class DifficultyCategory(Enum):
    """Enum for difficulty categories used in problem generation."""

    ONE_TOOL_CALL = 1
    TWO_TOOL_CALLS = 2
    THREE_TOOL_CALLS = 3

    def __str__(self) -> str:
        """Return the string value for compatibility with existing code."""
        if self == DifficultyCategory.ONE_TOOL_CALL:
            return "easy (1 tool call)"
        elif self == DifficultyCategory.TWO_TOOL_CALLS:
            return "medium (2 tool calls)"
        elif self == DifficultyCategory.THREE_TOOL_CALLS:
            return "hard (3 tool calls)"
        else:
            raise ValueError(f"Invalid difficulty category: {self}")


@dataclass
class Question:
    """Represents a generated question with its answer."""

    question: str
    answer: str
    difficulty: DifficultyCategory
    topic: Topic
    problem_type: Task
    is_valid: bool = True
    entropy_used: float = 0.0
    tool_calls_required: int = 0
    stepwise: list[dict[str, str]] = field(default_factory=list)
    golden: dict[str, str] = field(default_factory=dict)


@dataclass
class ProblemTemplate:
    """
    Data class with the main components for a problem.
    """

    expression: Expr
    variables: dict[str, Expr]
    sympy_solution: Expr | list[Expr] | str
    lib_result: LibTypes
    context_info: dict[str, Any]
    difficulty_markers: dict[str, float | tuple]
    question_templates: list[str] | None
    difficulty: DifficultyCategory | None = None


class CompositionType(Enum):
    """
    Types of problem composition strategies

    The mathematics_dataset package contains the following composition types:
    - Sequential composition feeds the output of one component into the next
    - Hierarchical composition with peel() method for parent-child relationships
    - Parallel composition for independent sub-problems
    - Conditional composition that adapts based on intermediate results
    """

    # NOTE[Future]: Implement other composition types here
    SEQUENTIAL = "sequential"


@dataclass
class ComponentResult:
    """Result from executing a problem component."""

    template: ProblemTemplate
    generator: "SympyProblemGenerator"
    entropy_consumed: float = 0.0
    tool_calls_used: int = 0


@dataclass
class CompositeResultBuilder:
    """Builder for combining component results into a unified template."""

    def __init__(self, composition_strategy: "CompositionStrategy"):
        self.composition_strategy = composition_strategy
        self.expressions: list = []
        self.solutions: list = []
        self.lib_results: list = []
        self.question_templates: list[str] = []
        self.context_info: dict[str, Any] = {}
        self.component_templates: list[ProblemTemplate] = []

    def add_component_result(self, result: ComponentResult) -> None:
        """Add a component result to the builder."""
        template = result.template

        self.expressions.append(template.expression)
        self.component_templates.append(template)
        # Variables are accessed directly from component results via sources system
        # No need to aggregate here as it would cause naming conflicts

        self.solutions.append(template.sympy_solution)
        self.lib_results.append(template.lib_result)

        self.context_info.update(template.context_info)

        # This is omitted because it is calculated by the composite problem base class
        # self.question_templates.extend(template.question_templates)

    def build_template(
        self, comp_context: "CompositionContext", component_results: list[ComponentResult]
    ) -> ProblemTemplate:
        """Build the final composite template."""
        return ProblemTemplate(
            expression=self._build_main_expression(),
            variables=self._deduplicate_variables(),
            sympy_solution=self.solutions,
            lib_result=self.lib_results,
            question_templates=self.question_templates or ["Solve the following problem:"],
            context_info=self._build_context_info(comp_context, component_results),
            difficulty_markers=self._build_difficulty_markers(comp_context),
        )

    def _build_main_expression(self) -> Expr | list[Expr]:
        """Build the main expression (single vs list)."""
        return self.expressions[0] if len(self.expressions) == 1 else self.expressions

    def _deduplicate_variables(self) -> dict[str, Expr]:
        """Return empty dict since composite problems don't aggregate variables."""
        # Variables are accessed directly from individual component results
        # via the sources system in composition constraints
        return {}

    def _build_context_info(
        self, comp_context: "CompositionContext", component_results: list[ComponentResult]
    ) -> dict[str, Any]:
        """Build combined context info with composition metadata."""
        return {
            **self.context_info,
            "composition_type": self.composition_strategy.__class__.__name__,
            "component_count": len(self.component_templates),
            "total_entropy_used": comp_context.used_entropy,
            "total_tool_calls": comp_context.tool_calls_count,
            "component_templates": self.component_templates,
            "component_results": component_results,
        }

    def _build_difficulty_markers(self, comp_context: "CompositionContext") -> dict[str, Any]:
        """Build difficulty markers for the composite problem."""
        return {
            "composition_complexity": len(self.component_templates),  # the number of components
            "entropy_per_component": comp_context.used_entropy / len(self.component_templates),
            "variable_count": len(self._deduplicate_variables()),
        }
