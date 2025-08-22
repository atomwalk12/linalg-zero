import random
from dataclasses import dataclass
from typing import Any, ClassVar

import sympy
from sympy.matrices import MutableDenseMatrix

from linalg_zero.generator.utils.difficulty import get_difficulty_category
from linalg_zero.shared.utils import get_logger

logger = get_logger(__name__)


@dataclass
class QuestionTemplate:
    """
    Data class template for generating natural language questions.
    """

    template_string: str
    required_variables: list[str]
    difficulty_level: int
    question_type: str


class MathFormatter:
    """
    Utilities for formatting mathematical expressions in text format.
    """

    @staticmethod
    def format_expression(expr: sympy.Expr) -> str:
        return str(expr)

    @staticmethod
    def format_equation(lhs: sympy.Expr, rhs: sympy.Expr) -> str:
        return f"{MathFormatter.format_expression(lhs)} = {MathFormatter.format_expression(rhs)}"

    @staticmethod
    def format_matrix(matrix: list[list[sympy.Expr]]) -> str:
        return str(matrix)


class TemplateEngine:
    """
    Main engine for generating natural language questions from mathematical templates.

    This class coordinates the process of converting SymPy content into
    human-readable questions using templates and formatters.
    """

    SOLVE_VERBS: ClassVar[dict[str, list[str]]] = {
        "easy": ["Find", "Calculate"],
        "medium": ["Find", "Solve", "Determine"],
        "hard": ["Solve", "Determine", "Evaluate", "Derive"],
    }

    QUESTION_STARTERS: ClassVar[dict[str, list[str]]] = {
        "easy": ["What is", "Find"],
        "medium": ["Find", "Calculate", "Determine", "Compute"],
        "hard": ["Evaluate", "Determine", "Derive", "Establish"],
    }

    COMPUTE_VERBS: ClassVar[dict[str, list[str]]] = {
        "easy": ["Find"],
        "medium": ["Calculate", "Compute"],
        "hard": ["Determine", "Evaluate"],
    }

    def __init__(self):
        self.math_formatter = MathFormatter()

    def generate_question(self, template: QuestionTemplate, variables: dict[str, Any]) -> str:
        """
        Generate natural language question text that will be included as the
        "query" field in the final dataset entry.

        This method validates variable types upfront, then formats expressions
        before performing template substitution.
        """
        # 1. Validation checks
        for var_name, var_value in variables.items():
            if not isinstance(var_value, MutableDenseMatrix):
                raise TypeError(
                    f"Variable '{var_name}' has unsupported type {type(var_value).__name__}. "
                    f"Supported types: sympy.Expr, str, list, tuple, int, float"
                )

        missing_vars = set(template.required_variables) - set(variables.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")

        # 2. Format mathematical expressions stored within variables
        formatted_variables = {}
        for var_name, var_value in variables.items():
            if isinstance(var_value, sympy.Expr):
                formatted_variables[var_name] = self.math_formatter.format_expression(var_value)
            elif isinstance(var_value, (list, tuple)) and len(var_value) == 2:
                if all(isinstance(x, sympy.Expr) for x in var_value):
                    formatted_variables[var_name] = self.math_formatter.format_equation(var_value[0], var_value[1])
                else:
                    formatted_variables[var_name] = str(var_value)
            elif isinstance(var_value, (int, float)):
                formatted_variables[var_name] = str(var_value)
            else:
                raise TypeError(f"Variable '{var_name}' has unsupported type {type(var_value).__name__}.")

        try:
            # 3. Apply template substitution
            question_text = template.template_string.format(**formatted_variables)
        except KeyError as e:
            raise ValueError(f"Template substitution failed: missing variable {e}") from e

        return question_text

    def create_default_templates(self, question_type: str, difficulty: int) -> list[QuestionTemplate]:
        """
        Create default question templates for common problem types.
        This simplifies the creation of question/answer pairs.
        """
        templates = []

        # Convert numeric difficulty to category
        difficulty_category = get_difficulty_category(difficulty)

        if question_type == "solve":
            solve_verb = random.choice(self.SOLVE_VERBS[difficulty_category])
            templates.extend([
                QuestionTemplate(
                    template_string=f"{solve_verb} {{equation}} for {{variable}}.",
                    required_variables=["equation", "variable"],
                    difficulty_level=difficulty,
                    question_type="solve",
                ),
                QuestionTemplate(
                    template_string="What is {variable} in {equation}?",
                    required_variables=["equation", "variable"],
                    difficulty_level=difficulty,
                    question_type="solve",
                ),
            ])

        elif question_type == "compute_product":
            compute_verb = random.choice(self.COMPUTE_VERBS[difficulty_category])

            templates.extend([
                QuestionTemplate(
                    template_string=f"{compute_verb} {{matrix}} * {{vector}}.",
                    required_variables=["matrix", "vector"],
                    difficulty_level=difficulty,
                    question_type="compute_product",
                ),
                QuestionTemplate(
                    template_string="What is {matrix} * {vector}?",
                    required_variables=["matrix", "vector"],
                    difficulty_level=difficulty,
                    question_type="compute_product",
                ),
                QuestionTemplate(
                    template_string=f"{compute_verb} the product {{matrix}} * {{vector}}.",
                    required_variables=["matrix", "vector"],
                    difficulty_level=difficulty,
                    question_type="compute_product",
                ),
            ])

        return templates

    def select_template(
        self, templates: list[QuestionTemplate], question_type: str | None = None, difficulty: int | None = None
    ) -> QuestionTemplate:
        """
        Select an appropriate template from a list based on specified criteria.
        The selection checks the question type and problem difficulty.
        """
        if not templates:
            raise ValueError("No templates available")

        # Filter by both criteria simultaneously. We look for templates
        # that match both in terms of question type as well as difficulty level
        candidates = [
            t
            for t in templates
            if (question_type is None or t.question_type == question_type)
            and (difficulty is None or t.difficulty_level == difficulty)
        ]

        return random.choice(candidates if candidates else templates)
