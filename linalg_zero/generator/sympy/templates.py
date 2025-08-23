import random
from dataclasses import dataclass
from typing import Any, ClassVar

from sympy.matrices import MutableDenseMatrix

from linalg_zero.generator.utils.difficulty import DifficultyCategory
from linalg_zero.shared.utils import get_logger

logger = get_logger(__name__)


@dataclass
class QuestionTemplate:
    """
    Data class template for generating natural language questions.
    """

    template_string: str
    required_variables: list[str]
    difficulty_level: DifficultyCategory
    question_type: str


class MathFormatter:
    """
    Utilities for formatting mathematical expressions in text format.
    """

    @staticmethod
    def format_matrix(matrix: MutableDenseMatrix) -> str:
        """
        Format a SymPy matrix, to be displayed in question query.
        Expected format:
          Q: What is [[4, -1, 1], [4, 5, -2], [-4, -2/9, 1]] * [[4], [-2], [2/3]]?
          A: [[56/3], [14/3], [-134/9]]
        """
        rows = []
        for i in range(matrix.rows):
            row = []
            for j in range(matrix.cols):
                element = matrix[i, j]
                row.append(str(element))
            rows.append(row)

        formatted_rows = []
        for row in rows:
            formatted_rows.append(f"[{', '.join(row)}]")

        return f"[{', '.join(formatted_rows)}]"


class TemplateEngine:
    """
    Main engine for generating natural language questions from mathematical templates.

    This class coordinates the process of converting SymPy content into
    human-readable questions using templates and formatters.
    """

    SOLVE_VERBS: ClassVar[dict[DifficultyCategory, list[str]]] = {
        DifficultyCategory.EASY: ["Find", "Calculate"],
        DifficultyCategory.MEDIUM: ["Find", "Solve", "Determine"],
        DifficultyCategory.HARD: ["Solve", "Determine", "Evaluate", "Derive"],
    }

    QUESTION_STARTERS: ClassVar[dict[DifficultyCategory, list[str]]] = {
        DifficultyCategory.EASY: ["What is", "Find"],
        DifficultyCategory.MEDIUM: ["Find", "Calculate", "Determine", "Compute"],
        DifficultyCategory.HARD: ["Evaluate", "Determine", "Derive", "Establish"],
    }

    COMPUTE_VERBS: ClassVar[dict[DifficultyCategory, list[str]]] = {
        DifficultyCategory.EASY: ["Find"],
        DifficultyCategory.MEDIUM: ["Calculate", "Compute"],
        DifficultyCategory.HARD: ["Determine", "Evaluate"],
    }

    def __init__(self) -> None:
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
            if isinstance(var_value, MutableDenseMatrix):
                formatted_variables[var_name] = self.math_formatter.format_matrix(var_value)
            else:
                raise TypeError(f"Variable '{var_name}' has unsupported type {type(var_value).__name__}.")

        try:
            # 3. Apply template substitution
            question_text = template.template_string.format(**formatted_variables)
        except KeyError as e:
            raise ValueError(f"Template substitution failed: missing variable {e}") from e

        return question_text

    def format_answer(self, answer: Any) -> str:
        """
        Format a SymPy matrix (can also be a vector), to be displayed in question answer.
        """
        if isinstance(answer, MutableDenseMatrix):
            return self.math_formatter.format_matrix(answer)
        else:
            raise TypeError(f"Variable '{answer}' has unsupported type {type(answer).__name__}.")

    def create_default_templates(self, question_type: str, difficulty: DifficultyCategory) -> list[QuestionTemplate]:
        """
        Create default question templates for common problem types.
        This simplifies the creation of question/answer pairs.
        """
        templates = []

        if question_type == "solve":
            solve_verb = random.choice(self.SOLVE_VERBS[difficulty])
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
            compute_verb = random.choice(self.COMPUTE_VERBS[difficulty])

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
        self,
        templates: list[QuestionTemplate],
        question_type: str,
        difficulty: DifficultyCategory,
    ) -> QuestionTemplate:
        """
        Select an appropriate template from a list based on specified criteria.
        The selection checks the question type and problem difficulty.
        """
        if not templates:
            raise ValueError("No templates available")

        # Filter by both criteria simultaneously. We look for templates that
        # match both in terms of question type as well as difficulty level
        candidates = [t for t in templates if t.question_type == question_type and t.difficulty_level == difficulty]

        return random.choice(candidates if candidates else templates)
