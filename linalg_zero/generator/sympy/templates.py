import json
import random
from dataclasses import dataclass
from typing import Any, ClassVar

from sympy import Float, Integer, Mul, Number, Pow, Rational, Symbol
from sympy.core import Expr
from sympy.matrices import MutableDenseMatrix

from linalg_zero.generator.difficulty_config import Precision
from linalg_zero.generator.models import ComponentResult, DifficultyCategory, Task
from linalg_zero.shared.types import LibTypes
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
    question_type: Task


class MathFormatter:
    """
    Utilities for formatting mathematical expressions in text format.
    """

    @staticmethod
    def round_sympy_element(element: LibTypes, precision: Precision) -> LibTypes | str:
        """
        Round a SymPy element to a precision of 2. The string return type is used for
        symbol variables that may be part of specific lists.
        """
        if isinstance(element, int | float):
            if precision != Precision.FULL:
                return round(element, precision.value)
            else:
                return element
        elif isinstance(element, list):
            result = []
            for e in element:
                if isinstance(e, (int, float, list)):
                    result.append(MathFormatter.round_sympy_element(e, precision))
                elif isinstance(e, str):
                    # Symbol elements are appended as they are. These are used for instance
                    # for the linear_system_solver problem type.
                    result.append(e)
                else:
                    raise TypeError(f"Unsupported element type in list: {type(e)} (value: {e})")
            return result
        else:
            raise TypeError(f"Unsupported element type: {type(element)}")

    @staticmethod
    def sympy_to_primitive(sympy_result: Expr, precision: Precision) -> LibTypes | str:
        """Convert sympy result to primitive type for verification."""
        result: LibTypes | str | None = None
        if isinstance(sympy_result, MutableDenseMatrix):
            list_of_lists = sympy_result.tolist()
            result = [[MathFormatter._sympy_element_to_python(element) for element in _] for _ in list_of_lists]
        elif isinstance(sympy_result, (Number, Integer, Float)):
            result = MathFormatter._sympy_element_to_python(sympy_result)
        elif isinstance(sympy_result, Mul | Pow):
            # Frobenius norm requires Pow and Mul expressions
            result = float(sympy_result.evalf())
        else:
            raise TypeError(f"Unsupported element type: {type(sympy_result)}")

        if precision != Precision.FULL and not isinstance(result, str):
            return MathFormatter.round_sympy_element(result, precision)
        else:
            return result

    @staticmethod
    def _sympy_element_to_python(element: Integer | Float | Number | Symbol) -> float | int | str:
        """Convert SymPy element to Python primitive, following quantum matrixutils pattern."""
        if hasattr(element, "is_Integer") and element.is_Integer:
            value = element.__int__()
            if isinstance(value, int):
                return value
            else:
                raise ValueError(f"Expected int, got {type(value)}")
        elif (hasattr(element, "is_Float") and element.is_Float) or (
            hasattr(element, "is_Number") and element.is_Number
        ):
            value = element.__float__()
            if isinstance(value, float):
                return value
            else:
                raise ValueError(f"Expected float, got {type(value)}")
        elif isinstance(element, Symbol):
            # This is used by the inverse solver because of variables that are not numbers
            return str(element)
        raise ValueError(f"Unsupported element type: {type(element)}")


class TemplateEngine:
    """
    Main engine for generating natural language questions from mathematical templates.

    This class coordinates the process of converting SymPy content into
    human-readable questions using templates and formatters.
    """

    SOLVE_VERBS: ClassVar[dict[DifficultyCategory, list[str]]] = {
        DifficultyCategory.ONE_TOOL_CALL: ["Find", "Calculate"],
        DifficultyCategory.TWO_TOOL_CALLS: ["Find", "Solve", "Determine"],
        DifficultyCategory.THREE_TOOL_CALLS: ["Solve", "Determine", "Evaluate", "Derive"],
    }

    QUESTION_STARTERS: ClassVar[dict[DifficultyCategory, list[str]]] = {
        DifficultyCategory.ONE_TOOL_CALL: ["What is", "Find"],
        DifficultyCategory.TWO_TOOL_CALLS: ["Find", "Calculate", "Determine", "Compute"],
        DifficultyCategory.THREE_TOOL_CALLS: ["Evaluate", "Determine", "Derive", "Establish"],
    }

    COMPUTE_VERBS: ClassVar[dict[DifficultyCategory, list[str]]] = {
        DifficultyCategory.ONE_TOOL_CALL: ["Find"],
        DifficultyCategory.TWO_TOOL_CALLS: ["Calculate", "Compute"],
        DifficultyCategory.THREE_TOOL_CALLS: ["Determine", "Evaluate"],
    }

    def __init__(self) -> None:
        self.math_formatter = MathFormatter()

    def generate_question(self, template: QuestionTemplate, variables: dict[str, Any], precision: Precision) -> str:
        """
        Generate natural language question text that will be included as the
        "query" field in the final dataset entry.

        This method validates variable types upfront, then formats expressions
        before performing template substitution.
        """
        # 1. Validation checks
        missing_vars = set(template.required_variables) - set(variables.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")

        # 2. Format mathematical expressions stored within variables
        formatted_variables = {}
        for var_name, var_value in variables.items():
            if isinstance(var_value, MutableDenseMatrix):
                formatted_variables[var_name] = self.math_formatter.sympy_to_primitive(var_value, precision)
            elif isinstance(var_value, str):
                # String values are symbolic references
                formatted_variables[var_name] = var_value
            elif isinstance(var_value, bool):
                # Boolean flags is is_independent flag
                formatted_variables[var_name] = var_value
            else:
                raise TypeError(f"Variable '{var_name}' has unsupported type {type(var_value).__name__}.")

        try:
            # 3. Apply template substitution
            question_text = template.template_string.format(**formatted_variables)
        except KeyError as e:
            raise ValueError(f"Template substitution failed: missing variable {e}") from e

        return question_text

    def format_answer(self, answer: Any, precision: Precision) -> str:
        """
        Format a SymPy matrix (can also be a vector), to be displayed in question answer.
        """
        if isinstance(answer, MutableDenseMatrix | Integer | Float | Rational | Pow | Mul):
            result = self.math_formatter.sympy_to_primitive(answer, precision)
            return json.dumps(result)
        else:
            raise TypeError(f"Variable '{answer}' has unsupported type {type(answer).__name__}.")

    def format_composite_answer(self, sympy_solutions: list[Expr], component_results: list[ComponentResult]) -> str:
        """
        Format a composite answer to be displayed in question as a JSON string.
        """
        result_dict = {}
        for i, (sol, component_result) in enumerate(zip(sympy_solutions, component_results, strict=True), 1):
            precision = component_result.generator.precision
            formatted_sol = self.math_formatter.sympy_to_primitive(sol, precision=precision)
            result_dict[f"tool_{i}"] = formatted_sol
        return json.dumps(result_dict)

    def create_default_templates(
        self, question_type: Task, difficulty: DifficultyCategory, is_independent: bool = True
    ) -> list[QuestionTemplate]:
        """
        Create default question templates for common problem types.
        This simplifies the creation of question/answer pairs.
        """
        if is_independent:
            return self.create_independent_templates(question_type, difficulty)
        else:
            return self.create_composite_templates(question_type, difficulty)

    def create_independent_templates(
        self, question_type: Task, difficulty: DifficultyCategory
    ) -> list[QuestionTemplate]:
        templates = []
        solve_verb = random.choice(self.SOLVE_VERBS[difficulty])
        compute_verb = random.choice(self.COMPUTE_VERBS[difficulty])

        if question_type == Task.LINEAR_SYSTEM_SOLVER:
            templates.extend([
                QuestionTemplate(
                    template_string=f"{solve_verb} the linear system A{{x_symbols}} = b for {{x_symbols}}, where A = {{matrix}} and b = {{target_b}}.",
                    required_variables=["matrix", "x_symbols", "target_b"],
                    difficulty_level=difficulty,
                    question_type=Task.LINEAR_SYSTEM_SOLVER,
                ),
                QuestionTemplate(
                    template_string="Given matrix A = {matrix} and vector b = {target_b}, solve A*{x_symbols} = b for {x_symbols}.",
                    required_variables=["matrix", "x_symbols", "target_b"],
                    difficulty_level=difficulty,
                    question_type=Task.LINEAR_SYSTEM_SOLVER,
                ),
            ])

        elif question_type == Task.MATRIX_VECTOR_MULTIPLICATION:
            templates.extend([
                QuestionTemplate(
                    template_string=f"{compute_verb} the matrix-vector product Av, where A = {{matrix}} and v = {{vector}}.",
                    required_variables=["matrix", "vector"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_VECTOR_MULTIPLICATION,
                ),
                QuestionTemplate(
                    template_string="Given matrix A = {matrix} and vector v = {vector}, compute Av.",
                    required_variables=["matrix", "vector"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_VECTOR_MULTIPLICATION,
                ),
                QuestionTemplate(
                    template_string=f"{compute_verb} A * v where A = {{matrix}} and v = {{vector}}.",
                    required_variables=["matrix", "vector"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_VECTOR_MULTIPLICATION,
                ),
            ])

        elif question_type == Task.DETERMINANT:
            templates.extend([
                QuestionTemplate(
                    template_string=f"{compute_verb} the determinant of matrix A, where A = {{matrix}}.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.DETERMINANT,
                ),
                QuestionTemplate(
                    template_string="Given matrix A = {matrix}, find det(A).",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.DETERMINANT,
                ),
                QuestionTemplate(
                    template_string="For A = {matrix}, compute det(A).",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.DETERMINANT,
                ),
            ])

        elif question_type == Task.FROBENIUS_NORM:
            templates.extend([
                QuestionTemplate(
                    template_string=f"{compute_verb} the Frobenius norm of matrix A, where A = {{matrix}}.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.FROBENIUS_NORM,
                ),
                QuestionTemplate(
                    template_string="Given matrix A = {matrix}, find ||A||_F.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.FROBENIUS_NORM,
                ),
                QuestionTemplate(
                    template_string=f"{compute_verb} ||A||_F where A = {{matrix}}.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.FROBENIUS_NORM,
                ),
                QuestionTemplate(
                    template_string="For A = {matrix}, compute the Frobenius norm ||A||_F.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.FROBENIUS_NORM,
                ),
            ])

        elif question_type == Task.MATRIX_RANK:
            templates.extend([
                QuestionTemplate(
                    template_string=f"{compute_verb} the rank of matrix A, where A = {{matrix}}.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_RANK,
                ),
                QuestionTemplate(
                    template_string="Given matrix A = {matrix}, find rank(A).",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_RANK,
                ),
                QuestionTemplate(
                    template_string="For A = {matrix}, compute rank(A).",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_RANK,
                ),
                QuestionTemplate(
                    template_string=f"{compute_verb} the number of linearly independent rows in matrix A = {{matrix}}.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_RANK,
                ),
            ])

        elif question_type == Task.MATRIX_TRANSPOSE:
            templates.extend([
                QuestionTemplate(
                    template_string=f"{compute_verb} the transpose of matrix A, where A = {{matrix}}.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_TRANSPOSE,
                ),
                QuestionTemplate(
                    template_string="Given matrix A = {matrix}, find A^T.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_TRANSPOSE,
                ),
                QuestionTemplate(
                    template_string="For A = {matrix}, compute A^T.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_TRANSPOSE,
                ),
                QuestionTemplate(
                    template_string=f"{compute_verb} A^T where A = {{matrix}}.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_TRANSPOSE,
                ),
            ])

        elif question_type == Task.MATRIX_INVERSE:
            templates.extend([
                QuestionTemplate(
                    template_string=f"{compute_verb} the inverse of matrix A, where A = {{matrix}}.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_INVERSE,
                ),
                QuestionTemplate(
                    template_string="Given matrix A = {matrix}, find A^(-1).",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_INVERSE,
                ),
                QuestionTemplate(
                    template_string="For A = {matrix}, compute A^(-1).",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_INVERSE,
                ),
                QuestionTemplate(
                    template_string=f"{compute_verb} A^(-1) where A = {{matrix}}.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_INVERSE,
                ),
            ])

        elif question_type == Task.MATRIX_TRACE:
            templates.extend([
                QuestionTemplate(
                    template_string=f"{compute_verb} the trace of matrix A, where A = {{matrix}}.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_TRACE,
                ),
                QuestionTemplate(
                    template_string="Given matrix A = {matrix}, find tr(A).",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_TRACE,
                ),
                QuestionTemplate(
                    template_string="For A = {matrix}, compute tr(A).",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_TRACE,
                ),
                QuestionTemplate(
                    template_string=f"{compute_verb} the sum of the diagonal elements of matrix A = {{matrix}}.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_TRACE,
                ),
            ])

        return templates

    def create_composite_templates(
        self, question_type: Task, difficulty: DifficultyCategory
    ) -> list[QuestionTemplate]:
        """
        Create question templates specifically for composite problems.
        These templates handle multi-step operations with intermediate results.
        """
        templates = []
        compute_verb = random.choice(self.COMPUTE_VERBS[difficulty])
        solve_verb = random.choice(self.SOLVE_VERBS[difficulty])

        if question_type == Task.MATRIX_VECTOR_MULTIPLICATION:
            templates.extend([
                QuestionTemplate(
                    template_string=f"{compute_verb} the matrix product AB, where A = {{matrix_A_ref}} and B = {{matrix_B_ref}}.",
                    required_variables=["matrix_A_ref", "matrix_B_ref"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_VECTOR_MULTIPLICATION,
                ),
                QuestionTemplate(
                    template_string="Given A = {matrix_A_ref} and B = {matrix_B_ref}, find AB.",
                    required_variables=["matrix_A_ref", "matrix_B_ref"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_VECTOR_MULTIPLICATION,
                ),
                QuestionTemplate(
                    template_string=f"{compute_verb} A^2, where A = {{matrix_A_ref}}.",
                    required_variables=["matrix_A_ref"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_VECTOR_MULTIPLICATION,
                ),
                QuestionTemplate(
                    template_string="Given matrix A defined in {matrix_A_ref}, compute A^2.",
                    required_variables=["matrix_A_ref"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_VECTOR_MULTIPLICATION,
                ),
                QuestionTemplate(
                    template_string=f"{compute_verb} the matrix-vector product using A = {{matrix}} and the vector from {{vector}}.",
                    required_variables=["matrix", "vector"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_VECTOR_MULTIPLICATION,
                ),
                QuestionTemplate(
                    template_string="Using matrix A = {matrix}, multiply by the vector from {vector}.",
                    required_variables=["matrix", "vector"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_VECTOR_MULTIPLICATION,
                ),
            ])

        elif question_type == Task.LINEAR_SYSTEM_SOLVER:
            templates.extend([
                QuestionTemplate(
                    template_string=f"{solve_verb} the linear system Ax = b for x, where A = {{matrix}} and b is the result from {{target_b}}.",
                    required_variables=["matrix", "target_b"],
                    difficulty_level=difficulty,
                    question_type=Task.LINEAR_SYSTEM_SOLVER,
                ),
                QuestionTemplate(
                    template_string="TODO: fix this template.",
                    required_variables=["matrix", "target_b", "x_symbols"],
                    difficulty_level=difficulty,
                    question_type=Task.LINEAR_SYSTEM_SOLVER,
                ),
            ])

        elif question_type == Task.DETERMINANT:
            templates.extend([
                QuestionTemplate(
                    template_string=f"{compute_verb} the determinant of the matrix from {{matrix}}.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.DETERMINANT,
                ),
                QuestionTemplate(
                    template_string="Find det(A) where A is {matrix}.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.DETERMINANT,
                ),
            ])

        elif question_type == Task.FROBENIUS_NORM:
            templates.extend([
                QuestionTemplate(
                    template_string=f"{compute_verb} the Frobenius norm of the matrix from {{matrix}}.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.FROBENIUS_NORM,
                ),
                QuestionTemplate(
                    template_string="Find ||A||_F where A is {matrix}.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.FROBENIUS_NORM,
                ),
            ])

        elif question_type == Task.MATRIX_TRACE:
            templates.extend([
                QuestionTemplate(
                    template_string=f"{compute_verb} the trace of the matrix from {{matrix}}.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_TRACE,
                ),
                QuestionTemplate(
                    template_string="Find tr(A) where A is {matrix}.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_TRACE,
                ),
            ])

        elif question_type == Task.MATRIX_RANK:
            templates.extend([
                QuestionTemplate(
                    template_string=f"{compute_verb} the rank of the matrix from {{matrix}}.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_RANK,
                ),
                QuestionTemplate(
                    template_string="Find rank(A) where A is {matrix}.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_RANK,
                ),
            ])

        elif question_type == Task.MATRIX_TRANSPOSE:
            templates.extend([
                QuestionTemplate(
                    template_string=f"{compute_verb} the transpose of the matrix from {{matrix}}.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_TRANSPOSE,
                ),
                QuestionTemplate(
                    template_string="Find A^T where A is {matrix}.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_TRANSPOSE,
                ),
            ])

        elif question_type == Task.MATRIX_INVERSE:
            templates.extend([
                QuestionTemplate(
                    template_string=f"{compute_verb} the inverse of the matrix from {{matrix}}.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_INVERSE,
                ),
                QuestionTemplate(
                    template_string="Find A^(-1) where A is {matrix}.",
                    required_variables=["matrix"],
                    difficulty_level=difficulty,
                    question_type=Task.MATRIX_INVERSE,
                ),
            ])

        return templates

    def select_template(
        self,
        templates: list[QuestionTemplate],
        question_type: Task,
        difficulty: DifficultyCategory,
        available_variables: dict[str, Any],
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

        # Filter by available variables
        if available_variables:
            available_vars = set(available_variables.keys())
            variable_compatible = [t for t in candidates if set(t.required_variables) == available_vars]
            if variable_compatible:
                candidates = variable_compatible
            else:
                raise ValueError(f"No variable compatible templates found for {available_variables}")

        return random.choice(candidates if candidates else templates)
