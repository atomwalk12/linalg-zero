import random

from linalg_zero.generator.models import DifficultyCategory, QuestionTemplate, Task


def get_static_templates(  # noqa: C901
    question_type: Task, difficulty: DifficultyCategory
) -> list[QuestionTemplate]:
    verb = random.choice(["Find", "Calculate", "Compute", "Determine", "Evaluate"])

    templates = []
    if question_type == Task.LINEAR_SYSTEM_SOLVER:
        templates.extend([
            QuestionTemplate(
                template_string="Solve the linear system Ax = b for x, where A = {matrix_A} and b = {target_b}.",
                required_variables=["matrix_A", "target_b"],
                difficulty_level=difficulty,
                question_type=Task.LINEAR_SYSTEM_SOLVER,
            ),
            QuestionTemplate(
                template_string="Given matrix A = {matrix_A} and vector b = {target_b}, solve Ax = b for x.",
                required_variables=["matrix_A", "target_b"],
                difficulty_level=difficulty,
                question_type=Task.LINEAR_SYSTEM_SOLVER,
            ),
            QuestionTemplate(
                template_string="What is the solution x to the equation Ax = b, where A = {matrix_A} and b = {target_b}?",
                required_variables=["matrix_A", "target_b"],
                difficulty_level=difficulty,
                question_type=Task.LINEAR_SYSTEM_SOLVER,
            ),
        ])
    elif question_type == Task.MATRIX_VECTOR_MULTIPLICATION:
        templates.extend([
            QuestionTemplate(
                template_string=f"{verb} the matrix-vector product Av, where A = {{matrix}} and v = {{vector}}.",
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
                template_string="Multiply matrix A = {matrix} by vector v = {vector}.",
                required_variables=["matrix", "vector"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_VECTOR_MULTIPLICATION,
            ),
        ])
    elif question_type == Task.MATRIX_MATRIX_MULTIPLICATION:
        templates.extend([
            QuestionTemplate(
                template_string=f"{verb} the matrix product AB, where A = {{matrix_A}} and B = {{matrix_B}}.",
                required_variables=["matrix_A", "matrix_B"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_MATRIX_MULTIPLICATION,
            ),
            QuestionTemplate(
                template_string="Given matrix A = {matrix_A} and matrix B = {matrix_B}, compute AB.",
                required_variables=["matrix_A", "matrix_B"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_MATRIX_MULTIPLICATION,
            ),
            QuestionTemplate(
                template_string="Multiply matrix A = {matrix_A} by matrix B = {matrix_B}.",
                required_variables=["matrix_A", "matrix_B"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_MATRIX_MULTIPLICATION,
            ),
        ])
    elif question_type == Task.DETERMINANT:
        templates.extend([
            QuestionTemplate(
                template_string=f"{verb} the determinant of matrix A, where A = {{matrix}}.",
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
                template_string=f"{verb} the Frobenius norm of matrix A = {{matrix}}.",
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
                template_string="What is ||A||_F for A = {matrix}?",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.FROBENIUS_NORM,
            ),
        ])
    elif question_type == Task.MATRIX_RANK:
        templates.extend([
            QuestionTemplate(
                template_string=f"{verb} the rank of matrix A = {{matrix}}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_RANK,
            ),
            QuestionTemplate(
                template_string="What is the rank of matrix A = {matrix}?",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_RANK,
            ),
            QuestionTemplate(
                template_string="Find rank(A) for A = {matrix}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_RANK,
            ),
        ])
    elif question_type == Task.MATRIX_TRANSPOSE:
        templates.extend([
            QuestionTemplate(
                template_string=f"{verb} the transpose of matrix A = {{matrix}}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_TRANSPOSE,
            ),
            QuestionTemplate(
                template_string="Find A^T for A = {matrix}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_TRANSPOSE,
            ),
            QuestionTemplate(
                template_string="What is the transpose of A = {matrix}?",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_TRANSPOSE,
            ),
        ])
    elif question_type == Task.MATRIX_INVERSE:
        templates.extend([
            QuestionTemplate(
                template_string=f"{verb} the inverse of matrix A = {{matrix}}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_INVERSE,
            ),
            QuestionTemplate(
                template_string="Find A^(-1) for A = {matrix}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_INVERSE,
            ),
            QuestionTemplate(
                template_string="What is the inverse of matrix A = {matrix}?",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_INVERSE,
            ),
        ])
    elif question_type == Task.MATRIX_TRACE:
        templates.extend([
            QuestionTemplate(
                template_string=f"{verb} the trace of matrix A = {{matrix}}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_TRACE,
            ),
            QuestionTemplate(
                template_string="Find tr(A) for A = {matrix}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_TRACE,
            ),
            QuestionTemplate(
                template_string="What is the trace of A = {matrix}?",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_TRACE,
            ),
        ])
    elif question_type == Task.MATRIX_COFACTOR:
        templates.extend([
            QuestionTemplate(
                template_string=f"{verb} the cofactor matrix of A = {{matrix}}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_COFACTOR,
            ),
            QuestionTemplate(
                template_string="Find the cofactor matrix for A = {matrix}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_COFACTOR,
            ),
            QuestionTemplate(
                template_string="What is the matrix of cofactors for A = {matrix}?",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_COFACTOR,
            ),
        ])
    return templates
