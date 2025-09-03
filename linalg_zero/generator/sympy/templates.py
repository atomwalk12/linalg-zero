from linalg_zero.generator.models import DifficultyCategory, QuestionTemplate, Task


def get_independent_templates(  # noqa: C901
    question_type: Task, difficulty: DifficultyCategory, verb: str, variables: dict[str, str]
) -> list[QuestionTemplate]:
    templates = []
    if question_type == Task.LINEAR_SYSTEM_SOLVER:
        templates.extend([
            QuestionTemplate(
                template_string=f"{verb} the linear system Ax = b for x, where A = {{matrix_A}} and b = {{target_b}}.",
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
    if question_type == Task.MATRIX_VECTOR_MULTIPLICATION:
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
    if question_type == Task.MATRIX_MATRIX_MULTIPLICATION:
        templates.extend([
            QuestionTemplate(
                template_string=f"{verb} the matrix-matrix product AB, where A = {{matrix_A}} and B = {{matrix_B}}.",
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
    if question_type == Task.DETERMINANT:
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
    if question_type == Task.FROBENIUS_NORM:
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
    if question_type == Task.MATRIX_RANK:
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
    if question_type == Task.MATRIX_TRANSPOSE:
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
    if question_type == Task.MATRIX_INVERSE:
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
    if question_type == Task.MATRIX_TRACE:
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
    if question_type == Task.MATRIX_COFACTOR:
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


def get_composite_templates(question_type: Task, difficulty: DifficultyCategory, verb: str) -> list[QuestionTemplate]:
    composite_templates = {
        Task.MATRIX_VECTOR_MULTIPLICATION: [
            QuestionTemplate(
                template_string=f"{verb} the matrix-vector product using A = {{matrix}} and the vector from {{vector}}.",
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
        ],
        Task.MATRIX_MATRIX_MULTIPLICATION: [
            QuestionTemplate(
                template_string=f"{verb} the matrix-matrix product AB, where A = {{matrix_A}} and B = {{matrix_B}}.",
                required_variables=["matrix_A", "matrix_B"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_MATRIX_MULTIPLICATION,
            ),
            QuestionTemplate(
                template_string="Given matrix A = {matrix_A} and matrix B = {matrix_B}, find AB.",
                required_variables=["matrix_A", "matrix_B"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_MATRIX_MULTIPLICATION,
            ),
        ],
        Task.LINEAR_SYSTEM_SOLVER: [
            QuestionTemplate(
                template_string=f"{verb} the linear system Ax = b for x, where A = {{matrix}} and b is the result from {{target_b}}.",
                required_variables=["matrix", "target_b"],
                difficulty_level=difficulty,
                question_type=Task.LINEAR_SYSTEM_SOLVER,
            ),
            QuestionTemplate(
                template_string="What is the solution x to Ax = b, where A = {matrix} and b comes from {target_b}?",
                required_variables=["matrix", "target_b"],
                difficulty_level=difficulty,
                question_type=Task.LINEAR_SYSTEM_SOLVER,
            ),
        ],
        Task.DETERMINANT: [
            QuestionTemplate(
                template_string=f"{verb} the determinant of the matrix from {{matrix}}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.DETERMINANT,
            ),
            QuestionTemplate(
                template_string="What is the determinant of the resulting matrix from {matrix}?",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.DETERMINANT,
            ),
        ],
        Task.FROBENIUS_NORM: [
            QuestionTemplate(
                template_string=f"{verb} the Frobenius norm of the matrix from {{matrix}}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.FROBENIUS_NORM,
            ),
            QuestionTemplate(
                template_string="What is the Frobenius norm of the resulting matrix from {matrix}?",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.FROBENIUS_NORM,
            ),
        ],
        Task.MATRIX_TRACE: [
            QuestionTemplate(
                template_string=f"{verb} the trace of the matrix from {{matrix}}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_TRACE,
            ),
            QuestionTemplate(
                template_string="What is the trace of the resulting matrix from {matrix}?",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_TRACE,
            ),
        ],
        Task.MATRIX_RANK: [
            QuestionTemplate(
                template_string=f"{verb} the rank of the matrix from {{matrix}}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_RANK,
            ),
            QuestionTemplate(
                template_string="What is the rank of the resulting matrix from {matrix}?",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_RANK,
            ),
        ],
        Task.MATRIX_TRANSPOSE: [
            QuestionTemplate(
                template_string=f"{verb} the transpose of the matrix from {{matrix}}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_TRANSPOSE,
            ),
            QuestionTemplate(
                template_string="What is the transpose of the resulting matrix from {matrix}?",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_TRANSPOSE,
            ),
        ],
        Task.MATRIX_INVERSE: [
            QuestionTemplate(
                template_string=f"{verb} the inverse of the matrix from {{matrix}}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_INVERSE,
            ),
            QuestionTemplate(
                template_string="What is the inverse of the resulting matrix from {matrix}?",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_INVERSE,
            ),
        ],
        Task.MATRIX_COFACTOR: [
            QuestionTemplate(
                template_string=f"{verb} the cofactor matrix of the matrix from {{matrix}}.",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_COFACTOR,
            ),
            QuestionTemplate(
                template_string="What is the cofactor matrix of the resulting matrix from {matrix}?",
                required_variables=["matrix"],
                difficulty_level=difficulty,
                question_type=Task.MATRIX_COFACTOR,
            ),
        ],
    }
    return composite_templates[question_type]
