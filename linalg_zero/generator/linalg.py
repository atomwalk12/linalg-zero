"""Linear algebra question factory functions."""

import random

from .models import Question


def matrix_determinant_2x2_factory() -> Question:
    """Factory function for 2x2 matrix determinant questions."""
    # Generate simple 2x2 matrix
    a, b = random.randint(1, 5), random.randint(1, 5)
    c, d = random.randint(1, 5), random.randint(1, 5)
    determinant = a * d - b * c

    return Question(
        text=f"What is the determinant of the matrix [[{a}, {b}], [{c}, {d}]]?",
        answer=str(determinant),
        difficulty="medium",
        topic="linear_algebra",
    )


def vector_dot_product_factory() -> Question:
    """Factory function for vector dot product questions."""
    # Generate simple 2D vectors
    v1 = [random.randint(1, 5), random.randint(1, 5)]
    v2 = [random.randint(1, 5), random.randint(1, 5)]
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]

    return Question(
        text=f"What is the dot product of vectors [{v1[0]}, {v1[1]}] and [{v2[0]}, {v2[1]}]?",
        answer=str(dot_product),
        difficulty="medium",
        topic="linear_algebra",
    )


def matrix_addition_factory() -> Question:
    """Factory function for 2x2 matrix addition questions."""
    # Generate two 2x2 matrices
    A = [[random.randint(1, 9) for _ in range(2)] for _ in range(2)]
    B = [[random.randint(1, 9) for _ in range(2)] for _ in range(2)]
    result = [[A[i][j] + B[i][j] for j in range(2)] for i in range(2)]

    return Question(
        text=f"Add the matrices {A} + {B}. Give your answer as a 2x2 matrix.",
        answer=str(result),
        difficulty="medium",
        topic="linear_algebra",
    )
