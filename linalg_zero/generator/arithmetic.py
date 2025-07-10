"""Arithmetic question factory functions."""

import random

from .models import Question


def arithmetic_addition_factory() -> Question:
    """Factory function for basic arithmetic addition questions."""
    a, b = random.randint(1, 10), random.randint(1, 10)
    return Question(text=f"What is {a} + {b}?", answer=str(a + b), difficulty="easy", topic="arithmetic")
