"""Difficulty modification utilities."""

from enum import Enum

# Minimum and maximum tool calls for difficulty
MIN_DIFFICULTY = 1
MAX_DIFFICULTY = 10


class DifficultyCategory(Enum):
    """Enum for difficulty categories used in problem generation."""

    EASY = 1
    MEDIUM = 2
    HARD = 3

    def __str__(self) -> str:
        """Return the string value for compatibility with existing code."""
        if self == DifficultyCategory.EASY:
            return "easy"
        elif self == DifficultyCategory.MEDIUM:
            return "medium"
        elif self == DifficultyCategory.HARD:
            return "hard"
        else:
            raise ValueError(f"Invalid difficulty category: {self}")
