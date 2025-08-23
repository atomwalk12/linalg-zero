"""Difficulty modification utilities."""

from collections.abc import Callable
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


def clamp_difficulty(difficulty: int) -> int:
    """Clamp difficulty to valid range (1 to 10 tool calls)."""
    return max(MIN_DIFFICULTY, min(difficulty, MAX_DIFFICULTY))


def make_difficulty_booster(boost_level: int) -> Callable[[int], int]:
    """Creates a difficulty modifier function that increases difficulty by boost_level tool calls."""

    def modify_difficulty(current: int) -> int:
        new_difficulty = current + boost_level
        return clamp_difficulty(new_difficulty)

    return modify_difficulty


def make_difficulty_reducer(reduction_level: int) -> Callable[[int], int]:
    """Creates a difficulty reducer function that decreases difficulty by reduction_level tool calls."""

    def reduce_difficulty(current: int) -> int:
        new_difficulty = current - reduction_level
        return clamp_difficulty(new_difficulty)

    return reduce_difficulty


def is_valid_difficulty(difficulty: int) -> bool:
    """Check if a difficulty level is valid (between 1 and 10 tool calls)."""
    return MIN_DIFFICULTY <= difficulty <= MAX_DIFFICULTY


def get_difficulty_range() -> tuple[int, int]:
    """Get the valid difficulty range as (min, max) tuple."""
    return (MIN_DIFFICULTY, MAX_DIFFICULTY)


def categorize_by_tool_calls(difficulty: int) -> str:
    """Get a descriptive string for the number of tool calls needed."""
    if difficulty == 1:
        return "1 tool call"
    else:
        return f"{difficulty} tool calls"
