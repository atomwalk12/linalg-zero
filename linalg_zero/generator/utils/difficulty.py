"""Difficulty modification utilities."""

from collections.abc import Callable

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]


def get_difficulty_index(difficulty: str) -> int:
    """Get the index of a difficulty level (0=easy, 1=medium, 2=hard)."""
    try:
        return DIFFICULTY_LEVELS.index(difficulty.lower())
    except ValueError:
        return 1  # Default to medium if unknown


def get_difficulty_by_index(index: int) -> str:
    """Get difficulty name by index, clamped to valid range."""
    clamped_index = max(0, min(index, len(DIFFICULTY_LEVELS) - 1))
    return DIFFICULTY_LEVELS[clamped_index]


def make_difficulty_booster(boost_level: int) -> Callable[[str], str]:
    """Creates a difficulty modifier function that increases difficulty by boost_level steps."""

    def modify_difficulty(current: str) -> str:
        current_index = get_difficulty_index(current)
        new_index = current_index + boost_level
        return get_difficulty_by_index(new_index)

    return modify_difficulty


def make_difficulty_reducer(reduction_level: int) -> Callable[[str], str]:
    """Creates a difficulty reducer function that decreases difficulty by reduction_level steps."""

    def reduce_difficulty(current: str) -> str:
        current_index = get_difficulty_index(current)
        new_index = current_index - reduction_level
        return get_difficulty_by_index(new_index)

    return reduce_difficulty


def is_valid_difficulty(difficulty: str) -> bool:
    """Check if a difficulty level is valid."""
    return difficulty.lower() in DIFFICULTY_LEVELS


def get_all_difficulties() -> list[str]:
    """Get all available difficulty levels."""
    return DIFFICULTY_LEVELS.copy()
