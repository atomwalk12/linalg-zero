"""Core data models for question generation."""

from dataclasses import dataclass, field

from linalg_zero.generator.utils.difficulty import DifficultyCategory


@dataclass
class Question:
    """Represents a generated question with its answer."""

    text: str
    answer: str
    difficulty: DifficultyCategory
    topic: str = "general"
    problem_type: str = "general"
    is_valid: bool = True
    entropy_used: float = 0.0
    tool_calls_required: int = 0
    stepwise: list[dict[str, str]] = field(default_factory=list)
    golden: dict[str, str] = field(default_factory=dict)
