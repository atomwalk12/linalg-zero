"""Core data models for question generation."""

from dataclasses import dataclass


@dataclass
class Question:
    """Represents a generated question with its answer."""

    text: str
    answer: str
    difficulty: str = "medium"
    topic: str = "general"
    is_valid: bool = True
