from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from linalg_zero.generator.entropy_control import SampleArgs
from linalg_zero.generator.models import DifficultyCategory, Task, Topic
from linalg_zero.shared.utils import get_logger

logger = get_logger(__name__)


class ToolCallDifficulty(Enum):
    """Tool call based difficulty levels."""

    SINGLE_TOOL = 1
    DUAL_TOOL = 2
    MULTI_TOOL = 3


@dataclass(frozen=True)
class ProblemConfig:
    """Configuration parameters for problems based on tool calls and difficulty."""

    target_tool_calls: int
    matrix_size_range: tuple[int, int]
    allow_rationals: bool
    special_constraints: list[str]
    entropy_range: tuple[float, float]

    @property
    def sample_entropy(self) -> float:
        """Get entropy from configuration."""
        entropy = self._sample_entropy()
        return entropy

    def get_random_matrix_size(self) -> int:
        """Get a random matrix size within the allowed range."""
        return random.randint(*self.matrix_size_range)

    def _sample_entropy(self) -> float:
        """Sample entropy within the configured range for this difficulty."""
        return random.uniform(*self.entropy_range)

    def create_sample_args_for_composition(self, num_components: int) -> SampleArgs:
        """Create SampleArgs for compositions - always uses entropy for Dirichlet distribution."""
        entropy = self._sample_entropy()
        return SampleArgs(num_modules=num_components, entropy=entropy)


DIFFICULTY_CONFIGS = {
    DifficultyCategory.EASY: ProblemConfig(
        target_tool_calls=1,
        matrix_size_range=(2, 2),
        allow_rationals=False,
        special_constraints=[],
        entropy_range=(0.5, 1.5),
    ),
    DifficultyCategory.MEDIUM: ProblemConfig(
        target_tool_calls=1,
        matrix_size_range=(2, 3),
        allow_rationals=False,
        special_constraints=["ensure_invertible"],
        entropy_range=(1.5, 3.0),
    ),
    DifficultyCategory.HARD: ProblemConfig(
        target_tool_calls=1,
        matrix_size_range=(3, 4),
        allow_rationals=False,
        special_constraints=["complex_decomposition"],
        entropy_range=(1.5, 4.0),
    ),
}


def get_problem_config(difficulty: DifficultyCategory, topic: Topic, problem_type: Task) -> ProblemConfig:
    """Get configuration for a given difficulty level."""
    return DIFFICULTY_CONFIGS[difficulty]


def validate_tool_calls(expected: int, actual: int, problem_type: Task) -> bool:
    """Validate that a problem uses the expected number of tool calls."""
    if actual != expected:
        raise ValueError(
            f"Problem type '{problem_type}' expected {expected} tool calls, "
            f"but used {actual}. This violates the difficulty system."
        )
    return True
