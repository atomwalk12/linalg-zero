from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from linalg_zero.generator.entropy_control import SampleArgs, sample_entropy_from_range
from linalg_zero.generator.models import DifficultyCategory, Task, Topic
from linalg_zero.shared.utils import get_logger

logger = get_logger(__name__)


class Precision(Enum):
    """Precision for formatting mathematical expressions."""

    MATRIX_VECTOR_MULTIPLICATION = 2
    LINEAR_SYSTEM_SOLVER = 2
    DETERMINANT = 2
    FROBENIUS_NORM = 2
    MATRIX_RANK = 2
    MATRIX_TRANSPOSE = 2
    MATRIX_INVERSE = 2
    MATRIX_TRACE = 2
    MATRIX_COFACTOR = 2
    FULL = -1


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
    entropy_range: tuple[float, float]
    # NOTE: for small entropy ranges it is better to set this to False
    center_biased_draw: bool = False

    @property
    def sample_entropy(self) -> float:
        """Get entropy from configuration."""
        entropy = self._sample_entropy()
        return entropy

    def get_random_matrix_size(self) -> int:
        """Get a random matrix size within the allowed range."""
        return random.randint(*self.matrix_size_range)

    def _sample_entropy(self) -> float:
        """Sample entropy within the configured range for this difficulty.

        To avoid extreme values while preserving diversity, we sample from a
        symmetric Beta distribution (center-biased) and scale to the range.
        """
        return sample_entropy_from_range(self.entropy_range, self.center_biased_draw)

    def create_sample_args_for_composition(self, num_components: int) -> SampleArgs:
        """Create SampleArgs for compositions - always uses entropy for Dirichlet distribution."""
        entropy = self._sample_entropy()
        return SampleArgs(num_modules=num_components, entropy=entropy)


# Possible entropy ranges:
# Moderate variability:
#   - 1 tool call: (1.2, 1.8)
#   - 2 tool calls: (2.6, 3.6)
#   - 3 tool calls: (3.8, 5.2)

# High variability:
#   - 1 tool call: (1.0, 2.0)
#   - 2 tool calls: (2.0, 4.0)
#   - 3 tool calls: (3.0, 6.0)

# Low variability:
#   - 1 tool call: (1.4, 1.6)
#   - 2 tool calls: (2.8, 3.2)
#   - 3 tool calls: (4.2, 4.8)

DIFFICULTY_CONFIGS = {
    DifficultyCategory.ONE_TOOL_CALL: ProblemConfig(
        target_tool_calls=1,
        matrix_size_range=(2, 2),
        allow_rationals=False,
        entropy_range=(1.2, 1.8),
    ),
    DifficultyCategory.TWO_TOOL_CALLS: ProblemConfig(
        target_tool_calls=1,
        matrix_size_range=(2, 3),
        allow_rationals=False,
        entropy_range=(2.6, 3.6),
    ),
    DifficultyCategory.THREE_TOOL_CALLS: ProblemConfig(
        target_tool_calls=1,
        matrix_size_range=(3, 3),
        allow_rationals=True,
        entropy_range=(2.6, 3.6),
    ),
}


def get_problem_config(difficulty: DifficultyCategory, topic: Topic, problem_type: Task) -> ProblemConfig:
    """Get problem configuration for a given difficulty level, topic, and problem type."""
    return DIFFICULTY_CONFIGS[difficulty]


def sample_entropy(difficulty: DifficultyCategory, topic: Topic, problem_type: Task) -> float:
    """Get sampled entropy for a given difficulty level."""
    return get_problem_config(difficulty, topic, problem_type).sample_entropy


def validate_tool_calls(expected: int, actual: int, problem_type: Task) -> bool:
    """Validate that a problem uses the expected number of tool calls."""
    if actual != expected:
        raise ValueError(
            f"Problem type '{problem_type}' expected {expected} tool calls, "
            f"but used {actual}. This violates the difficulty system."
        )
    return True
