from dataclasses import dataclass

from linalg_zero.generator.entropy_control import sample_entropy_from_range


@dataclass
class EntropyConstraints:
    entropy: float | None = None
    entropy_range: tuple[float, float] | None = None

    def sample_entropy(self, center_biased_draw: bool) -> float | None:
        if self.entropy is not None and self.entropy_range is not None:
            raise ValueError("Cannot specify both 'entropy' and 'entropy_range'")
        if self.entropy is not None:
            return self.entropy
        if self.entropy_range is not None:
            return sample_entropy_from_range(self.entropy_range, center_biased_draw)
        return None


@dataclass
class GenerationConstraints:
    """Matrix generation constraints for controlling output properties.

    Attributes:
        square: Generate square matrix (rows == cols)
        invertible: Generate invertible matrix (requires square=True)
        size: Specific size for square matrices (overrides rows/cols)
        rows: Specific number of rows
        cols: Specific number of columns
    """

    square: bool = False
    invertible: bool = False
    size: int | None = None
    rows: int | None = None
    cols: int | None = None

    def __post_init__(self) -> None:
        """Validate constraint combinations."""
        # Invertible matrices must be square
        if self.invertible and not self.square:
            raise ValueError("Invertible matrices must be square (set square=True)")

        # If invertible is True, automatically set square to True
        if self.invertible:
            self.square = True

        # Can't specify both size and rows/cols
        if self.size is not None and (self.rows is not None or self.cols is not None):
            raise ValueError("Cannot specify both 'size' and 'rows'/'cols' parameters")

        # Square matrices can't have different rows/cols specified
        if (self.square and self.rows is not None and self.cols is not None) and (self.rows != self.cols):
            raise ValueError("Square matrices must have equal rows and cols")

    def merge(self, other: "GenerationConstraints") -> "GenerationConstraints":
        """Merge two GenerationConstraints objects with conflict detection."""
        if not isinstance(other, GenerationConstraints):
            raise TypeError(f"Can only merge with GenerationConstraints, got {type(other)}")

        conflicts = []

        # Check for conflicts in each field
        if self.square is not False and other.square is not False and self.square != other.square:
            conflicts.append(f"square: {self.square} vs {other.square}")

        if self.invertible is not False and other.invertible is not False and self.invertible != other.invertible:
            conflicts.append(f"invertible: {self.invertible} vs {other.invertible}")

        if self.size is not None and other.size is not None and self.size != other.size:
            conflicts.append(f"size: {self.size} vs {other.size}")

        if self.rows is not None and other.rows is not None and self.rows != other.rows:
            conflicts.append(f"rows: {self.rows} vs {other.rows}")

        if self.cols is not None and other.cols is not None and self.cols != other.cols:
            conflicts.append(f"cols: {self.cols} vs {other.cols}")

        if conflicts:
            raise ValueError(f"Conflicting constraints found: {', '.join(conflicts)}")

        # Merge constraints (other takes precedence for non-None/non-False values)
        merged = GenerationConstraints(
            square=other.square if other.square else self.square,
            invertible=other.invertible if other.invertible else self.invertible,
            size=other.size if other.size is not None else self.size,
            rows=other.rows if other.rows is not None else self.rows,
            cols=other.cols if other.cols is not None else self.cols,
        )

        return merged
