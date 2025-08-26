"""Core question generation functionality."""

from collections.abc import Callable

from linalg_zero.generator.models import Question
from linalg_zero.generator.registry import create_default_registry
from linalg_zero.shared.utils import get_logger

logger = get_logger(__name__)


class QuestionGenerator:
    """
    Question generator using Instance Attribute Factory pattern. Here factories are passed as
    callables (i.e. functions, lambda expressions, methods, classes or partial functions).
    """

    def __init__(
        self, question_factory: Callable[[], Question], validator_factory: Callable[[Question], bool] | None = None
    ) -> None:
        """
        Initialize with factory callables.

        Args:
            question_factory: Any callable that returns a Question
            validator_factory: Optional callable to validate questions
        """
        self.question_factory = question_factory
        self.validator_factory = validator_factory or self._default_validator

    def generate(self) -> Question:
        """Generate a single question using the configured factories."""
        question = self.question_factory()

        # Set validation status using the configured validator
        question.is_valid = self.validator_factory(question)

        return question

    @staticmethod
    def _default_validator(question: Question) -> bool:
        """Default validator - checks basic requirements."""
        return len(question.question) > 0 and len(question.answer) > 0


class DatasetGenerator:
    """
    Dataset generator using Instance Attribute Factory pattern.

    Following python-patterns.guide recommendations - instead of a function
    with many parameters, use a class that accepts configuration in __init__.
    """

    def __init__(
        self,
        topic: str = "linear_algebra",
        validator_factory: Callable[[Question], bool] | None = None,
        max_attempts: int = 100,
    ):
        """Initialize with generation configuration."""
        self.topic = topic
        self.validator_factory = validator_factory or QuestionGenerator._default_validator
        self.max_attempts = max_attempts
        self.registry = create_default_registry()

    def generate_dataset(self, num_questions: int) -> list[Question]:
        """Generate a dataset with the configured parameters."""
        generator = QuestionGenerator(
            question_factory=lambda: self.registry.get_random_factory(self.topic)(),
            validator_factory=self.validator_factory,
        )

        questions: list[Question] = []
        attempts = 0

        while len(questions) < num_questions and attempts < self.max_attempts:
            question = generator.generate()
            if question.is_valid:
                questions.append(question)
            attempts += 1

        if len(questions) < num_questions:
            logger.warning(
                "Only generated %d/%d valid questions after %d attempts",
                len(questions),
                num_questions,
                self.max_attempts,
            )

        return questions


def print_dataset(questions: list[Question], include_invalid: bool = False) -> None:  # pragma: no cover
    """Display a formatted dataset of questions."""

    questions_to_print = questions if include_invalid else [q for q in questions if q.is_valid]

    if not questions_to_print:
        logger.info("No questions to display.")
        return

    logger.info("=" * 30)
    logger.info("GENERATED DATASET")
    logger.info("=" * 30)

    # Metadata
    topics = {q.topic for q in questions_to_print}
    problem_types = {q.problem_type for q in questions_to_print}
    difficulties = {q.difficulty for q in questions_to_print}
    entropy_values = [q.entropy_used for q in questions_to_print]
    tool_calls = [q.tool_calls_required for q in questions_to_print]

    # Summary
    logger.info("Dataset Summary:")
    logger.info("  Total Questions: %d", len(questions_to_print))
    logger.info("  Topics: %s", ", ".join(sorted(topics)))
    logger.info("  Problem Types: %s", ", ".join(sorted(problem_types)))
    logger.info("  Difficulties: %s", ", ".join(sorted(str(difficulty) for difficulty in difficulties)))
    logger.info(
        "  Entropy Used: %.2f - %.2f (avg: %.2f)",
        min(entropy_values),
        max(entropy_values),
        sum(entropy_values) / len(entropy_values),
    )
    logger.info(
        "  Tool Calls Required: %d - %d (avg: %.1f)",
        min(tool_calls),
        max(tool_calls),
        sum(tool_calls) / len(tool_calls),
    )
    logger.info("=" * 30)

    # Questions
    for i, question in enumerate(questions_to_print, 1):
        status = " [INVALID]" if not question.is_valid else ""
        logger.info("Question %d:%s", i, status)
        logger.info("Q: %s", question.question)
        logger.info("A: %s", question.answer)
        logger.info("")
