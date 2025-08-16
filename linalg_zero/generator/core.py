"""Core question generation functionality."""

from collections.abc import Callable

from linalg_zero.generator.models import Question
from linalg_zero.generator.registry import create_default_registry
from linalg_zero.shared.utils import get_logger

# Set up logger
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
        return len(question.text) > 0 and len(question.answer) > 0


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
    # Filter questions based on include_invalid flag
    questions_to_print = questions if include_invalid else [q for q in questions if q.is_valid]

    logger.info("=" * 30)
    logger.info("GENERATED DATASET")
    logger.info("=" * 30)

    for i, question in enumerate(questions_to_print, 1):
        status = " [INVALID]" if not question.is_valid else ""
        logger.info("Question %d:%s", i, status)
        logger.info("Topic: %s | Difficulty: %s", question.topic, question.difficulty)
        logger.info("Q: %s", question.text)
        logger.info("A: %s", question.answer)
