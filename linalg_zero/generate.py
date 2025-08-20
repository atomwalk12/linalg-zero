import logging

from linalg_zero.generator.core import DatasetGenerator, print_dataset
from linalg_zero.generator.models import Question
from linalg_zero.generator.registry import create_default_registry
from linalg_zero.shared.utils import get_logger, setup_logging


def main() -> None:  # pragma: no cover
    # Set up logging
    setup_logging(level=logging.INFO, include_timestamp=False)
    logger = get_logger(__name__)

    logger.info("Linear Algebra Dataset Generator")

    # Show available topics
    registry = create_default_registry()
    logger.info("Available topics: %s", registry.list_topics())

    # -----------------------------------------------
    # Generate and display the linear algebra dataset
    # -----------------------------------------------
    def matrix_only_validator(question: Question) -> bool:
        return "matrix" in question.text.lower() and len(question.answer) > 0

    generator = DatasetGenerator(topic="linear_algebra", validator_factory=matrix_only_validator)
    questions = generator.generate_dataset(num_questions=3)
    print_dataset(questions)

    # -------------------------------------------
    # Generate and display the arithmetic dataset
    # -------------------------------------------
    arithmetic_generator = DatasetGenerator(topic="arithmetic")
    arithmetic_questions = arithmetic_generator.generate_dataset(num_questions=2)
    print_dataset(arithmetic_questions)


if __name__ == "__main__":  # pragma: no cover
    main()
