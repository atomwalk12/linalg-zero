import argparse
import logging
import random

import numpy as np
from sympy.core.random import seed

from linalg_zero.generator.core import DatasetGenerator
from linalg_zero.generator.models import Question, Topic
from linalg_zero.generator.registry import create_default_registry
from linalg_zero.generator.utils import print_dataset, verify_dataset
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
        # A filter to only include questions that satisfy specific conditions
        return len(question.answer) > 0

    generator = DatasetGenerator(topic=Topic.LINEAR_ALGEBRA, validator_factory=matrix_only_validator)
    dataset = generator.generate_dataset(num_questions=3000)
    print_dataset(dataset)
    verify_dataset(dataset)

    # --------------------------------------------------
    # This is an example on generating other topic types
    # --------------------------------------------------
    # arithmetic_generator = DatasetGenerator(topic="arithmetic")
    # arithmetic_questions = arithmetic_generator.generate_dataset(num_questions=2)
    # print_dataset(arithmetic_questions)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    argv = parser.parse_args()

    if argv.seed is not None:
        random.seed(argv.seed)
        np.random.seed(argv.seed)
        seed(argv.seed)

    main()
