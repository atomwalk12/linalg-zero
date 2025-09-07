import argparse
import logging

from linalg_zero.generator.analysis.utils import (
    compute_stepwise_value_statistics,
    print_statistics_summary,
)
from linalg_zero.generator.core import DatasetGenerator
from linalg_zero.generator.models import DifficultyCategory, Question, Topic
from linalg_zero.generator.registry import create_default_registry, create_optimized_registry
from linalg_zero.generator.utils import (
    convert_to_dataset_splits,
    print_dataset,
    set_seed,
    verify_dataset,
)
from linalg_zero.shared.utils import get_logger, push_to_hub, setup_logging


def main(push_dataset: bool = False, use_optimized_registry: bool = False) -> None:  # pragma: no cover
    # Set up logging
    setup_logging(level=logging.INFO, include_timestamp=False)
    logger = get_logger(__name__)

    logger.info("Linear Algebra Dataset Generator")

    # Create registry (either default or optimized)
    if use_optimized_registry:
        registry = create_optimized_registry(filename="results/entropy_analysis/top_entropy_choices.json")
        logger.info("Using optimized entropy settings from analysis results")
    else:
        registry = create_default_registry()

    logger.info("Available topics: %s", registry.list_topics())

    # -----------------------------------------------
    # Generate and display the linear algebra dataset
    # -----------------------------------------------
    def matrix_only_validator(question: Question) -> bool:
        # A filter to only include questions that satisfy specific conditions
        return len(question.answer) > 0

    generator = DatasetGenerator(
        topic=Topic.LINEAR_ALGEBRA, validator_factory=matrix_only_validator, registry=registry
    )

    # Generate custom amounts per difficulty category
    # Easy: 3000, Medium: 2000, Hard: 1000 (total: 6000)
    dataset = generator.generate_exact_for_categories(
        requests={
            # Total: 12.4k problems
            DifficultyCategory.ONE_TOOL_CALL: 800,
            DifficultyCategory.TWO_TOOL_CALLS: 1000,
            DifficultyCategory.THREE_TOOL_CALLS: 1200,
        }
        # Alternative: 1.5k problems for each difficulty category
    )
    statistics = compute_stepwise_value_statistics(dataset)
    print_dataset(dataset)
    print_statistics_summary(statistics)
    verify_dataset(dataset)

    if push_dataset:
        # Create stratified splits by difficulty for balanced evaluation
        splits = convert_to_dataset_splits(
            dataset,
            test_size=0.1,
            val_size=0.1,
            seed=argv.seed or 42,
            stratify_by="difficulty",
        )
        push_to_hub(splits, "atomwalk12/linalg-zero-dataset", private=False)

    # --------------------------------------------------
    # This is an example on generating other topic types
    # --------------------------------------------------
    # arithmetic_generator = DatasetGenerator(topic="arithmetic")
    # arithmetic_questions = arithmetic_generator.generate_dataset(num_questions=2)
    # print_dataset(arithmetic_questions)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--push_dataset", action="store_true", default=True)
    parser.add_argument(
        "--use_optimized_registry",
        action="store_true",
        default=True,
        help="Use optimized entropy settings from analysis results for dataset generation",
    )
    argv = parser.parse_args()

    set_seed()
    main(argv.push_dataset, argv.use_optimized_registry)
