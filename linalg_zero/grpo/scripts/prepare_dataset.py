import json
import logging
import re
from argparse import ArgumentParser
from typing import Any

from datasets import Dataset, DatasetDict, DownloadMode, load_dataset

from linalg_zero.shared.lib import get_tools
from linalg_zero.shared.utils import get_logger, setup_logging

# Log both to file and console
setup_logging(level=logging.INFO, include_timestamp=True)
logger = get_logger(__name__)


def load_datasets(src_train: str, src_test: str) -> DatasetDict:
    """Load datasets for GRPO training."""
    # Load training dataset (has solutions)
    logger.info(f"Loading train dataset from https://huggingface.co/datasets/{src_train}")
    train_dataset = load_dataset(src_train, split="train", download_mode=DownloadMode.FORCE_REDOWNLOAD)

    # Load validation dataset (no solutions)
    logger.info(f"Loading validation dataset from https://huggingface.co/datasets/{src_test}")
    test_dataset = load_dataset(src_test, split="validation")

    # Prepare results
    assert isinstance(train_dataset, Dataset)  # noqa: S101
    assert isinstance(test_dataset, Dataset)  # noqa: S101

    return DatasetDict({"train": train_dataset, "validation": test_dataset})


def fix_think_tags(content: str) -> str:
    """Ensure exactly one newline after <think> and before </think>"""
    # First remove any existing whitespace around tags
    content = re.sub(r"<think>\s*", "<think>\n", content)
    content = re.sub(r"\s*</think>", "\n</think>", content)
    return content


def process_dataset_for_grpo(dataset: DatasetDict) -> DatasetDict:
    """Process dataset specifically for GRPO training."""

    # The necessary columns for GRPO training
    keep_columns = [
        "query",
        "ground_truth",
        "stepwise_ground_truths",
        "tools",
    ]

    def ensure_tools(example: dict[str, Any]) -> dict[str, Any]:
        """Ensure tools field is present."""
        if "tools" not in example or example["tools"] is None:
            example["tools"] = get_tools()
        return example

    def parse_messages_for_grpo(example: dict[str, Any]) -> dict[str, Any]:
        """Convert messages from JSON string to array and fix think tag formatting for GRPO."""
        if example.get("messages"):
            messages = json.loads(example["messages"])

            # Fix think tags in assistant messages
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "assistant" and "content" in msg:
                    msg["content"] = fix_think_tags(msg["content"])

            # Store processed messages for reference (optional)
            example["processed_messages"] = messages

        return example

    # Process training dataset (has messages field with solutions)
    train_dataset = dataset["train"]
    train_dataset = train_dataset.map(parse_messages_for_grpo)
    train_dataset = train_dataset.map(ensure_tools)

    # Process validation dataset (no messages, just problems)
    test_dataset = dataset["validation"]
    test_dataset = test_dataset.map(ensure_tools)

    # Remove unnecessary columns
    strip_cols = set(train_dataset.column_names) - set(keep_columns)
    if strip_cols:
        logger.info(f"Removing columns from train dataset: {strip_cols}")
        train_dataset = train_dataset.remove_columns(list(strip_cols))

    strip_cols = set(test_dataset.column_names) - set(keep_columns)
    if strip_cols:
        logger.info(f"Removing columns from validation dataset: {strip_cols}")
        test_dataset = test_dataset.remove_columns(list(strip_cols))

    # Ensure the two schemas align
    test_dataset = test_dataset.cast(train_dataset.features)

    # Prepare results
    assert isinstance(train_dataset, Dataset)  # noqa: S101
    assert isinstance(test_dataset, Dataset)  # noqa: S101

    return DatasetDict({"train": train_dataset, "validation": test_dataset})


def validate_grpo_dataset(dataset: DatasetDict) -> None:
    """Validate that the dataset is properly formatted for GRPO training."""
    logger.info("*** Validating GRPO dataset ***")

    required_columns = ["query", "ground_truth", "stepwise_ground_truths", "tools"]

    for split_name, split_data in dataset.items():
        logger.info(f"Validating {split_name} split...")

        # Check required columns
        missing_cols = set(required_columns) - set(split_data.column_names)
        if missing_cols:
            raise ValueError(f"Missing required columns in {split_name}: {missing_cols}")

        # Validate sample entries
        if len(split_data) > 0:
            sample = split_data[0]

            # Check query field
            if not sample["query"] or not sample["query"].strip():
                raise ValueError(f"Empty query in {split_name} split")

            # Check ground_truth is valid JSON
            try:
                json.loads(sample["ground_truth"])
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid ground_truth JSON in {split_name}: {e}") from e

            # Check stepwise_ground_truths is valid JSON list
            try:
                stepwise = json.loads(sample["stepwise_ground_truths"])
                if not isinstance(stepwise, list):
                    raise TypeError(f"stepwise_ground_truths must be a list in {split_name}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid stepwise_ground_truths JSON in {split_name}: {e}") from e

            # Check tools field
            if not isinstance(sample["tools"], list):
                raise ValueError(f"Tools field must be a list in {split_name}")

        logger.info(f"✓ {split_name} split validation passed ({len(split_data)} examples)")

    logger.info("*** Dataset validation completed successfully ***")


def prepare_debug(train: Dataset, validation: Dataset, dataset_size: int) -> DatasetDict:
    """Prepare debug dataset with limited size."""
    train = train.select(range(min(dataset_size, len(train))))
    validation = validation.select(range(min(dataset_size, len(validation))))
    return DatasetDict({"train": train, "validation": validation})


def main(output_repo: str, push_to_hub: bool, debug_mode: bool) -> None:
    """Main processing function for GRPO dataset preparation."""
    # Source datasets
    train_repo = "atomwalk12/linalgzero-distilled"  # Has solutions
    test_repo = "atomwalk12/linalgzero"  # No solutions

    logger.info("*** Loading datasets for GRPO training ***")
    dataset = load_datasets(train_repo, test_repo)

    # For debugging
    if debug_mode:
        size = 60
        logger.info(f"*** Preparing debug dataset (size: {size}) ***")
        dataset = prepare_debug(dataset["train"], dataset["validation"], dataset_size=size)

    # Process for GRPO
    logger.info("*** Processing dataset for GRPO training ***")
    dataset = process_dataset_for_grpo(dataset)

    # Validate the processed dataset
    validate_grpo_dataset(dataset)

    # Log dataset info
    logger.info("*** Dataset processing completed ***")
    logger.info("Processed dataset contains:")
    for split_name, split_data in dataset.items():
        logger.info(f"  - {split_name}: {len(split_data)} examples")
        logger.info(f"    Columns: {split_data.column_names}")

        # Show sample for verification
        if len(split_data) > 0:
            sample = split_data[0]
            logger.info(f"    Sample query: {sample['query'][:100]}...")
            logger.info(f"    Has ground_truth: {bool(sample['ground_truth'])}")
            logger.info(f"    Has stepwise_ground_truths: {bool(sample['stepwise_ground_truths'])}")
            logger.info(f"    Number of tools: {len(sample['tools'])}")

    # Push to hub
    if push_to_hub:
        logger.info("*** Pushing to Hub ***")
        try:
            dataset.push_to_hub(output_repo)
            logger.info(f"Successfully pushed dataset to https://huggingface.co/datasets/{output_repo}")
            logger.info("Dataset is now ready for GRPO training!")
        except Exception:
            logger.exception("Failed to push to hub")
    else:
        logger.info("Dataset processing completed. Use --push_to_hub to upload to HuggingFace Hub.")


if __name__ == "__main__":
    """Script entry point for GRPO dataset preparation."""
    parser = ArgumentParser(description="Prepare dataset for GRPO training")
    parser.add_argument(
        "--output_repo", default="atomwalk12/linalgzero-grpo", type=str, help="Output repository name for GRPO dataset"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the dataset to HuggingFace Hub")
    parser.add_argument("--debug_mode", action="store_true", help="Reduces dataset size to 60 examples for testing")
    args = parser.parse_args()

    main(args.output_repo, args.push_to_hub, args.debug_mode)
