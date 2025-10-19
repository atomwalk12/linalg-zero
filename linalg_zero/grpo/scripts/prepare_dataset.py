import json
import logging
import re
from argparse import ArgumentParser
from typing import Any

from datasets import Dataset, DatasetDict, DownloadMode, load_dataset

from linalg_zero.shared.lib import get_tools
from linalg_zero.shared.system_prompts import get_math_system_prompt
from linalg_zero.shared.utils import get_logger, setup_logging

# Log both to file and console
setup_logging(level=logging.INFO, include_timestamp=True)
logger = get_logger(__name__)


def load_datasets(src_train: str, src_test: str) -> DatasetDict:
    """Load datasets"""
    # Load
    logger.info(f"Loading train dataset from https://huggingface.co/datasets/{src_train}")
    train_dataset = load_dataset(src_train, split="train", download_mode=DownloadMode.FORCE_REDOWNLOAD)

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


def process_dataset(dataset: DatasetDict) -> DatasetDict:
    """Load and process dataset for SFT training."""

    # The necessary columns for SFT
    keep_columns = [
        "tools",
        "prompt",
        "answer",
        "stepwise_ground_truths",
    ]

    # Add missing columns (messages & tools)
    def ensure_messages(example: dict[str, Any]) -> dict[str, Any]:
        example["prompt"] = [
            {"role": "system", "content": get_math_system_prompt(include_examples=False)},
            {"role": "user", "content": example["query"]},
        ]
        return example

    def ensure_tools(example: dict[str, Any]) -> dict[str, Any]:
        if "tools" not in example or example["tools"] is None:
            example["tools"] = get_tools()
        return example

    def parse_messages(example: dict[str, Any]) -> dict[str, Any]:
        """Convert messages from JSON string to array and fix think tag formatting"""
        messages = json.loads(example["messages"])

        # Fix think tags in assistant messages
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant" and "content" in msg:
                msg["content"] = fix_think_tags(msg["content"])

        example["prompt"] = messages
        return example

    train_dataset = dataset["train"]
    train_dataset = train_dataset.map(parse_messages)
    train_dataset = train_dataset.rename_column("ground_truth", "answer")

    test_dataset = dataset["validation"]
    test_dataset = test_dataset.map(ensure_messages)
    test_dataset = test_dataset.map(ensure_tools)
    test_dataset = test_dataset.rename_column("ground_truth", "answer")

    # Ensure only relevant columns are preserved
    strip_cols = set(train_dataset.column_names) - set(keep_columns)
    train_dataset = train_dataset.remove_columns(strip_cols)

    strip_cols = set(test_dataset.column_names) - set(keep_columns)
    test_dataset = test_dataset.remove_columns(strip_cols)

    # Ensure the two schemas align (in tools field)
    test_dataset = test_dataset.cast(train_dataset.features)

    # Prepare results
    assert isinstance(train_dataset, Dataset)  # noqa: S101
    assert isinstance(test_dataset, Dataset)  # noqa: S101

    return DatasetDict({"train": train_dataset, "test": test_dataset})


def prepare_debug(train: Dataset, validation: Dataset, dataset_size: int) -> DatasetDict:
    train = train.select(range(dataset_size))
    validation = validation.select(range(dataset_size))
    return DatasetDict({"train": train, "validation": validation})


def main(output_repo: str, push_to_hub: bool, debug_mode: bool) -> None:
    """Main processing function."""
    # Load
    train_repo = "atomwalk12/linalgzero-distilled"
    test_repo = "atomwalk12/linalgzero"

    logger.info("*** Loading datasets ***")
    dataset = load_datasets(train_repo, test_repo)

    # For debugging
    if debug_mode:
        size = 60
        logger.info(f"*** Preparing debug dataset (size: {size}) ***")
        dataset = prepare_debug(dataset["train"], dataset["validation"], dataset_size=size)

    # Process
    logger.info("*** Processing dataset ***")
    dataset = process_dataset(dataset)

    # Push to hub
    if push_to_hub:
        logger.info("*** Pushing to Hub ***")
        try:
            dataset.push_to_hub(output_repo)
            logger.info(f"Successfully pushed dataset to https://huggingface.co/datasets/{output_repo}")
        except Exception:
            logger.exception("Failed to push to hub")


if __name__ == "__main__":
    """Script entry point for SFT training."""
    parser = ArgumentParser()
    parser.add_argument("--output_repo", default="atomwalk12/linalgzero-grpo", type=str, help="Output repository name")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the dataset to HuggingFace")
    parser.add_argument("--debug_mode", action="store_true", help="Reduces dataset size to 60 examples")
    args = parser.parse_args()

    main(args.output_repo, args.push_to_hub, args.debug_mode)
