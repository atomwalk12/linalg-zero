#!/usr/bin/env python3
"""
Script to create and push debug dataset to Hugging Face Hub.

Usage:
    python scripts/push_debug_dataset.py --dataset-name your-username/linalg-debug --private
"""

import argparse
import logging
import sys
from typing import Any

from datasets import Dataset, DatasetDict
from linalg_zero.shared import get_logger, setup_logging


def create_debug_dataset() -> list[dict[str, Any]]:
    """Create a comprehensive debug dataset with various linear algebra problems."""

    return [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the Frobenius norm of the product of matrices [[1, 2], [3, 4]] and [[2, 0], [1, 3]]?",
                },
            ],
            "ground_truth_result": "17.204650534085253",
            "ground_tool_calls": [
                {
                    "function": {
                        "arguments": '{"matrix_a": [[1, 2], [3, 4]], "matrix_b": [[2, 0], [1, 3]]}',
                        "name": "multiply_matrices",
                    },
                    "id": "tool_call_0",
                    "type": "function",
                },
                {
                    "function": {"arguments": '{"matrix": "[result_of_call_0]"}', "name": "frobenius_norm"},
                    "id": "tool_call_1",
                    "type": "function",
                },
            ],
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the Frobenius norm of the product of matrices [[1, 2], [3, 4]] and [[2, 0], [1, 3]]?",
                },
            ],
            "ground_truth_result": "17.204650534085253",
            "ground_tool_calls": [
                {
                    "function": {
                        "arguments": '{"matrix_a": [[1, 2], [3, 4]], "matrix_b": [[2, 0], [1, 3]]}',
                        "name": "multiply_matrices",
                    },
                    "id": "tool_call_0",
                    "type": "function",
                },
                {
                    "function": {"arguments": '{"matrix": "[result_of_call_0]"}', "name": "frobenius_norm"},
                    "id": "tool_call_1",
                    "type": "function",
                },
            ],
        },
    ]


def push_debug_dataset_to_hub(dataset_name: str, private: bool = True) -> None:
    """Create debug dataset and push it to Hugging Face Hub."""
    logger = get_logger(__name__)

    # Create the debug dataset
    debug_data = create_debug_dataset()

    # Split data into train/test (80/20)
    split_idx = int(0.8 * len(debug_data))
    train_data = debug_data[:split_idx]
    test_data = debug_data[split_idx:] if split_idx < len(debug_data) else debug_data[:1]

    # Convert to HuggingFace DatasetDict with train/test splits

    dataset_dict = DatasetDict({"train": Dataset.from_list(train_data), "test": Dataset.from_list(test_data)})

    # Push to hub
    logger.info(f"Pushing debug dataset to: {dataset_name}")
    logger.info(f"Train examples: {len(train_data)}, Test examples: {len(test_data)}")
    _ = dataset_dict.push_to_hub(dataset_name, private=private)
    logger.info("Debug dataset successfully pushed to hub!")


def main() -> None:
    """Main function to push debug dataset to hub."""
    parser = argparse.ArgumentParser(description="Push debug dataset to Hugging Face Hub")
    _ = parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name of the dataset on HuggingFace Hub (e.g., 'username/linalg-debug')",
    )
    _ = parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private (default: True)",
    )
    _ = parser.add_argument(
        "--public",
        action="store_true",
        help="Make the dataset public",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=logging.INFO, include_timestamp=True)

    # Determine privacy setting
    private = not args.public  # Default to private unless --public is specified

    try:
        push_debug_dataset_to_hub(args.dataset_name, private=private)
        print(f"✅ Debug dataset successfully pushed to: {args.dataset_name}")
        print(f"   Privacy: {'Private' if private else 'Public'}")
        print(f"   Access URL: https://huggingface.co/datasets/{args.dataset_name}")
    except Exception as e:
        print(f"❌ Error pushing dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
