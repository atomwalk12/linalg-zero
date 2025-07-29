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

from datasets import Dataset  # type: ignore[attr-defined]
from linalg_zero.shared import get_logger, setup_logging


def create_debug_dataset() -> list[dict[str, Any]]:
    """Create a comprehensive debug dataset with various linear algebra problems."""
    from linalg_zero.distillation.components.planner_for_tool_calling import UNIFIED_PLANNING_PROMPT

    return [
        {
            "messages": [
                {"role": "system", "content": UNIFIED_PLANNING_PROMPT},
                {
                    "role": "user",
                    "content": "What is the Frobenius norm of the product of matrices [[1, 2], [3, 4]] and [[2, 0], [1, 3]]?",
                },
            ],
            "ground_truth_result": "17.204650534085253",
        },
        {
            "messages": [
                {"role": "system", "content": UNIFIED_PLANNING_PROMPT},
                {
                    "role": "user",
                    "content": "Calculate the determinant of the matrix [[3, 1], [2, 4]].",
                },
            ],
            "ground_truth_result": "10.0",
        },
        {
            "messages": [
                {"role": "system", "content": UNIFIED_PLANNING_PROMPT},
                {
                    "role": "user",
                    "content": "Find the trace of the matrix [[1, 2, 3], [4, 5, 6], [7, 8, 9]].",
                },
            ],
            "ground_truth_result": "15.0",
        },
        {
            "messages": [
                {"role": "system", "content": UNIFIED_PLANNING_PROMPT},
                {
                    "role": "user",
                    "content": "Compute the L2 norm of the vector [3, 4, 5].",
                },
            ],
            "ground_truth_result": "7.0710678118654755",
        },
        {
            "messages": [
                {"role": "system", "content": UNIFIED_PLANNING_PROMPT},
                {
                    "role": "user",
                    "content": "What is the rank of the matrix [[1, 2], [2, 4]]?",
                },
            ],
            "ground_truth_result": "1",
        },
    ]


def push_debug_dataset_to_hub(dataset_name: str, private: bool = True) -> None:
    """Create debug dataset and push it to Hugging Face Hub."""
    logger = get_logger(__name__)

    # Create the debug dataset
    debug_data = create_debug_dataset()

    # Convert to HuggingFace Dataset
    hf_dataset = Dataset.from_list(debug_data)

    # Push to hub
    logger.info(f"Pushing debug dataset to: {dataset_name}")
    _ = hf_dataset.push_to_hub(dataset_name, private=private)
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
