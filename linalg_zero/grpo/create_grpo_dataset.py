#!/usr/bin/env python3
"""
Download HuggingFace dataset and convert to parquet format for VERL training.
"""

import argparse
import os

from argilla import Dataset

import datasets
from linalg_zero.distillation.utils import get_function_schema

# TODO: add more useful examples
json_string = """{\"name\": \"Tool name\", \"arguments\": {\"Argument name\": \"Argument content\", \"... ...\": \"... ...\"}}
{\"name\": \"... ...\", \"arguments\": {\"... ...\": \"... ...\", \"... ...\": \"... ...\"}}"""


SYS = """You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks and provide structured chat responses.

**Available Tools**
In your response, you can use the following tools:
{tools}

**Steps for Each Turn**
1. **Think:** Recall relevant context and analyze the current user goal.
2. **Decide on Tool Usage:** If a tool is needed, specify the tool and its arguments.
3. **Respond Appropriately:** If a response is needed, generate one while maintaining consistency across user queries.

**Output Format**
```plaintext
<think> Your thoughts and reasoning </think>
<tool_call>
{json_string}
...
</tool_call>
<response> AI's final response </response>
```

**Important Notes**
1. You must always include the `<think>` field to outline your reasoning. Provide at least one of `<tool_call>` or `<response>`. Decide whether to use `<tool_call>` (possibly multiple times), `<response>`, or both.
2. You can invoke multiple tool calls simultaneously in the `<tool_call>` fields. Each tool call should be a JSON object with a "name" field and an "arguments" field containing a dictionary of arguments. If no arguments are needed, leave the "arguments" field an empty dictionary.
3. Refer to the previous dialogue records in the history, including the user's queries, previous `<tool_call>`, `<response>`, and any tool feedback noted as `<obs>` (if exists).
""".format(tools=get_function_schema(), json_string=json_string)  # noqa: UP032


def remove_redundant_columns(dataset: Dataset, required_columns: list[str]) -> Dataset:
    return dataset.remove_columns([col for col in dataset.column_names if col not in required_columns])


def make_map_fn(split_name: str):
    """Create mapping function for dataset transformation, similar to GSM8k approach."""

    def process_fn(example, idx):
        # Extract user question from messages
        user_content = example["messages"][0]["content"]
        ground_truth = example["ground_truth_result"]

        # Create VERL-compatible format similar to GSM8k
        data = {
            "data_source": "atomwalk12/linalg-debug",
            "prompt": [
                {
                    "role": "system",
                    "content": SYS,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "interaction_kwargs": {
                # This must match the name from linalg_interaction_config.yaml
                "name": "linalg",
                "query": user_content,
                "ground_truth": ground_truth,
            },
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": split_name,
                "index": idx,
                "original_messages": example["messages"],
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "linalg_calculator": {
                        "create_kwargs": {"ground_truth": ground_truth},
                    },
                },
                "interaction_kwargs": {
                    "query": user_content,
                    "ground_truth": ground_truth,
                },
            },
        }
        return data

    return process_fn


def main():
    parser = argparse.ArgumentParser(description="Download HF dataset and convert to parquet")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="atomwalk12/linalg-debug",
        help="HuggingFace dataset name (e.g., 'atomwalk12/linalg-debug-distilled')",
    )
    parser.add_argument(
        "--output_dir", type=str, default="~/data/linalg-zero", help="Output directory for parquet files"
    )

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset_name}")

    # Load the dataset from HuggingFace
    dataset = datasets.load_dataset(args.dataset_name)
    splits = dataset

    if "train" not in splits or "test" not in splits:
        raise ValueError("Dataset must contain train and test splits")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save each split as parquet
    for split_name, split_data in splits.items():
        output_path = os.path.join(args.output_dir, f"{split_name}.parquet")
        print(f"Saving {split_name} split to {output_path}")
        print(f"Split contains {len(split_data)} examples")

        # Apply transformation using single map function like GSM8k
        dataset = split_data.map(function=make_map_fn(split_name), with_indices=True)
        dataset = remove_redundant_columns(dataset, ["extra_info", "reward_model", "prompt", "ability", "data_source"])

        # Display first example for verification
        if len(dataset) > 0:
            print("First example:")
            print(dataset[0])  # type: ignore[reportIndexIssue]
            print()

        dataset.to_parquet(output_path)
        print(f"Saved {output_path}")

    print("Dataset download and conversion complete!")
    print(f"Parquet files saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
