"""
See reference script: linalg_zero/grpo/verl/examples/data_preprocess/gsm8k_multiturn_w_tool.py
"""

import argparse
import json
import os
from collections.abc import Callable
from typing import Any

import yaml
from argilla import Dataset
from verl.tools.schemas import OpenAIFunctionToolSchema

import datasets
from linalg_zero.shared.lib import LibTypesList, get_lib
from linalg_zero.shared.system_prompts import MATH_TOOL_PROMPT
from linalg_zero.shared.utils import get_function_schema, push_to_hub


def remove_redundant_columns(dataset: Dataset, required_columns: list[str]) -> Dataset:
    return dataset.remove_columns([col for col in dataset.column_names if col not in required_columns])


def generate_tool_specification() -> None:
    """
    Update the linalg_tool_config.yaml file with current function schema.
    This gives the model information on how to use tools.
    """
    config_path = os.path.join(os.path.dirname(__file__), "config", "linalg_tool_config.yaml")

    # Get current function schema - it returns a JSON string, so parse it
    function_schemas_json = get_function_schema()
    if isinstance(function_schemas_json, str):
        function_schemas = json.loads(function_schemas_json)
    else:
        function_schemas = function_schemas_json

    # Create a new tool configuration
    validated_schemas: dict[str, list] = {"tools": []}
    for func_schema in function_schemas:
        try:
            OpenAIFunctionToolSchema.model_validate(func_schema, strict=True)

            config = {
                "class_name": "linalg_zero.grpo.linalg_zero_tool.LinalgZeroTool",
                "config": {"type": "native"},
                "tool_schema": func_schema,
            }
            validated_schemas["tools"].append(config)
            print(f"✓ Validated function schema: {func_schema['function']['name']}")

        except Exception as e:
            print(f"✗ Schema validation failed for {func_schema.get('function', {}).get('name', 'unknown')}: {e}")
            raise

    # Write updated configuration back to file
    with open(config_path, "w") as f:
        yaml.dump(validated_schemas, f, default_flow_style=False, indent=2)

    print(f"Updated tool configuration at {config_path}")
    print(f"Available functions: {[func['function']['name'] for func in function_schemas]}")


def normalize_dataset_schema(ground_truth: str, stepwise_ground_truths: str) -> tuple[dict, dict]:
    """
    This function ensures that the tool arguments and interaction arguments are normalized.
    This is necessary otherwise validation fails. The ground truth and stepwise ground truths
    are both in JSON format.
    """

    if type(json.loads(ground_truth)) not in LibTypesList:
        raise ValueError(f"Ground truth is not a valid type: {type(ground_truth)}")

    if len(json.loads(stepwise_ground_truths)) == 0:
        raise ValueError("Stepwise ground truth is empty")

    tool_kwargs = {}
    interaction_kwargs = {}
    lib_names = get_lib().keys()

    for lib_name in lib_names:
        tool_kwargs[lib_name] = {
            "create_kwargs": {"stepwise_ground_truth": json.dumps({}), "ground_truth": json.dumps({})}
        }

    stepwise_ground_arr = json.loads(stepwise_ground_truths)
    for stepwise_truth in stepwise_ground_arr:
        for key, value in stepwise_truth.items():
            if type(value) not in LibTypesList:
                raise ValueError(f"Stepwise truth is not a valid type: {type(stepwise_truth)}")

            if key not in lib_names:
                raise ValueError(f"Key {key} not in lib_names")

            value = json.dumps(value)
            tool_kwargs[key]["create_kwargs"].update({"stepwise_ground_truth": value, "ground_truth": ground_truth})

    interaction_kwargs = {
        "stepwise_ground_truths": stepwise_ground_truths,
        "ground_truth": ground_truth,
        "name": "linalg",
    }

    return tool_kwargs, interaction_kwargs


def make_map_fn(split_name: str) -> Callable[[dict[str, Any], int], dict[str, Any]]:
    """Create mapping function for dataset transformation."""

    def process_fn(example: dict[str, Any], idx: int) -> dict[str, Any]:
        # Process the user messages
        user_content = example["messages"][0]["content"]
        messages = []
        messages.append({
            "role": "system",
            "content": MATH_TOOL_PROMPT,
        })
        messages.append({
            "role": "user",
            "content": user_content,
        })

        # Ensure that the kwargs schema is normalized across all examples, otherwise training fails
        stepwise_ground_truths = example["stepwise_ground_truths"]
        ground_truth = example["ground_truth"]

        tool_kwargs, interaction_kwargs = normalize_dataset_schema(ground_truth, stepwise_ground_truths)

        # Generate VERL-compatible data
        data = {
            "data_source": "atomwalk12/linalg-debug",
            "prompt": messages,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": split_name,
                "index": idx,
                "original_messages": example["messages"],
                "need_tools_kwargs": True,
                "tools_kwargs": tool_kwargs,
                "interaction_kwargs": interaction_kwargs,
            },
        }
        return data

    return process_fn


def generate_parquet_files(dataset: Dataset, output_dir: str, args: argparse.Namespace) -> dict[str, Dataset]:
    """Generate parquet files for the dataset."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Store processed datasets for huggingface upload
    processed_datasets = {}

    # Save each split as parquet
    for split_name, split_data in dataset.items():
        output_path = os.path.join(args.output_dir, f"{split_name}.parquet")
        print(f"Saving {split_name} split to {output_path}")
        print(f"Split contains {len(split_data)} examples")

        # Apply transformation using single map function similar to GSM8k example
        dataset = split_data.map(function=make_map_fn(split_name), with_indices=True)
        dataset = remove_redundant_columns(dataset, ["extra_info", "reward_model", "prompt", "ability", "data_source"])

        # Store and save to parquet
        processed_datasets[split_name] = dataset
        dataset.to_parquet(output_path)
        print(f"Saved {output_path}")

    print("Dataset download and conversion complete!")
    print(f"Parquet files saved in: {args.output_dir}")
    return processed_datasets


def main(args: argparse.Namespace) -> None:
    # Generation tool specification file
    generate_tool_specification()

    # Load the base dataset from HuggingFace
    print(f"Loading dataset: {args.dataset_name}")
    dataset = datasets.load_dataset(args.dataset_name)  # type: ignore[attr-defined]

    if "train" not in dataset or "test" not in dataset:
        raise ValueError("Dataset must contain train and test splits")

    dataset = generate_parquet_files(dataset, args.output_dir, args)
    push_to_hub(dataset, args.hub_dataset_name, private=False)

    print(f"   Dataset contains {len(dataset)} splits:")
    for split_name, split_dataset in dataset.items():
        print(f"   - {split_name}: {len(split_dataset)} examples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HF dataset and convert to parquet")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="atomwalk12/linalg-debug",
        help="HuggingFace dataset name (e.g., 'atomwalk12/linalg-debug-distilled')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/data/linalg-zero",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--hub_dataset_name",
        type=str,
        default="atomwalk12/linalg-zero-grpo-training-dataset",
        help="Name for the dataset on Hugging Face Hub",
    )
    args = parser.parse_args()
    main(args)
