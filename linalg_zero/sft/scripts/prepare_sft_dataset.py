import json
import sys
from typing import (
    Any,
)

from datasets import Dataset, DatasetDict
from trl import ModelConfig, TrlParser

from linalg_zero.config.data import (
    ScriptArguments,
    SFTRunConfig,
)
from linalg_zero.distillation.utils import load_dataset_split
from linalg_zero.grpo.process_dataset import remove_redundant_columns
from linalg_zero.shared.lib import get_tools
from linalg_zero.shared.system_prompts import get_math_system_prompt
from linalg_zero.shared.utils import get_logger

logger = get_logger(__name__)


def process_dataset_for_sft(dataset: Dataset) -> Dataset:
    """Process a dataset for SFT training by keeping only required columns and parsing messages."""
    # Preserve minimal columns needed for SFT + optional correctness metrics
    # "messages" is required; "tools" helps validate tool names; ground truth fields enable answer correctness.
    keep_columns = [
        "tools",
        "messages",
        "ground_truth",
        "stepwise_ground_truths",
    ]
    dataset = remove_redundant_columns(dataset, keep_columns)
    # if "messages" in dataset.column_names:
    #    dataset = dataset.map(lambda x: {"messages": json.loads(x["messages"])})
    assert isinstance(dataset, Dataset)  # noqa: S101
    return dataset


def replace_system_prompt(dataset: Dataset) -> Dataset:
    """Replace the system prompt in all messages with the current version."""
    system_prompt = get_math_system_prompt()

    def update_system_prompt(example: dict[str, Any]) -> dict[str, Any]:
        if "messages" in example:
            # Parse messages if they're JSON strings
            messages = json.loads(example["messages"]) if isinstance(example["messages"], str) else example["messages"]

            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = system_prompt
            example["messages"] = messages
        return example

    return dataset.map(update_system_prompt)


def add_missing_fields_for_eval(dataset: Dataset) -> Dataset:
    """Add missing tools and messages fields to evaluation dataset from query field."""

    def add_fields(example: dict[str, Any]) -> dict[str, Any]:
        # Add tools if missing
        if "tools" not in example:
            example["tools"] = get_tools()

        # Add messages if missing, build from query field
        if "messages" not in example and "query" in example:
            # Create messages with the full schema including optional tool calling fields
            # to match the training dataset schema
            # Note: tool_calls should be None (not empty list) to match training data format
            example["messages"] = [
                {
                    "role": "system",
                    "content": get_math_system_prompt(),
                    "name": None,
                    "tool_call_id": None,
                    "tool_calls": None,
                },
                {
                    "role": "user",
                    "content": example["query"],
                    "name": None,
                    "tool_call_id": None,
                    "tool_calls": None,
                },
            ]

        return example

    return dataset.map(add_fields)


def load_datasets_for_sft(args: ScriptArguments, do_eval: bool = True) -> DatasetDict:
    """Loads train and optionally validation splits from separate datasets."""

    # Load training dataset
    if args.dataset_name is None:
        raise ValueError("dataset_name must be provided")

    train_dataset = load_dataset_split(args.dataset_name, args.dataset_config, "train", args.take_n)
    train_dataset = replace_system_prompt(train_dataset)
    train_dataset = process_dataset_for_sft(train_dataset)

    dataset_dict = {"train": train_dataset}

    if do_eval:
        # Load evaluation dataset from separate dataset if specified
        eval_dataset_name = args.eval_dataset_name
        eval_dataset_config = args.eval_dataset_config

        if eval_dataset_name is None or eval_dataset_config is None:
            raise ValueError("eval_dataset_name and eval_dataset_config must be provided when do_eval=True")

        eval_dataset = load_dataset_split(eval_dataset_name, eval_dataset_config, "validation", args.take_n)
        eval_dataset = add_missing_fields_for_eval(eval_dataset)
        eval_dataset = replace_system_prompt(eval_dataset)
        eval_dataset = process_dataset_for_sft(eval_dataset)

        # Cast eval dataset to match train dataset schema exactly
        eval_dataset = eval_dataset.cast(train_dataset.features)

        dataset_dict["test"] = eval_dataset

    return DatasetDict(dataset_dict)


if "--config" not in sys.argv:
    sys.argv.append("--config")
    sys.argv.append("./config/sft/sft_debug_config.yaml")

parser = TrlParser([ScriptArguments, SFTRunConfig, ModelConfig])
script_args, training_args, model_args = parser.parse_args_and_config()

dataset = load_datasets_for_sft(script_args)

hf_dataset_name = "atomwalk12/linalgzero-sft"
dataset.push_to_hub(hf_dataset_name)
print(f"✅ Successfully uploaded entropy settings to: https://huggingface.co/datasets/{hf_dataset_name}")
