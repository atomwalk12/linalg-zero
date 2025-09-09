import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from datasets import DatasetDict
from datasets import load_dataset as hf_load_dataset
from linalg_zero.config.data import ScriptArguments, SFTConfig

logger = logging.getLogger(__name__)


def is_using_deepspeed() -> bool:
    """Check if DeepSpeed is being used via environment variables"""
    return (
        os.environ.get("LOCAL_RANK") is not None
        or os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true"
        or "deepspeed" in os.environ.get("ACCELERATE_CONFIG_FILE", "").lower()
    )


def get_tokenizer(model_args: ModelConfig, training_args: SFTConfig) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def get_model(model_args: ModelConfig, training_args: SFTConfig) -> AutoModelForCausalLM:
    """Get the model"""
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    using_deepspeed = is_using_deepspeed()
    device_map = None
    if quantization_config is not None and not using_deepspeed:
        device_map = get_kbit_device_map()
        logger.info(f"Setting device_map: {device_map}")
    else:
        # Device map is not compatible with quantization and deepspeed ZeRO-3``
        logger.info("Not setting device_map (DeepSpeed detected or no quantization)")

    model_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation,
        "torch_dtype": torch_dtype,
        "use_cache": not training_args.gradient_checkpointing,
        "device_map": device_map,
        "quantization_config": quantization_config,
    }
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    return model


def load_dataset(args: ScriptArguments) -> DatasetDict:
    """Load the dataset produced during the distillation step, removing unnecessary columns for SFT."""

    def remove_redundant_columns(dataset: DatasetDict) -> DatasetDict:
        """Remove columns from a dataset."""
        if dataset.column_names:
            splits = dict(dataset.column_names.items())

            # Remove any redundant columns not using during SFT training. Only 'tools' and 'messages' are relevant.
            dataset = dataset.remove_columns([
                col
                for split in splits.values()
                if split is not None
                for col in split
                if col not in ["tools", "messages"]
            ])
        return dataset

    dataset = hf_load_dataset(args.dataset_name, args.dataset_config)

    if args.take_n is not None:
        dataset = dataset.select(range(args.take_n))

    # Only the ["messages", "tools"] columns are relevant for SFT
    return remove_redundant_columns(dataset)
