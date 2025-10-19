import logging
import os
from pathlib import Path

import torch
from datasets import DatasetDict
from datasets import load_dataset as hf_load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from trl.trainer.model_config import ModelConfig
from trl.trainer.utils import get_kbit_device_map, get_quantization_config
from unsloth import FastLanguageModel
from unsloth.tokenizer_utils import SFTConfig

from linalg_zero.config.data import ScriptArguments, SFTModelConfig, SFTRunConfig

logger = logging.getLogger(__name__)


def is_using_deepspeed() -> bool:
    """Check if DeepSpeed is being used via environment variables"""
    return (
        os.environ.get("LOCAL_RANK") is not None
        or os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true"
        or "deepspeed" in os.environ.get("ACCELERATE_CONFIG_FILE", "").lower()
    )


def init_wandb_training(training_args: SFTRunConfig) -> None:
    """Initialize Weights & Biases for training logging."""
    try:
        # Set environment variables for wandb
        if training_args.wandb_entity is not None:
            os.environ["WANDB_ENTITY"] = training_args.wandb_entity
        if training_args.wandb_project is not None:
            os.environ["WANDB_PROJECT"] = training_args.wandb_project
        if training_args.wandb_run_group is not None:
            os.environ["WANDB_RUN_GROUP"] = training_args.wandb_run_group

        logger.info("Set wandb environment variables from training args")

    except Exception:
        logger.exception("Failed to initialize wandb environment")


def get_tokenizer(model_args: ModelConfig, training_args: SFTRunConfig) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def get_unsloth_model(
    model_args: SFTModelConfig, training_args: SFTRunConfig, trl_training_args: SFTConfig
) -> tuple[FastLanguageModel, PreTrainedTokenizer]:
    """Fetch the model and optimizer."""

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        max_seq_length=training_args.max_seq_length,
        load_in_4bit=model_args.load_in_4bit,
        load_in_8bit=model_args.load_in_8bit,
        max_lora_rank=model_args.lora_r,
        # enforce_eager=True,
        # fast_inference=True,
        # gpu_memory_utilization=training_args.gpu_memory_utilization,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=model_args.lora_r,
        target_modules=model_args.lora_target_modules,
        lora_alpha=model_args.lora_alpha,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    if trl_training_args.chat_template_path is not None:
        template_path = Path(trl_training_args.chat_template_path)
        tokenizer.chat_template = template_path.read_text()

    return model, tokenizer


def get_model(model_args: ModelConfig, training_args: SFTRunConfig) -> AutoModelForCausalLM:
    """Get the model"""
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)  # type: ignore[arg-type]
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
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(  # type: ignore[assignment]
        model_args.model_name_or_path,  # type: ignore[arg-type]
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
