from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """Configuration for a single dataset in a mixture."""

    id: str = field(metadata={"help": "Dataset ID on HuggingFace Hub"})
    config: str | None = field(default=None, metadata={"help": "Dataset configuration name"})
    split: str = field(default="train", metadata={"help": "Dataset split to use"})
    columns: list[str] | None = field(default=None, metadata={"help": "Columns to select from dataset"})
    weight: float | None = field(default=None, metadata={"help": "Weight for sampling in mixture"})


@dataclass
class DatasetMixture:
    """Configuration for dataset mixture."""

    datasets: list[DatasetConfig] = field(metadata={"help": "List of datasets to mix"})
    seed: int = field(default=42, metadata={"help": "Random seed for shuffling"})
    test_split_size: float | None = field(default=None, metadata={"help": "Size of test split"})


@dataclass
class ScriptArguments:
    """Script arguments for SFT training."""

    # Dataset configuration
    dataset_name: str | None = field(default=None, metadata={"help": "Name of the dataset to use for training"})
    dataset_config: str | None = field(default=None, metadata={"help": "Configuration name of the dataset"})
    dataset_mixture: DatasetMixture | None = field(default=None, metadata={"help": "Dataset mixture configuration"})
    dataset_train_split: str = field(default="train", metadata={"help": "Dataset split to use for training"})
    dataset_test_split: str = field(default="test", metadata={"help": "Dataset split to use for testing"})

    # Training configuration
    max_seq_length: int = field(default=2048, metadata={"help": "Maximum sequence length for tokenization"})

    # Chat template
    chat_template: str | None = field(
        default=None, metadata={"help": "Chat template to use for formatting conversations"}
    )


@dataclass
class SFTConfig:
    """Configuration for Supervised Fine-Tuning."""

    # Basic training parameters
    output_dir: str = field(default="./results", metadata={"help": "Output directory for model and checkpoints"})
    per_device_train_batch_size: int = field(default=4, metadata={"help": "Batch size per device during training"})
    per_device_eval_batch_size: int = field(default=4, metadata={"help": "Batch size per device during evaluation"})
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of updates steps to accumulate gradients"}
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "Initial learning rate"})
    num_train_epochs: int = field(default=3, metadata={"help": "Total number of training epochs"})
    max_steps: int = field(
        default=-1, metadata={"help": "Total number of training steps (-1 means use num_train_epochs)"}
    )

    # Optimization parameters
    warmup_ratio: float = field(default=0.1, metadata={"help": "Ratio of total training steps for warmup"})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay for AdamW optimizer"})
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Enable gradient checkpointing to save memory"}
    )

    # Precision and hardware
    bf16: bool = field(default=False, metadata={"help": "Use bfloat16 precision"})
    fp16: bool = field(default=False, metadata={"help": "Use float16 precision"})

    # Evaluation and logging
    eval_strategy: str = field(default="steps", metadata={"help": "Evaluation strategy ('no', 'steps', 'epoch')"})
    eval_steps: int = field(default=500, metadata={"help": "Number of steps between evaluations"})
    logging_steps: int = field(default=10, metadata={"help": "Number of steps between logging"})
    save_steps: int = field(default=500, metadata={"help": "Number of steps between checkpoints"})
    save_total_limit: int = field(default=3, metadata={"help": "Maximum number of checkpoints to keep"})

    # Hub and reporting
    push_to_hub: bool = field(default=False, metadata={"help": "Push model to HuggingFace Hub after training"})
    hub_model_id: str | None = field(default=None, metadata={"help": "HuggingFace Hub model ID"})
    report_to: str = field(default="none", metadata={"help": "Reporting tool ('wandb', 'tensorboard', 'none')"})

    # Other
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})
    resume_from_checkpoint: str | None = field(default=None, metadata={"help": "Path to checkpoint to resume from"})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run evaluation"})

    # Chat template
    chat_template: str | None = field(
        default=None, metadata={"help": "Chat template to use for formatting conversations"}
    )

    # Callbacks
    callbacks: list[str] = field(default_factory=list, metadata={"help": "List of callback names to use"})

    def get_process_log_level(self) -> int:
        """Get the log level for the current process."""
        import logging

        return logging.INFO


@dataclass
class DatasetGenerationConfig:
    """
    Data class that stores the dataset generation parameters.

    Args:
        dataset_name (str): The name of the dataset to generate.
    """

    dataset_name: str | None = field(
        metadata={"help": "Should be the name used to store the dataset on the Hugging Face Hub."},
    )


@dataclass
class LlamaCppServerConfig:
    """
    Data class that stores LlamaCPP server parameters with llama_cpp_ prefix.
    """

    def __post_init__(self) -> None:
        pass

    # Server parameters
    host: str = field(
        metadata={"help": "Host address to bind to"},
    )
    port: int = field(
        metadata={"help": "Port to listen on"},
    )
    n_ctx: int = field(
        metadata={"help": "Context size"},
    )
    split_mode: int = field(
        metadata={"help": "Split mode (0=none, 1=layer, 2=row)"},
    )

    # Model parameters
    n_gpu_layers: int = field(
        metadata={"help": "Number of GPU layers to offload"},
    )

    model: str = field(
        metadata={"help": "Model URL to download (GGUF format)"},
    )

    hf_pretrained_model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Huggingface repository ID to ensure that the correct tokenizer is used."},
    )

    hf_model_repo_id: str | None = field(
        default=None,
        metadata={"help": "Path to the repository where the model is stored."},
    )


@dataclass
class VllmServerConfig:
    """
    Data class that stores vLLM server parameters with vllm_ prefix.
    """

    # Model parameters
    model: str = field(
        metadata={"help": "Model name (HuggingFace format)"},
    )
    quantization: str | None = field(
        metadata={"help": "Quantization method (awq, gptq, etc.)"},
    )

    # Server parameters
    host: str = field(
        metadata={"help": "Host address to bind to"},
    )
    port: int = field(
        metadata={"help": "Port to listen on"},
    )
    enable_auto_tool_choice: bool = field(
        metadata={"help": "Enable automatic tool choice"},
    )
    tool_call_parser: str = field(
        metadata={"help": "Tool call parser to use"},
    )
    chat_template: str = field(
        metadata={"help": "Chat template to use"},
    )


@dataclass
class DistillationConfig:
    """
    Data class that stores the distillation pipeline parameters.
    """

    # Dataset parameters
    hf_dataset: str | None = field(
        metadata={"help": "HuggingFace dataset to load"},
    )
    hf_dataset_config: str | None = field(
        metadata={"help": "Dataset config to use"},
    )
    hf_dataset_split: str = field(
        metadata={"help": "Dataset split to use"},
    )

    # Prompt parameters
    prompt_column: str = field(
        metadata={"help": "Column name for prompt data"},
    )
    prompt_template: str = field(
        metadata={"help": "Template string for formatting prompts"},
    )

    # Generation parameters
    temperature: float | None = field(
        metadata={"help": "Temperature for generation"},
    )
    top_p: float | None = field(
        metadata={"help": "Top-p value for generation"},
    )
    max_new_tokens: int = field(
        metadata={"help": "Maximum number of new tokens to generate"},
    )
    num_generations: int = field(
        metadata={"help": "Number of generations per problem"},
    )

    # Processing parameters
    input_batch_size: int = field(
        metadata={"help": "Batch size for input processing"},
    )
    use_cache: bool = field(
        metadata={"help": "Whether to use cache for the pipeline. This can enable error recovery."},
    )
    client_replicas: int = field(
        metadata={"help": "Number of client replicas for parallel processing"},
    )
    timeout: int = field(
        metadata={"help": "Request timeout in seconds"},
    )
    retries: int = field(
        metadata={"help": "Number of retries for failed requests"},
    )

    # Output parameters
    hf_output_dataset: str | None = field(
        metadata={"help": "HuggingFace repo to push results to"},
    )
    argilla_output_dataset: str | None = field(
        metadata={"help": "Argilla dataset to push results to. This is used for manual annotation."},
    )
    private: bool = field(
        metadata={"help": "Whether to make the output dataset private when pushing to HF Hub"},
    )
