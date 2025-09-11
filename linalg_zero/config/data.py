from dataclasses import dataclass, field

import trl


@dataclass
class ScriptArguments(trl.ScriptArguments):
    """
    Extended version of ScriptArguments with support for dataset mixtures.
    """

    dataset_name: str | None = field(
        default=None, metadata={"help": "Training dataset name. Contains chain-of-thought solutions."}
    )
    eval_dataset_name: str | None = field(
        default=None, metadata={"help": "Evaluation dataset name. Contains ground-truth solutions only."}
    )
    eval_dataset_config: str | None = field(default=None, metadata={"help": "Evaluation dataset config."})

    take_n: int | None = field(default=None, metadata={"help": "Number of examples to take from the dataset."})


@dataclass
class SFTConfig(trl.SFTConfig):
    early_stopping_patience: int = field(
        default=3, metadata={"help": "The number of epochs to wait before early stopping."}
    )
    early_stopping_threshold: float = field(
        default=0.0, metadata={"help": "Minimum improvement required to reset patience counter."}
    )

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    chat_template: str | None = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: str | None = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: str | None = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: str | None = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: str | None = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: str | None = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )

    eval_max_new_tokens: int | None = field(
        default=None,
        metadata={"help": "Max new tokens for evaluation callbacks (does not affect training)."},
    )


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
    chat_template: str | None = field(
        default=None,
        metadata={"help": "Chat template to use"},
    )

    # Memory / performance tuning parameters
    dtype: str | None = field(
        default=None,
        metadata={"help": "Computation dtype for model weights and activations (e.g., float16)"},
    )
    kv_cache_dtype: str | None = field(
        default=None,
        metadata={"help": "KV cache dtype (auto, fp8, fp8_e4m3, fp8_e5m2)"},
    )
    max_model_len: int | None = field(
        default=None,
        metadata={"help": "Maximum model context length (tokens)"},
    )
    max_num_seqs: int | None = field(
        default=None,
        metadata={"help": "Maximum number of concurrent sequences"},
    )
    gpu_memory_utilization: float | None = field(
        default=None,
        metadata={"help": "Fraction of GPU memory to be used by vLLM (0-1)"},
    )
    enforce_eager: bool | None = field(
        default=None,
        metadata={"help": "Disable CUDA graphs to reduce memory usage"},
    )
    swap_space: int | None = field(
        default=None,
        metadata={"help": "CPU swap space in GB per GPU for paging KV cache"},
    )
    max_num_batched_tokens: int | None = field(
        default=None,
        metadata={"help": "Limit number of tokens processed per batch (prefill)"},
    )
    tensor_parallel_size: int | None = field(
        default=None,
        metadata={"help": "Tensor parallelism degree"},
    )
    enable_chunked_prefill: bool | None = field(
        default=None,
        metadata={"help": "Enable chunked prefill to reduce peak prefill memory"},
    )
    reasoning_parser: str | None = field(
        default=None,
        metadata={"help": "Reasoning parser to use"},
    )


@dataclass
class DistillationConfig:
    """
    Data class that stores the distillation pipeline parameters.
    """

    # Dataset parameters
    dataset_name: str | None = field(
        metadata={"help": "HuggingFace dataset to load"},
    )
    dataset_config: str | None = field(
        metadata={"help": "Dataset config to use"},
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

    # Generation parameters
    n_turns: int = field(
        metadata={"help": "Number of turns to generate"},
    )

    # Optional stopping sequences (must come after non-default fields)
    stop: list[str] | None = field(
        default=None,
        metadata={"help": "Stop sequences for generation (each string is a stop token)"},
    )

    do_eval: bool = field(
        default=False,
        metadata={"help": "Whether to do evaluation"},
    )

    take_n: int | None = field(
        default=None,
        metadata={"help": "Number of examples to take from the dataset."},
    )
