from dataclasses import dataclass, field


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

    # Model parameters
    model: str = field(
        metadata={"help": "Model URL to download (GGUF format)"},
    )

    n_gpu_layers: int = field(
        metadata={"help": "Number of GPU layers to offload"},
    )

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
    private: bool = field(
        metadata={"help": "Whether to make the output dataset private when pushing to HF Hub"},
    )
