from dataclasses import dataclass, field


@dataclass
class DatasetGenerationConfig:
    """
    Data class that stores the dataset generation parameters.

    Args:
        dataset_name (str): The name of the dataset to generate.
    """

    dataset_name: str | None = field(
        default="linalg_zero",
        metadata={"help": "Should be the name used to store the dataset on the Hugging Face Hub."},
    )


@dataclass
class DistillationConfig:
    """
    Data class that stores the distillation pipeline parameters.
    """

    # Dataset parameters
    hf_dataset: str | None = field(
        default=None,
        metadata={"help": "HuggingFace dataset to load"},
    )
    hf_dataset_config: str | None = field(
        default=None,
        metadata={"help": "Dataset config to use"},
    )
    hf_dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use"},
    )

    # Prompt parameters
    prompt_column: str = field(
        default="prompt",
        metadata={"help": "Column name for prompt data"},
    )
    prompt_template: str = field(
        default="{{ instruction }}",
        metadata={"help": "Template string for formatting prompts"},
    )

    # Model parameters
    model: str = field(
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        metadata={"help": "Model name to use for generation"},
    )
    vllm_server_url: str = field(
        default="http://localhost:8000/v1",
        metadata={"help": "URL of the vLLM server"},
    )

    # Generation parameters
    temperature: float | None = field(
        default=None,
        metadata={"help": "Temperature for generation"},
    )
    top_p: float | None = field(
        default=None,
        metadata={"help": "Top-p value for generation"},
    )
    max_new_tokens: int = field(
        default=8192,
        metadata={"help": "Maximum number of new tokens to generate"},
    )
    num_generations: int = field(
        default=1,
        metadata={"help": "Number of generations per problem"},
    )

    # Processing parameters
    input_batch_size: int = field(
        default=64,
        metadata={"help": "Batch size for input processing"},
    )
    client_replicas: int = field(
        default=1,
        metadata={"help": "Number of client replicas for parallel processing"},
    )
    timeout: int = field(
        default=600,
        metadata={"help": "Request timeout in seconds"},
    )
    retries: int = field(
        default=0,
        metadata={"help": "Number of retries for failed requests"},
    )

    # Output parameters
    hf_output_dataset: str | None = field(
        default=None,
        metadata={"help": "HuggingFace repo to push results to"},
    )
    private: bool = field(
        default=False,
        metadata={"help": "Whether to make the output dataset private when pushing to HF Hub"},
    )
