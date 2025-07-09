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
