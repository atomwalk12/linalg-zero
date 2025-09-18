import datasets
from linalg_zero.config.data import ScriptArguments
from linalg_zero.distillation.utils import load_datasets_for_dpo


def load_dataset_for_dpo(script_args: ScriptArguments) -> datasets.DatasetDict:
    """Load the dataset for DPO."""
    return load_datasets_for_dpo(script_args)
