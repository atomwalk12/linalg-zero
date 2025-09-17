import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi

from datasets.dataset_dict import DatasetDict

logger = logging.getLogger(__name__)

LLAMA_CPP_DIR = Path(__file__).parent / "distillation" / "llama-cpp" / "models"


def get_config_dir() -> str:
    """Get the path of the config directory"""
    script_dir = Path(__file__).parent.parent
    return str(script_dir / "config")


def get_log_file_path() -> str:
    """
    Finds and returns the file path of the first FileHandler found in the logger's handlers.
    Raises ValueError if no FileHandler is found.
    """
    logger = logging.getLogger()  # Get root logger
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return handler.baseFilename
    raise ValueError("No FileHandler found in the logger's handlers")


def setup_logging(
    level: int = logging.INFO, include_timestamp: bool = False, file_suffix: str = "linalg_zero.log"
) -> None:  # pragma: no cover
    """
    Set up simple logging configuration. Will log to console and file.

    Args:
        level: Logging level (default: INFO)
        include_timestamp: Whether to include timestamp in logs
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("logs").mkdir(exist_ok=True)

    format_string = "%(asctime)s - %(levelname)s: %(message)s" if include_timestamp else "%(levelname)s: %(message)s"

    logging.basicConfig(
        level=level,
        format=format_string,
        force=True,
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(f"logs/{timestamp}_{file_suffix}")],
    )

    logging.info(f"Logging to {Path('logs') / f'{timestamp}_{file_suffix}'}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name."""
    return logging.getLogger(name)


def get_libpath() -> Path:
    """Returns the path to the library of functions."""
    return Path(__file__).parent / "lib.py"


def get_function_schema() -> str:
    """Return a JSON string with the full tool function schemas (sorted by name)."""
    from distilabel.steps.tasks.apigen.execution_checker import load_module_from_path

    libpath_module = load_module_from_path(get_libpath())
    tools = libpath_module.get_tools()
    # Ensure deterministic order for readability
    tools = sorted(tools, key=lambda t: t["function"]["name"])

    # For prompts, show only the inner function object (cleaner to read than the wrapper)
    extracted_functions = [tool_info["function"] for tool_info in tools]
    return json.dumps(extracted_functions, indent=2)


def push_to_hub(
    dataset: DatasetDict | dict, hub_dataset_name: str, private: bool = False, config_path: str | None = None
) -> None:
    """Push the dataset to Hugging Face Hub, optionally including entropy settings."""

    if isinstance(dataset, dict):
        dataset = DatasetDict(dataset)

    try:
        dataset.push_to_hub(hub_dataset_name, private=private)
        logger.info(f"Successfully pushed dataset to: https://huggingface.co/datasets/{hub_dataset_name}")

        # Upload entropy settings as an additional file if it exists
        if config_path and Path(config_path).exists():
            api = HfApi()
            api.upload_file(
                path_or_fileobj=config_path,
                path_in_repo="entropy_settings.json",
                repo_id=hub_dataset_name,
                repo_type="dataset",
            )
            logger.info(
                f"Successfully uploaded entropy settings to: https://huggingface.co/datasets/{hub_dataset_name}"
            )
        elif config_path:
            logger.warning(f"Warning: Entropy settings file not found at {config_path}")
    except Exception:
        logger.exception("Failed to push dataset to Hugging Face Hub.")
        raise
