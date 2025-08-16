import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from distilabel.steps.tasks.apigen.execution_checker import load_module_from_path

LLAMA_CPP_DIR = Path(__file__).parent / "distillation" / "llama-cpp" / "models"


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


def get_function_schema(descriptions_only: bool = False) -> str:
    """Return a string representation of the tool schema. This can be a short list of descriptions or a full schema."""
    libpath_module = load_module_from_path(Path(__file__).parent / "shared" / "lib.py")
    tools = libpath_module.get_tools()

    if descriptions_only:
        # Return only the descriptions
        return "\n".join(
            f'"{tool_info["function"]["name"]}": {tool_info["function"]["description"]}' for tool_info in tools
        )

    return json.dumps(tools, indent=2)
