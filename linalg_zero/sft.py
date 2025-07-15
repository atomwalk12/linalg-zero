import logging

from trl import TrlParser

from linalg_zero.config.data import DatasetGenerationConfig
from linalg_zero.shared import get_logger, setup_logging

if __name__ == "__main__":  # pragma: no cover
    """Script used to fine-tune a base model."""

    # Set up logging
    setup_logging(level=logging.INFO, include_timestamp=True)
    logger = get_logger(__name__)

    trl = TrlParser(DatasetGenerationConfig)  # type: ignore[reportArgumentType]
    config: DatasetGenerationConfig = trl.parse_args_and_config()[0]
    logger.info("Configuration: %s", config)
