import logging

from linalg_zero.shared import get_logger, setup_logging

if __name__ == "__main__":  # pragma: no cover
    """Script used to fine-tune a base model."""
    from trl import TrlParser

    from linalg_zero.config.configs import DatasetGenerationConfig

    # Set up logging
    setup_logging(level=logging.INFO, include_timestamp=True)
    logger = get_logger(__name__)

    trl = TrlParser(DatasetGenerationConfig)
    config: DatasetGenerationConfig = trl.parse_args_and_config()[0]
    logger.info("Configuration: %s", config)
