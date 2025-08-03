import logging
import os

from linalg_zero.config.data import SFTConfig

logger = logging.getLogger(__name__)


def init_wandb_training(training_args: SFTConfig) -> None:
    """Initialize Weights & Biases for training logging."""
    try:
        # Set environment variables for wandb
        if training_args.wandb_entity is not None:
            os.environ["WANDB_ENTITY"] = training_args.wandb_entity
        if training_args.wandb_project is not None:
            os.environ["WANDB_PROJECT"] = training_args.wandb_project
        if training_args.wandb_run_group is not None:
            os.environ["WANDB_RUN_GROUP"] = training_args.wandb_run_group

        logger.info("Set wandb environment variables from training args")

    except Exception:
        logger.exception("Failed to initialize wandb environment")
