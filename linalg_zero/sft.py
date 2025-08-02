import logging
import os
import sys

import transformers
from transformers.trainer_utils import get_last_checkpoint, set_seed
from trl import ModelConfig, SFTTrainer, TrlParser, get_peft_config, setup_chat_format

import datasets
from linalg_zero.config.data import ScriptArguments, SFTConfig
from linalg_zero.sft.callbacks import get_callbacks
from linalg_zero.sft.model_utils import get_model, get_tokenizer, load_dataset
from linalg_zero.sft.wandb_logging import init_wandb_training
from linalg_zero.shared import get_logger, setup_logging


def main(script_args: ScriptArguments, training_args: SFTConfig, model_args: ModelConfig) -> None:  # noqa: C901
    """Main training function."""
    set_seed(training_args.seed)

    #################
    # Setup logging #
    #################
    # Log both to file and console
    setup_logging(level=logging.INFO, include_timestamp=True)
    logger = get_logger(__name__)

    # Adjust script logging level based on the node logging level (main process or replica)
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Script parameters: {script_args}")
    logger.info(f"Training parameters: {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}")

    # Initialize wandb if requested
    if training_args.report_to and "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ######################################
    # Load dataset, tokenizer, and model #
    ######################################
    logger.info(f"Loading dataset from {script_args.dataset_name}...")
    dataset = load_dataset(script_args)

    assert isinstance(dataset, datasets.DatasetDict)  # noqa: S101

    if training_args.do_eval:
        # TODO: this is a workaround since we are currently missing the evaluation dataset
        logger.info(f"Loading eval dataset from {script_args.dataset_name}...")
        dataset["test"] = dataset["train"]

    logger.info("Loading tokenizer...")
    tokenizer = get_tokenizer(model_args, training_args)

    logger.info("Loading model...")
    model = get_model(model_args, training_args)

    # Setup chat format if no template provided
    if tokenizer.chat_template is None:
        logger.warning("No chat template provided, defaulting to ChatML for tool use compatibility.")
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    ##############################
    # Initialize the SFT Trainer #
    ##############################
    logger.info("Initializing SFT Trainer...")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
    )

    #################
    # Training loop #
    #################
    logger.info("*** Starting Training ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    try:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        logger.info("Training completed successfully!")

    except Exception:
        logger.exception("Training failed with error")
        raise

    ####################################
    # Save model and create model card #
    ####################################
    logger.info("*** Saving Model ***")
    try:
        # Align the model's generation config with the tokenizer's eos token
        # to avoid unbounded generation in the transformers `pipeline()` function
        if trainer.model is not None and trainer.model.generation_config is not None:
            trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

        # Save everything else on main process
        kwargs = {
            "dataset_name": script_args.dataset_name,
            "tags": ["linalg-zero", "sft", "tool-use"],
        }
        if trainer.accelerator.is_main_process:
            trainer.create_model_card(**kwargs)
            # Restore k,v cache for fast inference
            if trainer.model is not None:
                trainer.model.config.use_cache = True
                trainer.model.config.save_pretrained(training_args.output_dir)  # type: ignore[reportCallIssue]

    except Exception:
        logger.exception("Failed to save model")
        raise

    ############
    # Evaluate #
    ############
    if training_args.do_eval:
        logger.info("*** Evaluation ***")
        try:
            metrics = trainer.evaluate()
            metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            logger.info("Evaluation completed successfully!")

        except Exception:
            logger.exception("Evaluation failed")

    ###############
    # Push to hub #
    ###############
    if training_args.push_to_hub:
        logger.info("*** Pushing to Hub ***")
        try:
            trainer.push_to_hub(**kwargs)
            logger.info("Successfully pushed model to HuggingFace Hub!")
        except Exception:
            logger.exception("Failed to push to hub")


if __name__ == "__main__":
    """Script entry point for SFT training."""
    if "--config" not in sys.argv:
        sys.argv.append("--config")
        sys.argv.append("linalg_zero/config/sft/sft_debug_config.yaml")

    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    main(script_args, training_args, model_args)
