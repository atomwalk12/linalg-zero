import os

os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
import unsloth  # noqa: I001, F401
import logging
import sys
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
import transformers
from datasets import DatasetDict, load_dataset
from datasets.utils.logging import set_verbosity
from transformers.trainer_utils import get_last_checkpoint, set_seed
from trl.scripts.utils import TrlParser
from trl.trainer.sft_config import SFTConfig
from linalg_zero.shared.lib import get_tools
from linalg_zero.config.data import ScriptArguments, SFTModelConfig, SFTRunConfig
from linalg_zero.sft.utils import get_unsloth_model, init_wandb_training
from linalg_zero.shared.utils import get_logger, setup_logging

# Toggle between datasets
USE_MATH_DATASET = True  # Set to False to use linalg_zero dataset


def load_math_dataset(tokenizer, logger):
    """Load open-r1/DAPO-Math-17k-Processed dataset exactly as in working implementation"""

    import numpy as np

    logger.info("Loading dataset from open-r1/DAPO-Math-17k-Processed...")
    dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")

    def extract_hash_answer(text):
        # if "####" not in text: return None
        # return text.split("####")[1].strip()
        return text

    reasoning_start = "<start_working_out>"  # Acts as <think>
    reasoning_end = "<end_working_out>"  # Acts as </think>
    solution_start = "<SOLUTION>"
    solution_end = "</SOLUTION>"

    # Get system prompt
    system_prompt = f"""You are given a problem.
    Think about the problem and provide your working out.
    Place it between {reasoning_start} and {reasoning_end}.
    Then, provide your solution between {solution_start}{solution_end}"""

    dataset = dataset.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x["prompt"]},
            ],
            "answer": extract_hash_answer(x["solution"]),
        },
        remove_columns=dataset.column_names,
    )

    # Filter by length (90th percentile)
    tokenized = dataset.map(
        lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
        batched=True,
    )
    tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})

    maximum_length = int(np.quantile(tokenized["L"], 0.9))
    print(f"Maximum prompt length (90th percentile): {maximum_length}")
    logger.info(f"Maximum prompt length (90th percentile): {maximum_length}")

    # Filter only samples smaller than 90% max length
    dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
    del tokenized

    return dataset, maximum_length


def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!

        # No need to reward <start_working_out> since we always prepend it!
        # score += 0.5 if response.count(reasoning_start) == 1 else -1.0
        score += 0.5 if response.count("</think>") == 1 else -1.0
        score += 0.5 if response.count("<answer>") == 1 else -1.0
        score += 0.5 if response.count("</answer>") == 1 else -1.0
        scores.append(score)
    return scores


def main(  # noqa: C901
    script_args: ScriptArguments, training_args: SFTRunConfig, trl_training_args: SFTConfig, model_args: SFTModelConfig
) -> None:
    """Main training function."""
    # Reproducibility
    set_seed(trl_training_args.seed)

    #################
    # Setup logging #
    #################
    # Log both to file and console
    setup_logging(level=logging.INFO, include_timestamp=True)
    logger = get_logger(__name__)

    # Adjust script logging level based on the node logging level (main process or replica)
    log_level = trl_training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Script parameters: {script_args}")
    logger.info(f"Training parameters: {training_args}")
    logger.info(f"TRL training parameters: {trl_training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if trl_training_args.output_dir and os.path.isdir(trl_training_args.output_dir):
        last_checkpoint = get_last_checkpoint(trl_training_args.output_dir)
    if last_checkpoint is not None and trl_training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}")

    # Initialize wandb if requested
    if trl_training_args.report_to and "wandb" in trl_training_args.report_to:
        init_wandb_training(training_args)

    ######################################
    # Load dataset, tokenizer, and model #
    ######################################
    # Model, tokenizer, dataset
    logger.info("Loading model and tokenizer...")
    model, tokenizer = get_unsloth_model(model_args, training_args, trl_training_args, use_vllm=True)
    tokenizer.tools = get_tools()

    # Load dataset based on flag
    if USE_MATH_DATASET:
        dataset, maximum_length = load_math_dataset(tokenizer, logger)
        # Calculate lengths exactly as in working implementation
        max_seq_length = 2048  # Same as working script
        max_prompt_length = maximum_length + 1  # + 1 just in case!
        max_completion_length = max_seq_length - max_prompt_length
        logger.info(
            f"Using max_seq_length={max_seq_length}, max_prompt_length={max_prompt_length}, max_completion_length={max_completion_length}"
        )
    else:
        logger.info(f"Loading dataset from {script_args.dataset_name}...")
        dataset = load_dataset(script_args.dataset_name, script_args.dataset_config)

        if not isinstance(dataset, DatasetDict):
            raise TypeError(f"Expected dataset to be a DatasetDict, but got {type(dataset)}")

        # Use original lengths for linalg_zero dataset
        max_prompt_length = 4096
        max_completion_length = 2048

    ##############################
    # Initialize the SFT Trainer #
    ##############################
    logger.info("Initializing SFT Trainer...")

    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=3407,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    training_args = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=1.0,
        learning_rate=5e-6,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,  # Increase to 4 for smoother training
        num_generations=4,  # Decrease if out of memory
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps=100,
        save_steps=100,
        report_to="none",  # Can use Weights & Biases
        output_dir="outputs",
        # For optional training + evaluation
        # fp16_full_eval = True,
        # per_device_eval_batch_size = 4,
        # eval_accumulation_steps = 1,
        # eval_strategy = "steps",
        # eval_steps = 1,
    )

    # For optional training + evaluation
    # new_dataset = dataset.train_test_split(test_size = 0.01)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[match_format_approximately],
        args=training_args,
        train_dataset=dataset["train"] if not USE_MATH_DATASET else dataset,
        # For optional training + evaluation
        # train_dataset = new_dataset["train"],
        # eval_dataset = new_dataset["test"],
    )
    trainer.train()

    #################
    # Training loop #
    #################
    logger.info("*** Starting Training ***")
    checkpoint = None
    if trl_training_args.resume_from_checkpoint is not None:
        checkpoint = trl_training_args.resume_from_checkpoint
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

    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception:
        logger.exception("Training failed with an unexpected error")
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
        trainer.save_model(trl_training_args.output_dir)
        logger.info(f"Model saved to {trl_training_args.output_dir}")

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
                trainer.model.config.save_pretrained(trl_training_args.output_dir)

    except Exception:
        logger.exception("Failed to save model")
        raise

    ############
    # Evaluate #
    ############
    if trl_training_args.do_eval:
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
    if trl_training_args.push_to_hub:
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
        sys.argv.append("linalg_zero/config/sft/qwen3-4b-base/grpo_debug_config.yaml")

    parser = TrlParser([ScriptArguments, SFTRunConfig, SFTConfig, SFTModelConfig])
    script_args, training_args, trl_training_args, model_args = parser.parse_args_and_config()

    main(script_args, training_args, trl_training_args, model_args)
