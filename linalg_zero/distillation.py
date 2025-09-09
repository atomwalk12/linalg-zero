import logging
import os
import sys
from sys import argv

import argilla as rg
from distilabel.distiset import Distiset
from distilabel.pipeline import Pipeline
from trl import TrlParser

from linalg_zero.config.data import DistillationConfig, LlamaCppServerConfig, VllmServerConfig
from linalg_zero.distillation.components.multi_turn_generation import MultiTurnWithToolUseGenerator
from linalg_zero.distillation.data import ThoughtSchema
from linalg_zero.distillation.utils import (
    cleanup,
    create_argilla_dataset,
    create_llm_clients,
    load_datasets,
    push_to_huggingface,
)
from linalg_zero.shared.lib import get_lib
from linalg_zero.shared.system_prompts import get_math_system_prompt
from linalg_zero.shared.utils import get_logger, setup_logging


def main(args: DistillationConfig, server: LlamaCppServerConfig | VllmServerConfig, take_n: int | None) -> None:  # noqa: C901
    ################################
    # Initialize and load datasets #
    ################################
    enable_thinking = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}

    # Setup the logging and environment variables
    setup_logging(level=logging.INFO, include_timestamp=True)
    logger = get_logger(__name__)

    logger.info("Running with configuration:")
    for field_name in args.__dataclass_fields__:
        value = getattr(args, field_name)
        logger.info(f"  {field_name}: {value}")
    logger.info("")

    # Initialize Argilla client (if needed for dataset creation)
    argilla_client = None
    if args.hf_output_dataset:
        try:
            # Try to initialize Argilla client - this might fail if not configured
            argilla_client = rg.Argilla(
                api_url=os.environ.get("ARGILLA_API_URL", "http://localhost:6900"),
                api_key=os.environ.get("ARGILLA_API_KEY", "admin.apikey"),
            )
        except Exception as e:
            logger.warning(f"Could not initialize Argilla client: {e}")
            logger.warning("Argilla dataset creation will be skipped")

    # Load dataset splits and LLM clients
    llm, _ = create_llm_clients(server, args, ThoughtSchema)
    datasets = load_datasets(args, take_n=take_n)
    for split_name, split_ds in datasets.items():
        logger.info(f"Loaded {len(split_ds)} examples for split '{split_name}'")

    ##########################
    # Run the training split #
    ##########################

    # Run the pipeline
    logger.info("Running generation pipeline for available splits...")
    logger.info("Monitor progress in Ray dashboard: http://localhost:8265")

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        **enable_thinking,
    }

    if args.temperature is not None:
        generation_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        generation_kwargs["top_p"] = args.top_p
    if args.stop is not None:
        generation_kwargs["stop"] = args.stop

    # Run train split first
    with Pipeline("train-generation-pipeline", cache_dir="./distilabel_cache") as pipeline:
        multi_turn_generator = MultiTurnWithToolUseGenerator(
            name="multi_turn_generator",
            llm=llm,
            dataset=datasets["train"],
            batch_size=args.input_batch_size,
            n_turns=args.n_turns,
            system_prompt=get_math_system_prompt(),
            thought_schema=ThoughtSchema,
            library=get_lib(),
        )

        distiset: Distiset = pipeline.run(
            parameters={
                multi_turn_generator.name: {"llm": {"generation_kwargs": generation_kwargs}},
            },
            use_cache=args.use_cache,
            dataset_batch_size=args.input_batch_size,
        )

    ############################
    # Run the validation split #
    ############################

    if "validation" in datasets:
        with Pipeline("validation-generation-pipeline", cache_dir="./distilabel_cache") as pipeline:
            multi_turn_generator = MultiTurnWithToolUseGenerator(
                name="multi_turn_generator",
                llm=llm,
                dataset=datasets["validation"],
                batch_size=args.input_batch_size,
                n_turns=args.n_turns,
                system_prompt=get_math_system_prompt(),
                thought_schema=ThoughtSchema,
                library=get_lib(),
            )

            val_distiset: Distiset = pipeline.run(
                parameters={
                    multi_turn_generator.name: {"llm": {"generation_kwargs": generation_kwargs}},
                },
                use_cache=args.use_cache,
                dataset_batch_size=args.input_batch_size,
            )

        distiset["default"]["validation"] = val_distiset["default"]["train"]

    # The run interferes with the logger, this restores its state
    cleanup()

    ###############################
    # Push the results to the hub #
    ###############################
    logger.info("Generation complete!")
    distilabel_train = distiset["default"]["train"]
    total_train = len(distilabel_train)
    train_correct = sum(1 for row in distilabel_train if row["is_correct"])
    logger.info("Pipeline completed (train):")
    logger.info(f"  Math verify successes: {train_correct}/{total_train}")

    if "validation" in distiset["default"]:
        distilabel_val = distiset["default"]["validation"]
        total_val = len(distilabel_val)
        val_correct = sum(1 for row in distilabel_val if row["is_correct"])
        logger.info("Pipeline completed (validation):")
        logger.info(f"  Math verify successes: {val_correct}/{total_val}")

    if args.hf_output_dataset:
        logger.info(f"Pushing dataset to: {args.hf_output_dataset}")

        try:
            push_to_huggingface(distiset, args.hf_output_dataset, args.private)

            # Create Argilla dataset for annotation if client is available
            if argilla_client and args.argilla_output_dataset:
                try:
                    dataset_data = distiset["default"]["train"]
                    create_argilla_dataset(
                        dataset_name=f"{args.argilla_output_dataset}-train",
                        distiset_data=dataset_data,
                        client=argilla_client,
                        private=args.private,
                    )

                    dataset_data = distiset["default"]["validation"]
                    create_argilla_dataset(
                        dataset_name=f"{args.argilla_output_dataset}-validation",
                        distiset_data=dataset_data,
                        client=argilla_client,
                        private=args.private,
                    )
                except Exception as e:
                    logger.warning(f"Failed to create Argilla dataset: {e}")

        except Exception:
            logger.exception("❌ Error pushing dataset")
            sys.exit(1)


if __name__ == "__main__":
    take_n = None
    if "--config" not in argv:
        argv.append("--config")
        argv.append("linalg_zero/config/distillation/vllm_debug.yaml")
        take_n = 4

    # Parse configuration from YAML file stored in the --config argument
    parser = TrlParser(dataclass_types=[DistillationConfig, VllmServerConfig])
    (distillation_config, backend_config) = parser.parse_args_and_config()

    main(distillation_config, backend_config, take_n)
