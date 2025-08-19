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
    load_dataset,
    prepare_dataset_for_sft,
)
from linalg_zero.shared.lib import get_lib
from linalg_zero.shared.system_prompts import get_math_system_prompt
from linalg_zero.shared.utils import get_logger, setup_logging


def main(args: DistillationConfig, server: LlamaCppServerConfig | VllmServerConfig) -> None:
    ################
    # Initialization
    ################
    USING_VLLM = isinstance(server, VllmServerConfig)
    enable_thinking = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}} if USING_VLLM else {}

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

    ##########################
    # Load dataset/LLM clients
    ##########################
    llm, _ = create_llm_clients(server, args, ThoughtSchema)
    dataset = load_dataset(args)
    logger.info(f"Loaded {len(dataset)} examples")

    ############################
    # Build and run the pipeline
    ############################
    with Pipeline("generation-pipeline") as pipeline:
        # Single step: generate multi-turn conversations
        multi_turn_generator = MultiTurnWithToolUseGenerator(
            name="multi_turn_generator",
            llm=llm,
            dataset=dataset,
            batch_size=5,  # TODO(A): tweak this
            n_turns=args.n_turns,
            system_prompt=get_math_system_prompt(summary=False),
            include_system_prompt=True,
            thought_schema=ThoughtSchema,
            library=get_lib(),
        )

    # Run the pipeline
    logger.info("Running generation pipeline...")
    logger.info(f"Processing {len(dataset)} examples with batch size {args.input_batch_size}")
    logger.info("Monitor progress in Ray dashboard: http://localhost:8265")

    distiset: Distiset = pipeline.run(
        parameters={
            multi_turn_generator.name: {
                "llm": {"generation_kwargs": {"max_new_tokens": 4096, **enable_thinking, "temperature": 0.0}}
            },
        },
        use_cache=False,
        dataset_batch_size=args.input_batch_size,
    )

    # The run interferes with the logger, this restores its state
    cleanup()

    #############################
    # Push the results to the hub
    #############################
    logger.info("Generation complete!")
    train_data = distiset["default"]["train"]

    total_examples = len(train_data)
    total_inputs = len(dataset)

    # Count successes at each stage
    execution_successes = sum(1 for row in train_data if row.get("keep_row_after_execution_check", False))
    math_verify_successes = sum(1 for row in train_data if row.get("keep_row_after_semantic_check", False))

    logger.info("Pipeline completed:")
    logger.info(f"  Total results: {total_examples}/{total_inputs}")
    logger.info(f"  Execution successes: {execution_successes}/{total_inputs}")
    logger.info(f"  Math verify successes: {math_verify_successes}/{total_inputs}")

    if args.hf_output_dataset:
        logger.info(f"Pushing dataset to: {args.hf_output_dataset}")

        try:
            # Add the tools column to the dataset, required for SFT
            prepare_dataset_for_sft(distiset)

            # Push to HuggingFace Hub
            distiset.push_to_hub(
                args.hf_output_dataset,
                private=args.private,
            )
            logger.info(f"✅ Dataset successfully pushed to: {args.hf_output_dataset}")
            logger.info(f"   Privacy: {'Private' if args.private else 'Public'}")
            logger.info(f"   Access URL: https://huggingface.co/datasets/{args.hf_output_dataset}")

            # Create Argilla dataset for annotation if client is available
            if argilla_client and args.argilla_output_dataset:
                try:
                    dataset_data = distiset["default"]["train"]
                    create_argilla_dataset(
                        dataset_name=args.argilla_output_dataset, distiset_data=dataset_data, client=argilla_client
                    )
                    logger.info("✅ Argilla dataset created successfully")
                    logger.info(f"   Privacy: {'Private' if args.private else 'Public'}")
                    logger.info(f"   Access URL: https://{args.argilla_output_dataset.replace('/', '-')}.hf.space")
                except Exception as e:
                    logger.warning(f"Failed to create Argilla dataset: {e}")

        except Exception:
            logger.exception("❌ Error pushing dataset")
            sys.exit(1)


if __name__ == "__main__":
    # TODO: remove these lines if not developing locally
    if "--config" not in argv:
        argv.append("--config")
        argv.append("linalg_zero/config/distillation/llamacpp_debug.yaml")

    # Check backend type (vllm or llama-cpp)
    USING_VLLM = os.environ.get("USING_VLLM", "False").lower() == "true"
    server_config = VllmServerConfig if USING_VLLM else LlamaCppServerConfig

    # Parse configuration from YAML file stored in the --config argument
    parser = TrlParser(dataclass_types=[DistillationConfig, server_config])
    (distillation_config, backend_config) = parser.parse_args_and_config()

    main(distillation_config, backend_config)
