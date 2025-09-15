import logging
import os
from sys import argv

import argilla as rg
from distilabel.distiset import Distiset
from distilabel.pipeline import Pipeline
from trl import TrlParser

from linalg_zero.config.data import DistillationConfig, LlamaCppServerConfig, VllmServerConfig
from linalg_zero.distillation.components.models import DefaultConfig, ModelType, Qwen3ThinkingConfig
from linalg_zero.distillation.components.multi_turn_generation import MultiTurnWithToolUseGenerator
from linalg_zero.distillation.data import ThoughtSchema
from linalg_zero.distillation.utils import (
    cleanup,
    create_llm_clients,
    load_datasets_for_distillation,
    print_statistics,
    push_argilla_dataset,
    push_datasets_to_huggingface,
    save_distiset_to_disk,
)
from linalg_zero.shared.lib import get_lib_fn_names
from linalg_zero.shared.system_prompts import get_math_system_prompt
from linalg_zero.shared.utils import get_logger, setup_logging


def main(args: DistillationConfig, server: LlamaCppServerConfig | VllmServerConfig) -> None:
    ################################
    # Initialize and load datasets #
    ################################

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
    llm = create_llm_clients(server, args, ThoughtSchema)
    dataset = load_datasets_for_distillation(args)
    for split_name, split_ds in dataset.items():
        logger.info(f"Loaded {len(split_ds)} examples for split '{split_name}'")

    ##########################
    # Run the training split #
    ##########################

    # Run the pipeline
    logger.info("Running generation pipeline for available splits...")
    enable_thinking = {"extra_body": {"chat_template_kwargs": {"enable_thinking": args.enable_reasoning}}}

    generation_kwargs = {"max_new_tokens": args.max_new_tokens, **enable_thinking}
    if args.stop is not None:
        generation_kwargs["stop"] = args.stop

    available_functions = get_lib_fn_names()

    # Delegate all sampling defaults to parameters; only determinism toggled by user
    model_config = Qwen3ThinkingConfig() if args.model_type == ModelType.QWEN3_THINKING else DefaultConfig()
    model_config.set_recommended_defaults(generation_kwargs, deterministic=args.deterministic)

    # Run train split first
    pipeline_obj = Pipeline("train-generation-pipeline")
    if not args.debug_mode:
        logger.info("Monitor progress in Ray dashboard: http://localhost:8265")
        pipeline_obj = pipeline_obj.ray()

    with pipeline_obj as pipeline:
        multi_turn_generator = MultiTurnWithToolUseGenerator(
            name="multi_turn_generator",
            llm=llm,
            dataset=dataset["train"],
            batch_size=args.input_batch_size,
            n_turns=args.n_turns,
            system_prompt=get_math_system_prompt(),
            library=available_functions,
            model_name=args.model_type,
        )

        distiset: Distiset = pipeline.run(
            parameters={
                multi_turn_generator.name: {"llm": {"generation_kwargs": generation_kwargs}},
            },
            use_cache=args.use_cache if not args.debug_mode else False,
            dataset_batch_size=args.input_batch_size,
        )

    cleanup()
    logger.info("Generation complete!")

    save_distiset_to_disk(distiset, "results/distiset/")

    ###############################
    # Push the results to the hub #
    ###############################
    logger.info("Pipeline completed (train):")
    print_statistics(distiset["default"]["train"])

    if argilla_client and args.argilla_output_dataset:
        logger.info(f"Creating Argilla dataset: {args.argilla_output_dataset}")
        push_argilla_dataset(argilla_client, distiset, args)

    if args.hf_output_dataset:
        logger.info(f"Pushing dataset to: {args.hf_output_dataset}")
        push_datasets_to_huggingface(distiset, args)


if __name__ == "__main__":
    if "--config" not in argv:
        argv.append("--config")
        argv.append("linalg_zero/config/distillation/llamacpp_qwen3_30b_A3B_instruct.yaml")

    # Parse configuration from YAML file stored in the --config argument
    parser = TrlParser(dataclass_types=[DistillationConfig, LlamaCppServerConfig])
    (distillation_config, backend_config) = parser.parse_args_and_config()

    main(distillation_config, backend_config)
