"""Distillation pipeline for generating CoT-based solutions to linear algebra problems."""

import json
import logging

from trl import TrlParser

from linalg_zero.config.data import DistillationConfig
from linalg_zero.distillation.utils import build_fc_dataset, build_fc_pipeline, prepare_tools
from linalg_zero.shared import LLAMA_CPP_DIR, get_logger, setup_logging


def main() -> None:
    """Main function to run the distillation pipeline."""
    ################
    # Initialization
    ################
    # Set up logging
    setup_logging(level=logging.INFO, include_timestamp=True)
    logger = get_logger(__name__)

    # Parse configuration from YAML file
    parser = TrlParser(dataclass_types=[DistillationConfig])
    config: DistillationConfig = parser.parse_args_and_config()[0]

    logger.info("Running with configuration:")
    for field_name in config.__dataclass_fields__:
        value = getattr(config, field_name)
        logger.info(f"  {field_name}: {value}")
    logger.info("")

    logger.info(f"Loading '{config.hf_dataset}' (split: {config.hf_dataset_split}) dataset...")

    ##################
    # Load the dataset
    ##################
    # TODO: Currently a mock dataset is used.
    tools, target_fns = prepare_tools()
    dataset = build_fc_dataset(tools=tools)
    logger.info("Dataset loaded!")

    ############################
    # Build and run the pipeline
    ############################
    # The pipeline uses OpenAI compatible APIs to streamline both debugging and deployment.
    # TODO: Not all config parameters are used here, optimise parameter use.
    pipeline = build_fc_pipeline(
        model=str(LLAMA_CPP_DIR / config.model),
        dataset=dataset,
        target_fns=target_fns,
        base_url=config.vllm_server_url,
        temperature=config.temperature,
        top_p=config.top_p,
        max_new_tokens=config.max_new_tokens,
        timeout=config.timeout,
        retries=config.retries,
    )

    logger.info("Running generation pipeline...")
    distiset = pipeline.run(
        use_cache=False,
    )

    #############################
    # Push the results to the hub
    #############################
    # Distilabel uses ray which interferes with the logger, so we switch to print statements after
    # running the pipeline
    if config.hf_output_dataset:
        print(f"Pushing dataset to: {config.hf_output_dataset}")
        distiset.push_to_hub(
            config.hf_output_dataset,
            private=config.private,
        )
    else:
        print("No output dataset specified. Results:")
        result = distiset["default"]["train"][0]
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
