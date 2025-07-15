"""Distillation pipeline for generating CoT-based solutions to linear algebra problems."""

import json
import logging
from collections.abc import Generator
from typing import Any

from trl import TrlParser

from datasets import Dataset
from linalg_zero.config.data import DistillationConfig
from linalg_zero.distillation.utils import build_distilabel_pipeline
from linalg_zero.shared import get_logger, setup_logging


def main() -> None:
    """Main function to run the distillation pipeline."""
    ################
    # Initialization
    ################
    # Set up logging
    setup_logging(level=logging.INFO, include_timestamp=True)
    logger = get_logger(__name__)

    # Parse configuration from YAML file
    parser = TrlParser(dataclass_types=[DistillationConfig])  # type: ignore[reportArgumentType]
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
    # TODO: This will be replaced with the actual dataset
    def gen() -> Generator[dict[str, Any], None, None]:
        yield {
            "prompt": "Calculate 100 divided by 20, then multiply the result by 2.",
            "system_prompt": "You are a helpful assistant that can do maths.",
        }

    dataset = Dataset.from_generator(gen)
    logger.info("Dataset loaded!")

    ############################
    # Build and run the pipeline
    ############################
    # The pipeline uses OpenAI compatible APIs to streamline both debugging and deployment.
    pipeline = build_distilabel_pipeline(
        model=config.model,
        base_url=config.vllm_server_url,
        prompt_template=config.prompt_template,
        prompt_column=config.prompt_column,
        temperature=config.temperature,
        top_p=config.top_p,
        max_new_tokens=config.max_new_tokens,
        num_generations=config.num_generations,
        input_batch_size=config.input_batch_size,
        client_replicas=config.client_replicas,
        timeout=config.timeout,
        retries=config.retries,
    )

    logger.info("Running generation pipeline...")
    distiset = pipeline.run(
        dataset=dataset,  # type: ignore[reportArgumentType]
        dataset_batch_size=config.input_batch_size,
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
