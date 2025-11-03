import asyncio

import art
import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from run_rl import train
from tau_bench.types import TauBenchPolicyConfig


@hydra.main(version_base=None, config_path="../../config/grpo/Qwen/Qwen3-4B", config_name="config")
def main(cfg: DictConfig) -> None:
    # Load environment variables
    load_dotenv()

    # Convert all configs to plain dicts
    training_config = OmegaConf.to_container(cfg.training, resolve=True)
    run_config = OmegaConf.to_container(cfg.run, resolve=True)
    engine_args = OmegaConf.to_container(cfg.engine, resolve=True)
    torchtune_args = OmegaConf.to_container(cfg.torchtune, resolve=True)

    assert isinstance(engine_args, dict), "Engine args must be provided"
    assert isinstance(torchtune_args, dict), "Torchtune args must be provided"
    assert isinstance(run_config, dict), "Run config must be provided"
    assert isinstance(training_config, dict), "Training config must be provided"

    # Set dynamic values
    if "tensor_parallel_size" not in engine_args:
        engine_args["tensor_parallel_size"] = torch.cuda.device_count()

    # Build model and run training
    model = art.TrainableModel(
        name=run_config["project_id"],
        project=run_config["project"],
        base_model=run_config["base_model"],
        config=TauBenchPolicyConfig(
            training_config=training_config,
            run_config=run_config,
        ),
        _internal_config=art.dev.InternalModelConfig(
            engine_args=engine_args,
            torchtune_args=torchtune_args,
        ),
    )
    asyncio.run(train(model))


if __name__ == "__main__":
    main()
