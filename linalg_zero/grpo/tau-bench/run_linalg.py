import os

os.environ["IMPORT_PEFT"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
os.environ["LITELLM_LOG"] = "DEBUG"
import asyncio

import art
import torch
from dotenv import load_dotenv
from run import RunConfig
from run_rl import train
from tau_bench.types import TauBenchPolicyConfig, TauBenchTrainingConfig

if __name__ == "__main__":
    """Script entry point for SFT training."""
    load_dotenv()
    BASE_MODEL = "Qwen/Qwen3-4B"
    # BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
    MODEL_NAME = "003"
    model = art.TrainableModel(
        name=MODEL_NAME,
        project="tau-bench",
        base_model=BASE_MODEL,
        config=TauBenchPolicyConfig(
            training_config=TauBenchTrainingConfig(
                trajectories_per_group=16,
                groups_per_step=8,
                learning_rate=2e-6,
                eval_steps=10,
                val_set_size=60,
                training_dataset_size=32,
                num_epochs=1000,
                train_mode="sync_rl",
            ),
            run_config=RunConfig(
                env="linear_algebra",
                model_provider="hosted_vllm",
                user_model_provider="openai",
                model=MODEL_NAME,
                user_strategy="mathematician",
                user_model="gpt-4o",
                agent_strategy="tool-calling-rl",
                temperature=1.0,
                task_split="test",
                log_dir="rl_results",
                skip_eval=False,
                in_process=True,
            ),
        ),
        _internal_config=art.dev.InternalModelConfig(
            engine_args=art.dev.EngineArgs(
                tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.65
            ),
            torchtune_args=art.dev.TorchtuneArgs(
                model="qwen2_5_32b_instruct", model_type="QWEN2", async_weight_syncing=True
            ),
        ),
    )
    # Run training
    asyncio.run(train(model))
