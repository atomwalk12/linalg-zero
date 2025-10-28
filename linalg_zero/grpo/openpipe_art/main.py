import os

os.environ["IMPORT_PEFT"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

import asyncio
import concurrent
import logging
import random
import signal
import sys
from typing import Any

import art
import weave
from art import TrainableModel
from art.local import LocalBackend
from dotenv import load_dotenv
from trl.scripts.utils import TrlParser

from linalg_zero.grpo.openpipe_art.base_types import SolveResult
from linalg_zero.grpo.openpipe_art.data_types import LinearAlgebraScenario, LinearAlgebraTrainingConfig, RunConfig
from linalg_zero.grpo.openpipe_art.linalg_env import create_linalg_environment
from linalg_zero.sft.utils import init_wandb_training
from linalg_zero.shared.utils import get_logger, setup_logging


@weave.op
@art.retry(exceptions=())
async def rollout(model: art.Model, scenario: LinearAlgebraScenario) -> Any:
    """
    Execute a single GRPO training rollout using LinAlg environment.

    This function creates an environment, runs an agent episode, and converts
    the results to the art.Trajectory format for GRPO training.

    Args:
        model: art.Model instance for inference
        scenario: LinearAlgebraScenario with step information

    Returns:
        art.Trajectory formatted for GRPO training
    """
    logger = get_logger(__name__)
    logger.info(f"Starting rollout for scenario step {scenario.step}")
    global _global_run_config
    if _global_run_config is None:
        raise ValueError(
            "Global run_config is not set. Ensure main() function sets _global_run_config before training."
        )
    run_config = _global_run_config

    try:
        # Get run_config from global scope

        # Create environment instance
        environment = create_linalg_environment(run_config)
        logger.debug("Created LinAlg environment")

        # Create agent with model integration
        from linalg_zero.grpo.openpipe_art.linalg_agent import create_linalg_agent

        agent = create_linalg_agent(
            env=environment,
            model=model.name if hasattr(model, "name") else "art-model",
            provider="art",  # Custom provider for art integration
            temperature=run_config.temperature,
            art_model=model,  # Pass the art.Model instance directly
        )

        logger.debug("Created LinAlg agent with art model integration")

        # Run episode with agent
        solve_result = agent.solve(
            env=environment,
            task_index=None,  # Random task selection
            max_num_steps=run_config.max_num_steps,
        )

        logger.info(f"Episode completed with reward: {solve_result.reward}")

    except Exception as e:
        logger.exception(f"Rollout failed for scenario step {scenario.step}")
        # Return empty solve result on failure
        return SolveResult(
            reward=0.0,
            messages=[],
            info={"error": str(e), "scenario_step": scenario.step},
            total_cost=0.0,
        )
    else:
        # Return the solve result directly - art framework will handle trajectory conversion
        logger.debug(f"Rollout completed successfully with reward {solve_result.reward}")
        return solve_result


# Global variable to store run_config for rollout function access
_global_run_config: RunConfig | None = None


async def main(run_config: RunConfig, train_config: LinearAlgebraTrainingConfig):
    #####################
    # Setup and logging #
    #####################
    # Store run_config globally for rollout function access
    global _global_run_config
    _global_run_config = run_config

    # Load environment/setup seed
    load_dotenv()
    random.seed(42)

    # Log both to file and console
    setup_logging(level=logging.INFO, include_timestamp=True)
    logger = get_logger(__name__)

    logger.info(f"Run parameters: {run_config}")
    logger.info(f"Training parameters: {train_config}")

    # Initialize wandb if requested
    if run_config.report_to and "wandb" in run_config.report_to:
        # Create a minimal training args object for wandb initialization
        class WandbArgs:
            def __init__(self):
                self.wandb_entity = None
                self.wandb_project = "linear-algebra"
                self.wandb_run_group = None

        wandb_args = WandbArgs()
        init_wandb_training(wandb_args)

    # Concurrency parameters
    loop = asyncio.get_event_loop()
    big_pool = concurrent.futures.ThreadPoolExecutor(max_workers=50)
    loop.set_default_executor(big_pool)

    ##################
    # Training setup #
    ##################
    backend = LocalBackend(path="./.art")

    # Create TrainableModel with configuration
    model = TrainableModel(
        name=f"linalg-{run_config.model}",
        project="linear-algebra",
        base_model=run_config.base_model,
    )

    print("initializing weave")
    try:
        weave.init(model.project, settings={"print_call_link": False})
    except Exception as e:
        logger.warning(f"Failed to initialize weave: {e}")
        logger.info("Continuing without weave initialization")

    await model.register(backend)
    print("Model registered successfully!")

    #########
    # Train #
    #########
    # Use configuration parameters
    TRAINING_STEPS = train_config.num_epochs
    ROLLOUTS_PER_STEP = train_config.trajectories_per_group * train_config.groups_per_step
    LEARNING_RATE = train_config.learning_rate

    logger.info(f"Starting training: {TRAINING_STEPS} steps, {ROLLOUTS_PER_STEP} rollouts per step")

    for i in range(await model.get_step(), TRAINING_STEPS):
        logger.info(f"Training step {i + 1}/{TRAINING_STEPS}")

        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(rollout(model, LinearAlgebraScenario(step=i)) for _ in range(ROLLOUTS_PER_STEP))
                for _ in range(1)
            ),
            pbar_desc=f"gather step {i + 1}",
        )
        await model.delete_checkpoints()
        await model.train(train_groups, config=art.TrainConfig(learning_rate=LEARNING_RATE))

        logger.info(f"Completed training step {i + 1}/{TRAINING_STEPS}")


async def shutdown(signal_type, loop):
    """Gracefully shutdown on receiving signals."""
    print(f"\nReceived exit signal {signal_type.name}...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    print(f"Cancelling {len(tasks)} outstanding tasks")
    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


if __name__ == "__main__":
    """Script entry point for GRPO training."""

    # Get event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Setup signal handlers for graceful shutdown
    signals = (signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(s, lambda s=s: asyncio.create_task(shutdown(s, loop)))

    try:
        """Script entry point for SFT training."""
        if "--config" not in sys.argv:
            sys.argv.append("--config")
            sys.argv.append("linalg_zero/config/grpo/qwen2.5-3b/config.yaml")

        parser = TrlParser((RunConfig, LinearAlgebraTrainingConfig))
        run_args, training_args = parser.parse_args_and_config()

        loop.run_until_complete(main(run_args, training_args))
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    finally:
        loop.close()
        print("Event loop closed")
