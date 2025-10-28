from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel


class EnvResetResponse:
    pass


class LinearAlgebraScenario(BaseModel):
    step: int


@dataclass
class RunConfig:
    """Run configuration"""

    model_provider: str = field(
        default="openai",
        metadata={"help": "The provider for the primary model."},
    )
    user_model_provider: str = field(
        default="openai",
        metadata={"help": "The provider for the user model."},
    )
    model: str = field(
        default="gpt-4.1",
        metadata={"help": "The primary model to use for the run."},
    )
    user_model: str = field(
        default="gpt-4o",
        metadata={"help": "The user model to use for the run."},
    )
    num_trials: int = field(
        default=1,
        metadata={"help": "The number of trials to run."},
    )
    env_name: str = field(
        default="retail",
        metadata={"help": "The environment to run the trials in."},
    )
    agent_strategy: str = field(
        default="tool-calling",
        metadata={"help": "The strategy for the agent."},
    )
    temperature: float = field(
        default=0.0,
        metadata={"help": "The temperature for model generation."},
    )
    task_split: str = field(
        default="test",
        metadata={"help": "The split of the task to use."},
    )
    start_index: int = field(
        default=0,
        metadata={"help": "The starting index for tasks."},
    )
    end_index: int = field(
        default=-1,
        metadata={"help": "The ending index for tasks. -1 for all."},
    )
    task_ids: list[int] | None = field(
        default=None,
        metadata={"help": "A list of specific task IDs to run."},
    )
    log_dir: str = field(
        default="results",
        metadata={"help": "The directory to save logs to."},
    )
    max_concurrency: int = field(
        default=1,
        metadata={"help": "The maximum number of concurrent runs."},
    )
    seed: int = field(
        default=10,
        metadata={"help": "The random seed for the run."},
    )
    shuffle: int = field(
        default=0,
        metadata={"help": "Whether to shuffle the tasks."},
    )
    user_strategy: str = field(
        default="llm",
        metadata={"help": "The strategy for the user model."},
    )
    few_shot_displays_path: str | None = field(
        default=None,
        metadata={"help": "Path to the few-shot displays file."},
    )
    # art related configs
    api_key: str | None = field(
        default=None,
        metadata={"help": "API key for the model provider."},
    )
    base_url: str | None = field(
        default=None,
        metadata={"help": "Base URL for the model provider API."},
    )
    reward_type: str = field(
        default="real",
        metadata={"help": "The type of reward to use."},
    )
    judge_model: str = field(
        default="o3",
        metadata={"help": "The model to use for judging."},
    )
    max_num_steps: int = field(
        default=30,
        metadata={"help": "The maximum number of steps per trial."},
    )
    skip_eval: bool = field(
        default=False,
        metadata={"help": "Whether to skip evaluation."},
    )
    add_shadow_trajectory: bool = field(
        default=False,
        metadata={"help": "Whether to add a shadow trajectory."},
    )
    messages_only: bool = field(
        default=False,
        metadata={"help": "Whether to only use messages."},
    )
    base_model: str = field(
        default="unsloth/Qwen2.5-14B-Instruct",
        metadata={"help": "The base model for the run."},
    )
    is_multi_gpu: bool = field(
        default=False,
        metadata={"help": "Whether the run is on multiple GPUs."},
    )
    add_no_think: bool = field(
        default=False,
        metadata={"help": "Whether to add a no-think step."},
    )
    plot_tensors: bool = field(
        default=False,
        metadata={"help": "Whether to plot tensors."},
    )
    report_to: list[str] | None = field(
        default=None,
        metadata={"help": "List of services to report statistics to (i.e. wandb)."},
    )


@dataclass
class LinearAlgebraTrainingConfig:
    """Training configuration"""

    trajectories_per_group: int = field(
        default=6,
        metadata={"help": "The number of trajectories per group."},
    )
    groups_per_step: int = field(
        default=10,
        metadata={"help": "The number of groups per step."},
    )
    learning_rate: float = field(
        default=1.2e-5,
        metadata={"help": "The learning rate for training."},
    )
    eval_steps: int = field(
        default=10,
        metadata={"help": "The number of steps between evaluations."},
    )
    val_set_size: int = field(
        default=85,
        metadata={"help": "The size of the validation set."},
    )
    training_dataset_size: int = field(
        default=30,
        metadata={"help": "The size of the training dataset."},
    )
    num_epochs: int = field(
        default=50,
        metadata={"help": "The number of epochs to train for."},
    )
    train_mode: str = field(
        default="sync_rl",
        metadata={"help": "The training mode."},
    )
    importance_sampling_level: Literal["token", "sequence"] = field(
        default="token",
        metadata={"help": "The level of importance sampling."},
    )

    # Environment-specific configuration
    environment_type: str = field(
        default="linalg",
        metadata={"help": "The type of environment (linalg for linear algebra)."},
    )
    eval_dataset_split: str = field(
        default="validation",
        metadata={"help": "Dataset split to use for evaluation (validation/test)."},
    )
