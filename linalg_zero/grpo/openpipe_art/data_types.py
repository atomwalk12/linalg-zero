from dataclasses import field
from typing import Literal

from pydantic import BaseModel


class EnvResetResponse:
    pass


class LinearAlgebraScenario(BaseModel):
    step: int


class RunConfig(BaseModel):
    """Run configuration"""

    model_provider: str
    user_model_provider: str
    model: str = "gpt-4.1"
    user_model: str = "gpt-4o"
    num_trials: int = 1
    env: str = "retail"
    agent_strategy: str = "tool-calling"
    temperature: float = 0.0
    task_split: str = "test"
    start_index: int = 0
    end_index: int = -1
    task_ids: list[int] | None = None
    log_dir: str = "results"
    max_concurrency: int = 1
    seed: int = 10
    shuffle: int = 0
    user_strategy: str = "llm"
    few_shot_displays_path: str | None = None
    # art related configs
    api_key: str | None = None
    base_url: str | None = None
    reward_type: str = "real"
    judge_model: str = "o3"
    max_num_steps: int = 30
    skip_eval: bool = False
    add_shadow_trajectory: bool = False
    messages_only: bool = False
    base_model: str = "unsloth/Qwen2.5-14B-Instruct"
    is_multi_gpu: bool = False
    add_no_think: bool = False
    plot_tensors: bool = False
    report_to: list[str] | None = field(
        default=None,
        metadata={"help": "List of services to report statistics to (i.e. wandb)."},
    )


class LinearAlgebraTrainingConfig(BaseModel):
    """Training configuration"""

    trajectories_per_group: int = 6
    groups_per_step: int = 10
    learning_rate: float = 1.2e-5
    eval_steps: int = 10
    val_set_size: int = 85
    training_dataset_size: int = 30
    num_epochs: int = 50
    train_mode: str = "sync_rl"
    importance_sampling_level: Literal["token", "sequence"] = "token"
