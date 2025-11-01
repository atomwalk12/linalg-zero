# Copyright Sierra

import ast
import json
from typing import Any, Literal

from pydantic import BaseModel

from linalg_zero.shared.types import LibTypes

RESPOND_ACTION_NAME = "respond"
RESPOND_ACTION_FIELD_NAME = "content"


class Action(BaseModel):
    name: str
    kwargs: dict[str, Any]
    content: str | None = None


class Task(BaseModel):
    user_id: str
    actions: list[Action]
    instruction: str
    outputs: list[LibTypes]

    @classmethod
    def from_dataset_entry(cls, entry: dict[str, Any]) -> "Task":
        stepwise = json.loads(entry["stepwise_ground_truths"])
        return cls(
            user_id=entry.get("user_id", "user"),
            actions=[
                Action(name=fn_name, kwargs={"matrix": ground_truth})
                for step in stepwise
                for fn_name, ground_truth in step.items()
            ],
            instruction=entry["query"],
            outputs=[ast.literal_eval(entry["ground_truth"])],  # Keep as list of strings
        )


class RewardOutputInfo(BaseModel):
    r_outputs: float
    outputs: dict[str, bool]


class RewardActionInfo(BaseModel):
    r_actions: float
    gt_data_hash: str


class RewardResult(BaseModel):
    reward: float
    info: RewardOutputInfo | RewardActionInfo
    actions: list[Action]


class SolveResult(BaseModel):
    reward: float
    messages: list[dict[str, Any]]
    info: dict[str, Any]
    total_cost: float | None = None


class EnvInfo(BaseModel):
    task: Task
    source: str | None = None
    user_cost: float | None = None
    reward_info: RewardResult | None = None


class EnvResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: EnvInfo


class EnvResetResponse(BaseModel):
    observation: str
    info: EnvInfo


class EnvRunResult(BaseModel):
    task_id: int
    reward: float
    info: dict[str, Any]
    traj: list[dict[str, Any]]
    trial: int


class RunConfig(BaseModel):
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
    in_process: bool = False


class TauBenchTrainingConfig(BaseModel):
    """Training configuration for ART RL on tau-bench tasks"""

    trajectories_per_group: int = 6
    groups_per_step: int = 10
    learning_rate: float = 1.2e-5
    eval_steps: int = 10
    val_set_size: int = 85
    training_dataset_size: int = 30
    num_epochs: int = 50
    train_mode: str = "sync_rl"
    importance_sampling_level: Literal["token", "sequence"] = "token"


class TauBenchPolicyConfig(BaseModel):
    """Policy configuration for tau-bench agent"""

    # Training configuration
    training_config: TauBenchTrainingConfig | None = None

    # tau-bench specific configs
    run_config: RunConfig
