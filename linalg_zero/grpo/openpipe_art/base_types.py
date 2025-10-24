"""
Base types for the linear algebra environment system.
Self-contained types without external dependencies.
"""

from typing import Any

from pydantic import BaseModel

RESPOND_ACTION_NAME = "respond"
RESPOND_ACTION_FIELD_NAME = "content"


class Action(BaseModel):
    """Represents an action taken by an agent."""

    name: str
    kwargs: dict[str, Any]


class Task(BaseModel):
    """Represents a task with instructions and expected outputs."""

    user_id: str
    actions: list[Action]
    instruction: str
    outputs: list[str]


class RewardOutputInfo(BaseModel):
    """Information about reward calculation based on outputs."""

    r_outputs: float
    outputs: dict[str, bool]


class RewardActionInfo(BaseModel):
    """Information about reward calculation based on actions."""

    r_actions: float
    gt_data_hash: str


class RewardResult(BaseModel):
    """Result of reward calculation."""

    reward: float
    info: RewardOutputInfo | RewardActionInfo
    actions: list[Action]


class SolveResult(BaseModel):
    """Result of solving a task."""

    reward: float
    messages: list[dict[str, Any]]
    info: dict[str, Any]
    total_cost: float | None = None


class EnvInfo(BaseModel):
    """Information about environment state."""

    task: Task
    source: str | None = None
    user_cost: float | None = None
    reward_info: RewardResult | None = None


class EnvResponse(BaseModel):
    """Response from environment step."""

    observation: str
    reward: float
    done: bool
    info: EnvInfo


class EnvResetResponse(BaseModel):
    """Response from environment reset."""

    observation: str
    info: EnvInfo


class EnvRunResult(BaseModel):
    """Result of running an environment episode."""

    task_id: int
    reward: float
    info: dict[str, Any]
    traj: list[dict[str, Any]]
    trial: int
