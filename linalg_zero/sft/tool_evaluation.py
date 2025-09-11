import logging
from typing import Any

from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.training_args import TrainingArguments
from trl import ModelConfig

from linalg_zero.sft.hub import push_to_hub_revision

logger = logging.getLogger(__name__)


class EvaluationState:
    """Tracks evaluation state across turns."""

    def __init__(self) -> None:
        self.total_tool_calls = 0
        self.malformed_turns = 0
        self.all_tools_json_valid = True
        self.all_tools_known = True
        self.all_tools_exec_success = True
        self.final_answer_correct = False
        self.has_final_answer = False
        self.has_any_tool_call = False
        self.turns_used = 0
        self.overall_format_valid = True


class DummyConfig:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class PushToHubRevisionCallback(TrainerCallback):
    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if state.is_world_process_zero:
            global_step = state.global_step

            # WARNING: if you use dataclasses.replace(args, ...) the accelerator dist state will be broken
            # Also if you instantiate a new SFTConfig, the accelerator dist state will also be broken
            dummy_config = DummyConfig(
                hub_model_id=args.hub_model_id,
                hub_model_revision=f"{args.hub_model_revision}-step-{global_step:09d}",  # type: ignore[attr-defined]
                output_dir=f"{args.output_dir}/checkpoint-{global_step}",
                system_prompt=args.system_prompt,  # type: ignore[attr-defined]
            )

            _ = push_to_hub_revision(dummy_config, extra_ignore_patterns=["*.pt"])  # don't push the optimizer states
