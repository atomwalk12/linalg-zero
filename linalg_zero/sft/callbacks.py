from typing import Any

from transformers.trainer_callback import (
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.training_args import TrainingArguments
from trl import ModelConfig

from linalg_zero.config.data import SFTConfig
from linalg_zero.sft.hub import push_to_hub_revision
from linalg_zero.sft.tool_calling_accuracy import ToolCallingAccuracyCallback
from linalg_zero.shared.lib import get_lib


class DummyConfig:
    def __init__(self, **kwargs: Any):
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


CALLBACKS = {
    "push_to_hub_revision": PushToHubRevisionCallback,
    "tool_calling_accuracy": ToolCallingAccuracyCallback,
    "early_stopping": EarlyStoppingCallback,
}


def get_callbacks(train_config: SFTConfig, model_config: ModelConfig) -> list[TrainerCallback]:
    callbacks = []
    for callback_name in train_config.callbacks:
        if callback_name not in CALLBACKS:
            raise ValueError(f"Callback {callback_name} not found in CALLBACKS.")

        # Different callbacks have different constructor signatures
        if callback_name == "tool_calling_accuracy":
            callbacks.append(CALLBACKS[callback_name](library=get_lib()))
        elif callback_name == "early_stopping":
            patience = train_config.early_stopping_patience
            threshold = train_config.early_stopping_threshold
            callbacks.append(
                CALLBACKS[callback_name](early_stopping_patience=patience, early_stopping_threshold=threshold)
            )
        else:
            callbacks.append(CALLBACKS[callback_name](model_config))

    return callbacks
