"""
Tool calling accuracy callback for SFT training.

Evaluates structural and correctness metrics for tool-use generations on all eval data.
"""

from __future__ import annotations

import json as _json
from typing import Any

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

import wandb
from linalg_zero.distillation.components.models import DefaultConfig
from linalg_zero.distillation.data import FunctionInvocationInfo, ThoughtSchema
from linalg_zero.grpo.compute_score import get_interaction_reward
from linalg_zero.grpo.verifiers.xml_parser import XMLParser
from linalg_zero.grpo.verify import parse_string
from linalg_zero.sft.tool_evaluation import EvaluationState
from linalg_zero.shared.lib import get_lib, get_lib_fn_names
from linalg_zero.shared.utils import get_logger

logger = get_logger(__name__)


class ToolCallingAccuracyCallback(TrainerCallback):
    """
    Callback to evaluate tool calling accuracy during SFT training.

    Evaluates all samples in the eval_dataset for robust metric computation.
    """

    def __init__(self, eval_dataset: Any) -> None:
        self.eval_dataset = eval_dataset
        self.library = get_lib()
        self._parser = XMLParser()
        self.model_config = DefaultConfig()

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if not state.is_world_process_zero:
            return

        model = kwargs.get("model")
        tokenizer = kwargs.get("processing_class")

        if model is None or tokenizer is None:
            return

        # Get max_new_tokens from training args (eval_max_new_tokens config)
        max_new_tokens = getattr(args, "eval_max_new_tokens", 8192)

        logger.info(f"Computing tool-calling metrics on all {len(self.eval_dataset)} eval samples...")

        eval_metrics, per_sample_rewards = self._compute_metrics(
            model=model,
            tokenizer=tokenizer,
            dataset=self.eval_dataset,
            max_new_tokens=max_new_tokens,
        )
        if eval_metrics:
            self._log_evaluation_metrics(eval_metrics, state, prefix="eval")
            if metrics is not None:
                metrics.update(eval_metrics)

        # Log per-sample distributions for better W&B visualization
        if per_sample_rewards:
            self._log_per_sample_distributions(per_sample_rewards, state)

    def _log_evaluation_metrics(self, metrics: dict[str, float], state: TrainerState, prefix: str = "eval") -> None:
        """Log evaluation metrics to trainer state and logger (Trainer will forward to W&B)."""
        for metric_name, value in metrics.items():
            state.log_history.append({
                "epoch": state.epoch if state.epoch is not None else -1,
                "step": state.global_step,
                f"{prefix}_{metric_name}": float(value),
            })

            logger.info(f"tool_use/{metric_name}: {value:.3f}")

    def _log_per_sample_distributions(self, per_sample_rewards: dict[str, list[float]], state: TrainerState) -> None:
        """Log per-sample rewards to W&B for richer visualization."""
        if not wandb.run:
            logger.debug("No active wandb run, skipping per-sample logging")
            return

        # Log each sample individually to create multiple data points
        for metric_name, values in per_sample_rewards.items():
            for reward_value in values:
                wandb.log(
                    {
                        f"eval_samples/{metric_name}": float(reward_value),
                        "eval_samples/global_step": state.global_step,
                    },
                    commit=False,
                )

        # Commit all the batched logs at once
        wandb.log({}, commit=True)

    def _compute_metrics(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Any,
        max_new_tokens: int,
    ) -> tuple[dict[str, float], dict[str, list[float]]]:
        """Compute metrics on all eval samples with fair turn allocation per sample."""
        model.eval()

        if not dataset or len(dataset) == 0:
            return {}, {}

        # Evaluate all samples in the eval dataset
        samples: list[dict[str, Any]] = [dataset[i] for i in range(len(dataset))]

        # Initialize totals and per-sample lists
        sum_reward_final = 0.0
        sum_reward_response_format = 0.0
        sum_reward_interaction = 0.0
        per_sample_final: list[float] = []
        per_sample_format: list[float] = []
        per_sample_interaction: list[float] = []

        denom = float(len(samples))
        pbar = tqdm(samples, desc="Evaluating tool calling", unit="sample", disable=False)
        for sample in pbar:
            # Determine fair n_turns based on sample complexity
            steps = sample["stepwise_ground_truths"]
            arr = _json.loads(steps)
            num_tool_turns = len(arr)

            # Allocate num_tool_turns + 1 turn for the answer
            n_turns = num_tool_turns + 1

            state = self._run_evaluation_turns(model, tokenizer, sample, n_turns, max_new_tokens)
            reward_final = float(state.reward_final_answer)
            reward_format = float(state.reward_response_format)
            reward_interaction = float(state.reward_interaction)

            sum_reward_final += reward_final
            sum_reward_response_format += reward_format
            sum_reward_interaction += reward_interaction
            per_sample_final.append(reward_final)
            per_sample_format.append(reward_format)
            per_sample_interaction.append(reward_interaction)

            # Update progress bar with running averages
            current_count = pbar.n + 1  # +1 because we just processed this sample
            avg_final = sum_reward_final / current_count
            avg_format = sum_reward_response_format / current_count
            avg_interaction = sum_reward_interaction / current_count
            pbar.set_postfix({
                "turns": n_turns,
                "reward_final": f"{avg_final:.3f}",
                "reward_format": f"{avg_format:.3f}",
                "reward_total": f"{avg_interaction:.3f}",
            })

        aggregated_metrics = {
            "reward_final_answer": (sum_reward_final / denom) if denom > 0 else 0.0,
            "reward_response_format": (sum_reward_response_format / denom) if denom > 0 else 0.0,
            "reward_interaction": (sum_reward_interaction / denom) if denom > 0 else 0.0,
        }

        per_sample_rewards = {
            "reward_final_answer": per_sample_final,
            "reward_response_format": per_sample_format,
            "reward_interaction": per_sample_interaction,
        }

        return aggregated_metrics, per_sample_rewards

    def _generate(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt_text: str, max_new_tokens: int
    ) -> str:
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            padding=bool(getattr(tokenizer, "pad_token_id", None)),
        )
        if inputs["input_ids"].shape[1] == tokenizer.model_max_length:
            logger.warning(f"Input truncated to {tokenizer.model_max_length} tokens during tool calling evaluation")

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)

            outputs = model.generate(  # type: ignore[operator]
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                top_k=None,  # Disable sampling parameters when do_sample=False
                top_p=None,
                temperature=None,
            )

        # Extract only the generated tokens (after the input)
        prompt_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, prompt_length:]

        # Check if generation was truncated due to max_new_tokens
        if (
            generated_tokens.shape[1] == max_new_tokens
            and getattr(tokenizer, "eos_token_id", None) is not None
            and generated_tokens[0, -1].item() != tokenizer.eos_token_id
        ):
            logger.warning(f"Generation may have been truncated at max_new_tokens={max_new_tokens}")

        return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    def _run_evaluation_turns(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        sample: dict[str, Any],
        n_turns: int,
        max_new_tokens: int,
    ) -> EvaluationState:
        """Run evaluation using simplified GRPO-based conversation processing."""
        state = EvaluationState()

        context = list(sample["messages"])

        # Multi-turn conversation loop
        for _ in range(n_turns):
            # Generate assistant response
            prompt = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
            if not isinstance(prompt, str):
                break

            output = self._generate(model, tokenizer, prompt, max_new_tokens)
            message = self.extract_non_structured_output(output, context)

            # Check if message extraction worked
            if message is None:
                break

            if message.fi is None and message.tool_call is None:
                break

            self.add_message("assistant", context, message)

            # Execute tool call if it exists
            if message.tool_call is not None:
                tool_call = self._execute(message)
                self.add_message("tool", context, tool_call)

            if message.final_answer is not None:
                state.has_final_answer = True
                break

        # Calculate state based on conversation history
        self._calculate_conversation_metrics(context, sample, state)
        return state

    def _execute(self, msg: ThoughtSchema) -> dict[str, str]:
        if msg.tool_call is None:
            raise ValueError("Tool call is required for execution")

        name = msg.tool_call.name
        arguments = msg.tool_call.arguments

        try:
            if name not in self.library:
                return {
                    "function_name": name,
                    "execution_result": f"ERROR: Function '{name}' not found in library",
                }

            result = self.library[name](**arguments)
            return {"function_name": name, "execution_result": str(result)}
        except Exception as exc:
            return {
                "function_name": name,
                "execution_result": f"ERROR: {type(exc).__name__}: {exc}",
            }

    def extract_non_structured_output(self, message: str, context: list[dict]) -> ThoughtSchema | None:
        """Extract output from messages that do not enforce structured output."""
        analysis = self._parser.analyze_message_in_context(context, message=message, tool_names=get_lib_fn_names())

        if not bool(analysis["is_valid_think_then_tool_or_answer"]):
            return None

        if analysis["has_answer"] and not bool(analysis["answer_policy_valid"]):
            return None

        thought = analysis["thought"] or ""

        # Enforce a single tool call per turn: take only the last tool block
        tool_call: FunctionInvocationInfo | None = None
        tool_info = analysis["tool"]
        if tool_info and tool_info["json_valid"]:
            tool_call = FunctionInvocationInfo(
                name=str(tool_info["name"]),
                arguments=dict(tool_info["arguments"]),
            )

        # Mark completion based on presence of answer
        answer = analysis["answer"]
        return ThoughtSchema(
            thought=thought,
            tool_call=tool_call,
            final_answer=answer,
            completed=answer is not None,
        )

    def add_message(self, role: str, context: list[dict[str, Any]], message: ThoughtSchema | dict[str, str]) -> None:
        if role == "assistant":
            assert isinstance(message, ThoughtSchema)  # noqa: S101
            msg = self.model_config.format_assistant_message(message)
        elif role == "tool":
            assert isinstance(message, dict)  # noqa: S101
            msg = self.model_config.create_tool_message(context, message)
        else:
            raise ValueError(f"Invalid role: {role}")

        assert msg is not None, f"Message is None for role: {role}"  # noqa: S101
        context.append(msg)

    def _calculate_conversation_metrics(
        self, context: list[dict[str, Any]], sample: dict[str, Any], state: EvaluationState
    ) -> None:
        """Calculate conversation-wide metrics using GRPO reward functions."""
        if ground_truth := sample["ground_truth"]:
            # Calculate reward
            gt_parsed = parse_string(ground_truth)
            if gt_parsed is not None:
                _reward, metadata = get_interaction_reward(
                    parser=self._parser, ground_truth=gt_parsed, completion=context
                )
            else:
                _reward, metadata = 0.0, {}

            # Extract metrics from metadata
            state.reward_final_answer = float(metadata.get("reward_final_answer", 0.0))
            state.reward_response_format = float(metadata.get("reward_response_format", 0.0))
            state.reward_interaction = float(_reward)
