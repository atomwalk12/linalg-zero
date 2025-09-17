"""
Tool calling accuracy callback for SFT training.

Evaluates structural and correctness metrics for tool-use generations on a subset of eval data.
"""

from __future__ import annotations

import json as _json
import random
from collections.abc import Callable
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from datasets import Dataset as HFDataset  # type: ignore[attr-defined]
from linalg_zero.grpo.compute_score import get_interaction_reward
from linalg_zero.grpo.verifiers.xml_parser import XMLParser
from linalg_zero.grpo.verify import parse_string
from linalg_zero.sft.tool_evaluation import EvaluationState
from linalg_zero.shared.utils import get_logger

logger = get_logger(__name__)


class ToolCallingAccuracyCallback(TrainerCallback):
    """
    Callback to evaluate tool calling accuracy during SFT training.

    """

    def __init__(
        self,
        eval_dataset: HFDataset,
        library: dict[str, Callable[..., Any]],
        eval_subset: int = 256,
        max_new_tokens: int = 1024,
        seed: int = 42,
        n_turns: int = 4,
    ) -> None:
        self.eval_subset = eval_subset
        self.max_new_tokens = max_new_tokens
        self.n_turns = int(n_turns)
        self.eval_dataset = eval_dataset
        self._eval_indices: list[int] | None = None
        self.rng = random.Random(seed)
        self._parser = XMLParser()

        self.seed = seed
        self.library = library

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

        logger.info(f"Computing tool-calling metrics on subset={self.eval_subset}...")

        model = kwargs.get("model")
        tokenizer = kwargs.get("processing_class")
        if model is None or tokenizer is None:
            return

        try:
            self._ensure_partitions()
            eval_metrics = self._run_unified_evaluation(model, tokenizer)
            if eval_metrics:
                self._log_evaluation_metrics(eval_metrics, state, prefix="eval")
                if metrics is not None:
                    metrics.update(eval_metrics)
        except Exception:
            logger.exception("Tool-calling evaluation failed")

    def _ensure_partitions(self) -> None:
        if self._eval_indices is not None:
            return

        if self.eval_dataset is None:
            self._eval_indices = []
            return

        # Separate single-turn vs multi-turn samples for deterministic mixing
        single_candidates = []
        multi_candidates = []

        for i in range(len(self.eval_dataset)):
            row = self.eval_dataset[int(i)]
            steps = row["stepwise_ground_truths"]
            num_steps = 0
            try:
                if isinstance(steps, str):
                    arr = _json.loads(steps)
                    if isinstance(arr, list):
                        num_steps = len(arr)
                elif isinstance(steps, list):
                    num_steps = len(steps)
            except Exception:
                num_steps = 0

            if num_steps <= 1:
                single_candidates.append(int(i))
            else:
                multi_candidates.append(int(i))

        # Create mixed sample: 70% single (fast), 30% multi (reasoning)
        single_count = min(int(0.7 * self.eval_subset), len(single_candidates))
        multi_count = min(self.eval_subset - single_count, len(multi_candidates))

        # Deterministic sampling using seeded RNG
        selected_single = self.rng.sample(single_candidates, single_count) if single_candidates else []
        selected_multi = self.rng.sample(multi_candidates, multi_count) if multi_candidates else []

        self._eval_indices = selected_single + selected_multi

    def _run_unified_evaluation(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> dict[str, float]:
        """Run unified evaluation on mixed single/multi-turn samples."""
        if self.eval_dataset is None or not self._eval_indices:
            return {}

        return self._compute_metrics(
            model=model,
            tokenizer=tokenizer,
            indices=self._eval_indices,
        )

    def _log_evaluation_metrics(self, metrics: dict[str, float], state: TrainerState, prefix: str = "eval") -> None:
        """Log evaluation metrics to trainer state and logger (Trainer will forward to W&B)."""
        for name, value in metrics.items():
            state.log_history.append({
                "epoch": state.epoch if state.epoch is not None else -1,
                "step": state.global_step,
                f"{prefix}_{name}": float(value),
            })

            metric_name = name if prefix == "eval" else f"multi_{name}"
            logger.info(f"tool_use/{metric_name}: {value:.3f}")

    def _compute_metrics(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        indices: list[int],
    ) -> dict[str, float]:
        """Unified method to compute metrics on selected indices with fair turn allocation."""
        model.eval()

        if not indices:
            return {}

        # Use pre-selected indices (already mixed and sampled)
        samples: list[dict[str, Any]] = [self.eval_dataset[int(i)] for i in indices]

        if not samples:
            return {}

        # Initialize totals
        sum_reward_final = 0.0
        sum_reward_response_format = 0.0
        sum_tool_success = 0
        sum_tool_total = 0

        denom = float(len(samples))
        for sample in samples:
            try:
                # Determine fair n_turns based on sample complexity
                steps = sample.get("stepwise_ground_truths", [])
                num_tool_turns = 0
                try:
                    if isinstance(steps, str):
                        arr = _json.loads(steps)
                        if isinstance(arr, list):
                            num_tool_turns = len(arr)
                    elif isinstance(steps, list):
                        num_tool_turns = len(steps)
                except Exception:
                    num_tool_turns = 0

                # Allocate num_tool_turns + 1 turn for the answer
                n_turns = num_tool_turns + 1

                state = self._evaluate_sample(sample, model, tokenizer, n_turns)
                sum_reward_final += float(state.reward_final_answer)
                sum_reward_response_format += float(state.reward_response_format)
                sum_tool_success += int(state.successful_tool_calls)
                sum_tool_total += int(state.total_tool_calls)
            except Exception:
                logger.debug("Failed evaluating one sample", exc_info=True)
                continue

        tool_success_rate = (sum_tool_success / sum_tool_total) if sum_tool_total > 0 else 0.0
        return {
            "reward_final_answer": (sum_reward_final / denom) if denom > 0 else 0.0,
            "reward_response_format": (sum_reward_response_format / denom) if denom > 0 else 0.0,
            "tool_success_rate": tool_success_rate,
        }

    def _build_evaluation_context(self, sample: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Build minimal context: system (optional) + first user message."""
        messages = sample.get("messages")
        if not isinstance(messages, list):
            return None
        context: list[dict[str, Any]] = []

        # Add system message if present
        system_msgs = self._parser.get_messages(messages, "system")
        if system_msgs:
            context.append(system_msgs[0])

        # Add first user message
        user_msgs = self._parser.get_messages(messages, "user")
        if not user_msgs:
            return None
        context.append(user_msgs[0])
        return context

    def _evaluate_sample(
        self, sample: dict[str, Any], model: PreTrainedModel, tokenizer: PreTrainedTokenizer, n_turns: int = 1
    ) -> EvaluationState:
        """Evaluate a single sample with the model across multiple turns."""
        context = self._build_evaluation_context(sample)
        if context is None:
            return EvaluationState()

        state = self._run_evaluation_turns(context, model, tokenizer, n_turns, sample)
        return state

    def _generate(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt_text: str) -> str:
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
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )

        # Extract only the generated tokens (after the input)
        prompt_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, prompt_length:]

        # Check if generation was truncated due to max_new_tokens
        if (
            generated_tokens.shape[1] == self.max_new_tokens
            and getattr(tokenizer, "eos_token_id", None) is not None
            and generated_tokens[0, -1].item() != tokenizer.eos_token_id
        ):
            logger.warning(f"Generation may have been truncated at max_new_tokens={self.max_new_tokens}")

        return tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def _run_evaluation_turns(
        self,
        context: list[dict[str, Any]],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        n_turns: int,
        sample: dict[str, Any],
    ) -> EvaluationState:
        """Run evaluation using simplified GRPO-based conversation processing."""
        state = EvaluationState()

        # Multi-turn conversation loop
        for _ in range(n_turns):
            # Generate assistant response
            prompt = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
            if not isinstance(prompt, str):
                break

            message = self._generate(model, tokenizer, prompt)
            context.append({"role": "assistant", "content": message})

            # Analyze message using existing GRPO parser
            analysis = self._parser.analyze_message_in_context(
                context,
                message=message,
                tool_names=list(self.library.keys()) if self.library else None,
            )

            # Update minimal state tracking
            if analysis["has_tool_call"]:
                state.total_tool_calls += 1
                # Process tool call
                tool_success = self._process_tool_call(analysis["tool"], context, state, sample)
                if tool_success:
                    state.successful_tool_calls += 1
                else:
                    break

            if analysis["answer"] is not None:
                state.has_final_answer = True
                break

        # Calculate conversation-wide metrics using GRPO components
        self._calculate_conversation_metrics(context, sample, state)
        return state

    def _process_tool_call(
        self, tool_info: dict[str, Any], context: list[dict[str, Any]], state: EvaluationState, sample: dict[str, Any]
    ) -> bool:
        """Process a single tool call and update context. Returns False if conversation should end."""
        # Simple tool execution - let GRPO metrics handle detailed evaluation
        if not tool_info["json_valid"] or not tool_info["name_known"]:
            return False  # Early termination on invalid calls

        try:
            # Execute tool
            tool_name = tool_info["name"]
            tool_args = tool_info["arguments"]
            fn = self.library[tool_name]
            result = fn(**tool_args)
            result_text = f"Tool {tool_name} returned: {result}"
        except Exception as e:
            result_text = f"ERROR executing {tool_info['name']}: {e!s}"
            return False  # Early termination on execution errors

        # Add tool result to conversation
        context.append({"role": "tool", "content": result_text})
        return True

    def _calculate_conversation_metrics(
        self, context: list[dict[str, Any]], sample: dict[str, Any], state: EvaluationState
    ) -> None:
        """Calculate conversation-wide metrics using GRPO reward functions."""
        if ground_truth := sample.get("ground_truth"):
            try:
                # Use GRPO's get_interaction_reward for clean conversation-wide metrics
                gt_parsed = parse_string(ground_truth if isinstance(ground_truth, str) else str(ground_truth))
                if gt_parsed is not None:
                    _reward, metadata = get_interaction_reward(
                        parser=self._parser, ground_truth=gt_parsed, completion=context
                    )
                else:
                    _reward, metadata = 0.0, {}

                # Extract clean metrics from GRPO metadata
                state.reward_final_answer = float(metadata.get("reward_final_answer", 0.0))
                state.reward_response_format = float(metadata.get("reward_response_format", 0.0))

            except Exception as e:
                logger.debug(f"Failed to calculate conversation metrics: {e}", exc_info=True)
                state.reward_final_answer = 0.0
                state.reward_response_format = 0.0
