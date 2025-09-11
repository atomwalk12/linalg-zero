"""
Tool calling accuracy callback for SFT training.

Evaluates structural and correctness metrics for tool-use generations on a subset of eval data.
"""

from __future__ import annotations

import json
import random
import uuid
from collections.abc import Callable
from typing import Any

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from datasets import Dataset as HFDataset  # type: ignore[attr-defined]
from linalg_zero.grpo.verifiers.xml_parser import (
    XMLDiagnostics,
    XMLParser,
    analyze_message_in_context,
)
from linalg_zero.grpo.verify import parse_string, verify_answers
from linalg_zero.sft.tool_evaluation import EvaluationState
from linalg_zero.shared.system_prompts import THINK_CLOSE, THINK_OPEN
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
        multi_turn_subset: int = 64,
        max_new_tokens: int = 1024,
        seed: int = 42,
        eval_mode: str = "auto",  # 'subset' | 'full' | 'auto'
        full_eval_every_n_evals: int = 5,
        n_turns: int = 4,
    ) -> None:
        self.single_turn_subset = eval_subset
        self.multi_turn_subset = int(multi_turn_subset)
        self.full_eval_every_n_evals = max(1, int(full_eval_every_n_evals))
        self.max_new_tokens = max_new_tokens

        self.n_turns = int(n_turns)
        self.eval_mode = eval_mode
        self.eval_dataset = eval_dataset
        self._single_indices: list[int] | None = None
        self._multi_indices: list[int] | None = None
        self.rng = random.Random(seed)
        self._parser = XMLParser()
        self._diag = XMLDiagnostics(self._parser)
        self._eval_call_count: int = 0

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

        logger.info(
            f"Evaluation configuration: single_turn_subset={self.single_turn_subset}, "
            f"multi_turn_subset={self.multi_turn_subset}, "
            f"full_eval_every_n_evals={self.full_eval_every_n_evals}, "
            f"max_new_tokens={self.max_new_tokens}"
        )

        model = kwargs.get("model")
        tokenizer = kwargs.get("processing_class")
        if model is None or tokenizer is None:
            return

        do_full = self._determine_evaluation_scope(args)
        target = "full" if do_full else f"subset={self.single_turn_subset}"
        logger.info(f"Computing tool-calling metrics on {target}...")

        try:
            self._ensure_partitions()
            single = self._run_single_turn_evaluation(model, tokenizer, do_full, state)
            if metrics is not None and single:
                metrics.update(single)

            multi = self._run_multi_turn_evaluation_if_needed(model, tokenizer, state)
            if metrics is not None and multi:
                metrics.update({f"multi_{k}": v for k, v in multi.items()})
        except Exception:
            logger.exception("Tool-calling evaluation failed")

    def _determine_evaluation_scope(self, args: TrainingArguments) -> bool:
        """Determine whether to do full evaluation or subset based on mode and strategy."""
        if self.eval_mode == "full":
            return True
        elif self.eval_mode == "subset":
            return False
        else:
            # auto: honor strategy; for steps do subset, and promote to full every N evals
            eval_strategy = getattr(args, "eval_strategy", getattr(args, "evaluation_strategy", None))
            if str(eval_strategy) == "epoch":
                return True
            else:
                self._eval_call_count += 1
                return (self._eval_call_count % self.full_eval_every_n_evals) == 0

    def _run_single_turn_evaluation(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, do_full: bool, state: TrainerState
    ) -> dict[str, float]:
        """Run single-turn evaluation and log results, returning metrics."""
        if self.eval_dataset is None or not self._single_indices:
            return {}

        # Single-turn problems need 2 turns (think + tool/answer)
        metrics = self._compute_metrics(
            model=model,
            tokenizer=tokenizer,
            indices=self._single_indices,
            sample_size=self.single_turn_subset,
            n_turns=2,
            do_full=do_full,
        )
        self._log_evaluation_metrics(metrics, state, prefix="eval")
        return metrics

    def _run_multi_turn_evaluation_if_needed(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, state: TrainerState
    ) -> dict[str, float]:
        """Run multi-turn evaluation if conditions are met and log results, returning metrics."""
        if not (self._multi_indices and (self._eval_call_count % self.full_eval_every_n_evals == 0)):
            return {}

        if self.eval_dataset is None:
            return {}

        # Multi-turn always uses subset for performance
        mt_metrics = self._compute_metrics(
            model=model,
            tokenizer=tokenizer,
            indices=self._multi_indices,
            sample_size=self.multi_turn_subset,
            n_turns=self.n_turns,
            do_full=False,
        )
        self._log_evaluation_metrics(mt_metrics, state, prefix="eval_multi")
        return mt_metrics

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
        sample_size: int,
        n_turns: int,
        do_full: bool = False,
    ) -> dict[str, float]:
        """Unified method to compute metrics on any set of indices with configurable parameters."""
        model.eval()

        if not indices:
            return {}

        n = len(indices)
        if do_full:
            selected_indices = list(indices)
        else:
            local_rng = random.Random(self.seed)
            pick = local_rng.sample(indices, k=min(sample_size, n))
            selected_indices = list(pick)

        samples: list[dict[str, Any]] = [self.eval_dataset[int(i)] for i in selected_indices]

        if not samples:
            return {}

        # Initialize totals with all unified metrics
        totals = self.init_metrics()

        denom = float(len(samples))
        for sample in samples:
            try:
                metrics = self._evaluate_sample(sample, model, tokenizer, n_turns)
                for k, v in metrics.items():
                    totals[k] += v
            except Exception:
                logger.debug("Failed evaluating one sample", exc_info=True)
                continue

        return {k: (v / denom if denom > 0 else 0.0) for k, v in totals.items()}

    def _build_evaluation_context(self, sample: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Build minimal context: system (optional) + first user message."""
        messages = sample.get("messages", [])
        context: list[dict[str, Any]] = []
        if messages and messages[0].get("role") == "system":
            context.append(messages[0])

        user_msg = next((m for m in messages if m.get("role") == "user"), None)
        if user_msg is None:
            return None
        context.append(user_msg)
        return context

    def _run_evaluation_turns(  # noqa: C901
        self,
        context: list[dict[str, Any]],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        n_turns: int,
        sample: dict[str, Any],
    ) -> EvaluationState:
        """Run the evaluation loop across multiple turns, preserving exact original logic."""
        state = EvaluationState()
        parser = self._parser

        for turn in range(n_turns):
            state.turns_used = turn + 1

            # Generate assistant response
            prompt = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
            if not isinstance(prompt, str):
                state.malformed_turns += 1
                state.overall_format_valid = False
                break

            message = self._generate(model, tokenizer, prompt)

            analysis = analyze_message_in_context(
                parser,
                context,
                message=message,
                tool_names=set(self.library.keys()) if self.library else None,
            )

            # Extract analyzed properties
            answer = analysis["answer"]
            has_tool_call = analysis["has_tool_call"]
            tool_info = analysis["tool"]
            turn_format_valid = analysis["is_valid_think_then_tool_or_answer"]

            if not turn_format_valid:
                state.malformed_turns += 1
                state.overall_format_valid = False
                break

            if has_tool_call:
                state.has_any_tool_call = True
            if answer is not None:
                state.has_final_answer = True

            # If we have a final answer, validate it and finish
            if answer is not None:
                # Enforce adjacency policy: answer must follow a tool message
                if not bool(analysis["answer_policy_valid"]):
                    state.malformed_turns += 1
                    state.overall_format_valid = False
                    break
                if (gt := sample["ground_truth"]) is not None:
                    try:
                        gt_parsed = parse_string(gt if isinstance(gt, str) else str(gt))
                        ans_parsed = parse_string(answer)
                        state.final_answer_correct = bool(verify_answers(gt_parsed, ans_parsed))
                    except Exception:
                        state.final_answer_correct = False
                break

            # Process tool call using analysis results
            if has_tool_call:
                state.total_tool_calls += 1

                # Update tool validity tracking from analysis
                if not tool_info["json_valid"]:
                    state.all_tools_json_valid = False
                    result_text = "ERROR: Invalid JSON in tool call"
                else:
                    # Tool name validation from analysis
                    if tool_info["name_known"] is False:
                        state.all_tools_known = False
                        tool_name = tool_info["name"]
                        result_text = f"ERROR: Function '{tool_name}' not available"
                    else:
                        # Execute valid tool
                        tool_name = tool_info["name"]
                        tool_args = tool_info["arguments"]
                        try:
                            result = self.library[tool_name](**tool_args)
                            result_text = str(result)
                        except Exception as exc:
                            state.all_tools_exec_success = False
                            result_text = f"ERROR: {type(exc).__name__}: {exc}"

                # Terminate early on invalid/unknown/failed tool calls to avoid deterministic repeats
                json_valid = bool(tool_info.get("json_valid"))
                name_known = tool_info.get("name_known") is not False
                if (not json_valid) or (not name_known) or ("ERROR:" in result_text):
                    break

                # Append assistant tool_call message and the corresponding tool response
                tool_name = str(tool_info.get("name", ""))
                tool_args = tool_info.get("arguments", {})
                tool_call_id = str(uuid.uuid4())

                # Assistant message with think + tool_calls
                think_text = analysis.get("thought") or ""
                assistant_msg = {
                    "role": "assistant",
                    "content": f"{THINK_OPEN}{think_text}{THINK_CLOSE}",
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args),
                            },
                        }
                    ],
                }
                context.append(assistant_msg)

                # Tool response message
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": result_text,
                }
                context.append(tool_msg)

        return state

    def _evaluate_sample(
        self, sample: dict[str, Any], model: PreTrainedModel, tokenizer: PreTrainedTokenizer, n_turns: int = 1
    ) -> dict[str, float]:
        """Evaluate a single sample with the model across multiple turns."""
        context = self._build_evaluation_context(sample)
        if context is None:
            return self.init_metrics()

        state = self._run_evaluation_turns(context, model, tokenizer, n_turns, sample)
        return self.init_metrics(state)

    def _generate(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt_text: str) -> str:
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True)
        if inputs["input_ids"].shape[1] == tokenizer.model_max_length:
            logger.warning(f"Input truncated to {tokenizer.model_max_length} tokens during tool calling evaluation")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(  # type: ignore[operator]
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]

        # Check if generation was truncated due to max_new_tokens
        if len(generated_tokens) == self.max_new_tokens and generated_tokens[-1] != tokenizer.eos_token_id:
            logger.warning(f"Generation may have been truncated at max_new_tokens={self.max_new_tokens}")

        return tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def _ensure_partitions(self) -> None:
        if self._single_indices is not None and self._multi_indices is not None:
            return
        self._single_indices = []
        self._multi_indices = []
        if self.eval_dataset is None:
            return
        for i in range(len(self.eval_dataset)):
            row = self.eval_dataset[int(i)]
            steps = row.get("stepwise_ground_truths")
            num_steps = 0
            try:
                if isinstance(steps, str):
                    import json as _json

                    arr = _json.loads(steps)
                    if isinstance(arr, list):
                        num_steps = len(arr)
                elif isinstance(steps, list):
                    num_steps = len(steps)
            except Exception:
                num_steps = 0
            if num_steps <= 1:
                self._single_indices.append(int(i))
            else:
                self._multi_indices.append(int(i))

    def init_metrics(self, state: EvaluationState | None = None) -> dict[str, float]:
        """Compute final metrics from evaluation state. Returns empty metrics if state is None."""
        if state is None:
            return {
                "format_valid_rate": 0.0,
                "tool_call_presence_rate": 0.0,
                "answer_presence_rate": 0.0,
                "tool_json_valid_rate": 0.0,
                "tool_name_known_rate": 0.0,
                "tool_execution_success_rate": 0.0,
                "answer_correct_rate": 0.0,
                "turns_to_success_avg": 0.0,
                "tool_calls_total_avg": 0.0,
                "malformed_turns_avg": 0.0,
            }

        return {
            "format_valid_rate": 1.0 if state.overall_format_valid else 0.0,
            "tool_call_presence_rate": 1.0 if state.has_any_tool_call else 0.0,
            "answer_presence_rate": 1.0 if state.has_final_answer else 0.0,
            "tool_json_valid_rate": 1.0 if (not state.has_any_tool_call or state.all_tools_json_valid) else 0.0,
            "tool_name_known_rate": 1.0 if (not state.has_any_tool_call or state.all_tools_known) else 0.0,
            "tool_execution_success_rate": 1.0
            if (not state.has_any_tool_call or state.all_tools_exec_success)
            else 0.0,
            "answer_correct_rate": 1.0 if state.final_answer_correct else 0.0,
            "turns_to_success_avg": float(state.turns_used) if state.final_answer_correct else 0.0,
            "tool_calls_total_avg": float(state.total_tool_calls),
            "malformed_turns_avg": float(state.malformed_turns),
        }
