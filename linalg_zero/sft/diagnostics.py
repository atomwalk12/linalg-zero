"""Diagnostic metrics tracker for tool calling evaluation."""

from __future__ import annotations

import json as _json
from typing import Any

from linalg_zero.grpo.verifiers.xml_parser import XMLParser
from linalg_zero.grpo.verify import parse_string, verify_answers
from linalg_zero.sft.tool_evaluation import EvaluationState
from linalg_zero.shared.lib import get_lib_fn_names


class DiagnosticTracker:
    """Tracks both primary reward metrics and diagnostic metrics during evaluation."""

    def __init__(self) -> None:
        # Primary reward metrics
        self.sum_reward_final = 0.0
        self.sum_reward_response_format = 0.0
        self.sum_reward_interaction = 0.0
        self.per_sample_final: list[float] = []
        self.per_sample_format: list[float] = []
        self.per_sample_interaction: list[float] = []

        # Diagnostic metrics
        self.sum_strict_format = 0
        self.sum_tool_parse_success = 0
        self.sum_answer_attempted = 0
        self.sum_turns_taken = 0
        self.sum_partial_format = 0.0
        self.early_stop_reasons: dict[str, int] = {}
        self.total_samples = 0

        # Store all messages and samples for Weave logging
        self.all_messages: list[list[dict[str, Any]]] = []
        self.all_samples: list[dict[str, Any]] = []

    def update(self, state: EvaluationState) -> None:
        """Update all counters from an evaluation state."""
        # Primary reward metrics
        reward_final = float(state.reward_final_answer)
        reward_format = float(state.reward_response_format)
        reward_interaction = float(state.reward_interaction)

        self.sum_reward_final += reward_final
        self.sum_reward_response_format += reward_format
        self.sum_reward_interaction += reward_interaction
        self.per_sample_final.append(reward_final)
        self.per_sample_format.append(reward_format)
        self.per_sample_interaction.append(reward_interaction)

        # Diagnostic metrics
        self.sum_strict_format += state.strict_format_match
        self.sum_partial_format += state.partial_format_score
        self.sum_tool_parse_success += int(state.tool_parse_success)
        self.sum_answer_attempted += int(state.answer_attempted)
        self.sum_turns_taken += state.turns_taken

        # Store messages and sample from this evaluation
        self.all_messages.append(state.messages)
        if state.sample is not None:
            self.all_samples.append(state.sample)

        self.total_samples += 1

        if state.early_stop_reason:
            self.early_stop_reasons[state.early_stop_reason] = (
                self.early_stop_reasons.get(state.early_stop_reason, 0) + 1
            )

    def get_aggregated_metrics(self) -> dict[str, float]:
        """Return all aggregated metrics (primary reward metrics + diagnostic metrics)."""
        if self.total_samples == 0:
            return {}

        denom = float(self.total_samples)

        metrics = {
            "reward_final_answer": self.sum_reward_final / denom,
            "reward_response_format": self.sum_reward_response_format / denom,
            "reward_interaction": self.sum_reward_interaction / denom,
        }

        metrics.update({
            "diagnostic/format_valid_ratio": self.sum_strict_format / denom,
            "diagnostic/tool_parse_success_ratio": self.sum_tool_parse_success / denom,
            "diagnostic/answer_attempted_ratio": self.sum_answer_attempted / denom,
            "diagnostic/avg_turns_taken": self.sum_turns_taken / denom,
            "diagnostic/partial_format_adherence": self.sum_partial_format / denom,
        })

        for reason, count in self.early_stop_reasons.items():
            metric_name = f"diagnostic/stop_reason/{reason}"
            metrics[metric_name] = count / denom

        return metrics

    def get_per_sample_rewards(self) -> dict[str, list[float]]:
        """Return per-sample reward lists for W&B logging."""
        return {
            "reward_final_answer": self.per_sample_final,
            "reward_response_format": self.per_sample_format,
            "reward_interaction": self.per_sample_interaction,
        }

    def get_all_messages(self) -> tuple[list[list[dict[str, Any]]], dict[str, int]]:
        """Return all messages and computed metadata for Weave logging."""
        metadata = self._compute_metadata()
        return self.all_messages, metadata

    def _compute_metadata(self) -> dict[str, int]:
        """Compute metadata statistics from messages and samples."""
        parser = XMLParser()
        tool_names = get_lib_fn_names()

        total_samples = len(self.all_samples)
        total_expected_tool_calls = 0
        total_actual_tool_calls = 0
        total_expected_answers = total_samples  # One answer per sample
        total_actual_answers = 0
        total_correct_answers = 0

        for messages, sample in zip(self.all_messages, self.all_samples, strict=True):
            # Calculate expected tool calls from ground truth
            if "stepwise_ground_truths" in sample:
                ground_truths = _json.loads(sample["stepwise_ground_truths"])
                total_expected_tool_calls += len(ground_truths)

            # Track the last answer found in this conversation
            last_answer = None

            # Count actual tool calls and answers from assistant messages
            # Skip first two messages (system and user)
            for message in messages[2:]:
                if message.get("role") != "assistant":
                    continue

                content = message.get("content", "")
                analysis = parser.analyze_message_in_context(
                    messages[: messages.index(message) + 1], message=content, tool_names=tool_names
                )

                # Count tool calls
                if analysis.get("tool") and analysis["tool"].get("json_valid"):
                    total_actual_tool_calls += 1

                # Count answers and track the last one
                if analysis.get("has_answer"):
                    total_actual_answers += 1
                    last_answer = analysis.get("answer")

            # Check if the last answer is correct
            if last_answer is not None and "ground_truth" in sample:
                ground_truth = sample["ground_truth"]
                parsed_gt = parse_string(ground_truth)
                parsed_answer = parse_string(last_answer)

                if verify_answers(parsed_gt, parsed_answer):
                    total_correct_answers += 1

        return {
            "total_samples": total_samples,
            "total_expected_tool_calls": total_expected_tool_calls,
            "total_actual_tool_calls": total_actual_tool_calls,
            "total_expected_answers": total_expected_answers,
            "total_actual_answers": total_actual_answers,
            "total_correct_answers": total_correct_answers,
        }

    def get_progress_info(self) -> dict[str, str]:
        """Return current progress info for progress bar (4 key metrics)."""
        if self.total_samples == 0:
            return {
                "partial": "0.000",
                "strict": "0.000",
                "turns": "0.0",
                "correct": "0.000",
            }

        partial_format = self.sum_partial_format / self.total_samples
        strict_format = self.sum_strict_format / self.total_samples
        avg_turns = self.sum_turns_taken / self.total_samples
        correctness = self.sum_reward_final / self.total_samples

        return {
            "turns": f"{avg_turns:.1f}",
            "strict": f"{strict_format:.3f}",
            "partial": f"{partial_format:.3f}",
            "correct": f"{correctness:.3f}",
        }
