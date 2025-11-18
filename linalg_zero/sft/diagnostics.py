"""Diagnostic metrics tracker for tool calling evaluation."""

from __future__ import annotations

import json as _json
from typing import Any

from linalg_zero.grpo.verifiers.xml_parser import XMLParser
from linalg_zero.grpo.verify import parse_string, verify_answers
from linalg_zero.sft.tool_evaluation import EvaluationState
from linalg_zero.shared.lib import get_lib_fn_names


class DiagnosticTracker:
    """Tracks messages and samples for evaluation logging."""

    def __init__(self) -> None:
        # Store all messages and samples for Weave logging
        self.all_messages: list[list[dict[str, Any]]] = []
        self.all_samples: list[dict[str, Any]] = []
        self.sum_strict_format = 0.0
        self.sum_partial_format = 0.0
        self.sum_turns_taken = 0.0

    def update(self, state: EvaluationState) -> None:
        """Update tracker from an evaluation state."""
        self.sum_strict_format += state.strict_format_match
        self.sum_partial_format += state.partial_format_score
        self.sum_turns_taken += state.turns_taken
        # Store messages and sample from this evaluation
        self.all_messages.append(state.messages)
        if state.sample is not None:
            self.all_samples.append(state.sample)

    def get_history(self) -> tuple[list[list[dict[str, Any]]], dict[str, int]]:
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
        total_samples = len(self.all_samples)
        if total_samples == 0:
            return {
                "partial": "0.000",
                "strict": "0.000",
                "turns": "0.0",
                "correct": "0.000",
            }

        partial_format = self.sum_partial_format / total_samples
        strict_format = self.sum_strict_format / total_samples
        avg_turns = self.sum_turns_taken / total_samples

        return {
            "turns": f"{avg_turns:.1f}",
            "strict": f"{strict_format:.3f}",
            "partial": f"{partial_format:.3f}",
        }
