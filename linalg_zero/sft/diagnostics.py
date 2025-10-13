"""Diagnostic metrics tracker for tool calling evaluation."""

from __future__ import annotations

from linalg_zero.sft.tool_evaluation import EvaluationState


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
        self.sum_format_valid = 0
        self.sum_tool_parse_success = 0
        self.sum_answer_attempted = 0
        self.sum_turns_taken = 0
        self.early_stop_reasons: dict[str, int] = {}
        self.total_samples = 0

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
        self.sum_format_valid += int(state.format_valid)
        self.sum_tool_parse_success += int(state.tool_parse_success)
        self.sum_answer_attempted += int(state.answer_attempted)
        self.sum_turns_taken += state.turns_taken
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
            "diagnostic/format_valid_ratio": self.sum_format_valid / denom,
            "diagnostic/tool_parse_success_ratio": self.sum_tool_parse_success / denom,
            "diagnostic/answer_attempted_ratio": self.sum_answer_attempted / denom,
            "diagnostic/avg_turns_taken": self.sum_turns_taken / denom,
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

    def get_progress_info(self) -> dict[str, str]:
        """Return current progress info for progress bar (simplified metrics)."""
        if self.total_samples == 0:
            return {
                "correctness": "0.000",
                "format_valid": "0.000",
            }

        correctness = self.sum_reward_final / self.total_samples
        format_valid = self.sum_format_valid / self.total_samples

        return {
            "correctness": f"{correctness:.3f}",
            "format_valid": f"{format_valid:.3f}",
        }
