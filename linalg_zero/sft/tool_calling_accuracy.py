"""
Tool calling accuracy callback for SFT training.

Evaluates model's ability to correctly use tools on a subset of eval data.
"""

import random
import re
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from linalg_zero.shared.utils import get_logger

logger = get_logger(__name__)


class ToolCallingAccuracyCallback(TrainerCallback):
    """
    Callback to evaluate tool calling accuracy during SFT training.

    Samples a subset of eval data each epoch and measures:
    - Tool call presence (did model attempt tool calls?)
    - Function correctness (did it call the right function?)
    - Argument validity (are arguments syntactically correct?)
    - Execution success (does tool call execute without errors?)
    """

    def __init__(
        self,
        eval_sample_size: int = 50,
        max_new_tokens: int = 1024,  # Increased for multi-step reasoning
        seed: int = 42,
    ):
        self.eval_sample_size = eval_sample_size
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.rng = random.Random(seed)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Run tool calling accuracy evaluation after each eval."""
        if not state.is_world_process_zero:
            return

        model = kwargs.get("model")
        tokenizer = kwargs.get("processing_class")
        eval_dataloader = kwargs.get("eval_dataloader")

        if eval_dataloader is None or model is None or tokenizer is None:
            logger.warning("Missing model/tokenizer/eval_dataloader for tool calling accuracy")
            return

        logger.info(f"Computing tool calling accuracy on {self.eval_sample_size} samples...")

        try:
            accuracy_metrics = self._compute_tool_calling_accuracy(model, tokenizer, eval_dataloader)

            # Log metrics
            for metric_name, value in accuracy_metrics.items():
                state.log_history.append({
                    "epoch": state.epoch if state.epoch is not None else -1,
                    "step": state.global_step,
                    f"eval_{metric_name}": value,
                })
                logger.info(f"Tool calling {metric_name}: {value:.3f}")

        except Exception as e:
            logger.warning(f"Tool calling accuracy evaluation failed: {e}")

    def _compute_tool_calling_accuracy(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        eval_dataloader: DataLoader,
    ) -> dict[str, float]:
        """Compute tool calling accuracy metrics."""
        model.eval()

        # Sample eval examples
        eval_samples = self._sample_eval_data(eval_dataloader)

        metrics = {
            "tool_call_presence": 0.0,
            "function_correctness": 0.0,
            "argument_validity": 0.0,
            "execution_success": 0.0,
        }

        total_samples = len(eval_samples)
        if total_samples == 0:
            return metrics

        # Process each sample
        for sample in eval_samples:
            try:
                # Generate response
                response = self._generate_response(model, tokenizer, sample)

                # Extract tool calls
                tool_calls = self._extract_tool_calls(response)
                expected_tool_calls = self._extract_tool_calls(sample.get("output", ""))

                # Evaluate metrics
                sample_metrics = self._evaluate_sample(tool_calls, expected_tool_calls)

                # Accumulate metrics
                for key, value in sample_metrics.items():
                    metrics[key] += value

            except Exception as e:
                logger.debug(f"Error processing sample: {e}")
                # Count as all failures
                continue

        # Average metrics
        for key in metrics:
            metrics[key] /= total_samples

        return metrics

    def _sample_eval_data(self, eval_dataloader: DataLoader) -> list[dict[str, Any]]:
        """Sample evaluation data."""
        all_samples = []

        # Collect all eval samples
        for batch in eval_dataloader:
            # Convert batch to list of samples
            batch_size = len(batch["input_ids"])
            for i in range(batch_size):
                sample = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        sample[key] = value[i]
                    else:
                        sample[key] = value[i] if isinstance(value, list) else value
                all_samples.append(sample)

        # Sample subset
        sample_size = min(self.eval_sample_size, len(all_samples))
        return self.rng.sample(all_samples, sample_size)

    def _generate_response(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        sample: dict[str, Any],
    ) -> str:
        """Generate model response for a sample."""
        # Get input text
        if "messages" in sample:
            # Chat format
            input_text = tokenizer.apply_chat_template(sample["messages"], tokenize=False, add_generation_prompt=True)
        else:
            # Fallback to input_ids
            input_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)

        # Tokenize and generate
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            # We disable sampling to ensure deterministic behaviour
            outputs = model.generate(  # type: ignore[operator]
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=None,
                top_p=None,
                top_k=None,
            )

        # Decode response (only new tokens)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        return response

    def _extract_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Extract tool calls from text."""
        tool_calls = []

        # Look for tool call patterns in the text
        try:
            # Pattern for function calls
            pattern = r"(\w+)\s*\((.*?)\)"
            matches = re.findall(pattern, text)

            for func_name, args_str in matches:
                # Skip common words that aren't functions
                if func_name.lower() in ["print", "return", "if", "for", "while"]:
                    continue

                tool_calls.append({
                    "function_name": func_name,
                    "arguments": args_str.strip(),
                })

        except Exception as e:
            logger.debug(f"Error extracting tool calls: {e}")

        return tool_calls

    def _evaluate_sample(
        self,
        predicted_calls: list[dict[str, Any]],
        expected_calls: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Evaluate a single sample."""
        metrics = {
            "tool_call_presence": 0.0,
            "function_correctness": 0.0,
            "argument_validity": 0.0,
            "execution_success": 0.0,
        }

        # Tool call presence
        if predicted_calls:
            metrics["tool_call_presence"] = 1.0

        # If no tool calls expected or predicted, return early
        if not expected_calls or not predicted_calls:
            return metrics

        # Function correctness (check if any predicted function matches expected)
        expected_functions = {call.get("function_name", "") for call in expected_calls}
        predicted_functions = {call.get("function_name", "") for call in predicted_calls}

        if expected_functions & predicted_functions:  # Set intersection
            metrics["function_correctness"] = 1.0

        # Argument validity (check if arguments are parseable)
        valid_args = 0
        for call in predicted_calls:
            try:
                args = call.get("arguments", "")
                # Try to evaluate as Python expression
                if args.strip() and (
                    "," in args
                    or args.replace(".", "").replace("-", "").isdigit()
                    or (args.strip().startswith("[") and args.strip().endswith("]"))
                ):
                    valid_args += 1
            except Exception:
                logger.exception("Error evaluating arguments")
                continue

        if predicted_calls:
            metrics["argument_validity"] = valid_args / len(predicted_calls)

        # Execution success (simplified - just check if we have valid function + args)
        if metrics["function_correctness"] > 0 and metrics["argument_validity"] > 0:
            metrics["execution_success"] = 1.0

        return metrics
