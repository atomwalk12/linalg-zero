import ast
import json
from typing import Any

from linalg_zero.generator.models import Question
from linalg_zero.grpo.verify import parse_string, verify_answers
from linalg_zero.shared.lib import get_lib
from linalg_zero.shared.utils import get_logger

logger = get_logger(__name__)


def _verify_step_result(step: dict[str, Any], lib: dict[str, Any]) -> None:
    """Verify a single step's result against library function output."""
    step_id = step["step_id"]

    if "result" not in step:
        raise ValueError(f"Step {step_id} has no result - implementation is bugged")

    result_value = parse_string(step["result"])
    if result_value is None:
        raise ValueError(f"Step {step_id}: invalid result - implementation is bugged")

    fn_type = step["tool"]
    lib_fn = lib[fn_type]
    input_data = json.loads(step["verification"]["input"])
    fn_result = lib_fn(**input_data)

    if not verify_answers(result_value, fn_result):
        raise ValueError(f"Step mismatch - step - {json.dumps(step)} - lib_fn - {fn_type}")


def _verify_step_dependencies(step: dict[str, Any], question_stepwise: list[dict[str, Any]]) -> None:
    """Verify step dependencies against referenced steps."""
    step_id = step.get("step_id", "unknown")
    dependent_on = step["verification"].get("dependent_on", None)

    if dependent_on is None:
        return

    if not isinstance(dependent_on, dict):
        raise TypeError(f"Step {step_id}: dependent_on must be a dict, got {type(dependent_on)}")

    # Verify each input_* field against its corresponding referenced step's result
    for field_name, field_value_json in step["verification"].items():
        if field_name.startswith("input_"):
            expected_step_index = dependent_on[field_name]

            # Validate the reference step exists
            if not isinstance(expected_step_index, int):
                raise TypeError(
                    f"Step {step_id}: dependency index for '{field_name}' must be an integer, got {type(expected_step_index)}"
                )

            if expected_step_index < 0 or expected_step_index >= len(question_stepwise):
                raise ValueError(
                    f"Step {step_id}: dependent_on index {expected_step_index} for '{field_name}' out of bounds "
                    f"(stepwise has {len(question_stepwise)} steps)"
                )

            referenced_step = question_stepwise[expected_step_index]
            referenced_result = parse_string(referenced_step["result"])

            if referenced_result is None:
                raise ValueError(f"Step {step_id}: referenced step {expected_step_index} has invalid result")

            field_value = json.loads(field_value_json)
            if not verify_answers(field_value, referenced_result) or field_value != referenced_result:
                raise ValueError(
                    f"Step {step_id}: dependency verification failed - "
                    f"{field_name} ({field_value}) does not match referenced step {expected_step_index} result ({referenced_result})"
                )


def _verify_golden_answer(question: Question, question_index: int) -> None:
    """Verify the golden answer matches the final stepwise result."""
    if not question.golden or "final_answer" not in question.golden:
        raise ValueError(f"Question {question_index} has no golden final answer - implementation is bugged")

    golden_value = parse_string(question.golden["final_answer"])
    answer_value = parse_string(question.stepwise[-1]["result"])

    if golden_value is None:
        raise ValueError(f"Question {question_index}: invalid golden answer - implementation is bugged")
    if answer_value is None:
        raise ValueError(f"Question {question_index}: invalid formatted answer - implementation is bugged")

    if not verify_answers(golden_value, answer_value):
        raise ValueError(
            f"Question {question_index}: Golden answer mismatch - implementation is bugged. "
            f"Golden={golden_value}, Answer={answer_value}"
        )


def verify_dataset(dataset: list[Question]) -> dict[str, Any]:
    """
    Verify a dataset of questions by checking constituent ground truths and target values.
    """
    # NOTE: this function is temporary

    verification_results = {
        "total_questions": len(dataset),
        "verified_questions": 0,
        "stepwise_verifications": 0,
        "golden_verifications": 0,
    }
    lib = get_lib()

    for i, question in enumerate(dataset):
        if len(question.stepwise) == 0:
            raise ValueError(f"Question {i} has no stepwise results - implementation is bugged")

        # Verify stepwise results
        for step in question.stepwise:
            _verify_step_result(step, lib)
            _verify_step_dependencies(step, question.stepwise)
            verification_results["stepwise_verifications"] += 1

        # Verify golden answer
        _verify_golden_answer(question, i)
        verification_results["golden_verifications"] += 1
        verification_results["verified_questions"] += 1

    logger.info(
        "Dataset verification complete: All %d questions verified successfully (%d stepwise checks, %d golden checks)",
        verification_results["total_questions"],
        verification_results["stepwise_verifications"],
        verification_results["golden_verifications"],
    )

    return verification_results


def print_dataset(questions: list[Question], include_invalid: bool = False) -> None:  # pragma: no cover
    """Display a formatted dataset of questions."""

    questions_to_print = questions if include_invalid else [q for q in questions if q.is_valid]

    if not questions_to_print:
        logger.info("No questions to display.")
        return

    logger.info("=" * 30)
    logger.info("GENERATED DATASET")
    logger.info("=" * 30)

    # Metadata
    topics = {q.topic for q in questions_to_print}
    problem_types = {q.problem_type for q in questions_to_print}
    difficulties = {q.difficulty for q in questions_to_print}
    entropy_values = [q.entropy_used for q in questions_to_print]
    tool_calls = [q.tool_calls_required for q in questions_to_print]

    # Summary
    logger.info("Dataset Summary:")
    logger.info("  Total Questions: %d", len(questions_to_print))
    logger.info("  Topics: %s", ", ".join(sorted(topic.value for topic in topics)))
    logger.info("  Problem Types: %s", ", ".join(sorted(pt.value for pt in problem_types)))
    logger.info("  Difficulties: %s", ", ".join(sorted(str(difficulty) for difficulty in difficulties)))
    logger.info(
        "  Entropy Used: %.2f - %.2f (avg: %.2f)",
        min(entropy_values),
        max(entropy_values),
        sum(entropy_values) / len(entropy_values),
    )
    logger.info(
        "  Tool Calls Required: %d - %d (avg: %.1f)",
        min(tool_calls),
        max(tool_calls),
        sum(tool_calls) / len(tool_calls),
    )
    logger.info("=" * 30)

    # Questions
    for i, question in enumerate(questions_to_print, 1):
        status = " [INVALID]" if not question.is_valid else ""
        logger.info("Question %d:%s", i, status)
        logger.info("Q: %s", question.question)
        logger.info("A: %s", ast.literal_eval(question.answer))
        logger.info("")
