import ast
from typing import Any

from linalg_zero.generator.models import Question
from linalg_zero.grpo.verify import parse_string, verify_answers
from linalg_zero.shared.utils import get_logger

logger = get_logger(__name__)


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

    for i, question in enumerate(dataset):
        if len(question.stepwise) == 0:
            raise ValueError(f"Question {i} has no stepwise results - implementation is bugged")

        # Verify stepwise results
        for step in question.stepwise:
            if "result" not in step:
                raise ValueError(f"Step {step.get('step_id', 'unknown')} has no result - implementation is bugged")

            result_value = parse_string(step["result"])

            if result_value is None:
                raise ValueError(
                    f"Step {step.get('step_id', 'unknown')}: unparseable result - implementation is bugged"
                )

            verification_results["stepwise_verifications"] += 1

        if not question.golden or "final_answer" not in question.golden:
            raise ValueError(f"Question {i} has no golden final answer - implementation is bugged")

        # Verify golden result against the final stepwise result
        golden_value = parse_string(question.golden["final_answer"])
        answer_value = parse_string(question.stepwise[-1]["result"])

        if golden_value is None:
            raise ValueError(f"Question {i}: unparseable golden answer - implementation is bugged")
        if answer_value is None:
            raise ValueError(f"Question {i}: unparseable formatted answer - implementation is bugged")

        if not verify_answers(golden_value, answer_value):
            raise ValueError(
                f"Question {i}: Golden answer mismatch - implementation is bugged. Golden={golden_value}, Answer={answer_value}"
            )

        # Log successful results
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
