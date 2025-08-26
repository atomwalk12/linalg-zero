from linalg_zero.generator.models import Question
from linalg_zero.shared.utils import get_logger

logger = get_logger(__name__)


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
    logger.info("  Topics: %s", ", ".join(sorted(topics)))
    logger.info("  Problem Types: %s", ", ".join(sorted(problem_types)))
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
        logger.info("A: %s", question.answer)
        logger.info("")
