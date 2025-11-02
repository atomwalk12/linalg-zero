import re
from typing import Any

# Define the regex patterns and global variables needed
match_format = re.compile(r"<solution>(.*?)</solution>")
match_numbers = re.compile(r"<answer>(.*?)</answer>")
reasoning_end = "</working_out>"
solution_start = "<solution>"
solution_end = "</solution>"

PRINTED_TIMES = 0
PRINT_EVERY_STEPS = 10


def match_format_exactly(completions: list[list[dict[str, Any]]], **kwargs: Any) -> list[float]:
    scores: list[float] = []
    for completion in completions:
        score: float = 0.0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None:
            score += 3.0
        scores.append(score)
    return scores


def match_format_approximately(completions: list[list[dict[str, Any]]], **kwargs: Any) -> list[float]:
    scores: list[float] = []
    for completion in completions:
        score: float = 0.0
        response = completion[0]["content"]
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!

        # No need to reward <start_working_out> since we always prepend it!
        # score += 0.5 if response.count(reasoning_start) == 1 else -1.0
        score += 0.5 if response.count(reasoning_end) == 1 else -1.0
        score += 0.5 if response.count(solution_start) == 1 else -1.0
        score += 0.5 if response.count(solution_end) == 1 else -1.0
        scores.append(score)
    return scores


def check_numbers(
    prompts: list[list[dict[str, Any]]],
    completions: list[list[dict[str, Any]]],
    answer: list[str],
    **kwargs: Any,
) -> list[float]:
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1) if (guess := match_numbers.search(r)) is not None else None for r in responses
    ]

    scores: list[float] = []
    # Print only every few steps
    global PRINTED_TIMES
    global PRINT_EVERY_STEPS
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print(
            "*" * 20 + f"Question:\n{question}",
            f"\nAnswer:\n{answer[0]}",
            f"\nResponse:\n{responses[0]}",
            f"\nExtracted:\n{extracted_responses[0]}",
        )
    PRINTED_TIMES += 1

    for guess, true_answer in zip(extracted_responses, answer, strict=False):
        if guess is None:
            scores.append(-2.5)
            continue
        # Convert to numbers
        try:
            true_answer_float = float(true_answer.strip())
            # Remove commas like in 123,456
            guess_float = float(guess.strip().replace(",", ""))
            scores.append(3.5 if guess_float == true_answer_float else -1.5)
        except Exception:
            scores.append(0.0)
            continue
    return scores


def check_answer(
    prompts: list[list[dict[str, Any]]],
    completions: list[list[dict[str, Any]]],
    answer: list[str],
    **kwargs: Any,
) -> list[float]:
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1) if (guess := match_format.search(r)) is not None else None for r in responses
    ]

    scores: list[float] = []
    for guess, true_answer in zip(extracted_responses, answer, strict=False):
        score: float = 0.0
        if guess is None:
            scores.append(-2.0)
            continue
        # Correct answer gets 5 points!
        if guess == true_answer:
            score += 5.0
        # Match if spaces are seen, but less reward
        elif guess.strip() == true_answer.strip():
            score += 3.5
        else:
            # We also reward it if the answer is close via ratios!
            # Ie if the answer is within some range, reward it!
            try:
                ratio = float(guess) / float(true_answer)
                if ratio >= 0.9 and ratio <= 1.1:
                    score += 2.0
                elif ratio >= 0.8 and ratio <= 1.2:
                    score += 1.5
                else:
                    score -= 2.5  # Penalize wrong answers
            except Exception:
                score -= 4.5  # Penalize
        scores.append(score)
    return scores
