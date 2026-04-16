import json
import re
from io import StringIO
from pathlib import Path
from typing import Any

from datasets import Dataset, DownloadMode, load_dataset, load_from_disk
from linalg_zero.grpo.verifiers.xml_parser import XMLParser
from linalg_zero.grpo.verify import parse_string, verify_answers
from transformers import AutoTokenizer

DISTILLED_DATASET = "atomwalk12/linalgzero-distilled"
LOCAL_DATASET_PATH = "atomwalk12/linalgzero-distilled-local"

output_buffer = StringIO()


def parse_messages(messages_field: Any) -> list[dict]:
    return json.loads(messages_field) if isinstance(messages_field, str) else messages_field


def parse_stepwise_ground_truths(stepwise_ground_truths_field: Any) -> list[dict]:
    if isinstance(stepwise_ground_truths_field, str):
        return json.loads(stepwise_ground_truths_field)
    return stepwise_ground_truths_field


def get_message_content(message: dict) -> str:
    return (message.get("content") or "").strip()


def load_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(tokenizer_name)


def count_tokens_in_message(content: str, tokenizer: AutoTokenizer) -> int:
    return len(tokenizer.encode(content))


def get_max_assistant_tokens(messages: list[dict], tokenizer: AutoTokenizer) -> int:
    assistant_token_counts = []
    for msg in messages:
        if msg.get("role") == "assistant":
            token_count = count_tokens_in_message(get_message_content(msg), tokenizer)
            assistant_token_counts.append(token_count)

    return max(assistant_token_counts) if assistant_token_counts else 0


def print_to_both(*args, **kwargs) -> None:
    print(*args, **kwargs)
    print(*args, **kwargs, file=output_buffer)


def print_messages(messages: list[dict]) -> None:
    for i, msg in enumerate(messages):
        print_to_both(f"\n--- Message {i} ({msg['role']}) ---")

        if msg["role"] == "user":
            print_to_both(msg["content"])
        elif msg["role"] == "assistant":
            print_to_both(msg["content"])

            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                print_to_both("\nTOOL CALLS:")
                for tool_call in tool_calls:
                    function = tool_call["function"]
                    print_to_both(f"  - {function['name']}: {function['arguments']}")
        elif msg["role"] == "tool":
            print_to_both(f"RESULT: {msg['content']}")

    print_to_both("\n")


def ensure_local_dataset(dataset_name: str, config: str, split: str, local_path: str) -> None:
    if Path(local_path).exists():
        return

    dataset = load_dataset(dataset_name, config, split=split)
    dataset.save_to_disk(local_path)


def check_tool_call_is_skipped_due_to_simple_op(
    messages: list[dict], parser: XMLParser, minimal_dataset: bool = False
) -> tuple[bool, str]:
    answer_msg = messages[-1]
    first_to_last_msg = messages[-2]
    if first_to_last_msg["role"] != "tool":
        return False, "No tool response before final answer"

    answer = parse_string(parser.extract_last_answer(answer_msg["content"]))
    tool_response = parse_string(first_to_last_msg["content"])

    if verify_answers(tool_response, answer):
        return False, "Answer matches tool response"

    if first_to_last_msg["name"] == "determinant":
        if answer in [1, 2, 3] and not minimal_dataset:
            return False, ""
        return (
            True,
            "Interesting case: using determinant as the last tool call, but final answer is not a possible rank",
        )

    return (
        True,
        f"Answer does not match tool response: {answer} (final answer) != {tool_response} "
        f"(final tool response -- {first_to_last_msg['name']})\n",
    )


def check_repeated_tool_calls(messages: list[dict]) -> tuple[bool, list[dict]]:
    tool_calls: dict[str, list[int]] = {}

    for i, msg in enumerate(messages):
        assistant_tool_calls = msg.get("tool_calls") or []
        if msg["role"] == "assistant" and assistant_tool_calls:
            assert len(assistant_tool_calls) == 1, "Expected only one tool call per assistant message"
            tool_name = assistant_tool_calls[0]["function"]["name"]
            tool_calls.setdefault(tool_name, []).append(i)

    repeated = []
    for tool_name, positions in tool_calls.items():
        if len(positions) > 1:
            repeated.append({"tool_name": tool_name, "positions": positions, "count": len(positions)})

    return (len(repeated) > 0, repeated)


def check_all_think_duplicates(messages: list[dict], parser: XMLParser) -> tuple[list[dict], list[dict]]:
    duplicates = []
    modified_messages = [msg.copy() for msg in messages]

    thinks = []
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            think_content = parser.extract_tag_contents(msg["content"], "think")
            if think_content and think_content[0].strip():
                thinks.append((i, think_content[0].strip()))

    seen_thinks = {}
    positions_to_replace = []

    for i in range(len(thinks)):
        for j in range(i + 1, len(thinks)):
            pos_i, content_i = thinks[i]
            pos_j, content_j = thinks[j]

            if content_i == content_j:
                duplicates.append({
                    "positions": (pos_i, pos_j),
                    "adjacent": pos_j - pos_i == 2,
                    "content_preview": content_i[:100],
                })

                if content_i not in seen_thinks:
                    seen_thinks[content_i] = pos_i

                if pos_j not in [seen_thinks[content] for content in seen_thinks]:
                    positions_to_replace.append(pos_j)

    for pos in positions_to_replace:
        old_content = modified_messages[pos]["content"]
        modified_messages[pos]["content"] = re.sub(
            r"<think>.*?</think>",
            "<think>Finalise based on tool result</think>",
            old_content,
            flags=re.DOTALL,
        )

    return duplicates, modified_messages


def check_exact_sequence(messages: list[dict], stepwise_ground_truths: list[dict]) -> tuple[bool, list[dict]]:
    issues = []

    for step_idx, ground_truth in enumerate(stepwise_ground_truths):
        msg_idx = 2 + (step_idx * 2)

        if msg_idx >= len(messages):
            issues.append({
                "step": step_idx,
                "error": "Missing tool call",
                "expected_tools": list(ground_truth.keys()),
                "actual_tool": None,
                "message_index": msg_idx,
            })
            continue

        tool_calls = messages[msg_idx].get("tool_calls")
        if not tool_calls:
            issues.append({
                "step": step_idx,
                "error": "Expected tool call but message has no tool_calls",
                "expected_tools": list(ground_truth.keys()),
                "actual_tool": None,
                "message_index": msg_idx,
            })
            continue

        tool_name = tool_calls[0]["function"]["name"]
        if tool_name not in ground_truth:
            issues.append({
                "step": step_idx,
                "error": "Tool name mismatch",
                "expected_tools": list(ground_truth.keys()),
                "actual_tool": tool_name,
                "message_index": msg_idx,
            })
            continue

        tool_response_idx = msg_idx + 1
        if tool_response_idx >= len(messages):
            issues.append({
                "step": step_idx,
                "error": "Missing tool response",
                "tool_name": tool_name,
                "expected_value": ground_truth[tool_name],
                "actual_value": None,
                "message_index": tool_response_idx,
            })
            continue

        tool_answer = parse_string(messages[tool_response_idx]["content"])
        expected_answer = ground_truth[tool_name]
        if not verify_answers(expected_answer, tool_answer):
            issues.append({
                "step": step_idx,
                "error": "Tool response value mismatch",
                "tool_name": tool_name,
                "expected_value": expected_answer,
                "actual_value": tool_answer,
                "message_index": tool_response_idx,
            })

    return (len(issues) > 0, issues)


def matches_exact_message_count(messages: list[dict], ground_truths: list[dict]) -> tuple[bool, str]:
    expected_message_count = len(ground_truths) * 2 + 3
    return len(messages) == expected_message_count, f"Expected {expected_message_count} messages, got {len(messages)}"


def print_messages_above_token_threshold(
    tokenizer: AutoTokenizer,
    ds: Dataset,
    token_threshold: int,
) -> None:
    def collect_token_counts(dataset: Dataset, tokenizer: AutoTokenizer):
        token_counts_by_role = {"user": [], "assistant": [], "tool": []}
        all_token_counts = []
        messages_with_tokens = []

        for sample_idx in range(len(dataset)):
            messages = parse_messages(dataset[sample_idx]["messages"])
            for msg_idx, msg in enumerate(messages):
                content = get_message_content(msg)
                role = msg.get("role", "unknown")

                if role == "system":
                    continue

                token_count = count_tokens_in_message(content, tokenizer)
                all_token_counts.append(token_count)
                messages_with_tokens.append((token_count, sample_idx, msg_idx, role, content))

                if role in token_counts_by_role:
                    token_counts_by_role[role].append(token_count)

        return all_token_counts, token_counts_by_role, messages_with_tokens

    _, _, messages_with_tokens = collect_token_counts(ds, tokenizer)

    print_to_both(f"\n{'=' * 80}")
    print_to_both(f"MESSAGES WITH MORE THAN {token_threshold} TOKENS (system messages excluded)")
    print_to_both(f"{'=' * 80}\n")

    high_token_messages = [msg for msg in messages_with_tokens if msg[0] > token_threshold]
    sorted_messages = sorted(high_token_messages, key=lambda x: x[0], reverse=True)

    print_to_both(f"Found {len(sorted_messages)} messages above {token_threshold} tokens\n")

    for rank, (token_count, sample_idx, msg_idx, role, content) in enumerate(sorted_messages, 1):
        print_to_both(f"\n{'-' * 80}")
        print_to_both(
            f"Rank #{rank} | Tokens: {token_count} | Sample: {sample_idx} | Message: {msg_idx} | Role: {role}"
        )
        print_to_both(f"{'-' * 80}")
        print_to_both(content)

    print_to_both(f"\n{'=' * 80}")
    print_to_both(f"SUMMARY: {len(sorted_messages)} messages found with more than {token_threshold} tokens")
    print_to_both(f"{'=' * 80}")


def remove_issues(all_issues: set[int], dataset: Dataset) -> Dataset:
    if all_issues:
        keep_mask = [i not in all_issues for i in range(len(dataset))]
        cleaned_dataset = dataset.select([i for i, keep in enumerate(keep_mask) if keep])

        print_to_both(f"\nInitial dataset size: {len(dataset)}")
        print_to_both(f"\nCleaned dataset size: {len(cleaned_dataset)}")
        print_to_both(f"Removed: {len(dataset) - len(cleaned_dataset)} examples")

        return cleaned_dataset

    print_to_both("No problematic examples found! Dataset is clean.")
    return dataset


def analyze_dataset(  # noqa: C901
    dataset_name: str,
    config: str,
    split: str,
    _load_from_disk: bool = True,
    verbose: bool = True,
    minimal_dataset: bool = False,
    assistant_token_threshold: int | None = None,
    tokenizer_name: str = "Qwen/Qwen2.5-3B-Instruct",
    push_to_hub: bool = False,
):
    print_to_both(f"Loading dataset: {dataset_name}/{config} ({split})")
    if _load_from_disk:
        ds = load_from_disk(dataset_name)
    else:
        ds = load_dataset(dataset_name, config, split=split, download_mode=DownloadMode.FORCE_REDOWNLOAD)

    print_to_both(f"Dataset size: {len(ds)}")
    print_to_both("Checking for answer reuse and duplicated thinking issues...")
    print_to_both()

    parser = XMLParser()
    tokenizer = None
    if assistant_token_threshold is not None:
        print_to_both(f"Loading tokenizer for token counting: {tokenizer_name}")
        tokenizer = load_tokenizer(tokenizer_name)

    reuse_issues = []
    all_think_duplicates = []
    exact_sequence_issues = []
    repeated_tool_calls = []
    message_count_issues = []
    token_threshold_issues = []

    for idx in range(len(ds)):
        messages = parse_messages(ds[idx]["messages"])
        stepwise_ground_truths = parse_stepwise_ground_truths(ds[idx]["stepwise_ground_truths"])

        has_reuse_issue, reuse_reason = check_tool_call_is_skipped_due_to_simple_op(
            messages, parser, minimal_dataset=minimal_dataset
        )
        if has_reuse_issue:
            reuse_issues.append((idx, reuse_reason))

        has_repeated, repeated_info = check_repeated_tool_calls(messages)
        if has_repeated:
            repeated_tool_calls.append((idx, repeated_info))

        has_exact_sequence, exact_sequence_info = check_exact_sequence(messages, stepwise_ground_truths)
        if has_exact_sequence and minimal_dataset:
            exact_sequence_issues.append((idx, exact_sequence_info))

        has_count_issue, count_reason = matches_exact_message_count(messages, stepwise_ground_truths)
        if not has_count_issue:
            message_count_issues.append((idx, count_reason))

        if assistant_token_threshold is not None and tokenizer is not None:
            max_tokens = get_max_assistant_tokens(messages, tokenizer)
            if max_tokens > assistant_token_threshold:
                token_threshold_issues.append((idx, max_tokens))

        duplicates, modified_messages = check_all_think_duplicates(messages, parser)
        if duplicates:
            print_to_both(f"\n{'=' * 80}")
            print_to_both(f"Example {idx}: Found {len(duplicates)} duplicate(s)")
            print_to_both(f"{'=' * 80}")

            for dup in duplicates:
                pos_i, pos_j = dup["positions"]
                print_to_both(
                    f"\nDuplicate between positions {pos_i} and {pos_j} "
                    f"(adjacent: {dup['adjacent']}, last: {pos_j == len(messages) - 1})"
                )
                is_answer = pos_j == len(messages) - 1

                if verbose:
                    print_messages(messages)
                else:
                    print_to_both(f"\n--- BEFORE (Message {pos_j}) ---")
                    print_to_both(messages[pos_j]["content"])
                    print_to_both(f"\n--- AFTER (Message {pos_j}) ---")
                    print_to_both(modified_messages[pos_j]["content"])

                all_think_duplicates.append((idx, dup, is_answer))

    print_to_both("=" * 80)
    print_to_both("ANALYSIS RESULTS")
    print_to_both("=" * 80)

    print_to_both(
        f"\n1. Answer-Tool Response Mismatch Issues: {len(reuse_issues)} "
        f"({len(reuse_issues) / len(ds) * 100:.2f}%)"
    )
    if reuse_issues:
        for idx, reason in reuse_issues:
            print_to_both(f"  - Example {idx}: {reason}")

    print_to_both(
        f"\n2. Repeated Tool Calls: {len(repeated_tool_calls)} "
        f"({len(repeated_tool_calls) / len(ds) * 100:.2f}%)"
    )
    if repeated_tool_calls:
        for idx, repeated_info in repeated_tool_calls:
            for tool_info in repeated_info:
                print_to_both(
                    f"  - Example {idx}: '{tool_info['tool_name']}' called {tool_info['count']} times "
                    f"at positions {tool_info['positions']}"
                )

    print_to_both(
        f"\n3. Exact Ground Truth Sequence Mismatches: {len(exact_sequence_issues)} "
        f"({len(exact_sequence_issues) / len(ds) * 100:.2f}%)"
    )
    if exact_sequence_issues:
        for idx, issues_list in exact_sequence_issues:
            for issue in issues_list:
                print_to_both(
                    f"  - Example {idx} step {issue['step']}: {issue['error']} "
                    f"(expected: {issue.get('expected_tools') or issue.get('expected_value')}, "
                    f"got: {issue.get('actual_tool') or issue.get('actual_value')})"
                )

    print_to_both(
        f"\n4. Message Count Mismatches: {len(message_count_issues)} "
        f"({len(message_count_issues) / len(ds) * 100:.2f}%)"
    )
    if message_count_issues:
        for idx, reason in message_count_issues:
            print_to_both(f"  - Example {idx}: {reason}")

    print_to_both(
        f"\n5. Assistant Token Threshold Exceeded: {len(token_threshold_issues)} "
        f"({len(token_threshold_issues) / len(ds) * 100:.2f}%)"
    )
    if token_threshold_issues:
        if assistant_token_threshold is not None:
            print_to_both(f"  Threshold: {assistant_token_threshold} tokens")
        for idx, max_tokens in token_threshold_issues:
            print_to_both(f"  - Example {idx}: max {max_tokens} tokens")
        if verbose and tokenizer is not None and assistant_token_threshold is not None:
            print_messages_above_token_threshold(tokenizer, ds, assistant_token_threshold)

    adjacent_dups = [d for d in all_think_duplicates if d[1]["adjacent"]]
    non_adjacent_dups = [d for d in all_think_duplicates if not d[1]["adjacent"]]

    print_to_both("\n6. ALL Think Block Duplicates (comprehensive check):")
    print_to_both(f"  Total: {len(all_think_duplicates)}")
    print_to_both(f"  Adjacent (i -> tool -> j): {len(adjacent_dups)}")
    print_to_both(f"  Non-adjacent: {len(non_adjacent_dups)}")

    if adjacent_dups:
        print_to_both("\n  Adjacent duplicates:")
        for idx, dup, is_answer in adjacent_dups:
            print_to_both(
                f"    Example {idx}, msgs {dup['positions']}, is answer: {is_answer}: "
                f"{dup['content_preview']}..."
            )

    if non_adjacent_dups:
        print_to_both("\n  Non-adjacent duplicates:")
        for idx, dup, is_answer in non_adjacent_dups:
            print_to_both(
                f"    Example {idx}, msgs {dup['positions']}, is answer: {is_answer}: "
                f"{dup['content_preview']}..."
            )

    print_to_both(f"Number of adjacent duplicates with final answer: {len([d for d in adjacent_dups if d[2]])}")

    all_issues = set(
        [idx for idx, _ in reuse_issues]
        + [idx for idx, _, _ in all_think_duplicates]
        + [idx for idx, _ in repeated_tool_calls]
        + [idx for idx, _ in exact_sequence_issues]
        + [idx for idx, _ in message_count_issues]
        + [idx for idx, _ in token_threshold_issues]
    )
    print_to_both(
        f"\n7. Total Unique Examples with Issues: {len(all_issues)} ({len(all_issues) / len(ds) * 100:.2f}%)"
    )

    print_to_both("\n" + "=" * 80)

    cleaned_ds = remove_issues(all_issues, ds)

    print_to_both("\n" + "=" * 80)
    print_to_both("Cleaned dataset ready.")
    print_to_both(f"Total examples removed: {len(all_issues)}")
    if push_to_hub:
        print_to_both(f"Pushing dataset to https://huggingface.co/datasets/{dataset_name}-clean")
        print_to_both(f"Dataset size: {len(cleaned_ds)}")
        cleaned_ds.push_to_hub(f"{dataset_name}-clean")
    print_to_both("=" * 80)

    return reuse_issues, all_think_duplicates, all_issues


if __name__ == "__main__":
    local_dataset = False

    if local_dataset:
        ensure_local_dataset(DISTILLED_DATASET, "default", "train", LOCAL_DATASET_PATH)

    dataset_path = LOCAL_DATASET_PATH if local_dataset else DISTILLED_DATASET
    output_file = "analyse_dataset.txt"
    reuse_issues, all_think_duplicates, all_issues = analyze_dataset(
        dataset_path,
        "default",
        "train",
        _load_from_disk=local_dataset,
        verbose=True,
        minimal_dataset=True,
        assistant_token_threshold=800,
        push_to_hub=False,
    )

    print_to_both(f"Wrote data to file: {output_file}")

    with open(output_file, "w") as f:
        f.write(output_buffer.getvalue())
