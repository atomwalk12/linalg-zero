import os
import sys
from collections.abc import Iterable

from datasets import load_dataset
from openai import OpenAI

# # In some shell, from anywhere
# CUDA_VISIBLE_DEVICES=0 \
# vllm serve atomwalk12/LinalgZero-SFT \
#   --host 0.0.0.0 \
#   --port 8000 \
#   --dtype bfloat16 \
#   --gpu-memory-utilization 0.6 \
#   --max-model-len 4096 \
#   --swap-space 16 \
#   --max-num-batched-tokens 2048 \
#   --api-key default \
#   --served-model-name LinAlgZero-GRPO \
#   --enable-auto-tool-choice \
#   --tool-call-parser hermes


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QUERY_IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0

BASE_URL = os.environ.get("LINALGZERO_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("LINALGZERO_API_KEY", "default")
MODEL = os.environ.get("LINALGZERO_MODEL_NAME", "LinAlgZero-GRPO")

MAX_TOKENS = 800


def format_stop(stop: Iterable[str] | None) -> str:
    if stop is None:
        return "None"
    return "[" + ", ".join(repr(s) for s in stop) + "]"


def run_scenario(
    label: str,
    *,
    use_tools: bool,
    temperature: float,
    stop: list[str] | None = None,
) -> None:
    """Run a single configuration against one dataset query."""
    dataset = load_dataset("atomwalk12/linalgzero-sft", split="test")
    sample = dataset[QUERY_IDX]

    # Clean messages: remove None values that vLLM doesn't accept
    messages = [{k: v for k, v in msg.items() if v is not None} for msg in sample["messages"]]

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    kwargs = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
        "extra_body": {"skip_special_tokens": False},
    }

    if use_tools:
        kwargs["tools"] = sample.get("tools")
    if stop is not None:
        kwargs["stop"] = stop

    print("\n" + "=" * 80)
    print(f"SCENARIO: {label}")
    print("-" * 80)
    print(f"use_tools     = {use_tools}")
    print(f"temperature   = {temperature}")
    print(f"stop          = {format_stop(stop)}")
    print("=" * 80)

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    message = choice.message

    # Show both content and tool_calls explicitly so we can see where tags live.
    print("Raw message.content (repr):")
    print(repr(message.content))
    print("\nRaw message.tool_calls:")
    print(message.tool_calls)

    # Show metadata on stop reason and token details
    print(f"\nFinish reason: {choice.finish_reason}")
    if hasattr(choice, "stop_reason") and choice.stop_reason is not None:
        print(f"Stop reason: {choice.stop_reason}")

    if hasattr(response, "usage") and hasattr(response.usage, "completion_tokens_details"):
        print(f"Token details: {response.usage.completion_tokens_details}")

    if hasattr(choice, "model_extra"):
        print(f"Extra info: {choice.model_extra}")


def main() -> None:
    print(f"\n=== Query {QUERY_IDX} ===")
    dataset = load_dataset("atomwalk12/linalgzero-sft", split="test")
    sample = dataset[QUERY_IDX]
    print(f"User query:\n{sample['messages'][-1]['content']}\n")

    run_scenario(
        "A) tools + temperature=0.0 + stop on control tags (tags will be cut)",
        use_tools=True,
        temperature=0.0,
        stop=[
            "<tool_response>",
            "</tool_response>",
            "User:",
        ],
    )

    run_scenario(
        "B) no tools + temperature=0.0 + stop on control tags (tags will be cut)",
        use_tools=False,
        temperature=0.0,
        stop=[
            "<tool_response>",
            "</tool_response>",
            "User:",
            "</tool_call>",
            "</answer>",
        ],
    )


if __name__ == "__main__":
    main()
