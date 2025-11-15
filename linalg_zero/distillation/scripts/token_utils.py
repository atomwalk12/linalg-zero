"""Shared utilities for token counting and filtering in datasets."""

import json
from typing import Any

from transformers import AutoTokenizer


def parse_messages(messages_field: Any) -> list[dict]:
    """
    Parse messages from dataset field (handles both JSON string and dict formats).

    Args:
        messages_field: Messages field from dataset (can be string or list)

    Returns:
        List of message dictionaries
    """
    return json.loads(messages_field) if isinstance(messages_field, str) else messages_field


def get_message_content(message: dict) -> str:
    """
    Extract and clean content from a message dictionary.

    Args:
        message: Message dictionary with 'content' field

    Returns:
        Stripped message content string
    """
    return (message.get("content") or "").strip()


def load_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    """
    Load a tokenizer by name.

    Args:
        tokenizer_name: HuggingFace model name for the tokenizer

    Returns:
        Loaded AutoTokenizer instance
    """
    return AutoTokenizer.from_pretrained(tokenizer_name)


def count_tokens_in_message(content: str, tokenizer: AutoTokenizer) -> int:
    """Count tokens in a message content using the provided tokenizer."""
    return len(tokenizer.encode(content))


def get_max_assistant_tokens(messages: list[dict], tokenizer: AutoTokenizer) -> int:
    """
    Get the maximum token count among all assistant messages.

    Args:
        messages: List of message dictionaries
        tokenizer: Tokenizer to use for counting tokens

    Returns:
        Maximum token count found in assistant messages, or 0 if no assistant messages
    """
    assistant_token_counts = []
    for msg in messages:
        if msg.get("role") == "assistant":
            content = get_message_content(msg)
            token_count = count_tokens_in_message(content, tokenizer)
            assistant_token_counts.append(token_count)

    return max(assistant_token_counts) if assistant_token_counts else 0
