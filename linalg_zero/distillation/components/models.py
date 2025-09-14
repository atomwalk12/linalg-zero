import json
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

from distilabel.errors import DistilabelUserError

from linalg_zero.distillation.data import ThoughtSchema
from linalg_zero.shared.system_prompts import (
    ANSWER_CLOSE,
    ANSWER_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
)

if TYPE_CHECKING:
    pass


DIAG_PREFIX = "[diag]"


class ModelType(str, Enum):
    QWEN3_THINKING = "qwen3-thinking"
    DEFAULT = "default"


class ModelParameters(ABC):
    @abstractmethod
    def set_recommended_defaults(self, generation_kwargs: dict[str, Any], *, deterministic: bool) -> dict[str, Any]:
        """Inject recommended generation defaults for the model.

        deterministic=True should configure sampling deterministically (e.g., temperature=0, top_p=1).
        """
        raise NotImplementedError

    @abstractmethod
    def format_assistant_message(self, message: ThoughtSchema) -> dict[str, Any] | None:
        """Return an OpenAI-compatible assistant message for the given parsed output."""
        raise NotImplementedError

    @abstractmethod
    def append_policy(self) -> bool:
        """Return whether to append the assistant message to the conversation."""
        pass


class Qwen3ThinkingConfig(ModelParameters):
    def set_recommended_defaults(self, generation_kwargs: dict[str, Any], *, deterministic: bool) -> dict[str, Any]:
        # Deterministic vs recommended sampling per Qwen3 Thinking model card
        if deterministic:
            generation_kwargs["temperature"] = 0.0
            generation_kwargs["top_p"] = 1.0
        else:
            generation_kwargs["temperature"] = 0.6
            generation_kwargs["top_p"] = 0.95

            extra_body = generation_kwargs.setdefault("extra_body", {})
            extra_body.setdefault("top_k", 20)
            extra_body.setdefault("min_p", 0)
        return generation_kwargs

    def append_policy(self) -> bool:
        """Return whether to append the assistant message to the conversation."""
        return False

    def format_assistant_message(self, message: ThoughtSchema) -> dict[str, Any] | None:
        # Qwen3 Thinking best practice: do NOT include <think> content in history
        if message.completed:
            if message.final_answer is None:
                raise DistilabelUserError("final_answer cannot be None when completed=True")
            return {
                "role": "assistant",
                "content": f"{ANSWER_OPEN}{message.final_answer}{ANSWER_CLOSE}",
            }
        if message.tool_call is not None:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": str(uuid.uuid4()),
                        "type": "function",
                        "function": {
                            "name": message.tool_call.name,
                            "arguments": json.dumps(message.tool_call.arguments),
                        },
                    }
                ],  # type: ignore[dict-item]
            }
        return None


class DefaultConfig(ModelParameters):
    def set_recommended_defaults(self, generation_kwargs: dict[str, Any], *, deterministic: bool) -> dict[str, Any]:
        if deterministic:
            generation_kwargs["temperature"] = 0.0
            generation_kwargs["top_p"] = 1.0
        return generation_kwargs

    def append_policy(self) -> bool:
        """Return whether to append the assistant message to the conversation."""
        return False

    def format_assistant_message(self, message: ThoughtSchema) -> dict[str, Any] | None:
        if message.completed:
            if message.final_answer is None:
                raise DistilabelUserError("final_answer cannot be None when completed=True")
            return {
                "role": "assistant",
                "content": f"{THINK_OPEN}{message.thought}{THINK_CLOSE}\n\n{ANSWER_OPEN}{message.final_answer}{ANSWER_CLOSE}",
            }
        if message.tool_call is not None:
            return {
                "role": "assistant",
                "content": THINK_OPEN + message.thought + THINK_CLOSE,
                "tool_calls": [
                    {
                        "id": str(uuid.uuid4()),
                        "type": "function",
                        "function": {
                            "name": message.tool_call.name,
                            "arguments": json.dumps(message.tool_call.arguments),
                        },
                    }
                ],  # type: ignore[dict-item]
            }
        return None
