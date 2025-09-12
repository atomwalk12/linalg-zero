import json
import re


class XMLParser:
    # Checks exact format: <think>...</think> followed by <answer>...</answer>
    think_then_answer_regex = (
        r"^<think>\s*([^<]*(?:<(?!/?think>)[^<]*)*)\s*<\/think>\s*"
        r"<answer>\s*([\s\S]*?)\s*<\/answer>$"
    )
    # Checks exact format: <think>...</think> followed by <tool_call>...</tool_call>
    think_then_tool_regex = (
        r"^<think>\s*([^<]*(?:<(?!/?think>)[^<]*)*)\s*<\/think>\s*"
        r"<tool_call>\s*([\s\S]*?)\s*<\/tool_call>$"
    )

    def get_assistant_messages(self, completion: list[dict]) -> list[dict]:
        """Helper function to extract assistant messages from a completion."""
        return [msg for msg in completion if msg["role"] == "assistant"]

    def get_tool_messages(self, completion: list[dict]) -> list[dict]:
        """Helper function to extract tool messages from a completion."""
        return [msg for msg in completion if msg["role"] == "tool"]

    def is_last_msg_tool_call(self, messages: list[dict]) -> bool:
        prev_is_tool_response = False
        for prev in reversed(messages):
            if prev.get("role") == "system":
                continue
            if prev.get("role") == "tool":
                prev_is_tool_response = True
            break
        return prev_is_tool_response

    def _extract_last_answer(self, message: str) -> str | None:
        """Extract answer content from <answer> tags.

        Primary path: properly closed <answer>...</answer> (last occurrence).
        Fallback: if an opening <answer> exists but closing is missing (e.g.,
        stop sequences), return content from after <answer> to end of message.
        """
        contents = self._extract_tag_contents(message, "answer", last_only=True)
        if contents:
            return contents[0]
        return None

    def _check_format(self, message: str, regex: str, expected_groups: int) -> bool:
        """Check if message matches the expected format with correct number of groups."""
        last_think_pos = message.rfind("<think>")
        if last_think_pos == -1:
            return False

        # Find the last <think> token in the string, then verify format till the end
        last_think_token = message[last_think_pos:]

        match = re.search(regex, last_think_token, re.DOTALL)

        # We look for a match and assert that the number of matched groups is correct
        return match is not None and len(match.groups()) == expected_groups

    def _is_valid_think_then_answer(self, message: str) -> bool:
        """Validate '<think>...</think>' followed by '<answer>...</answer>'."""

        return self._check_format(message, self.think_then_answer_regex, expected_groups=2)

    def _is_valid_think_then_tool(self, message: str) -> bool:
        """Validate '<think>...</think>' followed by '<tool_call>...</tool_call>'."""

        return self._check_format(message, self.think_then_tool_regex, expected_groups=2)

    def _is_valid_think_then_tool_or_answer(self, message: str) -> bool:
        """Validate '<think>...</think>' followed by exactly one of '<tool_call>...</tool_call>' or '<answer>...</answer>'."""

        valid_tool = self._check_format(message, self.think_then_tool_regex, expected_groups=2)
        valid_answer = self._check_format(message, self.think_then_answer_regex, expected_groups=2)
        return valid_tool or valid_answer

    def ensure_think_prefix(self, message: str | None) -> str | None:
        """Ensure the message starts with a single '<think>' prefix without duplicating it."""
        if message is None:
            return None
        if message.startswith("<think>"):
            return message
        return "<think>" + message

    def _extract_tag_contents(
        self,
        message: str,
        tag: str,
        *,
        last_only: bool = False,
    ) -> list[str]:
        """
        Extract contents enclosed by a specific XML-like tag.

        - If last_only is True, returns at most one element: the content between the
          last occurrence of <tag> and its subsequent </tag>, if properly closed.
        - If last_only is False, returns all non-overlapping, properly closed tag
          contents in document order.
        - Whitespace around extracted content is stripped.
        """
        assert tag in ["tool_call", "answer", "think"]  # noqa: S101
        if not message:
            return []

        open_token = f"<{tag}>"
        close_token = f"</{tag}>"

        if last_only:
            last_open = message.rfind(open_token)
            if last_open == -1:
                return []
            after_open = message[last_open + len(open_token) :]
            close_pos = after_open.find(close_token)
            if close_pos == -1:
                return []
            return [after_open[:close_pos].strip()]

        # Find all properly closed occurrences using a non-greedy regex.
        # Using DOTALL so content can span multiple lines.
        pattern = re.compile(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", re.DOTALL)
        return [m.group(1).strip() for m in pattern.finditer(message)]

    def _extract_last_tool_call(self, message: str) -> list[str]:
        """Extract <tool_call>...</tool_call> block contents."""
        return self._extract_tag_contents(message, "tool_call", last_only=True)

    def _extract_thought(self, message: str) -> str | None:
        """Extract thought content from properly formed <think></think> tags.

        Supports normalization when the message begins with an auto-seeded
        "<think>" and the model also emitted its own "<think>", resulting in
        leading "<think><think>". In such case, we still return the content of
        the last properly closed think block.
        """
        if not message:
            return None

        contents = self._extract_tag_contents(message, "think", last_only=True)
        return contents[0] if contents else None

    def analyze_message(
        self,
        message: str,
        *,
        tool_names: set[str] | None = None,
    ) -> dict:
        """
        Parse a single assistant message and return diagnostics + extracted fields.

        Returns a dictionary with keys:
        - has_think, has_tool_call, has_answer: bool
        - valid_format: bool (think then tool_call|answer)
        - is_valid_think_then_answer: bool (think then answer)
        - first_turn_policy_valid: bool | None (if enforce_first_turn_tool)
        - think_count, tool_call_count, answer_count: int
        - orphan_closing: {think, tool_call, answer}: bool
        - unclosed: {think, tool_call, answer}: bool
        - stray_content: bool
        - code_fences_in_last_tool: bool
        - thought: str | None
        - answer: str | None
        - tool: {
            json_valid: bool,
            name: str | "",
            arguments: dict | {},
            name_known: bool | None,
            }
        """

        diagnostics = XMLDiagnostics(self)

        thought = self._extract_thought(message)
        answer = self._extract_last_answer(message)
        last_tool_block = self._extract_last_tool_call(message)
        tool_block = last_tool_block[0] if last_tool_block else None

        result: dict = {}
        result["thought"] = thought
        result["answer"] = answer
        result["has_think"] = bool(thought)
        result["has_tool_call"] = tool_block is not None
        result["has_answer"] = bool(answer)

        # Counts
        result["think_count"] = diagnostics._count_tags(message, "think")
        result["tool_call_count"] = diagnostics._count_tags(message, "tool_call")
        result["answer_count"] = diagnostics._count_tags(message, "answer")

        # Format validity
        result["is_valid_think_then_tool_or_answer"] = self._is_valid_think_then_tool_or_answer(message)
        result["is_valid_think_then_answer"] = self._is_valid_think_then_answer(message)

        # Structural diagnostics
        result["unopened"] = {
            "think": diagnostics._has_orphan_closing_tag(message, "think"),
            "tool_call": diagnostics._has_orphan_closing_tag(message, "tool_call"),
            "answer": diagnostics._has_orphan_closing_tag(message, "answer"),
        }
        result["unclosed"] = {
            "think": diagnostics._has_unclosed_tag(message, "think"),
            "tool_call": diagnostics._has_unclosed_tag(message, "tool_call"),
            "answer": diagnostics._has_unclosed_tag(message, "answer"),
        }
        result["stray_content"] = diagnostics._has_stray_content_outside_allowed(message)
        result["code_fences_in_last_tool"] = diagnostics._has_code_fences_in_last_tool(message)

        # Tool parsing
        tool_info: dict = {"json_valid": False, "name": "", "arguments": {}, "name_known": None}
        if tool_block is not None:
            data, err = _safe_json_loads(tool_block)
            if (
                isinstance(data, dict)
                and isinstance(data.get("name"), str)
                and isinstance(data.get("arguments"), dict)
            ):
                tool_info["json_valid"] = True
                tool_info["name"] = data["name"]
                tool_info["arguments"] = data["arguments"]
                if tool_names is not None:
                    tool_info["name_known"] = data["name"] in tool_names
            else:
                tool_info["json_valid"] = False

        result["tool"] = tool_info

        return result

    def analyze_message_in_context(
        self,
        context: list[dict],
        message: str,
        *,
        tool_names: set[str] | None = None,
    ) -> dict:
        """
        Like analyze_message, but also evaluates conversation-level policy:
        - answer_policy_valid: if an <answer> is present in message, the immediately
        previous message in the conversation MUST be a tool response (a system message
        whose content contains <tool_response> ... </tool_response>).
        Adds fields:
        - answer_allowed: bool (there is a prior adjacent tool_response)
        - answer_policy_valid: bool (no answer, or answer_allowed)
        """
        result = self.analyze_message(message, tool_names=tool_names)

        prev_is_tool_response = self.is_last_msg_tool_call(context)

        has_answer = bool(result["has_answer"])

        result["answer_allowed"] = prev_is_tool_response
        result["answer_policy_valid"] = (not has_answer) or prev_is_tool_response
        return result


class XMLDiagnostics:
    """Diagnostic helpers for analyzing malformed generations.

    Separated from XMLParser to keep core parsing/validation lean.
    """

    def __init__(self, parser: XMLParser):
        self.parser = parser

    def _count_tags(self, message: str, tag: str) -> int:
        return len(self.parser._extract_tag_contents(message, tag, last_only=False))

    def _has_unclosed_tag(self, message: str, tag: str) -> bool:
        assert tag in ["tool_call", "answer", "think"]  # noqa: S101
        if not message:
            return False
        open_token = f"<{tag}>"
        close_token = f"</{tag}>"
        last_open = message.rfind(open_token)
        if last_open == -1:
            return False
        after_open = message[last_open + len(open_token) :]
        return close_token not in after_open

    def _has_code_fences_in_last_tool(self, message: str) -> bool:
        block = self.parser._extract_tag_contents(message, "tool_call", last_only=True)
        if not block:
            return False
        return "```" in block[0]

    def _has_stray_content_outside_allowed(self, message: str) -> bool:
        if "<think>" not in message:
            return False
        has_tool = "<tool_call>" in message and "</tool_call>" in message
        has_answer = "<answer>" in message and "</answer>" in message
        if not (has_tool or has_answer):
            return False
        return not (self.parser._is_valid_think_then_tool(message) or self.parser._is_valid_think_then_answer(message))

    def _has_orphan_closing_tag(self, message: str, tag: str) -> bool:
        """Return True if a closing </tag> appears without any prior opening <tag>."""
        assert tag in ["tool_call", "answer", "think"]  # noqa: S101
        if not message:
            return False
        open_token = f"<{tag}>"
        close_token = f"</{tag}>"
        if close_token not in message:
            return False
        first_close = message.find(close_token)
        first_open = message.find(open_token)
        return first_open == -1 or first_close < first_open


def _safe_json_loads(s: str) -> tuple[dict | None, str | None]:
    try:
        return json.loads(s), None
    except Exception as e:
        return None, f"invalid tool JSON: {type(e).__name__}: {e}"
