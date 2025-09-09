import json
import re


class XMLParser:
    think_then_answer_regex = (
        r"^<think>\s*([^<]*(?:<(?!/?think>)[^<]*)*)\s*<\/think>\s*"
        r"<answer>\s*([\s\S]*?)\s*<\/answer>$"
    )
    think_then_tool_regex = (
        r"^<think>\s*([^<]*(?:<(?!/?think>)[^<]*)*)\s*<\/think>\s*"
        r"<tool>\s*([\s\S]*?)\s*<\/tool>$"
    )

    def get_assistant_messages(self, completion: list[dict]) -> list[dict]:
        """Helper function to extract assistant messages from a completion."""
        return [msg for msg in completion if msg["role"] == "assistant"]

    def get_tool_messages(self, completion: list[dict]) -> list[dict]:
        """Helper function to extract tool messages from a completion."""
        return [msg for msg in completion if msg["role"] == "tool"]

    def extract_answer(self, message: str) -> str | None:
        """Extract answer content from <answer> tags.

        Primary path: properly closed <answer>...</answer> (last occurrence).
        Fallback: if an opening <answer> exists but closing is missing (e.g.,
        stop sequences), return content from after <answer> to end of message.
        """
        contents = self._extract_tag_contents(message, "answer", last_only=True)
        if contents:
            return contents[0]

        if not message:
            return None
        open_token = "<answer>"  # noqa: S105
        last_open = message.rfind(open_token)
        if last_open == -1:
            return None
        # Return everything after the opening tag as best-effort fallback
        return message[last_open + len(open_token) :].strip()

    def check_format(self, message: str, regex: str, expected_groups: int) -> bool:
        """Check if message matches the expected format with correct number of groups."""
        last_think_pos = message.rfind("<think>")
        if last_think_pos == -1:
            return False

        # Find the last <think> token in the string, then verify format till the end
        last_think_token = message[last_think_pos:]

        match = re.search(regex, last_think_token, re.DOTALL)

        # We look for a match and assert that the number of matched groups is correct
        return match is not None and len(match.groups()) == expected_groups

    def is_valid_think_then_answer(self, message: str) -> bool:
        """Validate '<think>...</think>' followed by '<answer>...</answer>'."""

        return self.check_format(message, self.think_then_answer_regex, expected_groups=2)

    def is_valid_think_then_tool(self, message: str) -> bool:
        """Validate '<think>...</think>' followed by '<tool>...</tool>'."""

        return self.check_format(message, self.think_then_tool_regex, expected_groups=2)

    def is_valid_think_then_tool_or_answer(self, message: str) -> bool:
        """Validate '<think>...</think>' followed by exactly one of '<tool>...</tool>' or '<answer>...</answer>'."""

        valid_tool = self.check_format(message, self.think_then_tool_regex, expected_groups=2)
        valid_answer = self.check_format(message, self.think_then_answer_regex, expected_groups=2)
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

    def extract_tools(self, message: str, first_only: bool = False) -> list[str]:
        """Extract <tool>...</tool> block contents.

        If first_only is False (default), returns all tool blocks in order of appearance.
        If first_only is True, returns at most one element: the first properly closed
        <tool> block encountered in document order.
        """
        if first_only:
            # Find all, then slice to at most the first to keep return type stable (list[str])
            tools = self._extract_tag_contents(message, "tool", last_only=False)
            return tools[:1]
        return self._extract_tag_contents(message, "tool", last_only=False)

    def extract_thought(self, message: str) -> str | None:
        """Extract thought content from properly formed <think></think> tags.

        Supports normalization when the message begins with an auto-seeded
        "<think>" and the model also emitted its own "<think>", resulting in
        leading "<think><think>". In such case, we still return the content of
        the first properly closed think block.
        """
        if not message:
            return None

        # If message starts with duplicated '<think><think>', strip one to normalize
        if message.startswith("<think><think>"):
            message = message.replace("<think><think>", "<think>", 1)

        contents = self._extract_tag_contents(message, "think", last_only=True)
        return contents[0] if contents else None


class XMLDiagnostics:
    """Diagnostic helpers for analyzing malformed generations.

    Separated from XMLParser to keep core parsing/validation lean.
    """

    def __init__(self, parser: XMLParser):
        self.parser = parser

    def has_unclosed_answer(self, message: str) -> bool:
        return self.has_unclosed_tag(message, "answer")

    def count_tags(self, message: str, tag: str) -> int:
        return len(self.parser._extract_tag_contents(message, tag, last_only=False))

    def has_unclosed_tag(self, message: str, tag: str) -> bool:
        if not message:
            return False
        open_token = f"<{tag}>"
        close_token = f"</{tag}>"
        last_open = message.rfind(open_token)
        if last_open == -1:
            return False
        after_open = message[last_open + len(open_token) :]
        return close_token not in after_open

    def get_first_tool_block(self, message: str) -> str | None:
        tools = self.parser._extract_tag_contents(message, "tool", last_only=False)
        return tools[0] if tools else None

    def has_code_fences_in_first_tool(self, message: str) -> bool:
        block = self.get_first_tool_block(message)
        if not block:
            return False
        return "```" in block

    def extract_first_tool_call(self, message: str) -> tuple[dict | None, str | None]:
        block = self.get_first_tool_block(message)
        if block is None:
            return None, "no tool block found"
        try:
            data = json.loads(block)
        except Exception:
            return None, "invalid tool JSON"
        if not isinstance(data, dict):
            return None, "invalid tool JSON"
        name = data.get("name")
        args = data.get("arguments")
        if not isinstance(name, str):
            return None, "missing or invalid 'name' in tool JSON"
        if not isinstance(args, dict):
            return None, "missing or invalid 'arguments' in tool JSON"
        return {"name": name, "arguments": args}, None

    def has_stray_content_outside_allowed(self, message: str) -> bool:
        if "<think>" not in message:
            return False
        has_tool = "<tool>" in message and "</tool>" in message
        has_answer = "<answer>" in message and "</answer>" in message
        if not (has_tool or has_answer):
            return False
        return not (self.parser.is_valid_think_then_tool(message) or self.parser.is_valid_think_then_answer(message))
