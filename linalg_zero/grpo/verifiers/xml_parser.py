import re


class XMLParser:
    def get_assistant_messages(self, completion: list[dict]) -> list[dict]:
        """Helper function to extract assistant messages from a completion."""
        return [msg for msg in completion if msg["role"] == "assistant"]

    def get_tool_messages(self, completion: list[dict]) -> list[dict]:
        """Helper function to extract tool messages from a completion."""
        return [msg for msg in completion if msg["role"] == "tool"]

    def extract_answer(self, message: str) -> str | None:
        """Extract answer content from the last <answer> tag (handles truncated text)."""
        # Find the last occurrence of <answer>
        last_answer_pos = message.rfind("<answer>")
        if last_answer_pos == -1:
            return None

        # Extract text after the last <answer> tag
        after_tag = message[last_answer_pos + len("<answer>") :].strip()

        # If there's a closing </answer> tag, extract only the content before it
        end_tag_pos = after_tag.find("</answer>")
        if end_tag_pos != -1:
            return after_tag[:end_tag_pos].strip()

        # If no closing tag, return everything after <answer>
        return after_tag

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
