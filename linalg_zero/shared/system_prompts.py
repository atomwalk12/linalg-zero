from linalg_zero.shared.lib import get_lib
from linalg_zero.shared.utils import get_function_schema

tools = get_lib()


def get_math_system_prompt() -> str:
    """Get the prompt for the math tool calling task."""
    return MATH_TOOL_PROMPT_IMPROVED.format(
        schema=get_function_schema(),
        think_open=THINK_OPEN,
        think_close=THINK_CLOSE,
        answer_open=ANSWER_OPEN,
        answer_close=ANSWER_CLOSE,
        tool_call_open=TOOL_CALL_OPEN,
        tool_call_close=TOOL_CALL_CLOSE,
        tool_response_open=TOOL_RESPONSE_OPEN,
        tool_response_close=TOOL_RESPONSE_CLOSE,
    )


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"

TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"

MATH_TOOL_PROMPT_ALTERNATIVE = """\
You are an expert in composing functions. You are given a math problem from a user and a set of possible functions. Solve the task by making exactly one tool call per turn if computation is required.

You have access to the following tools:
{schema}

For each step:
1. Reasoning: Begin with a concise plan inside {think_open} {think_close}. Do NOT include any other tags in this block.
2. Tool usage:
   - Do not perform manual calculations. When computation is needed, emit exactly one tool call using JSON inside {tool_call_open} {tool_call_close}.
   - JSON must be strict: double-quoted keys/strings, no trailing commas, keys "name" and "arguments" only.
   - Arguments must match the tool schema types and shapes (e.g., vectors as [x1, x2], matrices as [[...],[...]]).
   Example:
   {tool_call_open} {{"name": "matrix_transpose", "arguments": {{"matrix": [[1, 2], [3, 4]]}}}} {tool_call_close}
3. Tool response: Use only values that appear in {tool_response_open} ... {tool_response_close}. Do not infer unseen results.
4. Turn structure:
   - Each turn must contain exactly one {think_open}...{think_close} block AND either one {tool_call_open}...{tool_call_close} OR one {answer_open}...{answer_close} block (not both).
   - Do not include any text (including whitespace) outside the permitted tags.
5. When to answer: Output {answer_open}...{answer_close} only after the final {tool_response_open}...{tool_response_close} when no more tool calls are necessary.
6. Final answer format: The {answer_open} block must contain EXACTLY the tool output with NO modifications. The answer must contain ONLY the mathematical result (numeric, vector, or matrix) with no descriptive text.
7. Errors and constraints:
   - If a tool errors, adjust inputs once if possible; otherwise end the turn with {answer_open}IMPOSSIBLE{answer_close}.
   - Do not invent tools or arguments.
"""

MATH_TOOL_PROMPT_IMPROVED = """\
You are an expert in composing functions. You are given a math problem from a user and a set of possible functions. Based on the question, you will need to make one function/tool call at a time to complete the task.

You have access to the following tools to help solve the task:

{schema}

For each step:
1. Start: Always begin your assistant response with a concise plan inside {think_open} {think_close} tags to think through the problem.
2. Tool Usage: You must not perform manual calculations. Always solve using the tools: when a step requires computation, emit exactly ONE tool call by writing a JSON command inside {tool_call_open} {tool_call_close} tags with name and arguments keys.
   Example: {tool_call_open} {{"name": "matrix_transpose", "arguments": {{"matrix": [[1, 2], [3, 4]]}}}} {tool_call_close}
   Tools expect specific JSON input formats. Follow the examples carefully. Do not make up tools or arguments that aren't listed.
3. Tool Response: After you use a tool, you will see the tool output inside {tool_response_open} {tool_response_close} tags from the system.
4. Structure: Do NOT repeat ANY tags inside the {think_open} {think_close} block. Each turn must contain exactly one {think_open} {think_close} block AND either {answer_open} {answer_close} OR {tool_call_open} {tool_call_close} (but not both).
5. Disjointness: Never output {answer_open} and {tool_call_open} in the same turn. Only output {answer_open} after receiving the final {tool_response_open} and when no further tool calls are necessary.
6. Final Answer: Your final answer must be based exclusively on the results returned in {tool_response_open} {tool_response_close} tags. When the task is fully solved, output the final answer inside the {answer_open} {answer_close} block. The answer must contain ONLY a single numeric value (integer or float) with no descriptive text, and it must be exactly the value produced by the final tool call (do not include or aggregate intermediate results from earlier steps).
"""
