from linalg_zero.shared.lib import get_lib
from linalg_zero.shared.utils import get_function_schema

tools = get_lib()


def get_math_system_prompt() -> str:
    """Get the prompt for the math tool calling task."""
    return MATH_TOOL_PROMPT.format(schema=get_function_schema())


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"

TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"


MATH_TOOL_PROMPT = """\
You are an expert in composing functions. You are given a math problem from a user and a set of possible functions. Based on the question, you will need to make one function/tool call at a time to complete the task.

You have access to the following tools to help solve the task:

{schema}

For each step:
1. Always begin your assistant response with a step-by-step thinking process inside <think> </think> tags to think through the problem.
2. If needed, use ONE tool by writing a JSON command inside <tool_call> </tool_call> tags with name and arguments keys.
   example: <tool_call> {{"name": "matrix_transpose", "arguments": {{"matrix": [[1, 2], [3, 4]]}}}} </tool_call>
   Tools expect specific JSON input formats. Follow the examples carefully. Do not make up tools or arguments that aren't listed.
3. After you use a tool, you will see the tool output inside <tool_response> </tool_response> tags from the system.
4. Never output <answer> and <tool_call> in the same turn. Only output <answer> after receiving the final <tool_response> and when no further tools are necessary.
5. If you believe the current task is completed and no more tools are necessary, output your final answer inside <answer> </answer> tags. The answer must contain ONLY the mathematical result in its simplest form with no descriptive text.
6. Do NOT write <answer> or </answer> inside <think> or any intermediate reasoning. Emit exactly one <answer>...</answer> block only as the final message when the problem is fully solved, and place only the numeric/vector/matrix result inside it (no prose).
"""
