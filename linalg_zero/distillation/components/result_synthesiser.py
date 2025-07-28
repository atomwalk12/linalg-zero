from distilabel.distiset import Distiset
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts

from linalg_zero.distillation.components.chat_generation import ChatGeneration
from linalg_zero.distillation.utils import get_patched_openai_client

RESULT_SUMMARIZER_PROMPT = """You are a helpful response formatter. Your job is to take tool execution results and present them clearly and professionally to the user.

**Your Task:**
- Take the exact numerical results from tool executions
- Format them in a clear, readable way
- Answer the user's original question directly and helpfully

**Critical Rules:**
- Use ONLY the tool results - never recalculate or modify values
- Do not explain calculation methods or show work
- Focus on presenting the final answer clearly
- Round appropriately for readability (e.g., 19.8997... → 19.9)
- ALWAYS include the exact numerical result in <RESULT></RESULT> tags for verification
- For floating point results, provide reasonable precision (typically 2-4 significant digits)
- Match the precision level shown in the tool results

**Response Style:**
- Be direct and helpful
- Use clear formatting with **bold** for final answers
- Present results in a logical, easy-to-read structure
- Always end your response with <RESULT >exact_numerical_value</RESULT> containing the precise answer for verification

---
**EXAMPLES:**

**Example 1:**
*User Query:* "What is the determinant of [[2, 1], [3, 4]]?"
*Tool Results:* `{"determinant": 5}`
*Response:*
The determinant is **5**.

<RESULT>5</RESULT>

**Example 2:**
*User Query:* "What is the Frobenius norm of the product of [[1, 2], [3, 4]] and [[2, 1], [1, 3]]?"
*Tool Results:* `{"product_matrix": [[4, 7], [10, 15]]}`, `{"frobenius_norm": 19.8997}`
*Response:*
The matrix product is:
[[4, 7], [10, 15]]

The Frobenius norm of this result is **19.9**.

<RESULT>19.8997</RESULT>
"""


if __name__ == "__main__":
    """The following code demonstrates how planning works. The code is not being used for other purposes."""
    llm = get_patched_openai_client(model="Qwen3-32B-Q4_K_M.gguf", base_url="http://localhost:8000/v1")

    with Pipeline("result-executor-pipeline") as pipeline:
        load_dataset = LoadDataFromDicts(
            name="load_instructions",
            data=[
                {
                    "messages": [
                        {"role": "user", "content": "Turn on the living room lights."},
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_abc123",
                                    "type": "function",
                                    "function": {
                                        "name": "control_light",
                                        "arguments": '{"room": "living room", "state": "on"}',
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_abc123",
                            "content": "Couldn't turn on the lights in the living room.",
                        },
                    ]
                }
            ],
        )

        # Create the TextGeneration step
        summary_generation = ChatGeneration(
            name="summary_generation",
            llm=llm,
            input_batch_size=8,
            output_mappings={"model_name": "generation_model"},
            system_prompt=RESULT_SUMMARIZER_PROMPT,
        )

        # Connect the steps
        load_dataset >> summary_generation

    # Run the pipeline
    distiset: Distiset = pipeline.run(
        parameters={
            summary_generation.name: {
                "llm": {
                    "generation_kwargs": {
                        "max_new_tokens": 1024,
                        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                    }
                }
            }
        },
        use_cache=False,
    )

    print("The results of the pipeline are:")
    for num, data in enumerate(distiset["default"]["train"]):
        print(f"\n--- Example {num + 1} ---")
        print(f"Generated: {data['generation']}")
        print("-" * 50)
