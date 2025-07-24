import json

from distilabel.distiset import Distiset
from distilabel.models import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import ChatGeneration

from linalg_zero.distillation.data import QueryAnswer
from linalg_zero.distillation.utils import get_openai_client, load_module_from_path

EXAMPLES = {
    "linear-algebra": """Example 1 for Tool Selection: Simple tool selection scenario.

User initial query:
What is the determinant of the matrix [[9, 10], [6, 11]]?

Execution plan:
1. Calculate the determinant of the matrix [[9, 10], [6, 11]].

Model Response:
{
    "tool_name": "determinant",
    "parameters": {
    "matrix": [[9, 10], [6, 11]]
    }
}

---
Example 2 for Tool Selection: It illustrates how dependencies between tools can be handled.

User Query: What is the Frobenius norm of the product of matrices [[1, 2], [6, 4]] and [[2, 4], [1, 3]]?

Decomposed Sub-questions:
1. Multiply the matrices [[1, 2], [6, 4]] and [[2, 4], [1, 3]].
2. Calculate the Frobenius norm of the resulting matrix.

Model Response:
[
  {
    "tool_name": "multiply_matrices",
    "parameters": {
      "matrix_a": [[1, 2], [6, 4]],
      "matrix_b": [[2, 4], [1, 3]]
    }
  },
  {
    "tool_name": "frobenius_norm",
    "parameters": {
      "matrix": "[result_of_call_1]"
    }
  }
]
"""
}

TOOL_SELECTION_PROMPT = """
You are an expert in linear algebra and a highly skilled tool-using assistant. Your current task is to meticulously select the most appropriate linear algebra tool(s) for a given sub-question and extract all necessary parameters for its execution. This is the Tool Selection stage, following the initial task planning phase.

Your Objective:
To take a single, specific linear algebra sub-question (from a previously decomposed user query) and, referencing a list of available linear algebra tools (including their JSON schema definitions), precisely identify the most relevant tool(s) and formulate their function calls with the correct input arguments.

Strict Instructions for Tool Selection and Parameter Extraction:
1.  Analyze the Sub-question: Read the provided sub-question carefully to understand the exact linear algebra operation or query it represents.
2.  Prioritise Overall User Intent: While focusing on the current sub-question, always keep the `User initial query` in mind. If the sub-question appears to over-complicate a simpler underlying need, or if a more direct tool application aligns better with the original query's intent, prioritise selecting the tool(s) that most directly and efficiently fulfil the `User initial query`.
3.  Evaluate Available Tools: Review the `available_tools` list and their detailed JSON schema definitions. Your selection must be based on the tool's `name`, `description`, and the parameters specified in its `properties`.
4.  Select the Best Fit: Choose the single most appropriate tool that directly addresses the sub-question. Avoid selecting multiple tools if one suffices, focusing on minimal, effective components.
5.  Extract Parameters Precisely: For the selected tool, extract ALL required parameters and any relevant optional parameters directly from the sub-question's phrasing. Ensure the extracted values strictly match the `type` and `format` specified in the tool's JSON schema. For matrices or vectors, ensure they are represented as nested arrays of numbers (e.g., `[[9, 10], [6, 11]]` for a matrix).
6.  Handling Inter-Tool Dependencies: If a sub-question requires the output of a preceding tool call as an input, represent this dependency using the specific placeholder [result_of_call_N], where 'N' corresponds to the sequential number of the tool call whose result is needed. For instance, [result_of_call_1] refers to the output of the first tool executed in the sequence. Ensure this placeholder is used exactly as specified, even if the parameter's expected type is a matrix or vector.
7.  Focus on Selection, Not Solution: DO NOT attempt to perform any linear algebra calculations, solve the problem, or generate any descriptive text about the solution. Your output must ONLY contain the selected tool's name and its extracted parameters.
8.  Output Format: Your response MUST be a valid JSON object.

Schema for Available Tools:
{function_schema}

---
{examples}
---
"""

LIBPATH = "/home/atomwalk12/repos/thesis/development/mathematics_dataset/steps/1.2.8.structured_output_multiple_fn_calls_with_verification_with_completion_carry_messages_improved_prompts_fns.py"
libpath_module = load_module_from_path(LIBPATH)
tools = libpath_module.get_tools()

function_definitions = [tool_info["function"] for tool_info in tools.values()]
function_schema_formatted = json.dumps(function_definitions, indent=2)


prompt = TOOL_SELECTION_PROMPT.format(function_schema=function_schema_formatted, examples=EXAMPLES["linear-algebra"])


if __name__ == "__main__":
    """The following code demonstrates how planning works. The code is not being used for other purposes."""
    llm: OpenAILLM = get_openai_client(
        model="Qwen3-32B-Q4_K_M.gguf", base_url="http://localhost:8000/v1", structured_output={"schema": QueryAnswer}
    )

    with Pipeline("planner-pipeline") as pipeline:
        load_dataset = LoadDataFromDicts(
            name="load_instructions",
            data=[
                {
                    "messages": [
                        {"role": "system", "content": prompt},
                        {
                            "role": "user",
                            "content": "What is the Frobenius norm of the product of matrices [[1, 2], [3, 4]] and [[2, 0], [1, 3]]?",
                        },
                    ]
                },
            ],
        )

        # Create the TextGeneration step
        tool_selection = ChatGeneration(
            name="generate_function_calls",
            llm=llm,
            input_batch_size=8,
            output_mappings={"model_name": "generation_model"},
        )

        # Connect the steps
        load_dataset >> tool_selection

    # Run the pipeline
    distiset: Distiset = pipeline.run(
        parameters={tool_selection.name: {"llm": {"generation_kwargs": {"max_new_tokens": 2048}}}},
        use_cache=False,
    )

    print("The results of the pipeline are:")
    for num, data in enumerate(distiset["default"]["train"]):
        print(f"\n--- Example {num + 1} ---")
        print(f"Generated: {data.get('generation', 'N/A')}")
        print("-" * 50)
