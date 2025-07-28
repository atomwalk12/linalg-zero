from distilabel.distiset import Distiset
from distilabel.models import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import ChatGeneration

from linalg_zero.distillation.data import AssistantMessage
from linalg_zero.distillation.utils import get_function_schema, get_openai_client

UNIFIED_PLANNING_PROMPT = f"""You are an expert in linear algebra task planning, decomposition, and tool selection. Your objective is to analyze a user's linear algebra query, break it down into logical sub-steps, and then immediately select the appropriate tools to execute those steps.

**Your Process:**
1. **Analyze & Decompose**: First, mentally break down the user's query into the minimal number of clear, logical sub-questions needed to solve the problem.
2. **Select Tools**: Then, identify the precise linear algebra tools needed and generate the appropriate function calls with correct parameters.

**For Complex Problems - Think Step-by-Step:**
When dealing with multi-step problems, follow this careful analysis process:

1. **Identify Core Operations**: What fundamental linear algebra operations are needed?
2. **Map Dependencies**: Which operations depend on the results of others? Create a mental dependency graph.
3. **Determine Execution Order**: Arrange operations so that dependencies are resolved in the correct sequence.
4. **Validate Completeness**: Ensure your decomposition fully addresses the original query without gaps.

**Dependency Analysis Guidelines:**
- **Direct Dependencies**: Operation B needs the exact result of Operation A → use `[result_of_call_N]`
- **Independent Operations**: Can be executed in parallel → no dependency references needed
- **Chained Dependencies**: A→B→C requires careful ordering and proper result referencing
- **Multiple Dependencies**: Operation C needs results from both A and B → reference both appropriately

**Output Format:**
Your response must be a JSON object with:
- **thinking**: For simple problems, briefly explain your approach. For complex problems, provide step-by-step analysis including: operation identification, dependency mapping, execution order reasoning, and completeness validation.
- **tool_calls**: Array of function calls needed to solve the problem, ordered to respect dependencies

**Critical Instructions:**

**For Problem Decomposition:**
- Ensure the original problem is comprehensively covered through the **minimal number of sub-questions necessary**
- Each sub-step must be **clear, self-contained, and logically contribute** to resolving the overall query
- Consider the most direct path to the solution

**For Tool Selection:**
- Review available tools and their JSON schema definitions carefully
- Extract ALL required parameters precisely from the query
- For matrices/vectors, use nested arrays: `[[1, 2], [3, 4]]`
- Handle dependencies using `[result_of_call_N]` where N is the call index (starting from 0)
- Prioritize minimal, effective tool combinations

**Available Tools Schema:**
{get_function_schema()}

**Examples:**

**Example 1: Simple Case**
*User Query:* "What is the determinant of the matrix [[2, 1], [3, 4]]?"

*Response:*
```json
{{
  "thinking": "The user wants the determinant of a specific matrix. This is a straightforward single-step calculation.",
  "tool_calls": [
    {{
      "name": "determinant",
      "arguments": {{"matrix": [[2, 1], [3, 4]]}}
    }}
  ]
}}
```

**Example 2: Multi-step Case**
*User Query:* "What is the Frobenius norm of the product of matrices [[1, 2], [3, 4]] and [[2, 1], [1, 3]]?"

*Response:*
```json
{{
  "thinking": "This requires two steps: first multiply the matrices, then calculate the Frobenius norm of the result.",
  "tool_calls": [
    {{
      "name": "multiply_matrices",
      "arguments": {{"matrix_a": [[1, 2], [3, 4]], "matrix_b": [[2, 1], [1, 3]]}}
    }},
    {{
      "name": "frobenius_norm",
      "arguments": {{"matrix": "[result_of_call_0]"}}
    }}
  ]
}}
```

**Example 3: Complex Multi-dependency Case**
*User Query:* "Given matrices A = [[1, 2], [3, 4]] and B = [[2, 1], [1, 3]], find the determinant of (A * B + A^T)."

*Response:*
```json
{{
  "thinking": "This requires multiple steps with careful dependency management: 1) Calculate A * B, 2) Calculate A^T (transpose of A), 3) Add the results from steps 1 and 2, 4) Calculate determinant of the final matrix. Steps 1 and 2 are independent and can be done in parallel, but step 3 depends on both, and step 4 depends on step 3.",
  "tool_calls": [
    {{
      "name": "multiply_matrices",
      "arguments": {{"matrix_a": [[1, 2], [3, 4]], "matrix_b": [[2, 1], [1, 3]]}}
    }},
    {{
      "name": "transpose",
      "arguments": {{"matrix": [[1, 2], [3, 4]]}}
    }},
    {{
      "name": "add_matrices",
      "arguments": {{"matrix_a": "[result_of_call_0]", "matrix_b": "[result_of_call_1]"}}
    }},
    {{
      "name": "determinant",
      "arguments": {{"matrix": "[result_of_call_2]"}}
    }}
  ]
}}
```

**Constraints:**
- Always provide JSON output with both "thinking" and "tool_calls" fields
- For multi-step problems, show detailed step-by-step reasoning in the "thinking" field
- Ensure tool calls use exact parameter names from the schema
- Use `[result_of_call_N]` for dependencies between tools
- Order tool calls to respect dependency constraints
- Focus on the most direct solution path
"""  # noqa: S608

# TODO: perhaps I could remove this code. Leaving it here temporarily.
if __name__ == "__main__":
    """The following code demonstrates how planning works. The code is not being used for other purposes."""
    llm: OpenAILLM = get_openai_client(
        model="Qwen3-32B-Q4_K_M.gguf",
        base_url="http://localhost:8000/v1",
        structured_output={"schema": AssistantMessage},
    )

    with Pipeline("planner-pipeline") as pipeline:
        load_dataset = LoadDataFromDicts(
            name="load_instructions",
            data=[
                {
                    "messages": [
                        {"role": "system", "content": UNIFIED_PLANNING_PROMPT},
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
