from distilabel.distiset import Distiset
from distilabel.models import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import ChatGeneration

from linalg_zero.distillation.utils import get_openai_client

# TODO: refine the prompts with challenging examples
EXAMPLES = {
    "linear-algebra": """Example 1 for Task Decomposition:

User Query: What is the determinant of the matrix [[2, 1], [3, 4]]?

Decomposed Sub-questions:
1. Calculate the determinant of the given matrix.

---
Example 2 for Task Decomposition:

User Query: What is the Frobenius norm of the product of matrices [[1, 2], [3, 4]] and [[2, 1], [1, 3]]?

Decomposed Sub-questions:
1. Multiply the matrices [[1, 2], [3, 4]] and [[2, 1], [1, 3]].
2. Calculate the Frobenius norm of the resulting matrix.

---
Example 3 for Task Decomposition:

User Query: Can you transpose the matrix [[1, 3], [4, 6]] and then calculate its dot product with the vector [1, 2]?

Decomposed Sub-questions:
1. What is the transpose of the matrix [[1, 3], [4, 6]]?
2. What is the dot product of the transposed matrix and the vector?
"""
}

PLANNER_PROMPT = """You are an expert in linear algebra task planning and decomposition, skilled at breaking down complex requests into simpler components to facilitate problem-solving.

Your primary objective is to take a user query requiring multi-step actions and reasoning, and subdivide it into a series of distinct, solvable sub-questions or sub-problems. This decomposition should logically prepare for subsequent stages of a workflow, such as tool selection and eventual execution via tool calling.

Ensure that the original problem is comprehensively covered through the minimal number of sub-questions necessary. Each sub-question should be clear, self-contained, and logically contribute to resolving the overall user query.

For this task, you are strictly to focus on the decomposition of the problem into its constituent sub-questions. Do not generate any tool calls, JSON function calls, or propose solutions or steps beyond the sub-questions themselves, as these aspects will be handled in subsequent stages of the workflow. Specifically, do not attempt to solve the linear algebra problems or perform the calculations yourself by detailing mathematical steps; your role is solely to break them down into conceptual sub-problems that could be addressed by an external linear algebra tool.

Present the decomposed sub-questions as a numbered list.

---
{examples}
---
""".format(examples=EXAMPLES["linear-algebra"])


if __name__ == "__main__":
    """The following code demonstrates how planning works. The code is not being used for other purposes."""
    llm: OpenAILLM = get_openai_client(
        model="Qwen3-32B-Q4_K_M.gguf",
        base_url="http://localhost:8000/v1",
    )

    with Pipeline("planner-pipeline") as pipeline:
        load_dataset = LoadDataFromDicts(
            name="load_instructions",
            data=[
                {
                    "messages": [
                        {"role": "system", "content": PLANNER_PROMPT.format(examples=EXAMPLES["linear-algebra"])},
                        {"role": "user", "content": "What is the trace of the matrix [[1, 2], [3, 4]]?"},
                    ]
                }
            ],
        )

        # Create the TextGeneration step
        planner = ChatGeneration(
            name="task_planning", llm=llm, input_batch_size=8, output_mappings={"model_name": "generation_model"}
        )

        # Connect the steps
        load_dataset >> planner

    # Run the pipeline
    distiset: Distiset = pipeline.run(
        parameters={planner.name: {"llm": {"generation_kwargs": {"max_new_tokens": 1024}}}},
        use_cache=False,
    )

    print("The results of the pipeline are:")
    for num, data in enumerate(distiset["default"]["train"]):
        print(f"\n--- Example {num + 1} ---")
        print(f"Generated: {data['generation']}")
        print("-" * 50)
