from distilabel.distiset import Distiset
from distilabel.models import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts

from linalg_zero.distillation.components.chat_generation import ChatGeneration
from linalg_zero.distillation.components.planner import PLANNER_PROMPT
from linalg_zero.distillation.components.tool_selection import TOOL_SELECTION_PROMPT
from linalg_zero.distillation.data import QueryAnswer
from linalg_zero.distillation.utils import (
    get_openai_client,
)


def main() -> None:
    """The following code demonstrates how planning works. The code is not being used for other purposes."""
    llm_unstructured: OpenAILLM = get_openai_client(
        model="Qwen3-32B-Q4_K_M.gguf",
        base_url="http://localhost:8000/v1",
    )

    llm_structured: OpenAILLM = get_openai_client(
        model="Qwen3-32B-Q4_K_M.gguf", base_url="http://localhost:8000/v1", structured_output={"schema": QueryAnswer}
    )

    with Pipeline("generation-pipeline") as pipeline:
        load_dataset = LoadDataFromDicts(
            name="load_instructions",
            data=[
                {
                    "messages": [
                        {"role": "system", "content": PLANNER_PROMPT},
                        {
                            "role": "user",
                            "content": "What is the Frobenius norm of the product of matrices [[1, 2], [3, 4]] and [[2, 0], [1, 3]]?",
                        },
                    ]
                },
            ],
        )

        # Create the components required for the pipeline
        planner = ChatGeneration(
            name="planning-step",
            llm=llm_unstructured,
            input_batch_size=8,
            output_mappings={"model_name": "generation_model"},
            use_default_structured_output=False,
            system_prompt=PLANNER_PROMPT,
        )

        tool_selection = ChatGeneration(
            name="tool-selection-step",
            llm=llm_structured,
            input_batch_size=8,
            output_mappings={"model_name": "generation_model"},
            use_default_structured_output=True,
            system_prompt=TOOL_SELECTION_PROMPT,
        )

        # Connect the steps
        load_dataset >> planner >> tool_selection

    # Run the pipeline
    distiset: Distiset = pipeline.run(
        parameters={
            planner.name: {"llm": {"generation_kwargs": {"max_new_tokens": 2048}}},
            tool_selection.name: {"llm": {"generation_kwargs": {"max_new_tokens": 2048}}},
        },
        use_cache=False,
    )

    print("The results of the pipeline are:")
    for num, data in enumerate(distiset["default"]["train"]):
        print(f"\n--- Example {num + 1} ---")
        print(f"Generated: {data.get('generation', 'N/A')}")
        print("-" * 50)


if __name__ == "__main__":
    main()
