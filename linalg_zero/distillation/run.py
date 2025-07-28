import os

from distilabel.distiset import Distiset
from distilabel.models import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts

from linalg_zero.distillation.components.chat_generation import ChatGeneration
from linalg_zero.distillation.components.execution_checker import LinAlgZeroExecutionChecker
from linalg_zero.distillation.components.planner_for_tool_calling import UNIFIED_PLANNING_PROMPT
from linalg_zero.distillation.components.result_synthesiser import RESULT_SUMMARIZER_PROMPT
from linalg_zero.distillation.data import AssistantMessage
from linalg_zero.distillation.utils import (
    get_libpath,
    get_openai_client,
    get_patched_openai_client,
)


def main() -> None:
    """The following code demonstrates how planning works. The code is not being used for other purposes."""
    USING_VLLM = os.environ.get("USING_VLLM", "False").lower() == "true"

    llm_structured: OpenAILLM = get_openai_client(
        model="Qwen3-32B-Q4_K_M.gguf",
        base_url="http://localhost:8000/v1",
        structured_output={"schema": AssistantMessage},
    )

    llm_tools: OpenAILLM = get_patched_openai_client(
        model="Qwen3-32B-Q4_K_M.gguf", base_url="http://localhost:8000/v1"
    )

    with Pipeline("generation-pipeline") as pipeline:
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

        # Step 1: planning and tool selection
        tool_selection = ChatGeneration(
            name="tool-selection-step",
            llm=llm_structured,
            input_batch_size=8,
            output_mappings={"model_name": "tool_selection_model", "generation": "answers"},
            use_default_structured_output=True,
            tool_calls=True,
            system_prompt=UNIFIED_PLANNING_PROMPT,
        )

        # Step 2: code execution
        execution_checker = LinAlgZeroExecutionChecker(
            name="verify_function_execution",
            libpath=str(get_libpath()),
            check_is_dangerous=True,
        )

        # Step 3: result summarization
        result_summarizer = ChatGeneration(
            name="summarize_results",
            llm=llm_tools,
            input_batch_size=8,
            system_prompt=RESULT_SUMMARIZER_PROMPT,
            output_mappings={"model_name": "summary_model"},
            tool_calls=False,
            thinking_mode="/no_think",
        )

        # Connect the steps
        load_dataset >> tool_selection >> execution_checker >> result_summarizer

    # Run the pipeline
    enable_thinking = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}} if USING_VLLM else {}
    distiset: Distiset = pipeline.run(
        parameters={
            tool_selection.name: {"llm": {"generation_kwargs": {"max_new_tokens": 4096, **enable_thinking}}},
            result_summarizer.name: {"llm": {"generation_kwargs": {"max_new_tokens": 2048, **enable_thinking}}},
        },
        use_cache=False,
    )

    print("The results of the pipeline are:")
    for num, data in enumerate(distiset["default"]["train"]):
        print(f"\n--- Example {num + 1} ---")
        print(f"Generated: {data['messages'][-1]['content']}")
        print("-" * 50)


if __name__ == "__main__":
    main()
