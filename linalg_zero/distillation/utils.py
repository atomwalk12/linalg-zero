import json
import logging
import logging as stdlib_logging
from pathlib import Path
from typing import (
    Any,
)

import argilla as rg
from distilabel.models import OpenAILLM
from distilabel.models.base_clients.openai import SecretStr
from distilabel.pipeline import Pipeline, RayPipeline
from distilabel.steps import StepResources
from distilabel.steps.tasks import (
    TextGeneration,
)
from distilabel.steps.tasks.apigen.execution_checker import load_module_from_path
from distilabel.typing import FormattedInput, GenerateOutput
from pydantic import NonNegativeInt, PositiveInt
from typing_extensions import override

from datasets import load_dataset as hf_load_dataset
from linalg_zero.config.data import DistillationConfig, LlamaCppServerConfig, VllmServerConfig
from linalg_zero.distillation.data import AssistantMessage
from linalg_zero.shared import get_logger, setup_logging


# TODO: is this the right file to store this class in?
class CustomOpenAILLM(OpenAILLM):
    """
    Patched OpenAI LLM that supports tool calls by bypassing the restrictive validation.
    This allows using the full OpenAI API format with tool_calls and tool roles.
    """

    @override
    async def agenerate(
        self,
        input: FormattedInput,
        num_generations: int = 1,
        max_new_tokens: NonNegativeInt = 128,
        logprobs: bool = False,
        top_logprobs: PositiveInt | None = None,
        echo: bool = False,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: str | list[str] | None = None,
        response_format: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> GenerateOutput:
        """Override agenerate to bypass validation and support tool calls."""

        if isinstance(input, str):
            return await self._generate_completion(
                input=input,
                num_generations=num_generations,
                max_new_tokens=max_new_tokens,
                echo=echo,
                top_logprobs=top_logprobs,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                temperature=temperature,
                top_p=top_p,
                extra_body=extra_body,
            )

        return await self._generate_chat_completion(
            input=input,
            num_generations=num_generations,
            max_new_tokens=max_new_tokens,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            response_format=response_format,
            extra_body=extra_body,
        )


def load_dataset(args: DistillationConfig) -> list[dict[str, Any]]:
    """Loads the dataset either from the hub or from a local file."""
    logger = get_logger(__name__)

    try:
        logger.info(
            f"Loading '{args.hf_dataset}' (config: {args.hf_dataset_config}, split: {args.hf_dataset_split}) dataset..."
        )

        dataset = hf_load_dataset(args.hf_dataset, args.hf_dataset_config, split=args.hf_dataset_split)
        dataset_dict = dataset.to_dict()
        logger.info("Dataset loaded!")
        # Convert the dict format back to list of dicts
        return [
            dict(zip(dataset_dict.keys(), values, strict=True)) for values in zip(*dataset_dict.values(), strict=True)
        ]
    except Exception:
        logger.exception(f"The dataset {args.hf_dataset} is not available on the Hugging Face Hub.")
        logger.warning("Run `make setup-dev` to push a debugging dataset to the hub, then rerun the script.")
        logger.warning("Make sure to configure the `setup-dev` target to use the correct username.")
        raise FileNotFoundError(f"The dataset {args.hf_dataset} is not available on the Hugging Face Hub.")


def get_openai_client(
    model: str,
    base_url: str,
    timeout: int = 900,
    retries: int = 3,
    max_new_tokens: int = 8192,
    temperature: float | None = None,
    top_p: float | None = None,
    structured_output: dict[str, Any] | None = None,
) -> OpenAILLM:
    generation_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}

    if temperature is not None:
        generation_kwargs["temperature"] = temperature

    if top_p is not None:
        generation_kwargs["top_p"] = top_p

    return CustomOpenAILLM(
        model=model,
        base_url=base_url,
        api_key=SecretStr("not-used"),
        timeout=timeout,
        max_retries=retries,
        generation_kwargs=generation_kwargs,
        structured_output=structured_output,
    )


def create_llm_clients(
    server: LlamaCppServerConfig | VllmServerConfig, args: DistillationConfig
) -> tuple[OpenAILLM, OpenAILLM]:
    """Create structured and non-structured LLM clients."""
    base_params: dict[str, Any] = {
        "model": server.model,
        "base_url": f"http://{server.host}:{server.port}/v1",
        "timeout": args.timeout,
        "retries": args.retries,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    llm_planner = get_openai_client(**base_params, structured_output={"schema": AssistantMessage})
    llm_synthesizer = get_openai_client(**base_params, structured_output=None)

    return llm_planner, llm_synthesizer


def get_function_schema() -> str:
    """Returns the tools for function calling."""
    libpath_module = load_module_from_path(get_libpath())
    tools = libpath_module.get_tools()

    function_definitions = [tool_info["function"] for tool_info in tools.values()]
    function_schema = json.dumps(function_definitions, indent=2)

    return function_schema


def get_libpath() -> Path:
    """Returns the path to the library of functions."""
    return Path(__file__).parent / "fc_fns.py"


def _get_target_fns(tools: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """These are the functions that will interactively be called during training.
    For each element in the dataset, each function below is called once."""
    return [
        {
            "func_name": "get_division",
            "func_desc": "Performs division of a and b, and returns the result.",
            "tools": [tools["get_division"]],
        },
        {
            "func_name": "get_multiplication",
            "func_desc": "Performs multiplication of a and b, and returns the result.",
            "tools": [tools["get_multiplication"]],
        },
    ]


def build_generation_pipeline(
    model: str,
    base_url: str = "http://localhost:8000/v1",
    prompt_column: str | None = None,
    prompt_template: str = "{{ instruction }}",
    temperature: float | None = None,
    top_p: float | None = None,
    max_new_tokens: int = 8192,
    num_generations: int = 1,
    input_batch_size: int = 64,
    client_replicas: int = 1,
    timeout: int = 900,
    retries: int = 0,
) -> Pipeline | RayPipeline:
    """Builds a pipeline for generation. Prior to this, the function calling pipeline is called."""
    with Pipeline().ray() as pipeline:
        _ = TextGeneration(
            llm=get_openai_client(
                model=model,
                base_url=base_url,
                timeout=timeout,
                retries=retries,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            ),
            template=prompt_template,
            input_mappings=({"instruction": prompt_column} if prompt_column is not None else {}),
            input_batch_size=input_batch_size,
            num_generations=num_generations,
            group_generations=True,
            resources=StepResources(replicas=client_replicas),
        )

    return pipeline


def is_openai_format(messages: Any) -> bool:
    """Checks if the input is in OpenAI chat-like format:

    ```python
    [
        {"role": "user", "content": "Turn on the living room lights."},
        {"role": "assistant", "tool_calls": [
            {"type": "function", "function": {
                "name": "control_light",
                "arguments": {"room": "living room", "state": "on"}
            }}]
        },
        {"role": "tool", "name": "control_light", "content": "The lights in the living room are now on."},
        {"role": "assistant", "content": "Done!"}
    ]
    ```

    Args:
        input: The input to check.

    Returns:
        A boolean indicating if the input is in OpenAI chat-like format.
    """
    if not isinstance(messages, list):
        return False
    return all(isinstance(x, dict) and "role" in x and ("content" in x or "tool_calls" in x) for x in messages)


def cleanup() -> None:
    """Cleans up logging to prevent multiprocessing queue errors."""
    root_logger = stdlib_logging.getLogger()
    queue_handlers = [h for h in root_logger.handlers if hasattr(h, "queue")]
    for handler in queue_handlers:
        root_logger.removeHandler(handler)

    # Reinitialize logging
    setup_logging(level=logging.INFO, include_timestamp=True)


def create_argilla_dataset_settings() -> rg.Settings:
    """Create Argilla dataset settings for linear algebra distillation results."""

    return rg.Settings(
        guidelines="""Review and validate the model's reasoning for linear algebra problems.""",
        fields=[
            rg.TextField(
                name="ground_truth",
                title="Ground Truth Result",
                use_markdown=False,
            ),
            rg.TextField(
                name="problem",
                title="User's Linear Algebra Problem",
                use_markdown=False,
            ),
            rg.TextField(
                name="tool_planning_thought",
                title="Model's Tool Planning Thought",
                use_markdown=False,
            ),
            rg.TextField(
                name="tool_calls",
                title="Tool Calls Made",
                use_markdown=False,
            ),
            rg.TextField(
                name="execution_result",
                title="Code Execution Result",
                use_markdown=False,
            ),
            rg.TextField(
                name="keep_row_after_execution_check",
                title="Keep Row After Execution Check",
                use_markdown=False,
            ),
            rg.TextField(
                name="final_answer",
                title="Model's Final Answer",
                use_markdown=False,
            ),
            rg.TextField(
                name="verification_result",
                title="Math-Verify Verification Result",
                use_markdown=False,
            ),
            rg.TextField(
                name="verification_details",
                title="Detailed Verification Information",
                use_markdown=False,
            ),
            rg.TextField(
                name="final_result_correct",
                title="Does the final answer match the ground truth?",
                use_markdown=False,
            ),
            rg.TextField(
                name="keep_row_after_semantic_check",
                title="Keep Row After Semantic Check",
                use_markdown=False,
            ),
        ],
        questions=[
            rg.LabelQuestion(
                name="reasoning_quality",
                title="How would you rate the overall reasoning quality?",
                labels=["excellent", "good", "fair", "poor"],
            ),
            rg.LabelQuestion(
                name="mathematical_accuracy",
                title="Is the mathematical reasoning correct?",
                labels=["correct", "minor_errors", "major_errors", "incorrect"],
            ),
            rg.LabelQuestion(
                name="tool_usage",
                title="Are the tool calls appropriate and effective?",
                labels=["optimal", "good", "suboptimal", "incorrect"],
            ),
            rg.LabelQuestion(
                name="final_correctness",
                title="Is the final answer correct?",
                labels=["correct", "close", "wrong", "no_answer"],
            ),
            rg.TextQuestion(
                name="feedback",
                title="Additional feedback or observations",
            ),
        ],
    )


def _delete_existing_argilla_dataset(client: rg.Argilla, dataset_name: str) -> None:
    """Delete existing Argilla dataset if it exists."""
    logger = get_logger(__name__)
    try:
        existing_dataset = client.datasets(name=dataset_name)
        if existing_dataset:
            existing_dataset.delete()
            logger.info(f"Deleted existing Argilla dataset: {dataset_name}")
    except Exception:
        logger.exception("Failed to delete existing Argilla dataset")
        # Dataset doesn't exist
        pass


def _extract_assistant_messages(messages: list[dict[str, Any]]) -> tuple[str, str, str]:
    """Extract tool planning thought, tool calls, and final answer from assistant messages."""
    tool_planning_thought = ""
    tool_calls = ""
    final_answer = ""

    for msg in messages:
        if msg.get("role") == "assistant":
            if msg.get("tool_calls"):
                tool_calls = str(msg.get("tool_calls", ""))
                if msg.get("content"):
                    tool_planning_thought = msg.get("content", "")
            elif msg.get("content"):
                # Preserve the raw content including special tags like <think></think> and <RESULT></RESULT>
                final_answer = msg.get("content", "")

    return tool_planning_thought, tool_calls, final_answer


def _convert_item_to_argilla_record(item: dict[str, Any]) -> dict[str, str] | None:
    """Convert a single distillation item to an Argilla record."""
    logger = get_logger(__name__)
    try:
        # Extract problem from messages
        problem = ""
        for msg in item.get("messages", []):
            if msg.get("role") == "user":
                problem = msg.get("content", "")
                break

        # Extract assistant message components
        tool_planning_thought, tool_calls, final_answer = _extract_assistant_messages(item.get("messages", []))

        return {
            "problem": problem,
            "ground_truth": str(item.get("ground_truth_result", "")),
            "tool_planning_thought": tool_planning_thought,
            "tool_calls": tool_calls,
            "execution_result": str(item.get("execution_result", "")),
            "final_answer": final_answer,
            "verification_result": str(item.get("verification_result", "")),
            "final_result_correct": str(item.get("final_result_correct", "")),
            "keep_row_after_semantic_check": str(item.get("keep_row_after_semantic_check", "")),
            "verification_details": str(item.get("verification_details", "")),
            "keep_row_after_execution_check": str(item.get("keep_row_after_execution_check", "")),
        }
    except Exception as e:
        logger.warning(f"Failed to process record: {e}")
        return None


def create_argilla_dataset(dataset_name: str, distiset_data: list[dict[str, Any]], client: rg.Argilla) -> None:
    """Create and populate an Argilla dataset from distillation results."""
    logger = get_logger(__name__)

    try:
        # Delete existing dataset if it exists to ensure clean reupload
        _delete_existing_argilla_dataset(client, dataset_name)

        # Create dataset with settings
        settings = create_argilla_dataset_settings()
        dataset = rg.Dataset(
            name=dataset_name,
            settings=settings,
            client=client,
        )
        _ = dataset.create()
        logger.info(f"Created Argilla dataset: {dataset_name}")

        # Convert distilabel data to Argilla records
        records = []
        for item in distiset_data:
            record = _convert_item_to_argilla_record(item)
            if record is not None:
                records.append(record)

        # Log records to dataset
        if records:
            dataset.records.log(records=records)
            logger.info(f"Logged {len(records)} records to Argilla dataset")
        else:
            logger.warning("No valid records found to log")

    except Exception:
        logger.exception("Failed to create Argilla dataset")
        raise
