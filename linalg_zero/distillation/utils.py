import json
import logging
import logging as stdlib_logging
from typing import (
    Any,
)

import argilla as rg
from distilabel.distiset import Distiset
from distilabel.models import OpenAILLM
from distilabel.models.base_clients.openai import SecretStr
from distilabel.pipeline import Pipeline, RayPipeline
from distilabel.steps import StepResources
from distilabel.steps.tasks import (
    TextGeneration,
)
from distilabel.steps.tasks.apigen.execution_checker import load_module_from_path
from distilabel.typing import FormattedInput, GenerateOutput
from pydantic import BaseModel, NonNegativeInt, PositiveInt
from typing_extensions import override

from datasets import load_dataset as hf_load_dataset
from linalg_zero.config.data import (
    DistillationConfig,
    LlamaCppServerConfig,
    VllmServerConfig,
)
from linalg_zero.shared.lib import get_tools
from linalg_zero.shared.utils import get_libpath, get_logger, setup_logging

logger = get_logger(__name__)


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
    server: LlamaCppServerConfig | VllmServerConfig, args: DistillationConfig, schema: type[BaseModel]
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

    llm = get_openai_client(**base_params, structured_output=None)
    llm_structured = get_openai_client(**base_params, structured_output={"schema": schema})

    return llm, llm_structured


def get_function_schema() -> str:
    """Returns the tools for function calling."""
    libpath_module = load_module_from_path(get_libpath())
    tools = libpath_module.get_tools()

    function_definitions = [tool_info["function"] for tool_info in tools]
    function_schema = json.dumps(function_definitions, indent=2)

    return function_schema


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
                name="query",
                title="User's Linear Algebra Problem Query",
                use_markdown=False,
            ),
            rg.TextField(
                name="ground_truth",
                title="Ground Truth Result",
                use_markdown=False,
            ),
            rg.TextField(
                name="stepwise_ground_truths",
                title="Stepwise Ground Truth Solutions",
                use_markdown=False,
            ),
            rg.TextField(
                name="tool_calls",
                title="Number of Tool Calls Made",
                use_markdown=False,
            ),
            rg.TextField(
                name="problem_type",
                title="Problem Type",
                use_markdown=False,
            ),
            rg.TextField(
                name="composition_type",
                title="Composition Type",
                use_markdown=False,
            ),
            rg.TextField(
                name="composition_dependencies",
                title="Composition Dependencies",
                use_markdown=False,
            ),
            rg.TextField(
                name="conversation",
                title="Full Conversation",
                use_markdown=True,
            ),
            rg.TextField(
                name="dependency_edges",
                title="Dependency Edges",
                use_markdown=False,
            ),
            rg.TextField(
                name="final_answer",
                title="Model's Final Answer",
                use_markdown=False,
            ),
            rg.TextField(
                name="is_correct",
                title="Is Answer Correct?",
                use_markdown=False,
            ),
            rg.TextField(
                name="model_name",
                title="Model Name Used",
                use_markdown=False,
            ),
            rg.TextField(
                name="diagnostics",
                title="Diagnostics (per turn)",
                use_markdown=False,
            ),
            rg.TextField(
                name="diagnostic_messages",
                title="Diagnostic raw messages (failed turns)",
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


def _format_indexed_list(items: list[Any]) -> str:
    """Format a list with indexed headers and separators for better readability."""
    if not items:
        return ""

    indexed_list = []
    for i, item in enumerate(items):
        indexed_list.append({"index": i, "content": item})

    return json.dumps(indexed_list, indent=2)


def _convert_item_to_argilla_record(item: dict[str, Any]) -> dict[str, str] | None:
    """Convert a single distillation item to an Argilla record."""
    logger = get_logger(__name__)
    try:
        # Extract problem from messages
        num_tool_calls = len(json.loads(item.get("stepwise_ground_truths", "N/A")))
        # Get diagnostics and find diagnostics_* key if present
        metadata = item.get("distilabel_metadata", {})
        diagnostics_key = next((k for k in metadata if k.startswith("diagnostics_")), None)
        diagnostic_msgs_key = next((k for k in metadata if k.startswith("diagnostic_messages_")), None)
        diagnostics_list = metadata.get(diagnostics_key, []) if diagnostics_key else []
        diagnostic_msgs_list = metadata.get(diagnostic_msgs_key, []) if diagnostic_msgs_key else []

        return {
            "query": str(item.get("query", "N/A")),
            "ground_truth": str(item.get("ground_truth", "N/A")),
            "stepwise_ground_truths": str(item.get("stepwise_ground_truths", "N/A")),
            "tool_calls": str(num_tool_calls),
            "problem_type": str(item.get("problem_type", "N/A")),
            "composition_type": str(item.get("composition_type", "N/A")),
            "composition_dependencies": str(item.get("composition_dependencies", "N/A")),
            "conversation": json.dumps(item.get("conversation", "N/A"), indent=2)
            if item.get("conversation") != "N/A"
            else "N/A",
            "dependency_edges": str(item.get("dependency_edges", "N/A")),
            "final_answer": str(item.get("final_answer", "N/A")),
            "is_correct": str(item.get("is_correct", "N/A")),
            "model_name": str(item.get("model_name", "N/A")),
            "diagnostics": _format_indexed_list(diagnostics_list),
            "diagnostic_messages": _format_indexed_list(diagnostic_msgs_list),
        }
    except Exception as e:
        logger.warning(f"Failed to process record: {e}")
        return None


def create_argilla_dataset(
    dataset_name: str, distiset_data: list[dict[str, Any]], client: rg.Argilla, private: bool
) -> None:
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
        domain = dataset_name.replace("/", "-").replace("-debug", "").replace("-train", "").replace("-validation", "")
        logger.info("✅ Argilla dataset created successfully")
        logger.info(f"   Privacy: {'Private' if private else 'Public'}")
        logger.info(f"   Access URL: https://{domain}.hf.space")
    except Exception:
        logger.exception("Failed to create Argilla dataset")
        raise


def push_to_huggingface(distiset: Distiset, dataset_name: str, private: bool) -> None:
    prepare_dataset_for_sft(distiset)
    strip_diagnostic_messages_from_metadata(distiset)
    normalize_schema(distiset)

    distiset.push_to_hub(
        dataset_name,
        private=private,
    )
    logger.info(f"✅ Dataset successfully pushed to: {dataset_name}")
    logger.info(f"   Privacy: {'Private' if private else 'Public'}")
    logger.info(f"   Access URL: https://huggingface.co/datasets/{dataset_name}")


def prepare_dataset_for_sft(distiset: Distiset) -> None:
    """Adds the tools column to the dataset."""
    TOOLS = get_tools()

    def add_tools_column(example: dict[str, Any]) -> dict[str, Any]:
        example["tools"] = TOOLS
        return example

    distiset["default"]["train"] = distiset["default"]["train"].map(add_tools_column)
    if "validation" in distiset["default"]:
        distiset["default"]["validation"] = distiset["default"]["validation"].map(add_tools_column)


def normalize_schema(distiset: Distiset) -> None:
    ns = distiset["default"]

    # 1) Stringify nested columns if present
    for split in list(ns.keys()):
        if "conversation" in ns[split].column_names:
            ns[split] = ns[split].map(lambda r: {"conversation": json.dumps(r.get("conversation", []))})
        if "distilabel_metadata" in ns[split].column_names:
            ns[split] = ns[split].map(lambda r: {"distilabel_metadata": json.dumps(r.get("distilabel_metadata", {}))})

    # 2) Align columns by UNION: add missing columns with empty placeholders
    all_cols = set()
    for split in ns:
        all_cols |= set(ns[split].column_names)

    for split in list(ns.keys()):
        missing = sorted(all_cols - set(ns[split].column_names))
        if missing:
            for col in missing:
                ns[split] = ns[split].add_column(col, [None] * len(ns[split]))


def load_dataset_split(args: DistillationConfig, split: str, take_n: int | None = None) -> list[dict[str, Any]]:
    """Loads a single dataset split either from the hub or from a local file."""
    logger = get_logger(__name__)

    try:
        logger.info(f"Loading '{args.hf_dataset}' (config: {args.hf_dataset_config}, split: {split}) dataset.")

        dataset = hf_load_dataset(args.hf_dataset, args.hf_dataset_config, split=split)

        logger.info("Dataset loaded!")
    except Exception as err:
        raise FileNotFoundError(f"The dataset {args.hf_dataset} is not available on the Hugging Face Hub.") from err
    else:
        if take_n is not None:
            dataset = dataset.select(range(take_n))
        # Convert the dict format back to list of dicts. This is the format expected by Argilla.
        dataset_dict = dataset.to_dict()
        return [dict(zip(dataset_dict.keys(), vals, strict=True)) for vals in zip(*dataset_dict.values(), strict=True)]


def load_datasets(args: DistillationConfig, take_n: int | None) -> dict[str, list[dict[str, Any]]]:
    """Loads train and optionally validation splits as lists of dicts."""
    datasets: dict[str, list[dict[str, Any]]] = {}
    datasets["train"] = load_dataset_split(args, args.hf_dataset_train_split, take_n)
    if args.hf_dataset_validation_split:
        datasets["validation"] = load_dataset_split(args, args.hf_dataset_validation_split, take_n)
    if args.hf_dataset_test_split:
        datasets["test"] = load_dataset_split(args, args.hf_dataset_test_split)
    return datasets


def strip_diagnostic_messages_from_metadata(distiset: Distiset) -> None:
    """Remove diagnostic_messages_* keys from distilabel_metadata for all splits (before HF push)."""
    ns = distiset["default"]

    def strip_md(record: dict[str, Any]) -> dict[str, Any]:
        md = record.get("distilabel_metadata", {})
        if isinstance(md, dict):
            keys_to_remove = [k for k in md if k.startswith("diagnostic_messages_")]
            if keys_to_remove:
                for k in keys_to_remove:
                    md.pop(k, None)
                return {"distilabel_metadata": md}
        return {"distilabel_metadata": md}

    for split in list(ns.keys()):
        if "distilabel_metadata" in ns[split].column_names:
            ns[split] = ns[split].map(strip_md)
