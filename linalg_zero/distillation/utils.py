import json
from collections.abc import Generator
from pathlib import Path
from types import ModuleType
from typing import (
    Any,
)

from distilabel.models import OpenAILLM
from distilabel.models.base_clients.openai import SecretStr
from distilabel.pipeline import Pipeline, RayPipeline
from distilabel.steps import CombineOutputs, DataSampler, LoadDataFromDicts, StepResources
from distilabel.steps.tasks import (
    APIGenExecutionChecker,
    APIGenGenerator,
    APIGenSemanticChecker,
    TextGeneration,
)
from distilabel.steps.tasks.apigen.execution_checker import load_module_from_path
from distilabel.steps.tasks.apigen.utils import PrepareExamples
from distilabel.typing import FormattedInput
from pydantic import NonNegativeInt, PositiveInt
from typing_extensions import override

from datasets import Dataset  # type: ignore[attr-defined]


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
    ):
        """Override agenerate to bypass validation and support tool calls."""

        # Handle string input
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


def get_patched_openai_client(
    model: str,
    base_url: str,
    timeout: int = 900,
    retries: int = 3,
    generation_kwargs: dict[str, Any] | None = None,
    structured_output: dict[str, Any] | None = None,
) -> OpenAILLM:
    return CustomOpenAILLM(
        base_url=base_url,
        api_key=SecretStr("something"),
        model=model,
        timeout=timeout,
        max_retries=retries,
        generation_kwargs=generation_kwargs,
        structured_output=structured_output,
    )


def get_openai_client(
    model: str,
    base_url: str,
    timeout: int = 900,
    retries: int = 3,
    generation_kwargs: dict[str, Any] | None = None,
    structured_output: dict[str, Any] | None = None,
) -> OpenAILLM:
    return OpenAILLM(
        base_url=base_url,
        api_key=SecretStr("something"),
        model=model,
        timeout=timeout,
        max_retries=retries,
        generation_kwargs=generation_kwargs,
        structured_output=structured_output,
    )


def build_fc_pipeline(
    model: str,
    dataset: Dataset,
    target_fns: list[dict[str, Any]],
    base_url: str = "http://localhost:8000/v1",
    temperature: float | None = None,
    top_p: float | None = None,
    max_new_tokens: int = 8192,
    timeout: int = 900,
    retries: int = 0,
) -> Pipeline:
    """Builds a pipeline for function calling. This is called prior to the generation pipeline."""
    generation_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}

    if temperature is not None:
        generation_kwargs["temperature"] = temperature

    if top_p is not None:
        generation_kwargs["top_p"] = top_p

    with Pipeline(name="APIGenPipeline") as pipeline:
        loader_seeds = LoadDataFromDicts(data=target_fns)
        sampler = DataSampler(
            data=dataset,
            size=2,
            samples=len(target_fns),
            batch_size=8,
        )

        prep_examples = PrepareExamples()

        llm = get_openai_client(
            model=model,
            base_url=base_url,
            timeout=timeout,
            retries=retries,
            generation_kwargs=generation_kwargs,
        )
        apigen = APIGenGenerator(
            llm=llm,
            use_default_structured_output=True,
        )
        combine_steps = CombineOutputs()

        execution_checker = APIGenExecutionChecker(libpath=str(get_libpath()))
        semantic_checker = APIGenSemanticChecker(llm=llm)

        sampler >> prep_examples
        ([loader_seeds, prep_examples] >> combine_steps >> apigen >> execution_checker >> semantic_checker)

    return pipeline


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


def prepare_tools() -> Any:
    """Prepares the tools for function calling."""

    # This module is used internally by the distilabel library to make dynamic calls while building the dataset.
    libpath = get_libpath()
    libpath_module: ModuleType = load_module_from_path(libpath)
    tools = libpath_module.get_tools()

    target_fns = _get_target_fns(tools)

    return tools, target_fns


def build_fc_dataset(tools: dict[str, dict[str, Any]]) -> Any:
    """Builds a dataset for function calling."""

    def gen() -> Generator[dict[str, Any], None, None]:
        yield {
            "query": "Calculate 100 divided by 20, then multiply the result by 2.",
            "answers": ["100/20 = 5, then 5*2 = 10"],
            "tools": [key["function"] for key in tools.values()],
        }

    return Dataset.from_generator(gen)


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
    generation_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}

    if temperature is not None:
        generation_kwargs["temperature"] = temperature

    if top_p is not None:
        generation_kwargs["top_p"] = top_p

    with Pipeline().ray() as pipeline:
        _ = TextGeneration(
            llm=get_openai_client(
                model=model,
                base_url=base_url,
                timeout=timeout,
                retries=retries,
                generation_kwargs=generation_kwargs,
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
