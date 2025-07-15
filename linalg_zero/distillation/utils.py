from typing import Any

from distilabel.models import OpenAILLM
from distilabel.models.base_clients.openai import SecretStr
from distilabel.pipeline import Pipeline
from distilabel.steps import StepResources
from distilabel.steps.tasks import TextGeneration


def get_openai_client(
    model: str,
    base_url: str,
    timeout: int,
    retries: int,
    generation_kwargs: dict,
) -> OpenAILLM:
    return OpenAILLM(
        base_url=base_url,
        api_key=SecretStr("something"),
        model=model,
        timeout=timeout,
        max_retries=retries,
        generation_kwargs=generation_kwargs,
    )


def build_distilabel_pipeline(
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
) -> Pipeline:
    generation_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}

    if temperature is not None:
        generation_kwargs["temperature"] = temperature

    if top_p is not None:
        generation_kwargs["top_p"] = top_p

    with Pipeline() as pipeline:
        TextGeneration(
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
