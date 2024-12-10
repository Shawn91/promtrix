import asyncio
from datetime import datetime
from itertools import product
from typing import Iterator

import litellm
from datasets import (
    load_dataset as load_hf_dataset,
    DatasetDict as HFDatasetDict,
)
from litellm import completion_cost

from app.entities_models.entities import (
    DatasetEntity,
    DatasetSplitEntity,
    PromptTemplateEntity,
    PromptEntity,
    LLMResponseEntity,
    ExecutionEntity,
    ExecutionGroupEntity,
    LLMParametersEntity,
    ExpectedResponseEntity,
)
from app.shared.utils import PROJECT_ROOT, asyncio_gather


def load_dataset(path, name) -> DatasetEntity:
    cache_dir = str(PROJECT_ROOT / "datasets" / "public_datasets")
    raw_dataset = load_hf_dataset(path=path, name=name, cache_dir=cache_dir)
    if isinstance(raw_dataset, HFDatasetDict):
        raw_dataset = raw_dataset["test"]
    splits = [
        DatasetSplitEntity(name=split.name, size=split.num_bytes, count=split.num_examples)
        for _, split in raw_dataset.info.splits.items()
    ]
    return DatasetEntity(
        name=raw_dataset.info.config_name if raw_dataset.info.config_name else raw_dataset.info.dataset_name,
        size=raw_dataset.info.size_in_bytes,
        splits=splits,
        version=raw_dataset.info.version.version_str,
        raw_dataset=raw_dataset,
        raw_dataset_dir=cache_dir,
        created_at=datetime.now(),
    )


def generate_prompt(
    prompt_templates: list[PromptTemplateEntity] | None = None,
    template_variables: list[dict[str, str]] | None = None,
    dataset: DatasetEntity | None = None,
) -> Iterator[PromptEntity]:
    assert (
        template_variables is not None or dataset is not None
    ), "Either template_variables or dataset should be provided"
    assert not (
        template_variables is not None and dataset is not None
    ), "Only one of template_variables or dataset should be provided"
    if prompt_templates is None:
        prompt_templates = [PromptTemplateEntity.create_empty()]
    if dataset is not None:
        template_variables = iter(dataset)
    for prompt_template, variables in product(prompt_templates, template_variables):
        expected_response = None
        if variables.get("output"):
            expected_response = ExpectedResponseEntity(content=variables["output"])
        yield prompt_template.generate_prompt(user=variables, expected_response=expected_response)


def create_execution_entity(
    prompt: PromptEntity, duration: int, response: litellm.ModelResponse, llm_params: LLMParametersEntity | None = None
) -> ExecutionEntity:
    reponse_entities = [
        LLMResponseEntity(
            content=choice.message.content,
            role=choice.message.role,
            finish_reason=choice.finish_reason,
            index=choice.index,
        )
        for choice in response.choices
    ]
    try:
        cost = completion_cost(completion_response=response)
    except litellm.exceptions.NotFoundError:
        cost = None
    return ExecutionEntity(
        prompt=prompt,
        request_id=response.id,
        duration=duration,
        responses=reponse_entities,
        token_count=response.usage.completion_tokens,
        llm_parameters=llm_params,
        cost=cost,
    )


async def send_to_llm(prompt: PromptEntity, llm_params: LLMParametersEntity) -> ExecutionEntity:
    messages = []
    if prompt.system:
        messages.append({"role": "system", "content": prompt.system})
    messages.append({"role": "user", "content": prompt.user})
    start_time = datetime.now()
    response = await litellm.acompletion(
        model="openrouter/deepseek/deepseek-chat",
        messages=messages,
        stream=False,
        api_key=llm_params.api_key,
    )
    duration = (datetime.now() - start_time).microseconds
    prompt.token_count = response.usage.prompt_tokens
    return create_execution_entity(prompt=prompt, duration=duration, llm_params=llm_params, response=response)


async def execute_task(
    llm_params: LLMParametersEntity,
    prompt_templates: list[PromptTemplateEntity] | None = None,
    dataset: DatasetEntity | None = None,
    batch_size: int = 5,
) -> ExecutionGroupEntity:
    llm_requests = []
    executions = []
    start_time = datetime.now()
    for prompts in generate_prompt(prompt_templates=prompt_templates, dataset=dataset):
        llm_requests.append(send_to_llm(prompts, llm_params=llm_params))
        if len(llm_requests) == batch_size:
            executions.extend(await asyncio_gather(*llm_requests))
            llm_requests = []
    if llm_requests:
        executions.extend(await asyncio_gather(*llm_requests))
    duration = (datetime.now() - start_time).microseconds
    return ExecutionGroupEntity(executions=executions, duration=duration)


if __name__ == "__main__":
    from app.config import Config

    dataset = load_dataset(path="lukaemon/bbh", name="boolean_expressions")
    # print(
    #     asyncio.run(
    #         execute_task(
    #             llm_params=LLMParametersEntity(api_key=Config.OPENROUTER_API_KEY),
    #             dataset=dataset,
    #         )
    #     )
    # )
