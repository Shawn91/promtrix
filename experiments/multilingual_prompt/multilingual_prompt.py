import asyncio
from datetime import datetime
from typing import Tuple, Iterator, AsyncIterator

import litellm
from datasets import (
    load_dataset as load_hf_dataset,
    load_from_disk,
)
from litellm import completion_cost
from nicegui import ui

from app.config import PROJECT_ROOT, Config
from app.models.db_models import (
    LLMParametersModel,
    LLMInteractionGroupModel,
    EvaluationGroupModel,
    LLMInteractionModel,
    LLMResponseModel,
    TaskModel,
    PromptModel,
    PromptTemplateModel,
    DatasetModel,
    LLMServiceModel,
    LLMInteractionCategory,
    EvaluationModel,
    EvaluationMetric,
    EvaluationMethod,
    EvaluationStep,
)
from app.repository import (
    RepositoryManager,
)
from app.shared.utils import logger, iterate

repository_manager = RepositoryManager()
repository_manager.init_db_sync(db_path=PROJECT_ROOT / "experiments" / "multilingual_prompt" / "database.sqlite")

litellm.drop_params = True
litellm.register_model(
    {
        "anthropic/claude-3.5-haiku-20241022": {
            "max_tokens": 8192,
            "max_input_tokens": 200000,
            "max_output_tokens": 8192,
            "input_cost_per_token": 0.000001,
            "output_cost_per_token": 0.000005,
            "litellm_provider": "openrouter",
            "mode": "chat",
            "supports_function_calling": True,
            "tool_use_system_prompt_tokens": 264,
        },
        "openrouter/qwen/qwen-2.5-72b-instruct": {
            "max_tokens": 8000,
            "max_input_tokens": 128000,
            "max_output_tokens": 8000,
            "input_cost_per_token": 0.23 / 1000000,
            "output_cost_per_token": 0.4 / 1000000,
            "litellm_provider": "openrouter",
            "mode": "chat",
        },
        "openrouter/openai/gpt-4o-2024-11-20": {
            "max_tokens": 4096,
            "max_input_tokens": 128000,
            "max_output_tokens": 4096,
            "input_cost_per_token": 0.000005,
            "output_cost_per_token": 0.000015,
            "litellm_provider": "openrouter",
            "mode": "chat",
            "supports_function_calling": True,
            "supports_parallel_function_calling": True,
            "supports_vision": True,
        },
        "openrouter/deepseek/deepseek-chat": {
            "max_tokens": 4000,
            "max_input_tokens": 64000,
            "max_output_tokens": 4000,
            "input_cost_per_token": 0.14 / 1000000,
            "output_cost_per_token": 0.28 / 1000000,
            "litellm_provider": "openrouter",
            "mode": "chat",
        },
    }
)
TASK_NAME = "test multilingual prompt"

LLM_PARAMS = LLMParametersModel(
    api_key=Config.OPENROUTER_API_KEY,
    seed=199199,
    temperature=0,
)

LLM_PARAMS_WITH_FP8_QUANTIZATION = LLMParametersModel(
    api_key=Config.OPENROUTER_API_KEY,
    seed=199199,
    temperature=0,
    custom_params={"provider": {"quantizations": ["fp8"], "allow_fallbacks": False}},
)
LLM_PARAMS_WITH_BF16_QUANTIZATION = LLMParametersModel(
    api_key=Config.OPENROUTER_API_KEY,
    seed=199199,
    temperature=0,
    custom_params={"provider": {"quantizations": ["bf16"], "allow_fallbacks": False}},
)

LLM_SERVICES = {
    "claude": {
        "service": LLMServiceModel(
            llm="claude-3.5-sonnet",
            llm_version="20241022",
            llm_provider="Anthropic",
            service_gateway="OpenRouter",
            api_endpoint="openrouter/anthropic/claude-3.5-sonnet",
        ),
        "params": LLM_PARAMS,
    },
    "gpt4o": {
        "service": LLMServiceModel(
            llm="gpt-4o",
            llm_version="20241120",
            llm_provider="OpenAI",
            service_gateway="OpenRouter",
            api_endpoint="openrouter/openai/gpt-4o-2024-11-20",
        ),
        "params": LLM_PARAMS,
    },
    "gemini": {
        "service": LLMServiceModel(
            llm="gemini-pro-1.5",
            llm_provider="Google",
            service_gateway="OpenRouter",
            api_endpoint="openrouter/google/gemini-pro-1.5",
        ),
        "params": LLM_PARAMS,
    },
    "qwen": {
        "service": LLMServiceModel(
            llm="qwen-2.5-72b-instruct",
            llm_provider="Alibaba",
            service_gateway="OpenRouter",
            quantization="bf16",
            api_endpoint="openrouter/qwen/qwen-2.5-72b-instruct",
        ),
        "params": LLM_PARAMS_WITH_BF16_QUANTIZATION,
    },
    "deepseek": {
        "service": LLMServiceModel(
            llm="deepseek-v3",
            llm_version="v3",
            llm_provider="DeepSeek",
            service_gateway="OpenRouter",
            quantization="fp8",
            api_endpoint="openrouter/deepseek/deepseek-chat",
        ),
        "params": LLM_PARAMS_WITH_FP8_QUANTIZATION,
    },
}

EVALUATION_LLM_SERVICE = {
    "model": LLMServiceModel(
        llm="anthropic/claude-3.5-haiku-20241022",
        llm_version="20241022",
        llm_provider="Anthropic",
        service_gateway="OpenRouter",
        api_endpoint="openrouter/anthropic/claude-3.5-haiku-20241022",
    ),
    "params": LLM_PARAMS,
}


def download_dataset():
    """download the high school math subdataset of CMMLU dataset"""
    load_hf_dataset(
        path="haonan-li/cmmlu",
        name="high_school_mathematics",
        cache_dir=str(PROJECT_ROOT / "datasets" / "public_datasets"),
    ).save_to_disk(str(PROJECT_ROOT / "datasets" / "public_datasets" / "cmmlu" / "high_school_mathematics"))


async def load_dataset() -> DatasetModel:
    """load the high school math subdataset of CMMLU dataset"""
    subdataset_dir = PROJECT_ROOT / "datasets" / "public_datasets" / "cmmlu" / "high_school_mathematics"
    cmmlu_dataset = DatasetModel(name="CMMLU", raw_dataset_dir=str(subdataset_dir.parent), is_split=False)
    raw_subdataset = load_from_disk(subdataset_dir)
    subdataset_dev = DatasetModel(
        name="dev",
        raw_dataset_dir=str(subdataset_dir / "dev"),
        is_split=True,
        count=raw_subdataset["dev"].num_rows,
    )
    subdataset_test = DatasetModel(
        name="test",
        raw_dataset_dir=str(subdataset_dir / "test"),
        is_split=True,
        count=raw_subdataset["test"].num_rows,
    )

    subdataset = DatasetModel(
        raw_dataset_dir=str(subdataset_dir),
        name="high_school_mathematics",
    )
    async with repository_manager.transaction() as session:
        cmmlu_dataset, cmmlu_dataset_created = await repository_manager.save(model=cmmlu_dataset, session=session)
        subdataset, subdataset_created = await repository_manager.save(model=subdataset, session=session)
        subdataset_dev, subdataset_dev_created = await repository_manager.save(model=subdataset_dev, session=session)
        subdataset_test, subdataset_test_created = await repository_manager.save(
            model=subdataset_test, session=session
        )
        assert cmmlu_dataset_created == subdataset_created == subdataset_test_created == subdataset_dev_created
        if not cmmlu_dataset_created:
            subdataset_test.parent_id = subdataset.id
            subdataset_dev.parent_id = subdataset.id
            subdataset.parent_id = cmmlu_dataset.id
            await repository_manager.save(model=cmmlu_dataset, session=session)
            subdataset, subdataset_created = await repository_manager.save(model=subdataset, session=session)
            await repository_manager.save(model=subdataset_dev, session=session)
            await repository_manager.save(model=subdataset_test, session=session)
        return subdataset


async def iter_llm_services() -> AsyncIterator[Tuple[LLMServiceModel, LLMParametersModel]]:
    """iterate over all available LLM services"""
    for service_name, service_and_params in LLM_SERVICES.items():
        # create new service in database if it doesn't exist
        await repository_manager.save(service_and_params["service"])
        yield service_and_params["service"], service_and_params["params"]


async def create_prompt_templates() -> dict[str, PromptTemplateModel]:
    raw_prompt = """{{Question}}
A. {{A}}
B. {{B}}
C. {{C}}
D. {{D}}"""
    prompt_template_english = PromptTemplateModel(
        user=f"""The following is a multiple-choice question about high school mathematics. Think step by step and find the correct answer.
Question: {raw_prompt}
The correct answer is:"""
    )
    prompt_template_chinese = PromptTemplateModel(
        user=f"""以下是关于高中数学的单项选择题，逐步分析并选出正确答案。
题目： {raw_prompt}
答案是："""
    )
    prompt_template_no_cot = PromptTemplateModel(
        user=f"""以下是关于高中数学的单项选择题
题目： {raw_prompt}
答案是：""",
    )
    await repository_manager.save_many([prompt_template_english, prompt_template_chinese, prompt_template_no_cot])
    return {"english": prompt_template_english, "chinese": prompt_template_chinese, "no_cot": prompt_template_no_cot}


def generate_prompts(
    prompt_templates: dict[str, PromptTemplateModel], dataset: DatasetModel
) -> Iterator[Tuple[str, PromptModel]]:
    """generate prompts for a given dataset"""
    for example in dataset:
        for prompt_name, prompt_template in prompt_templates.items():
            prompt_model = PromptModel(
                template_id=prompt_template.id,
                user=prompt_template.user_template.render(
                    Question=example["Question"], A=example["A"], B=example["B"], C=example["C"], D=example["D"]
                ),
                template=prompt_template,
                expected_response=example["Answer"],
            )
            yield prompt_name, prompt_model


async def create_task() -> TaskModel:
    task = TaskModel(name=TASK_NAME)
    await repository_manager.save(task)
    return task


async def create_llm_interaction_group(task: TaskModel, name: str):
    interaction_group = LLMInteractionGroupModel(name=name, task_id=task.id)
    await repository_manager.save(interaction_group)
    return interaction_group


class Evaluator:
    evaluate_answer_raw_prompt = """
Given the problem:
{{user_prompt}} 

The correct answer to the above problem is {{expected_response}}

An AI's reponse to the above problem is:
{{response}} 

Your job is not to solve the above problem, but to evaluate whether the AI's answer is correct against the expected answer. 
Just answer "Correct" or "Incorrect" without any explanations.
    """.strip()
    evaluate_answer_prompt_template = PromptTemplateModel(user=evaluate_answer_raw_prompt)

    async def init(self):
        await repository_manager.save(self.evaluate_answer_prompt_template)

    async def evaluate_llm_interaction_by_llm(
        self,
        llm_interaction: LLMInteractionModel,
        evaluation_group: EvaluationGroupModel,
        llm_service: LLMServiceModel,
        llm_params: LLMParametersModel,
    ):
        async with repository_manager.transaction() as session:
            # todo: how to prevent duplicate evaluations
            for response in llm_interaction.llm_responses:
                # evaluation = await evaluation_repository.find_by_llm_response(llm_response=response)
                # if evaluation:
                #     continue
                evaluation_prompt = self.evaluate_answer_prompt_template.generate_prompt(
                    user=dict(
                        user_prompt=llm_interaction.prompt.user,
                        expected_response=llm_interaction.prompt.expected_response,
                        response=response.content,
                    )
                )
                evaluation_prompt, _ = await repository_manager.save(evaluation_prompt, session=session)
                # use llm to evaluate the response
                response_llm_interaction = LLMInteractionModel(
                    prompt_id=evaluation_prompt.id,
                    llm_service_id=llm_service.id,
                    llm_params_id=llm_params.id,
                    category=LLMInteractionCategory.EVALUATION,
                )
                response_llm_interaction = await run_llm_interaction(response_llm_interaction)
                response.llm_interaction_id = response_llm_interaction.id
                if response.evaluations is None:
                    response.evaluations = []
                score = -1
                if response_llm_interaction.llm_responses[0].content.lower() == "correct":
                    score = 1
                elif response_llm_interaction.llm_responses[0].content.lower() == "incorrect":
                    score = 0
                evaluation = EvaluationModel(
                    group_id=evaluation_group.id,
                    metric=EvaluationMetric.CORRECTNESS,
                    duration=response_llm_interaction.duration,
                    cost=response_llm_interaction.cost,
                    score=score,
                    llm_response_id=response.id,
                    steps=[EvaluationStep(execution_id=response_llm_interaction.id, method=EvaluationMethod.LLM)],
                )
                await repository_manager.save(evaluation, session=session)
                await session.flush()
                await repository_manager.refresh(response)

    async def evaluate_llm_interaction_group_by_llm(
        self,
        group_name: str,
        llm_interaction_group: LLMInteractionGroupModel,
        llm_service: LLMServiceModel,
        llm_params: LLMParametersModel,
        evaluate_batch_size=10,
    ) -> EvaluationGroupModel:
        evaluation_group = EvaluationGroupModel(
            name=group_name,
            llm_interaction_group_id=llm_interaction_group.id,
        )
        await repository_manager.save(evaluation_group)

        # Collect all interactions that need evaluation
        pending_evaluations = []
        for llm_interaction in iterate(llm_interaction_group.llm_interactions, desc="Evaluating LLM interactions: "):
            pending_evaluations.append(llm_interaction)
            if len(pending_evaluations) >= evaluate_batch_size:
                await asyncio.gather(
                    *[
                        self.evaluate_llm_interaction_by_llm(
                            llm_interaction=interaction,
                            evaluation_group=evaluation_group,
                            llm_service=llm_service,
                            llm_params=llm_params,
                        )
                        for interaction in pending_evaluations
                    ]
                )
                pending_evaluations = []
        if pending_evaluations:
            await asyncio.gather(
                *[
                    self.evaluate_llm_interaction_by_llm(
                        llm_interaction=interaction,
                        evaluation_group=evaluation_group,
                        llm_service=llm_service,
                        llm_params=llm_params,
                    )
                    for interaction in pending_evaluations
                ]
            )
        await repository_manager.refresh(evaluation_group)
        return evaluation_group


async def run_llm_interaction(llm_interaction: LLMInteractionModel) -> LLMInteractionModel:
    """execute a task for a given prompt and LLM service
    Note: this function assumes that the prompt, LLM service and llm_interaction group have already been saved in the database
    """
    prompt = llm_interaction.prompt
    llm_service = llm_interaction.llm_service
    llm_params = llm_interaction.llm_params
    messages = []
    if prompt.system:
        messages.append({"role": "system", "content": prompt.system})
    messages.append({"role": "user", "content": prompt.user})
    start_time = datetime.now()
    raw_params = llm_params.model_dump(exclude_none=True, exclude={"custom_params"})
    if llm_params.custom_params:
        response = await litellm.acompletion(
            model=llm_service.api_endpoint,
            messages=messages,
            stream=False,
            extra_body=llm_params.custom_params,
            **raw_params,
        )
    else:
        response = await litellm.acompletion(
            model=llm_service.api_endpoint, messages=messages, stream=False, **raw_params
        )
    duration = (datetime.now() - start_time).microseconds

    # transform litellm response to LLMInteractionModel
    reponse_entities = [
        LLMResponseModel(
            content=choice.message.content,
            role=choice.message.role,
            finish_reason=choice.finish_reason,
            index=choice.index,
            llm_interaction_id=llm_interaction.id,
        )
        for choice in response.choices
    ]
    try:
        cost = completion_cost(completion_response=response)
    except litellm.exceptions.NotFoundError:
        cost = None
    llm_interaction.request_id = response.id
    llm_interaction.duration = duration
    llm_interaction.responses = reponse_entities
    llm_interaction.token_count = response.usage.completion_tokens
    llm_interaction.cost = cost
    llm_interaction, _ = await repository_manager.save(llm_interaction)
    return llm_interaction


async def execute_task(
    group_name: str, llm_interaction_batch_size=10
) -> Tuple[LLMInteractionGroupModel, EvaluationGroupModel]:
    task = await create_task()
    llm_interaction_group = await create_llm_interaction_group(task, group_name)
    dataset = await load_dataset()
    prompt_templates = await create_prompt_templates()
    evaluator = Evaluator()
    # Collect all services and params first
    servies_and_params = [x async for x in iter_llm_services()]

    # Generate all prompts first and store them in a list
    all_prompts = list(generate_prompts(prompt_templates, dataset.splits["test"]))

    # Process all combinations of services and prompts in batches
    pending_interactions = []
    for llm_service, llm_params in iterate(servies_and_params, desc="Using LLM services: "):
        for prompt_name, prompt in iterate(all_prompts, desc="Testing prompts: ", total=len(all_prompts)):
            await repository_manager.save(prompt)
            llm_interaction = LLMInteractionModel(
                prompt_id=prompt.id,
                llm_service_id=llm_service.id,
                llm_params_id=llm_params.id,
                group_id=llm_interaction_group.id,
            )
            existed_llm_interaction = await repository_manager.get(llm_interaction)
            if existed_llm_interaction:
                continue
            pending_interactions.append(llm_interaction)
            # Process in batches
            if len(pending_interactions) >= llm_interaction_batch_size:
                await asyncio.gather(*[run_llm_interaction(interaction) for interaction in pending_interactions])
                pending_interactions = []
            if await repository_manager.get(llm_interaction):
                logger.info(f"LLM interaction {llm_interaction.id} already exists, skipping")
            else:
                await run_llm_interaction(llm_interaction)
            break
        break
    # Process any remaining interactions
    if pending_interactions:
        await asyncio.gather(*[run_llm_interaction(interaction) for interaction in pending_interactions])
    evaluation_group = await evaluator.evaluate_llm_interaction_group_by_llm(
        group_name=group_name,
        llm_interaction_group=llm_interaction_group,
        evaluate_batch_size=llm_interaction_batch_size,
        llm_service=EVALUATION_LLM_SERVICE["model"],
        llm_params=EVALUATION_LLM_SERVICE["params"],
    )
    return llm_interaction_group, evaluation_group


class ResultPresenter:
    def __init__(self, llm_interaction_group: LLMInteractionGroupModel, evaluation_group: EvaluationGroupModel):
        self.columns = [
            {"name": "ID", "label": "ID", "field": "id", "align": "left"},
            {"name": "LLM", "label": "LLM", "field": "llm", "align": "left"},
            {"name": "SystemPrompt", "label": "System Prompt", "field": "system prompt", "align": "left"},
            {"name": "UserPrompt", "label": "User Prompt", "field": "user prompt", "align": "left"},
            {"name": "Response", "label": "Response", "field": "response", "align": "left"},
            {"name": "Evaluation", "label": "Evaluation", "field": "evaluation", "align": "left"},
        ]
        self.data = ResultPresenter.create_data(llm_interaction_group, evaluation_group)

    @staticmethod
    def create_data(llm_interaction_group: LLMInteractionGroupModel, evaluation_group: EvaluationGroupModel):
        evaluation_ids = {e.id for e in evaluation_group.evaluations}
        data = []
        for interaction in llm_interaction_group.llm_interactions:
            for response in interaction.responses:
                evaluation = [e for e in response.evaluations if e.id in evaluation_ids][0]
                data.append(
                    {
                        "id": response.id,
                        "llm": interaction.llm_service.llm,
                        "system prompt": interaction.prompt.system,
                        "user prompt": interaction.prompt.user,
                        "response": response.content,
                        "evaluation": evaluation.llm_response.content,
                    }
                )
        return data

    def render(self):
        ui.table(columns=self.columns, rows=self.data, row_key="id")


if __name__ in ["__main__"]:
    # download_dataset()
    async def main():
        await repository_manager.save(EVALUATION_LLM_SERVICE["model"])
        llm_interaction_group, evaluation_group = await execute_task("1", llm_interaction_batch_size=3)
        # ResultPresenter(llm_interaction_group, evaluation_group).render()
        # ui.run()

    asyncio.run(main())
