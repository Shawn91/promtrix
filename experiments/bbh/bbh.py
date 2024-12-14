import json

import litellm
from datasets import (
    load_dataset as load_hf_dataset,
    load_from_disk,
)
from litellm.types.utils import ModelResponse
from nicegui import ui

from app.config import Config, PROJECT_ROOT
from app.entities_models.entities import PromptEntity, LLMInteractionGroupEntity
from app.evaluate import evaluate_group_by_llm
from app.shared.utils import asyncio_gather
from app.task_execution import create_llm_interaction_entity

# List of all subdatasets
NAMES = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]


def download_dataset():
    for name in NAMES:
        load_hf_dataset(
            path="lukaemon/bbh",
            name=name,
            cache_dir=str(PROJECT_ROOT / "datasets" / "public_datasets"),
        ).save_to_disk(str(PROJECT_ROOT / "datasets" / "public_datasets" / "bbh" / name))


async def save_raw_llm_response(subdataset_name: str, index: int, question: str, answer: str):
    res = await litellm.acompletion(
        model="openrouter/deepseek/deepseek-chat",
        messages=[{"role": "user", "content": "Let's think step by step. \n" + question}],
        stream=False,
        api_key=Config.OPENROUTER_API_KEY,
    )

    with open(
        PROJECT_ROOT / "experiments" / "bbh" / "raw_llm_responses" / f"{subdataset_name}_{index}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "response": res.to_dict(),
                "question": question,
                "answer": answer,
                "subdataset_name": subdataset_name,
            },
            f,
        )


async def save_sample_questions_and_responses():
    for name in NAMES:
        dataset = load_from_disk(str(PROJECT_ROOT / "datasets" / "public_datasets" / "bbh" / name))
        tasks = []
        for i, example in enumerate(dataset["test"]):
            if i == 9:  # take 10 examples for each subdataset
                await asyncio_gather(*tasks)
                tasks = []
                print(f"Finished {name}")
            tasks.append(save_raw_llm_response(name, i, example["input"], example["target"]))


def load_sample_questions_and_responses() -> LLMInteractionGroupEntity:
    data_dir = PROJECT_ROOT / "experiments" / "bbh" / "raw_llm_responses"
    all_data: dict[str, list[dict]] = {}  # subdataset_name -> list of data
    for file_name in data_dir.iterdir():
        if file_name.suffix == ".json":
            with open(file_name, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["response"] = ModelResponse(**data["response"])  # check if response is valid
                if not data["subdataset_name"] in all_data:
                    all_data[data["subdataset_name"]] = []
                all_data[data["subdataset_name"]].append(data)
    interactions = []
    for subdataset_name, data in all_data.items():
        for i, example in enumerate(data):
            prompt = PromptEntity(
                user="Let's think step by step. \n" + example["question"],
                expected_response=example["answer"],
            )
            interaction = create_llm_interaction_entity(prompt=prompt, duration=0, response=example["response"])
            interactions.append(interaction)
    return LLMInteractionGroupEntity(llm_interactions=interactions, duration=0)


async def evaluate_and_save_results(llm_interaction_group: LLMInteractionGroupEntity):
    # await evaluate_group_with_llm(execution_group=execution_group)
    await evaluate_group_by_llm(llm_interaction_group=llm_interaction_group, batch_size=3)
    with open(PROJECT_ROOT / "experiments" / "bbh" / "evaluations.json", "w", encoding="utf-8") as f:
        f.write(llm_interaction_group.model_dump_json())


def visualize_llm_interaction_group():
    # load data
    with open(PROJECT_ROOT / "experiments" / "bbh" / "evaluations.json", "r", encoding="utf-8") as f:
        llm_interaction_group = LLMInteractionGroupEntity.model_validate_json(f.read())
    columns = [
        {
            "name": "Prompt",
            "label": "Prompt",
            ":field": "row => row.prompt.user",
            "align": "left",
            "style": "text-wrap: wrap; white-space: pre-wrap",
        },
        {
            "name": "Response",
            "label": "Response",
            ":field": "row => row.responses[0].content",
            "align": "left",
            "style": "text-wrap: wrap; white-space: pre-wrap",
        },
        {
            "name": "expected_response",
            "label": "Expected Response",
            ":field": "row => row.prompt.expected_response.content",
            "align": "left",
            "style": "text-wrap: wrap; white-space: pre-wrap",
        },
        {
            "name": "evaluation_extracted_response",
            "label": "Evaluation Extracted Response",
            ":field": "row => row.responses[0].evaluation.steps[0].responses[0].content",
            "align": "left",
            "style": "text-wrap: wrap; white-space: pre-wrap",
        },
    ]
    ui.table(
        columns=columns, rows=[interaction.model_dump() for interaction in llm_interaction_group.llm_interactions], row_key="id"
    ).classes("full-width")
    ui.run()


if __name__ in {"__main__", "__mp_main__"}:
    # asyncio.run(save_sample_questions_and_responses())
    # llm_interaction_group = load_sample_questions_and_responses()
    # asyncio.run(evaluate_and_save_results(llm_interaction_group))
    # 改为先保存LLM的结果，再读取然后用 LLM 评估
    visualize_llm_interaction_group()
