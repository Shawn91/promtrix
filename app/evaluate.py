from datetime import datetime

from app.config import Config
from app.entities_models.entities import (
    PromptEntity,
    LLMResponseEntity,
    PromptTemplateEntity,
    LLMParametersEntity,
    ExecutionGroupEntity,
    EvaluationEntity,
)
from app.shared.utils import iterate, asyncio_gather
from app.task_execution import send_to_llm, generate_prompt


evaluate_answer_raw_prompt = """
Given the problem:
{{user_prompt}} 

The correct answer to the above problem is {{expected_answer}}

An AI's reponse to the above problem is:
{{response}} 

Your job is not to solve the above problem, but to evaluate whether the AI's answer is correct against the expected answer. 
Just answer "Correct" or "Incorrect" without any explanations.
""".strip()
evaluate_answer_prompt_template = PromptTemplateEntity(user=evaluate_answer_raw_prompt)


async def evaluate_by_llm(prompt: PromptEntity, response: LLMResponseEntity) -> EvaluationEntity:
    """An evaluation function that requires llms to directly evaluate the response.

    If an LLM service returns multiple choices for one request (when n parameter is set to a value greater than 1),
    the response parameter should be one of the choices returned by the LLM service.
    """
    prompt = list(
        generate_prompt(
            prompt_templates=[evaluate_answer_prompt_template],
            template_variables=[
                {
                    "user_prompt": prompt.user,
                    "expected_answer": prompt.expected_response.content,
                    "response": response.content,
                }
            ],
        )
    )[0]
    start = datetime.now()
    evaluate_answer_execution = await send_to_llm(
        prompt=prompt, llm_params=LLMParametersEntity(api_key=Config.OPENROUTER_API_KEY)
    )
    score = 1 if evaluate_answer_execution.responses[0].content.lower().strip(".") == "correct" else 0
    duration = (datetime.now() - start).microseconds
    evaluation = EvaluationEntity(
        method="evaluate_answer_by_llm",
        score=score,
        duration=duration,
        cost=evaluate_answer_execution.cost,
        steps=[evaluate_answer_execution],
    )
    response.evaluation = evaluation
    return evaluation


async def evaluate_group_by_llm(execution_group: ExecutionGroupEntity, batch_size=5):
    batches = [[]]
    for execution in iterate(execution_group.executions):
        for response in execution.responses:
            batches[-1].append((execution, response))
            if len(batches[-1]) == batch_size:
                batches.append([])
    for batch in batches:
        await asyncio_gather(*[evaluate_by_llm(execution.prompt, res) for (execution, res) in batch])
