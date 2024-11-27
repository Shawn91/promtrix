from datetime import datetime
from uuid import UUID

import litellm

from app.features.execution_persistence.models import Execution as ExecutionModel, LLMResponse
from app.features.task_execution.domain.entities import (
    Prompt,
    LLMService,
    ExecutionLLMPrameters,
    Execution as ExecutionEntity,
    Task,
)
from app.features.task_execution.domain.repositories import LLMServiceRepository


class LLMServiceImpl(LLMServiceRepository):
    def __init__(self):
        pass

    @staticmethod
    def create_execution_model(
        response: dict,
        prompt: Prompt,
        llm_service: LLMService,
        llm_params: ExecutionLLMPrameters,
        task_id: UUID,
        start_time: datetime,
        duration: int,
    ) -> ExecutionModel:
        choices = []
        for choice in response["choices"]:
            choices.append(
                LLMResponse(
                    content=choice["message"]["content"],
                    role=choice["message"]["role"],
                    finish_reason=choice["finish_reason"],
                )
            )
        return ExecutionModel(
            request_id=response.get("id"),
            prompt_id=prompt.id,
            llm_service_id=llm_service.id,
            task_id=task_id,
            duration=duration,
            cost=0,
            response_ids=[choice.id for choice in choices],
            token_count=response["usage"]["completion_tokens"],
            created_at=start_time,
            **llm_params.to_dict(),
        )

    @staticmethod
    def convert_to_entity(
        execution_model: ExecutionModel,
        prompt: Prompt,
        llm_params: ExecutionLLMPrameters,
        llm_service: LLMService,
        task: Task,
    ) -> ExecutionEntity:
        return ExecutionEntity(
            id=execution_model.id,
            request_id=execution_model.request_id,
            prompt=prompt,
            llm_service=llm_service,
            llm_parameters=llm_params,
            duration=execution_model.duration,
            cost=execution_model.cost,
            token_count=execution_model.token_count,
            task=task,
            created_at=execution_model.created_at,
        )

    def execute_task(
        self, prompt: Prompt, llm_service: LLMService, llm_params: ExecutionLLMPrameters, task: Task
    ) -> ExecutionEntity:
        start_time = datetime.now()
        response = self.send_messages(prompt=prompt, llm_service=llm_service, llm_params=llm_params)
        duration = (datetime.now() - start_time).microseconds

        execution_model = self.create_execution_model(
            response=response,
            prompt=prompt,
            llm_service=llm_service,
            llm_params=llm_params,
            task_id=task.id,
            duration=duration,
            start_time=start_time,
        )
        return self.convert_to_entity(
            execution_model, prompt=prompt, llm_params=llm_params, llm_service=llm_service, task=task
        )

    @staticmethod
    def send_messages(prompt: Prompt, llm_service: LLMService, llm_params: ExecutionLLMPrameters) -> dict:
        messages = []
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.user})
        response = litellm.completion(
            model=llm_service.model,
            messages=messages,
            stream=False,
            **llm_params.to_dict(),
        )
        return response.to_dict()
