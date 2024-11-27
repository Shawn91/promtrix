from abc import ABC, abstractmethod

from app.features.task_execution.domain.entities import (
    Prompt,
    LLMService,
    ExecutionLLMPrameters,
    Execution,
    Task,
)


class LLMServiceRepository(ABC):
    @abstractmethod
    def execute_task(
        self, prompt: Prompt, llm_service: LLMService, llm_params: ExecutionLLMPrameters, task: Task
    ) -> Execution:
        """Sends a prompt to the LLM and returns the response."""
        pass

