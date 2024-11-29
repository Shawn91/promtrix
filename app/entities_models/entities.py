"""
The concept of Entity is from Clean Architecture.

It is a core concept of the application that represents a business entity, 
such as a customer, a product, or a purchase.

Entities are supposed to be passed around business logic, 
but do not interact with external systems like databases or APIs.

Entities should be converted to Models before being passed to external systems.

Data retrieved from external systems should be Models and thenconverted to Entities 
before being used in the application.
"""
from typing import Optional, Iterator

import datasets as HFDatasets
from jinja2 import Template
from pydantic import ConfigDict
from sqlmodel import Field, SQLModel

from app.entities_models.base import (
    PromptTemplate,
    Prompt,
    Dataset,
    DatasetSplit,
    LLMResponse,
    Execution,
    ExecutionGroup,
)


class PromptTemplateEntity(PromptTemplate):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    user: Template = Field(description="template for user prompt")
    system: Template | None = None  # Store as string, convert to jinja2 Template when needed
    is_placeholder: bool = False  # True if no template is actually required and a template is created for convenience

    @classmethod
    def create_empty(cls) -> "PromptTemplateEntity":
        template = cls(user=Template("{{input}}"))
        template.is_placeholder = True
        return template

    def generate_prompt(self, user: str, system: str | None = None) -> "PromptEntity":
        prompt_entity = PromptEntity(user=self.user.render(input=user))
        if system is not None:
            prompt_entity.system = self.system.render(system)
        if not self.is_placeholder:
            prompt_entity.template = self
        return prompt_entity


class PromptEntity(Prompt):
    template: PromptTemplateEntity | None = None


class LLMResponseEntity(LLMResponse):
    pass


class LLMParametersEntity(SQLModel):
    """
    Represents the parameters used for one call to an llm service
    """

    api_key: str = Field(description="The API key used to access the llm service")
    temperature: Optional[float] = Field(default=None, description="The temperature used for sampling")
    max_completion_tokens: Optional[int] = Field(default=None, description="The maximum number of tokens to generate")
    top_k: Optional[int] = Field(default=None, description="The number of top-k tokens to keep")
    top_p: Optional[float] = Field(default=None, description="The cumulative probability threshold")
    stop: Optional[str] = Field(default=None, description="The stop tokens for the generation")
    n: Optional[int] = Field(default=None, description="The number of responses to generate")
    logprobs: Optional[int] = Field(default=None, description="The number of logprobs to return")
    presence_penalty: Optional[float] = Field(default=None, description="The presence penalty")
    frequency_penalty: Optional[float] = Field(default=None, description="The frequency penalty")
    end_user_id: Optional[str] = Field(default=None, description="user id that represents the end user")


class ExecutionEntity(Execution):
    """
    Represents a prompt being sent to a llm_service
    """

    prompt: PromptEntity = Field(description="The prompt being sent to the llm service")
    llm_parameters: LLMParametersEntity = Field(description="The parameters used for the llm service call")


class ExecutionGroupEntity(ExecutionGroup):
    executions: list[ExecutionEntity]


class DatasetSplitEntity(DatasetSplit):
    pass


class DatasetEntity(Dataset):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    parent: Optional["DatasetEntity"] = None
    subsets: list["DatasetEntity"] | None = None
    raw_dataset: HFDatasets.Dataset | None = None
    splits: list[DatasetSplitEntity] | None = None

    def __iter__(self) -> Iterator[dict]:
        """
        Iterate over the samples in the dataset.
        """
        if not self.raw_dataset:
            raise ValueError("No raw dataset available to iterate over.")
        return iter(self.raw_dataset)

    def iter_batch(self, size: int) -> Iterator["DatasetEntity"]:
        """
        Create an iterator that yields batches of data samples from the current dataset entity.
        Each batch will be a new DatasetEntity with the specified size.

        Args:
            size (int): The size of each batch to return.

        Yields:
            DatasetEntity: A new DatasetEntity representing a batch.
        """
        if not self.raw_dataset:
            raise ValueError("No raw dataset available to create batches.")

        # Assume raw_dataset can be sliced or iterated. Adjust as necessary.
        total_size = len(self.raw_dataset)  # Get total number of samples
        for start in range(0, total_size, size):
            end = min(start + size, total_size)
            batch_data = self.raw_dataset[start:end]  # Slice the dataset into the batch

            # Create a new DatasetEntity for the batch
            batch_entity = DatasetEntity(
                raw_dataset=batch_data,
                name=f"{self.name}_batch_{start}_{end}",
                version=self.version,
                description=f"A batch of {size} samples from {self.name}",
                created_at=self.created_at,
                parent=self,
            )
            yield batch_entity
