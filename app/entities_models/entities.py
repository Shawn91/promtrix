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
from functools import cached_property
from typing import Optional, Iterator, Any, Literal, Union

import datasets as HFDatasets
from jinja2 import Template
from pydantic import ConfigDict
from pydantic.main import IncEx
from sqlmodel import Field
from typing_extensions import override

from app.entities_models.base import (
    PromptTemplate,
    Prompt,
    Dataset,
    DatasetSplit,
    LLMResponse,
    Execution,
    ExecutionGroup,
    MySQLModel,
    Evaluation,
)


class PromptTemplateEntity(PromptTemplate):
    is_placeholder: bool = False  # True if no template is actually required and a template is created for convenience

    @cached_property
    def user_template(self):
        return Template(self.user)

    @cached_property
    def system_template(self):
        if self.system is None:
            return None
        return Template(self.system)

    @classmethod
    def create_empty(cls) -> "PromptTemplateEntity":
        template = cls(user="{{input}}", is_placeholder=True)
        return template

    def generate_prompt(
        self,
        user: dict[str, str],
        system: dict[str, str] | None = None,
        expected_response: Optional["ExpectedResponseEntity"] = None,
    ) -> "PromptEntity":
        prompt_entity = PromptEntity(user=self.user_template.render(**user), expected_response=expected_response)
        if system is not None:
            prompt_entity.system = self.system_template.render(**system)
        if not self.is_placeholder:
            prompt_entity.template = self
        return prompt_entity


class PromptEntity(Prompt):
    template: PromptTemplateEntity | None = None
    expected_response: Optional["ExpectedResponseEntity"] = None


class LLMResponseEntity(LLMResponse):
    evaluation: Optional["EvaluationEntity"] = None


class EvaluationEntity(Evaluation):
    steps: list | None = Field(default=None, description="The steps taken to evaluate the response")


class LLMParametersEntity(MySQLModel):
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


class ExpectedResponseEntity(MySQLModel):
    """
    Represents the expected response for a prompt
    """

    content: str = Field(description="The expected response text")


class ExecutionEntity(Execution):
    """
    Represents a prompt being sent to a llm_service
    """

    prompt: PromptEntity = Field(description="The prompt being sent to the llm service")
    llm_parameters: LLMParametersEntity | None = Field(
        default=None, description="The parameters used for the llm service call"
    )
    responses: list[LLMResponseEntity] = Field(
        default_factory=list, description="The responses received from the llm service"
    )


class ExecutionGroupEntity(ExecutionGroup):
    executions: list[ExecutionEntity]


class DatasetSplitEntity(DatasetSplit):
    pass


class DatasetEntity(Dataset):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    parent: Union[None, "DatasetEntity"] = None
    subsets: Union[None, list["DatasetEntity"]] = None
    raw_dataset: HFDatasets.Dataset | None = None
    splits: list[DatasetSplitEntity] | None = None

    def __iter__(self) -> Iterator[dict]:
        """
        Iterate over the samples in the dataset.
        """
        if not self.raw_dataset:
            raise ValueError("No raw dataset available to iterate over.")
        return iter(self.raw_dataset)
