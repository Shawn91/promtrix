"""
The concept of Entity is from Clean Architecture.

It is a core concept of the application that represents a business entity, 
such as a customer, a product, or a purchase.

Entities are supposed to be passed around business logic, 
but do not interact with external systems like databases or APIs.

Entities should be converted to Models before being passed to external systems.

Data retrieved from external systems should be Models and then converted to Entities
before being used in the application.
"""
import json
from functools import cached_property
from typing import Optional, Iterator, Any, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from app.entities_models.db_models import (
        APIKeyModel,
        LLMServiceModel,
        PromptTemplateModel,
        PromptModel,
        LLMResponseModel,
        EvaluationModel,
        ExecutionModel,
        DatasetModel,
        ExecutionGroupModel,
    )

import datasets as HFDatasets
from jinja2 import Template
from pydantic import ConfigDict
from sqlmodel import Field, SQLModel

from app.entities_models.base import (
    PromptTemplate,
    Prompt,
    Dataset,
    LLMResponse,
    Execution,
    ExecutionGroup,
    MySQLModel,
    Evaluation,
    APIKey,
    LLMService,
)


class Entity:
    ...


class ToModelEntity(SQLModel):
    @property
    def model(self):
        raise NotImplementedError()

    def to_model(self):
        """Base conversion method from entity to model"""
        return self.model(**self.model_dump(exclude_unset=True))


class PromptTemplateEntity(PromptTemplate, ToModelEntity, Entity):
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
        template = cls(user="{{input}}")
        return template

    @property
    def model(self) -> type["PromptTemplateModel"]:
        from app.entities_models.db_models import PromptTemplateModel

        return PromptTemplateModel

    def to_model(self) -> "PromptTemplateModel":
        data = self.model_dump(exclude={"prompts"})
        return self.model(**data)

    def generate_prompt(
        self,
        user: dict[str, str],
        system: dict[str, str] | None = None,
        expected_response: Optional["ExpectedResponseEntity"] = None,
    ) -> "PromptEntity":
        prompt_entity = PromptEntity(user=self.user_template.render(**user), expected_response=expected_response)
        if system is not None:
            prompt_entity.system = self.system_template.render(**system)
        return prompt_entity


class PromptEntity(Prompt, ToModelEntity, Entity):
    template: PromptTemplateEntity | None = None
    expected_response: Optional["ExpectedResponseEntity"] = None

    @property
    def model(self) -> type["PromptModel"]:
        from app.entities_models.db_models import PromptModel

        return PromptModel

    def to_model(self) -> "PromptModel":
        data = self.model_dump(exclude={"template", "executions"})
        return self.model(**data)


class LLMResponseEntity(LLMResponse, ToModelEntity, Entity):
    evaluation: Optional["EvaluationEntity"] = None

    @property
    def model(self) -> type["LLMResponseModel"]:
        from app.entities_models.db_models import LLMResponseModel

        return LLMResponseModel

    def to_model(self) -> "LLMResponseModel":
        data = self.model_dump(exclude={"evaluation"})
        return self.model(**data)


class EvaluationEntity(Evaluation, ToModelEntity, Entity):
    steps: list | None = Field(default=None, description="The steps taken to evaluate the response")

    @property
    def model(self) -> type["EvaluationModel"]:
        from app.entities_models.db_models import EvaluationModel

        return EvaluationModel

    def to_model(self) -> "EvaluationModel":
        data = self.model_dump(exclude={"steps"})
        return self.model(**data)


class LLMParametersEntity(MySQLModel, Entity):
    """
    Represents the parameters used for one call to an llm service
    """

    api_key: str = Field(description="The API key used to access the llm service")
    temperature: Optional[float] = Field(default=None, description="The temperature used for sampling")
    max_completion_tokens: Optional[int] = Field(default=None, description="The maximum number of tokens to generate")
    top_k: Optional[int] = Field(default=None, description="The number of top-k tokens to keep")
    top_p: Optional[float] = Field(default=None, description="The cumulative probability threshold")
    min_p: Optional[float] = Field(default=None, description="The minimum probability for a token to be considered")
    top_a: Optional[float] = Field(
        default=None,
        description="Consider only the top tokens with 'sufficiently high' probabilities based on the probability of the most likely token",
    )
    stop: Optional[str] = Field(default=None, description="The stop tokens for the generation")
    n: Optional[int] = Field(default=None, description="The number of responses to generate")
    logprobs: Optional[int] = Field(default=None, description="The number of logprobs to return")
    presence_penalty: Optional[float] = Field(default=None, description="The presence penalty")
    frequency_penalty: Optional[float] = Field(default=None, description="The frequency penalty")
    repitition_penalty: Optional[float] = Field(default=None, description="The repitition penalty")
    end_user_id: Optional[str] = Field(default=None, description="user id that represents the end user")


class ExpectedResponseEntity(MySQLModel, Entity):
    """
    Represents the expected response for a prompt
    """

    content: str = Field(description="The expected response text")


class ExecutionEntity(Execution, ToModelEntity, Entity):
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

    @property
    def model(self) -> type["ExecutionModel"]:
        from app.entities_models.db_models import ExecutionModel

        return ExecutionModel

    def to_model(self) -> "ExecutionModel":
        data = self.model_dump(exclude={"prompt", "llm_parameters", "responses"})
        if self.llm_parameters:
            data.update(self.llm_parameters.model_dump())
        return self.model(**data)


class ExecutionGroupEntity(ExecutionGroup, ToModelEntity, Entity):
    executions: list[ExecutionEntity]

    @property
    def model(self) -> type["ExecutionGroupModel"]:
        from app.entities_models.db_models import ExecutionGroupModel

        return ExecutionGroupModel

    def to_model(self) -> "ExecutionGroupModel":
        data = self.model_dump(exclude={"executions"})
        return self.model(**data)


class DatasetEntity(Dataset, ToModelEntity, Entity):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_dataset_dir: str = Field(description="The path to the directory containing the raw dataset")
    parent: Optional["DatasetEntity"] = None
    children: Optional[dict[str, "DatasetEntity"]] = None  # Can contain both subdatasets and splits
    raw_dataset: HFDatasets.Dataset | None = None

    def __iter__(self) -> Iterator[dict]:
        """
        Iterate over the samples in the dataset.
        """
        if not self.raw_dataset:
            raise ValueError("No raw dataset available to iterate over.")
        return iter(self.raw_dataset)

    @property
    def model(self) -> type["DatasetModel"]:
        from app.entities_models.db_models import DatasetModel

        return DatasetModel

    def to_model(self) -> "DatasetModel":
        data = self.model_dump(exclude={"parent", "children", "raw_dataset"})

        # If there's a parent, add its ID
        if self.parent:
            data["parent_id"] = self.parent.id
        return self.model(**data)

    @property
    def splits(self) -> dict[str, "DatasetEntity"]:
        """Get all child datasets that are splits"""
        if not self.children:
            return {}
        return {name: dataset for name, dataset in self.children.items() if dataset.is_split}

    @property
    def subdatasets(self) -> dict[str, "DatasetEntity"]:
        """Get all child datasets that are not splits"""
        if not self.children:
            return {}
        return {name: dataset for name, dataset in self.children.items() if not dataset.is_split}

    def add_split(self, dataset: "DatasetEntity"):
        """Add a split to this dataset"""
        if not self.children:
            self.children = {}
        dataset.is_split = True
        dataset.parent = self
        self.children[dataset.name] = dataset

    def add_subdataset(self, dataset: "DatasetEntity"):
        """Add a subdataset to this dataset"""
        if not self.children:
            self.children = {}
        dataset.is_split = False
        dataset.parent = self
        self.children[dataset.name] = dataset


class APIKeyEntity(APIKey, ToModelEntity, Entity):
    service: Optional["LLMServiceEntity"]

    @property
    def model(self):
        from app.entities_models.db_models import APIKeyModel

        return APIKeyModel

    def to_model(self) -> "APIKeyModel":
        data = self.model_dump(exclude={"service"})
        return self.model(**data)


class LLMServiceEntity(LLMService, ToModelEntity, Entity):
    api_keys: list[APIKeyEntity] | None = None
    custom_config: dict[str, Any] | None = None

    @property
    def model(self):
        from app.entities_models.db_models import LLMServiceModel

        return LLMServiceModel

    def to_model(self) -> "LLMServiceModel":
        data = self.model_dump(exclude={"api_keys", "custom_config"})
        if self.custom_config:
            data["custom_config"] = json.dumps(self.custom_config)
        return self.model(**data)


EntityType = TypeVar("EntityType", bound=Entity)
ToModelEntityType = TypeVar("ToModelEntityType", bound=ToModelEntity)
