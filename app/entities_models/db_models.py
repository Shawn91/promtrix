"""
The concept of Model is from Clean Architecture.

Models are the objects that represent the raw data in the application and
are used to interact with external systems like databases or APIs.

All models in this file are database models using SQLModel library.

Entities should be converted to Models before being passed to external systems.

Data retrieved from external systems should be Models and then converted to Entities
before being used in the application.
"""
import json
from typing import Optional, TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from app.entities_models.entities import (
        APIKeyEntity,
        LLMServiceEntity,
        PromptTemplateEntity,
        PromptEntity,
        LLMResponseEntity,
        ExecutionEntity,
        ExecutionGroupEntity,
        EvaluationEntity,
        DatasetEntity,
    )

from sqlalchemy import Index
from sqlmodel import Field, Relationship, SQLModel

from app.entities_models.base import (
    PromptTemplate,
    Prompt,
    Dataset,
    LLMResponse,
    Execution,
    Evaluation,
    LLMService,
    APIKey,
    ExecutionGroup,
)


class ToEntityModel(SQLModel):
    @property
    def entity(self):
        raise NotImplementedError()

    def to_entity(self):
        """Base conversion method from model to entity"""
        return self.entity(**self.model_dump())


class PromptTemplateModel(PromptTemplate, ToEntityModel, table=True):
    __tablename__ = "prompt_template"
    __table_args__ = (Index("idx_unique_system_user", "system", "user", unique=True),)

    id: Optional[UUID] = Field(default=None, primary_key=True)

    prompts: list["PromptModel"] = Relationship(back_populates="template")

    @property
    def entity(self):
        from app.entities_models.entities import PromptTemplateEntity

        return PromptTemplateEntity

    def to_entity(self) -> "PromptTemplateEntity":
        data = self.model_dump(exclude={"prompts"})
        return self.entity(**data)


class PromptModel(Prompt, ToEntityModel, table=True):
    __tablename__ = "prompt"

    id: UUID | None = Field(default=None, primary_key=True)
    template_id: UUID | None = Field(default=None, foreign_key="prompt_template.id")

    template: PromptTemplateModel = Relationship(back_populates="prompts")
    executions: list["ExecutionModel"] = Relationship(back_populates="prompt")

    @property
    def entity(self):
        from app.entities_models.entities import PromptEntity

        return PromptEntity

    def to_entity(self) -> "PromptEntity":
        data = self.model_dump(exclude={"template", "executions"})
        if self.template:
            data["template"] = self.template.to_entity()
        return self.entity(**data)


class LLMResponseModel(LLMResponse, ToEntityModel, table=True):
    __tablename__ = "llm_response"

    execution_id: UUID = Field(foreign_key="execution.id")
    evaluation_id: UUID | None = Field(default=None, foreign_key="evaluation.id")

    execution: "ExecutionModel" = Relationship(back_populates="llm_responses")
    evaluation: "EvaluationModel" = Relationship(back_populates="llm_response")

    @property
    def entity(self):
        from app.entities_models.entities import LLMResponseEntity

        return LLMResponseEntity

    def to_entity(self) -> "LLMResponseEntity":
        data = self.model_dump(exclude={"execution", "evaluation"})
        if self.evaluation:
            data["evaluation"] = self.evaluation.to_entity()
        return self.entity(**data)


class ExecutionModel(Execution, ToEntityModel, table=True):
    __tablename__ = "execution"

    prompt_id: UUID | None = Field(default=None, foreign_key="prompt.id")
    group_id: UUID | None = Field(default=None, foreign_key="execution_group.id")
    llm_service_id: UUID = Field(foreign_key="llm_service.id")
    api_key_id: str = Field(foreign_key="api_key.key", description="The API key used to access the llm service")

    # llm parameters
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

    group: "ExecutionGroupModel" = Relationship(back_populates="executions")
    # task_id: uuid.UUID = Field(foreign_key="task.id")
    # execution_group_id: Optional[uuid.UUID] = Field(default=None, foreign_key="executiongroup.id")
    llm_responses: list["LLMResponseModel"] = Relationship(back_populates="execution")
    prompt: PromptModel = Relationship(back_populates="executions")
    llm_service: "LLMServiceModel" = Relationship(back_populates="executions")
    api_key: "APIKeyModel" = Relationship(back_populates="executions")

    @property
    def entity(self):
        from app.entities_models.entities import ExecutionEntity

        return ExecutionEntity

    def to_entity(self) -> "ExecutionEntity":
        data = self.model_dump(exclude={"group", "llm_responses", "prompt", "llm_service", "api_key"})
        if self.prompt:
            data["prompt"] = self.prompt.to_entity()
        if self.llm_responses:
            data["responses"] = [response.to_entity() for response in self.llm_responses]
        return self.entity(**data)


class ExecutionGroupModel(ExecutionGroup, ToEntityModel, table=True):
    __tablename__ = "execution_group"
    executions: list["ExecutionModel"] = Relationship(back_populates="group")

    @property
    def entity(self):
        from app.entities_models.entities import ExecutionGroupEntity

        return ExecutionGroupEntity

    def to_entity(self) -> "ExecutionGroupEntity":
        data = self.model_dump(exclude={"executions"})
        if self.executions:
            data["executions"] = [execution.to_entity() for execution in self.executions]
        return self.entity(**data)


class EvaluationModel(Evaluation, ToEntityModel, table=True):
    __tablename__ = "evaluation"

    llm_response: LLMResponseModel = Relationship(back_populates="evaluation")

    @property
    def entity(self):
        from app.entities_models.entities import EvaluationEntity

        return EvaluationEntity

    def to_entity(self) -> "EvaluationEntity":
        data = self.model_dump(exclude={"llm_response"})
        return self.entity(**data)

    # evaluation_group: Optional["EvaluationGroup"] = Relationship(back_populates="evaluations")


class DatasetModel(Dataset, ToEntityModel, table=True):
    __tablename__ = "dataset"

    parent_id: UUID | None = Field(default=None, foreign_key="dataset.id")
    raw_dataset_dir: str = Field(description="The path to the directory containing the raw dataset ", unique=True)

    # Define the relationship to child datasets
    children: list["DatasetModel"] = Relationship(
        back_populates="parent", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )
    # Define the relationship to parent dataset
    parent: Optional["DatasetModel"] = Relationship(
        back_populates="children", sa_relationship_kwargs={"remote_side": "[DatasetModel.id]"}
    )

    @property
    def entity(self):
        from app.entities_models.entities import DatasetEntity

        return DatasetEntity

    def to_entity(self) -> "DatasetEntity":
        data = self.model_dump(exclude={"parent", "children"})
        if self.parent:
            data["parent"] = self.parent.to_entity()
        if self.children:
            data["children"] = {child.name: child.to_entity() for child in self.children}
        return self.entity(**data)


class APIKeyModel(APIKey, ToEntityModel, table=True):
    __tablename__ = "api_key"
    service_id: UUID = Field(foreign_key="llm_service.id")

    service: "LLMServiceModel" = Relationship(back_populates="api_keys")
    executions: list["ExecutionModel"] = Relationship(back_populates="api_key")

    @property
    def entity(self):
        from app.entities_models.entities import APIKeyEntity

        return APIKeyEntity

    def to_entity(self) -> "APIKeyEntity":
        data = self.model_dump(exclude={"service", "executions"})
        return self.entity(**data)


class LLMServiceModel(LLMService, ToEntityModel, table=True):
    __tablename__ = "llm_service"
    __table_args__ = (Index("idx_unique_llm_version_quant", "llm", "llm_version", "quantization", unique=True),)

    custom_config: Optional[str] = Field(
        default=None, description="Custom configuration for the LLM service in JSON format"
    )

    api_keys: list["APIKeyModel"] = Relationship(back_populates="service")
    executions: list["ExecutionModel"] = Relationship(back_populates="llm_service")

    @property
    def entity(self):
        from app.entities_models.entities import LLMServiceEntity

        return LLMServiceEntity

    def to_entity(self) -> "LLMServiceEntity":
        data = self.model_dump(exclude={"api_keys", "executions", "custom_config"})
        if self.custom_config:
            data["custom_config"] = json.loads(self.custom_config)
        if self.api_keys:
            data["api_keys"] = [key.to_entity() for key in self.api_keys]
        return self.entity(**data)
