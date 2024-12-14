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
from typing import Optional, TYPE_CHECKING, Type
from uuid import UUID

if TYPE_CHECKING:
    from app.entities_models.entities import (
        LLMServiceEntity,
        PromptTemplateEntity,
        PromptEntity,
        LLMResponseEntity,
        LLMInteractionEntity,
        LLMInteractionGroupEntity,
        EvaluationEntity,
        DatasetEntity,
        TaskEntity,
    )

from sqlalchemy import Index
from sqlmodel import Field, Relationship, SQLModel

from app.entities_models.base import (
    PromptTemplate,
    Prompt,
    Dataset,
    LLMResponse,
    Evaluation,
    LLMService,
    Task, LLMInteraction,
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
    llm_interactions: list["LLMInteractionModel"] = Relationship(back_populates="prompt")

    @property
    def entity(self):
        from app.entities_models.entities import PromptEntity

        return PromptEntity

    def to_entity(self) -> "PromptEntity":
        data = self.model_dump(exclude={"template", "llm_interactions"})
        if self.template:
            data["template"] = self.template.to_entity()
        return self.entity(**data)


class LLMResponseModel(LLMResponse, ToEntityModel, table=True):
    __tablename__ = "llm_response"

    llm_interaction_id: UUID = Field(foreign_key="llm_interaction.id")
    evaluation_id: UUID | None = Field(default=None, foreign_key="evaluation.id")

    llm_interaction: "LLMInteractionModel" = Relationship(back_populates="llm_responses")
    evaluation: "EvaluationModel" = Relationship(back_populates="llm_response")

    @property
    def entity(self):
        from app.entities_models.entities import LLMResponseEntity

        return LLMResponseEntity

    def to_entity(self) -> "LLMResponseEntity":
        data = self.model_dump(exclude={"llm_interaction", "evaluation"})
        if self.evaluation:
            data["evaluation"] = self.evaluation.to_entity()
        return self.entity(**data)


class LLMInteractionModel(LLMInteraction, ToEntityModel, table=True):
    __tablename__ = "llm_interaction"

    prompt_id: UUID | None = Field(default=None, foreign_key="prompt.id")
    group_id: UUID | None = Field(default=None, foreign_key="llm_interaction_group.id")
    llm_service_id: UUID = Field(foreign_key="llm_service.id")

    # llm parameters
    api_key: str | None = Field(default=None, description="The API key used to access the llm service")
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
    seed: Optional[int] = Field(default=None, description="The seed for the generation")

    group: "LLMInteractionGroupModel" = Relationship(back_populates="llm_interactions")
    llm_responses: list["LLMResponseModel"] = Relationship(back_populates="llm_interaction")
    prompt: PromptModel = Relationship(back_populates="llm_interactions")
    llm_service: "LLMServiceModel" = Relationship(back_populates="llm_interactions")

    @property
    def entity(self):
        from app.entities_models.entities import LLMInteractionEntity

        return LLMInteractionEntity

    def to_entity(self) -> "LLMInteractionEntity":
        from app.entities_models.entities import LLMParametersEntity

        data = self.model_dump(exclude={"group", "llm_responses", "prompt", "llm_service", "api_key", "task"})

        # Create LLMParametersEntity from the llm parameters fields
        llm_params_fields = {
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "top_a": self.top_a,
            "stop": self.stop,
            "n": self.n,
            "logprobs": self.logprobs,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repitition_penalty": self.repitition_penalty,
            "end_user_id": self.end_user_id,
            "seed": self.seed,
        }
        data["llm_parameters"] = LLMParametersEntity(**{k: v for k, v in llm_params_fields.items() if v is not None})

        # Add related entities if they exist
        if self.prompt:
            data["prompt"] = self.prompt.to_entity()
        if self.llm_responses:
            data["responses"] = [response.to_entity() for response in self.llm_responses]
        if self.llm_service:
            data["llm_service"] = self.llm_service.to_entity()
        if self.group:
            data["group"] = self.group.to_entity()

        return self.entity(**data)


class LLMInteractionGroupModel(LLMInteraction, ToEntityModel, table=True):
    __tablename__ = "llm_interaction_group"
    __table_args__ = (Index("idx_unique_task_group_name", "task_id", "name", unique=True),)
    task_id: UUID = Field(foreign_key="task.id")
    llm_interactions: list["LLMInteractionModel"] = Relationship(back_populates="group")
    task: "TaskModel" = Relationship(back_populates="llm_interaction_groups")

    @property
    def entity(self):
        from app.entities_models.entities import LLMInteractionGroupEntity

        return LLMInteractionGroupEntity

    def to_entity(self) -> "LLMInteractionGroupEntity":
        data = self.model_dump(exclude={"llm_interactions"})
        if self.llm_interactions:
            data["llm_interactions"] = [interaction.to_entity() for interaction in self.llm_interactions]
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


class LLMServiceModel(LLMService, ToEntityModel, table=True):
    __tablename__ = "llm_service"
    __table_args__ = (Index("idx_unique_llm_version_quant", "llm", "llm_version", "quantization", unique=True),)

    custom_config: Optional[str] = Field(
        default=None, description="Custom configuration for the LLM service in JSON format"
    )

    llm_interactions: list["LLMInteractionModel"] = Relationship(back_populates="llm_service")

    @property
    def entity(self):
        from app.entities_models.entities import LLMServiceEntity

        return LLMServiceEntity

    def to_entity(self) -> "LLMServiceEntity":
        data = self.model_dump(exclude={"api_keys", "llm_interactions", "custom_config"})
        if self.custom_config:
            data["custom_config"] = json.loads(self.custom_config)
        return self.entity(**data)


class TaskModel(Task, ToEntityModel, table=True):
    __tablename__ = "task"
    llm_interaction_groups: list["LLMInteractionGroupModel"] = Relationship(back_populates="task")

    @property
    def entity(self) -> Type["TaskEntity"]:
        from app.entities_models.entities import TaskEntity

        return TaskEntity

    def to_entity(self) -> "TaskEntity":
        data = self.model_dump(exclude={"llm_interaction_groups"})
        if self.llm_interaction_groups:
            data["llm_interaction_groups"] = [group.to_entity() for group in self.llm_interaction_groups]
        return self.entity(**data)
