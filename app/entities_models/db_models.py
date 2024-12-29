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
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Type
from uuid import UUID

from datasets import load_from_disk, DatasetDict as HFDatasetDict

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
        EvaluationGroupEntity,
    )

from sqlalchemy import Index
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlmodel import Field, Relationship, SQLModel

from app.entities_models.base import (
    PromptTemplate,
    Prompt,
    Dataset,
    LLMResponse,
    Evaluation,
    LLMService,
    Task,
    LLMInteraction,
    LLMInteractionGroup,
    EvaluationGroup,
    EvaluationMetric,
    EvaluationMethod,
    ResponseRole,
)


class ToEntityModel(SQLModel, AsyncAttrs):
    @property
    def entity(self):
        raise NotImplementedError()

    async def to_entity(self, **kwargs):
        raise NotImplementedError()


class PromptTemplateModel(PromptTemplate, ToEntityModel, table=True):
    __tablename__ = "prompt_template"
    __table_args__ = (Index("idx_unique_system_user", "system", "user", unique=True),)

    id: Optional[UUID] = Field(default=None, primary_key=True)

    prompts: list["PromptModel"] = Relationship(back_populates="template")

    @property
    def entity(self):
        from app.entities_models.entities import PromptTemplateEntity

        return PromptTemplateEntity

    async def to_entity(self) -> "PromptTemplateEntity":
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

    async def to_entity(self) -> "PromptEntity":
        data = self.model_dump(exclude={"template", "llm_interactions", "template_id"})
        if await self.awaitable_attrs.template:
            data["template"] = await self.template.to_entity()
        return self.entity(**data)


class LLMResponseModel(LLMResponse, ToEntityModel, table=True):
    __tablename__ = "llm_response"

    llm_interaction_id: UUID = Field(foreign_key="llm_interaction.id")

    llm_interaction: "LLMInteractionModel" = Relationship(back_populates="llm_responses")
    evaluations: list["EvaluationModel"] = Relationship(back_populates="llm_response")

    @property
    def entity(self):
        from app.entities_models.entities import LLMResponseEntity

        return LLMResponseEntity

    async def to_entity(self, exclude_fields: list | None = None) -> "LLMResponseEntity":
        if exclude_fields is None:
            exclude_fields = []
        data = self.model_dump(exclude={"llm_interaction", "evaluations", "llm_interaction_id", *exclude_fields})
        if isinstance(data["role"], str):
            data["role"] = ResponseRole(data["role"])

        if "evaluations" not in exclude_fields and await self.awaitable_attrs.evaluations:
            data["evaluations"] = [
                await evaluation.to_entity(exclude_fields=["llm_response"])
                for evaluation in await self.awaitable_attrs.evaluations
            ]
        entity = self.entity(**data)
        if "evaluations" in data:
            entity.evaluations = data["evaluations"]
            for evaluation in entity.evaluations:
                evaluation.llm_response = entity
        return entity


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
    custom_params: Optional[str] = Field(
        default=None, description="Custom parameters for the LLM service in JSON format"
    )

    group: "LLMInteractionGroupModel" = Relationship(back_populates="llm_interactions")
    llm_responses: list["LLMResponseModel"] = Relationship(back_populates="llm_interaction")
    prompt: PromptModel = Relationship(back_populates="llm_interactions")
    llm_service: "LLMServiceModel" = Relationship(back_populates="llm_interactions")

    @property
    def entity(self):
        from app.entities_models.entities import LLMInteractionEntity

        return LLMInteractionEntity

    async def to_entity(self, exclude_fields: list | None = None) -> "LLMInteractionEntity":
        """
        Convert to entity with option to include related entities.
        """
        from app.entities_models.entities import LLMParametersEntity

        if exclude_fields is None:
            exclude_fields = []
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
            "custom_params": json.loads(self.custom_params) if self.custom_params else None,
        }
        data = self.model_dump(
            exclude={
                "prompt_id",
                "group_id",
                "llm_service_id",
                "group",
                "llm_responses",
                "prompt",
                "llm_service",
                *llm_params_fields.keys(),
                *exclude_fields,
            }
        )
        data["llm_parameters"] = LLMParametersEntity(**{k: v for k, v in llm_params_fields.items() if v is not None})

        # Add related entities if they exist
        if "prompt" not in exclude_fields and await self.awaitable_attrs.prompt:
            data["prompt"] = await (await self.awaitable_attrs.prompt).to_entity()
        if "llm_responses" not in exclude_fields and await self.awaitable_attrs.llm_responses:
            data["responses"] = [await response.to_entity() for response in await self.awaitable_attrs.llm_responses]
        if "llm_service" not in exclude_fields and await self.awaitable_attrs.llm_service:
            data["llm_service"] = await (await self.awaitable_attrs.llm_service).to_entity()
        if "group" not in exclude_fields and await self.awaitable_attrs.group:
            data["group"] = await (await self.awaitable_attrs.group).to_entity(exclude_fields=["llm_interactions"])

        entity = self.entity(**data)
        if "group" in data:
            if entity.group.llm_interactions is None:
                entity.group.llm_interactions = []
            if entity.id not in [llm_interaction.id for llm_interaction in entity.group.llm_interactions]:
                entity.group.llm_interactions.append(entity)
        return entity


class LLMInteractionGroupModel(LLMInteractionGroup, ToEntityModel, table=True):
    __tablename__ = "llm_interaction_group"
    __table_args__ = (Index("idx_unique_task_group_name", "task_id", "name", unique=True),)
    task_id: UUID = Field(foreign_key="task.id")
    llm_interactions: list["LLMInteractionModel"] = Relationship(back_populates="group")
    task: "TaskModel" = Relationship(back_populates="llm_interaction_groups")
    evalution_groups: list["EvaluationGroupModel"] = Relationship(back_populates="llm_interaction_group")

    @property
    def entity(self):
        from app.entities_models.entities import LLMInteractionGroupEntity

        return LLMInteractionGroupEntity

    async def to_entity(self, exclude_fields: list | None = None) -> "LLMInteractionGroupEntity":
        if exclude_fields is None:
            exclude_fields = []
        data = self.model_dump(exclude={"llm_interactions", "task_id", "task", "evaluation_groups", *exclude_fields})
        if "task" not in exclude_fields and await self.awaitable_attrs.task:
            data["task"] = await (await self.awaitable_attrs.task).to_entity(exclude_fields=["llm_interaction_groups"])
        if "llm_interactions" not in exclude_fields and await self.awaitable_attrs.llm_interactions:
            data["llm_interactions"] = [
                await interaction.to_entity(exclude_fields=["group"])
                for interaction in await self.awaitable_attrs.llm_interactions
            ]
        entity = self.entity(**data)
        if "llm_interactions" in data:
            for interaction in data["llm_interactions"]:
                interaction.group = entity

        return entity


class EvaluationModel(Evaluation, ToEntityModel, table=True):
    __tablename__ = "evaluation"

    group_id: UUID = Field(foreign_key="evaluation_group.id")
    llm_response_id: UUID = Field(foreign_key="llm_response.id")
    steps: str | None = Field(default=None, description="The steps taken to evaluate the response serialised as JSON")

    llm_response: LLMResponseModel = Relationship(back_populates="evaluations")
    group: "EvaluationGroupModel" = Relationship(back_populates="evaluations")

    @property
    def entity(self):
        from app.entities_models.entities import EvaluationEntity

        return EvaluationEntity

    async def to_entity(self, exclude_fields: list | None = None) -> "EvaluationEntity":
        from app.entities_models.entities import EvaluationStepEntity

        if exclude_fields is None:
            exclude_fields = []

        data = self.model_dump(exclude={"llm_response", "group_id", "llm_response_id", "group", *exclude_fields})
        if isinstance(data["metric"], str):
            data["metric"] = EvaluationMetric(data["metric"])
        if "llm_response" not in exclude_fields and await self.awaitable_attrs.llm_response:
            data["llm_response"] = await (await self.awaitable_attrs.llm_response).to_entity(
                exclude_fields=["evaluations"]
            )
        if "group" not in exclude_fields and await self.awaitable_attrs.group:
            data["group"] = await self.group.to_entity(exclude_fields=["evaluations"])
        if self.steps:
            steps_data = json.loads(self.steps)
            for step in steps_data:
                if isinstance(step["method"], str):
                    step["method"] = EvaluationMethod(step["method"])
            data["steps"] = [EvaluationStepEntity(**s) for s in steps_data]
        entity = self.entity(**data)
        if "llm_response" in data:
            if data["llm_response"].evaluations is None:
                data["llm_response"].evaluations = []
            data["llm_response"].evaluations.append(entity)
            entity.llm_response = data["llm_response"]
        if "group" in data:
            if not entity.group.evaluations:
                entity.group.evaluations = []
            if entity.id not in [e.id for e in entity.group.evaluations]:
                entity.group.evaluations.append(entity)
        return entity


class EvaluationGroupModel(EvaluationGroup, ToEntityModel, table=True):
    """Normally, an llm interaction group as an execution of a task should correspond to an evaluation group."""

    __tablename__ = "evaluation_group"
    __table_args__ = (Index("idx_unique_interaction_group_name", "llm_interaction_group_id", "name", unique=True),)
    llm_interaction_group_id: UUID = Field(foreign_key="llm_interaction_group.id")

    evaluations: list["EvaluationModel"] = Relationship(back_populates="group")
    llm_interaction_group: "LLMInteractionGroupModel" = Relationship(back_populates="evalution_groups")

    @property
    def entity(self):
        from app.entities_models.entities import EvaluationGroupEntity

        return EvaluationGroupEntity

    async def to_entity(self, exclude_fields: list | None = None) -> "EvaluationGroupEntity":
        if exclude_fields is None:
            exclude_fields = []
        data = self.model_dump(
            exclude={"evaluations", "llm_interaction_group", "llm_interaction_group_id", *exclude_fields}
        )
        if "evaluations" not in exclude_fields and await self.awaitable_attrs.evaluations:
            data["evaluations"] = [
                await evaluation.to_entity(exclude_fields=["group"])
                for evaluation in await self.awaitable_attrs.evaluations
            ]
        if await self.awaitable_attrs.llm_interaction_group:
            data["llm_interaction_group"] = await (await self.awaitable_attrs.llm_interaction_group).to_entity()
        entity = self.entity(**data)
        if "evaluations" in data:
            for evaluation in data["evaluations"]:
                evaluation.group = entity
            entity.evaluations = data["evaluations"]
        return entity


class DatasetModel(Dataset, ToEntityModel, table=True):
    __tablename__ = "dataset"

    parent_id: UUID | None = Field(default=None, foreign_key="dataset.id")
    # Define the relationship to child datasets
    children: list["DatasetModel"] = Relationship(
        back_populates="parent", sa_relationship_kwargs={"cascade": "all, delete-orphan", "lazy": "selectin"}
    )
    # Define the relationship to parent dataset
    parent: Optional["DatasetModel"] = Relationship(
        back_populates="children", sa_relationship_kwargs={"remote_side": "[DatasetModel.id]", "lazy": "selectin"}
    )

    @property
    def entity(self):
        from app.entities_models.entities import DatasetEntity

        return DatasetEntity

    async def to_entity(self, exclude_fields: list | None = None) -> "DatasetEntity":
        if exclude_fields is None:
            exclude_fields = []
        data = self.model_dump(exclude={"parent", "children", "parent_id", *exclude_fields})
        if "parent" not in exclude_fields and await self.awaitable_attrs.parent:
            data["parent"] = await (await self.awaitable_attrs.parent).to_entity(exclude_fields=["children"])
        if "children" not in exclude_fields and await self.awaitable_attrs.children:
            data["children"] = {
                self.name: await child.to_entity(exclude_fields=["parent"])
                for child in await self.awaitable_attrs.children
            }

        if await self.awaitable_attrs.raw_dataset_dir:
            try:
                raw_dataset = load_from_disk(Path(await self.awaitable_attrs.raw_dataset_dir))
                if isinstance(raw_dataset, HFDatasetDict) and self.name in raw_dataset:
                    raw_dataset = raw_dataset[self.name]
                data["raw_dataset"] = raw_dataset
            except Exception:
                # some datasets' raw_dataset_dir is not real, it's ok to skip it
                ...
        entity = self.entity(**data)
        if entity.parent:
            if entity.parent.children is None:
                entity.parent.children = {}
            entity.parent.children[entity.name] = entity
        if "children" in data:
            for child in data["children"].values():
                child.parent = entity
        return entity


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

    async def to_entity(self) -> "LLMServiceEntity":
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

    async def to_entity(self) -> "TaskEntity":
        data = self.model_dump(exclude={"llm_interaction_groups"})
        task_entity = self.entity(**data)
        for llm_interaction_group in await self.awaitable_attrs.llm_interaction_groups:
            llm_interaction_group_entity = await llm_interaction_group.to_entity(exclude_fields=["task"])
            llm_interaction_group_entity.task = task_entity
            if task_entity.llm_interaction_groups is None:
                task_entity.llm_interaction_groups = []
            task_entity.llm_interaction_groups.append(llm_interaction_group_entity)
        return task_entity
