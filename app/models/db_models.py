from datetime import datetime
from enum import Enum
from functools import cached_property
from typing import Optional, TypeVar, Literal, Iterator
from uuid import UUID, uuid4

from typing_extensions import TypedDict
from datasets import load_from_disk, Dataset as HFDataset, DatasetDict as HFDatasetDict
from jinja2 import Template
from pydantic import ConfigDict
from sqlalchemy import Index, Column, Enum as SQLAlchemyEnum, JSON, func
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlmodel import Field, Relationship, SQLModel


class LLMInteractionCategory(str, Enum):
    EVALUATION = "evaluation"
    GENERATION = "generation"


class EvaluationMethod(str, Enum):
    LLM = "llm"


class EvaluationMetric(str, Enum):
    CORRECTNESS = "correctness"


class ResponseRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

    @classmethod
    def _missing_(cls, value: str):
        """Handle case-insensitive lookup"""
        for member in cls:
            if member.value.lower() == value.lower():
                return member
        return None


class MyModel(SQLModel, AsyncAttrs):
    """Base class for all database models."""

    id: UUID = Field(default=uuid4(), primary_key=True)
    model_config = ConfigDict(extra="forbid")

    def update(self, other: "ModelType", overwrite_priority: Literal["self", "other"] | None = None) -> bool:
        """Update self with other
        return True if anyone gets updated
        """
        assert type(self) is type(other)
        updated = False
        for field in self.model_fields:
            if getattr(self, field) is None and getattr(other, field) is not None:
                setattr(self, field, getattr(other, field))
                updated = True
            if getattr(self, field) is not None and getattr(other, field) is None:
                setattr(other, field, getattr(self, field))
                updated = True
            if overwrite_priority:
                if overwrite_priority == "self" and getattr(self, field):
                    setattr(other, field, getattr(self, field))
                    updated = True
                elif overwrite_priority == "other" and getattr(other, field):
                    setattr(self, field, getattr(other, field))
                    updated = True
        return updated


ModelType = TypeVar("ModelType", bound=MyModel)


class PromptTemplateModel(MyModel, table=True):
    __tablename__ = "prompt_template"
    __table_args__ = (Index("idx_unique_system_user_in_prompt_template", "system", "user", unique=True),)

    user: str | None = Field(default=None, description="template for user prompt")
    system: str | None = None  # Store as string, convert to jinja2 Template when needed
    description: Optional[str] = None
    prompts: list["PromptModel"] = Relationship(back_populates="template")

    @cached_property
    def user_template(self) -> Template:
        return Template(self.user)

    @cached_property
    def system_template(self) -> Template | None:
        if self.system is None:
            return None
        return Template(self.system)

    def generate_prompt(
        self,
        user: dict[str, str],
        system: dict[str, str] | None = None,
        expected_response: Optional[str] = None,
    ) -> "PromptModel":
        prompt = PromptModel(
            template_id=self.id,
            user=self.user_template.render(**user),
            system=self.system_template.render(**system) if system and self.system_template else None,
            expected_response=expected_response,
        )
        return prompt

    #     prompt_entity = PromptEntity(user=self.user_template.render(**user), expected_response=expected_response)
    #     if system is not None:
    #         prompt_entity.system = self.system_template.render(**system)
    #     return prompt_entity


class PromptModel(MyModel, table=True):
    __tablename__ = "prompt"
    __table_args__ = (Index("idx_unique_system_user_in_template", "system", "user", unique=True),)

    template_id: UUID | None = Field(default=None, foreign_key="prompt_template.id")
    user: str = Field(description="user prompt")
    system: Optional[str] = Field(default=None, description="system prompt")
    created_at: datetime = Field(
        default_factory=datetime.now, description="The date and time when the prompt was created"
    )
    token_count: Optional[int] = Field(default=None, description="The number of tokens in the prompt")
    expected_response: Optional[str] = None
    template: PromptTemplateModel = Relationship(back_populates="prompts")
    llm_interactions: list["LLMInteractionModel"] = Relationship(back_populates="prompt")


class LLMResponseModel(MyModel, table=True):
    __tablename__ = "llm_response"

    content: str
    finish_reason: str = Field(description="The reason why the llm interaction was finished")
    index: int = Field(description="The index of the response in the list of responses generated by the llm_service")
    llm_interaction_id: UUID = Field(foreign_key="llm_interaction.id")
    role: ResponseRole = Field(
        sa_column=Column(SQLAlchemyEnum(ResponseRole, native_enum=False)),
        description="The role of the response (user/assistant/system)",
    )

    llm_interaction: "LLMInteractionModel" = Relationship(back_populates="llm_responses")
    evaluations: list["EvaluationModel"] = Relationship(back_populates="llm_response")


class LLMParametersModel(MyModel, table=True):
    __tablename__ = "llm_parameters"
    api_key: str | None = Field(default=None, description="The API key used to access the llm service")
    temperature: Optional[float] = Field(default=None, description="The temperature used for sampling")
    max_completion_tokens: Optional[int] = Field(default=None, description="The maximum number of tokens to generate")
    top_k: Optional[int] = Field(default=None, description="The number of top-k tokens to keep")
    top_p: Optional[float] = Field(default=None, description="The cumulative probability threshold")
    min_p: Optional[float] = Field(default=None, description="The minimum probability for a token to be considered")
    top_a: Optional[float] = Field(
        default=None,
        description="Consider only the top tokens with 'sufficiently high' probabilities "
        "based on the probability of the most likely token",
    )
    stop: Optional[str] = Field(default=None, description="The stop tokens for the generation")
    n: Optional[int] = Field(default=None, description="The number of responses to generate")
    logprobs: Optional[int] = Field(default=None, description="The number of logprobs to return")
    presence_penalty: Optional[float] = Field(default=None, description="The presence penalty")
    frequency_penalty: Optional[float] = Field(default=None, description="The frequency penalty")
    repetition_penalty: Optional[float] = Field(default=None, description="The repetition penalty")
    end_user_id: Optional[str] = Field(default=None, description="user id that represents the end user")
    seed: Optional[int] = Field(default=None, description="The seed for the generation")
    custom_params: Optional[dict] = Field(
        default=None, sa_column=Column(JSON), description="Custom parameters for the LLM service in JSON format"
    )

    llm_interactions: list["LLMInteractionModel"] = Relationship(back_populates="llm_params")


class LLMInteractionModel(MyModel, table=True):
    __tablename__ = "llm_interaction"

    prompt_id: UUID = Field(foreign_key="prompt.id")
    group_id: UUID | None = Field(
        default=None,
        foreign_key="llm_interaction_group.id",
        description="If an interaction is for evaluation, it doesn't have group",
    )
    llm_service_id: UUID = Field(foreign_key="llm_service.id")
    llm_params_id: UUID = Field(foreign_key="llm_parameters.id")

    request_id: str | None = Field(default=None, description="The id of the id provided by the llm service")
    duration: int | None = Field(default=None, description="The duration of the interaction in milliseconds")
    cost: Optional[float] = Field(default=None, description="The cost of the interaction")
    token_count: int | None = Field(default=None, description="The number of tokens in the response")
    created_at: datetime = Field(
        default_factory=datetime.now, description="The date and time when the interaction was created"
    )
    category: LLMInteractionCategory = Field(
        sa_column=Column(SQLAlchemyEnum(LLMInteractionCategory, native_enum=False)),
        description="The method used to evaluate the response",
        default=LLMInteractionCategory.GENERATION,
    )

    group: "LLMInteractionGroupModel" = Relationship(back_populates="llm_interactions")
    llm_responses: list["LLMResponseModel"] = Relationship(back_populates="llm_interaction")
    prompt: PromptModel = Relationship(back_populates="llm_interactions")
    llm_service: "LLMServiceModel" = Relationship(back_populates="llm_interactions")
    llm_params: LLMParametersModel = Relationship(back_populates="llm_interactions")


class LLMInteractionGroupModel(MyModel, table=True):
    __tablename__ = "llm_interaction_group"
    __table_args__ = (Index("idx_unique_task_group_name", "task_id", "name", unique=True),)
    task_id: UUID = Field(foreign_key="task.id")
    name: str = Field(description="The name of the group")
    created_at: datetime = Field(
        default_factory=datetime.now, description="The date and time when the group was created"
    )
    duration: int | None = Field(default=None, description="The total duration of the group in milliseconds")
    llm_interactions: list["LLMInteractionModel"] = Relationship(back_populates="group")
    task: "TaskModel" = Relationship(back_populates="llm_interaction_groups")
    evaluation_groups: list["EvaluationGroupModel"] = Relationship(back_populates="llm_interaction_group")


class EvaluationStep(TypedDict):
    execution_id: UUID
    method: EvaluationMethod


class EvaluationModel(MyModel, table=True):
    __tablename__ = "evaluation"

    group_id: UUID = Field(foreign_key="evaluation_group.id")
    llm_response_id: UUID = Field(foreign_key="llm_response.id")
    steps: list[EvaluationStep] = Field(
        sa_column=Column(JSON),
        description="The steps taken to evaluate the response serialised as JSON",
    )
    metric: EvaluationMetric = Field(
        sa_column=Column(SQLAlchemyEnum(EvaluationMetric, native_enum=False)),
        description="The metric used to evaluate the response",
    )
    score: float | None = Field(default=None, description="The score obtained by the evaluation")
    created_at: datetime = Field(
        default_factory=datetime.now, description="The date and time when the evaluation was created"
    )
    duration: int = Field(description="The duration of the evaluation in milliseconds")
    cost: Optional[float] = Field(default=None, description="The cost of the evaluation")

    llm_response: LLMResponseModel = Relationship(back_populates="evaluations")
    group: "EvaluationGroupModel" = Relationship(back_populates="evaluations")


class EvaluationGroupModel(MyModel, table=True):
    """Normally, an llm interaction group as an execution of a task should correspond to an evaluation group."""

    __tablename__ = "evaluation_group"
    __table_args__ = (Index("idx_unique_interaction_group_name", "llm_interaction_group_id", "name", unique=True),)
    llm_interaction_group_id: UUID = Field(foreign_key="llm_interaction_group.id")
    name: str = Field(description="The name of the group")
    created_at: datetime = Field(
        default_factory=datetime.now, description="The date and time when the group was created"
    )
    duration: int | None = Field(default=None, description="The total duration of the group in milliseconds")
    cost: Optional[float] = Field(default=None)
    evaluations: list["EvaluationModel"] = Relationship(back_populates="group")
    llm_interaction_group: "LLMInteractionGroupModel" = Relationship(back_populates="evaluation_groups")


class DatasetModel(MyModel, table=True):
    __tablename__ = "dataset"

    name: str = Field(
        index=True,
        description="The name of the dataset. If this is a subdataset, just include the name of the subdataset",
    )
    raw_dataset_dir: str = Field(
        unique=True, index=True, description="The path to the directory containing the raw dataset"
    )
    version: str | None = Field(default=None, index=True, description="The version of the dataset")
    description: str | None = None
    url: str | None = None
    size: int | None = Field(default=None, description="The size of the dataset in bytes")
    count: int | None = Field(default=None, description="The number of examples in the dataset")
    created_at: datetime = Field(
        default_factory=datetime.now, description="The date and time when the dataset was created"
    )
    is_split: bool = Field(
        default=False, description="True if this dataset is a split of another dataset (e.g., train/test/validation)"
    )

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
    def raw_dataset(self) -> HFDataset | HFDatasetDict:
        dataset = load_from_disk(dataset_path=self.raw_dataset_dir)
        if isinstance(dataset, HFDatasetDict) and self.name in dataset:
            dataset = dataset[self.name]
        return dataset

    def __iter__(self) -> Iterator[dict]:
        """
        Iterate over the samples in the dataset.
        """
        if not self.raw_dataset:
            raise ValueError("No raw dataset available to iterate over.")
        if not isinstance(self.raw_dataset, HFDataset):
            raise ValueError("Raw dataset is not a Hugging Face Dataset.")
        return iter(self.raw_dataset)

    def iter_split_data(self, split: str):
        return iter(self.splits[split])

    @property
    def splits(self) -> dict[str, "DatasetModel"]:
        """Get all child datasets that are splits"""
        if not self.children:
            return {}
        return {name: dataset for name, dataset in self.children if dataset.is_split}

    @property
    def subdatasets(self) -> dict[str, "DatasetModel"]:
        """Get all child datasets that are not splits"""
        if not self.children:
            return {}
        return {name: dataset for name, dataset in self.children if not dataset.is_split}


class LLMServiceModel(MyModel, table=True):
    __tablename__ = "llm_service"
    __table_args__ = (Index("idx_unique_llm_version_quant", "llm", "llm_version", "quantization", unique=True),)
    llm: str = Field(index=True)
    llm_provider: Optional[str] = Field(default=None, description="The provider of the llm model. E.g. OpenAI, etc")
    service_provider: Optional[str] = Field(
        default=None,
        description="The provider of the llm service. E.g. Amazon, Fireworks, etc",
    )
    service_gateway: Optional[str] = Field(
        default=None,
        description="The gateway used to access the llm service. E.g. OpenRouter, etc",
    )
    llm_version: Optional[str] = Field(default=None, description="The version of the llm model")
    api_endpoint: str = Field(
        description="Currently, litellm is used for service interaction so this should be used for the model "
        "parameter of the 'completion' function of litellm, instead of the actual API endpoint."
    )
    quantization: str | None = Field(default=None, description="The quantization used for the llm model")
    description: Optional[str] = None
    custom_config: Optional[dict] = Field(
        sa_column=Column(JSON), default=None, description="Custom configuration for the LLM service in JSON format"
    )

    llm_interactions: list["LLMInteractionModel"] = Relationship(back_populates="llm_service")


class TaskModel(MyModel, table=True):
    __tablename__ = "task"
    name: str = Field(description="The name of the task", unique=True, index=True)
    description: str | None = None
    llm_interaction_groups: list["LLMInteractionGroupModel"] = Relationship(back_populates="task")
