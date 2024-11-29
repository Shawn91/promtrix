"""
The concept of Model is from Clean Architecture.

Models are the objects that represent the raw data in the application and
are used to interact with external systems like databases or APIs.

All models in this file are database models using SQLModel library.

Entities should be converted to Models before being passed to external systems.

Data retrieved from external systems should be Models and thenconverted to Entities
before being used in the application.
"""
from typing import Optional
from uuid import UUID

from sqlmodel import Field, Relationship

from app.entities_models.base import PromptTemplate, Prompt, Dataset, LLMResponse, Execution


class PromptTemplateModel(PromptTemplate, table=True):
    __tablename__ = "prompt_template"

    id: UUID | None = Field(default=None, primary_key=True)
    user: str | None = Field(default=None, description="template for user prompt")
    system: str | None = None  # Store as string, convert to jinja2 Template when needed

    prompts: list["PromptModel"] = Relationship(back_populates="template")


class PromptModel(Prompt, table=True):
    __tablename__ = "prompt"

    id: UUID | None = Field(default=None, primary_key=True)
    template_id: UUID | None = Field(default=None, foreign_key="prompt_template.id")

    template: PromptTemplate = Relationship(back_populates="prompts")
    executions: list["ExecutionModel"] = Relationship(back_populates="prompt")


class LLMResponseModel(LLMResponse, table=True):
    __tablename__ = "llm_response"

    id: UUID | None = Field(default=None, primary_key=True)
    execution_id: UUID = Field(foreign_key="execution.id")

    execution: Execution = Relationship(back_populates="llm_responses")


class ExecutionModel(Execution, table=True):
    __tablename__ = "execution"

    id: UUID | None = Field(default=None, primary_key=True)
    prompt_id: UUID | None = Field(default=None, foreign_key="prompt.id")
    group_id: UUID | None = Field(default=None, foreign_key="execution_group.id")

    # llm parameters
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

    group: "ExecutionGroupModel" = Relationship(back_populates="executions")
    # llm_service_id: uuid.UUID = Field(foreign_key="llmservice.id")
    # task_id: uuid.UUID = Field(foreign_key="task.id")
    # execution_group_id: Optional[uuid.UUID] = Field(default=None, foreign_key="executiongroup.id")
    llm_responses: list["LLMResponseModel"] = Relationship(back_populates="execution")
    prompt: PromptModel = Relationship(back_populates="executions")


class ExecutionGroupModel(Execution, table=True):
    __tablename__ = "execution_group"
    id: UUID | None = Field(default=None, primary_key=True)
    executions: list["ExecutionModel"] = Relationship(back_populates="group")


class DatasetSplitModel(Dataset, table=True):
    __tablename__ = "dataset_split"

    id: UUID | None = Field(default=None, primary_key=True)
    dataset_id: UUID = Field(foreign_key="dataset.id")

    dataset: Dataset = Relationship(back_populates="splits")


class DatasetModel(Dataset, table=True):
    __tablename__ = "dataset"

    id: UUID | None = Field(default=None, primary_key=True)
    parent_id: UUID | None = Field(default=None, foreign_key="dataset.id")

    # Define the relationship to child datasets
    subsets: list["DatasetModel"] = Relationship(back_populates="parent", sa_relationship_kwargs={"remote_side": [id]})
    # Define the relationship to parent dataset
    parent: Optional["DatasetModel"] = Relationship(
        back_populates="subsets", sa_relationship_kwargs={"remote_side": [parent_id]}
    )
    splits: list["DatasetSplitModel"] = Relationship(back_populates="dataset")
