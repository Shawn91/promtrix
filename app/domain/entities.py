import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class PromptTemplate(BaseModel):
    """
    PromptTemplate represents a prompt template that may contain placeholders for variables.
    A plaint text prompt template without any placeholders is also supported.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    text: str
    description: str


class Target(BaseModel):
    """
    Target represents a llm model or a LLM-backed service
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str
    version: str
    endpoint: str
    description: str


class Task(BaseModel):
    """
    Represents a specific problem that needs to be solved
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str
    description: str


class Execution(BaseModel):
    """
    Represents a specific execution of a task on a prompt for a target
    If a dataset is used for testing or evaluating the model or prompt, each data point should be represented
        as a separate Execution object.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    prompt_template: PromptTemplate = Field(description='The original prompt template that was used for this execution')
    final_prompt: str = Field(description='The prompt that was used for this execution, after any modifications')
    target: Target
    duration: int = Field(description='The duration of the execution in milliseconds')
    cost: float | None = Field(default=None,
                               description='The cost of the execution. 0 if free. None if unknown or not applicable.')
    response: str = Field(description='The response generated by the target for the prompt')
    task: Task = Field(description='The task that was executed')
    created_at: datetime = Field(description='The date and time when the execution was created')

    def __hash__(self):
        return hash(self.id)


class GroupedExecution(BaseModel):
    """
    Represents a group of Executions that were executed together.
    Typically, this will be used for evaluating a model or prompt on a dataset, or evaluating various similar
        prompts for a task.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    executions: list[Execution] = Field(description='The list of Executions that were grouped together')
    created_at: datetime = Field(description='The date and time when the group was created')
    duration: int = Field(
        description='The total duration of the group in milliseconds. If asynchronous executions are performed, '
                    'this will be smaller than the sum of the durations of all Executions in the group.'
    )
    cost: float | None = Field(default=None)


class Evaluation(BaseModel):
    """
    Represents an evaluation of an execution using a specific method or metric
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    execution: Execution
    method: str = Field(description='The method or metric used for evaluating the execution')
    score: float = Field(description='The score obtained by the evaluation')
    created_at: datetime = Field(description='The date and time when the evaluation was created')
    duration: int = Field(description='The duration of the evaluation in milliseconds')
    cost: float | None = Field(default=None,
                               description='The cost of the evaluation. 0 if free. None if unknown or not applicable.')


class GroupedEvaluation(BaseModel):
    """
    Represents a group of Evaluation objects that were performed together.
    Note: given a dataset, each llm response could be evaluated using different metrics or methods creating multiple
        Evaluation objects.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    evaluations: dict[Execution, list[Evaluation]] = Field(
        description='Each Execution object is mapped to a list of Evaluation objects '
                    'that were performed on it using different methods or metrics.')
    created_at: datetime = Field(description='The date and time when the group was created')
    duration: int = Field(
        description='The total duration of the group in milliseconds. If asynchronous evaluations are performed, '
                    'this will be smaller than the sum of the durations of all Evaluation objects in the group.'
    )
    cost: float | None = Field(default=None)


class Dataset(BaseModel):
    """
    Represents a dataset used for testing or evaluating a model or a prompt
    A dataset can contain prompts only or both prompts and desired responses.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str
    description: str
    url: str
    size: int = Field(description='The size of the dataset in bytes')
    count: int = Field(description='The number of examples in the dataset')
    created_at: datetime = Field(description='The date and time when the dataset was created')