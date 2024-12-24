import asyncio
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Iterable

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import or_, select, and_, SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import Config
from app.entities_models.db_models import *
from app.entities_models.entities import (
    PromptEntity,
    LLMServiceEntity,
    ToModelEntityType,
    DatasetEntity,
    PromptTemplateEntity,
    TaskEntity,
    LLMInteractionGroupEntity,
    LLMInteractionEntity,
    LLMResponseEntity,
    EvaluationEntity,
)
from app.shared.utils import logger


class Repository:
    def __init__(self):
        self._engine = create_async_engine(f"sqlite+aiosqlite:///{Config.DATABASE_PATH}", echo=False)

    async def init_db(self):
        async with self._engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    def init_db_sync(self):
        """Synchronous wrapper to initialize database"""
        try:
            # Get or create an event loop
            if sys.platform == "win32":
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            else:
                loop = asyncio.get_event_loop()

            # Run the async init_db
            loop.run_until_complete(self.init_db())
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        async with AsyncSession(self._engine) as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def get(self, entity: ToModelEntityType) -> ToModelEntityType | None:
        """
        Get an entity from the database. If found, updates the entity's ID with the database ID.

        Args:
            entity: The entity to find in the database

        Returns:
            The entity with updated ID if found, None otherwise
        """
        async with self.session() as session:
            # First try to find by unique combination
            stmt = self.check_existance_statement(entity)
            result = await session.execute(stmt)
            db_model = result.scalar_one_or_none()
            if not db_model and entity.id:
                # If not found by unique combination, try to find by ID
                db_model = await session.get(entity.model, entity.id)

            if db_model:
                db_entity = db_model.to_entity()
                for field in entity.model_fields:
                    if getattr(entity, field) is None and getattr(db_entity, field) is not None:
                        setattr(entity, field, getattr(db_entity, field))
                return entity
            return None

    async def create(self, entity: ToModelEntityType) -> bool:
        """
        Create a new model corresponding to the given entity in the database.

        Args:
            entity: The entity to create

        Returns:
            bool: True if successfully created, False if already exists
        """
        return await self.create_many([entity])

    async def create_many(self, entities: Iterable[ToModelEntityType]) -> bool:
        """
        Create multiple new models corresponding to the given entities in the database.
        Skips entities that already exist and saves the remaining ones.
        If any entity fails to save, all new entities are rolled back.

        Args:
            entities: List of entities to create. All entities must be of the same type.

        Returns:
            True if all entities were saved successfully, False if any entity failed to save.
        """
        if not entities:
            return True
        first_entity = None
        try:
            async with self.session() as session:
                for entity in entities:
                    if not first_entity:
                        first_entity = entity
                    assert isinstance(
                        entity, type(first_entity)
                    ), f"All entities must be of the same type. Expected: {type(first_entity)} but got {type(entity)}"
                    stmt = self.check_existance_statement(entity)
                    if stmt is not None:
                        result = await session.execute(stmt)
                        existing = result.scalar_one_or_none()
                        if existing:
                            entity.id = existing.id
                            continue

                    model = entity.to_model()
                    session.add(model)

                # Attempt to save all new entities
                await session.flush()
            return True

        except SQLAlchemyError as e:
            # Handle specific database errors
            logger.error(f"Database error while creating models for entities {type(first_entity)}: {e}")
            # Session rollback is handled by context manager
            return False
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error while creating models for entities {type(first_entity)}: {e}")
            return False

    def check_existance_statement(self, entity: ToModelEntityType):
        raise NotImplementedError()


class LLMServiceRepository(Repository):
    def check_existance_statement(self, llm_service: LLMServiceEntity):
        conditions = []
        if llm_service.llm is not None:  # Make this optional too
            conditions.append(LLMServiceModel.llm == llm_service.llm)
        if llm_service.llm_version is not None:
            conditions.append(LLMServiceModel.llm_version == llm_service.llm_version)
        if llm_service.quantization is not None:
            conditions.append(LLMServiceModel.quantization == llm_service.quantization)

        if not conditions:
            return None
        return select(LLMServiceModel).where(and_(*conditions))


class DatasetRepository(Repository):
    def check_existance_statement(self, dataset: DatasetEntity):
        conditions = []
        if dataset.raw_dataset_dir:
            conditions.append(DatasetModel.raw_dataset_dir == dataset.raw_dataset_dir)
        if dataset.name and dataset.version:
            conditions.append(and_(DatasetModel.name == dataset.name, DatasetModel.version == dataset.version))

        if not conditions:
            return None
        return select(DatasetModel).where(or_(*conditions))


class PromptTemplateRepository(Repository):
    def check_existance_statement(self, prompt_template: PromptTemplateEntity):
        conditions = [PromptTemplateModel.user == prompt_template.user]
        if prompt_template.system is not None:
            conditions.append(PromptTemplateModel.system == prompt_template.system)
        return select(PromptTemplateModel).where(and_(*conditions))


class PromptRepository(Repository):
    def check_existance_statement(self, prompt: PromptEntity):
        conditions = [PromptModel.user == prompt.user]
        if prompt.system is not None:
            conditions.append(PromptModel.system == prompt.system)
        return select(PromptModel).where(and_(*conditions))


class TaskRepository(Repository):
    def check_existance_statement(self, task: TaskEntity):
        return select(TaskModel).where(TaskModel.name == task.name)


class LLMInteractionRepository(Repository):
    def check_existance_statement(self, llm_interaction: LLMInteractionEntity):
        conditions = []

        if llm_interaction.group and llm_interaction.group.id:
            conditions.append(LLMInteractionModel.group_id == llm_interaction.group.id)
        if llm_interaction.prompt and llm_interaction.prompt.id:
            conditions.append(LLMInteractionModel.prompt_id == llm_interaction.prompt.id)
        if llm_interaction.llm_service and llm_interaction.llm_service.id:
            conditions.append(LLMInteractionModel.llm_service_id == llm_interaction.llm_service.id)

        # Add LLM parameters that affect the response
        if llm_interaction.llm_parameters:
            param_mapping = {
                "temperature": llm_interaction.llm_parameters.temperature,
                "max_completion_tokens": llm_interaction.llm_parameters.max_completion_tokens,
                "top_k": llm_interaction.llm_parameters.top_k,
                "top_p": llm_interaction.llm_parameters.top_p,
                "min_p": llm_interaction.llm_parameters.min_p,
                "top_a": llm_interaction.llm_parameters.top_a,
                "stop": llm_interaction.llm_parameters.stop,
                "n": llm_interaction.llm_parameters.n,
                "presence_penalty": llm_interaction.llm_parameters.presence_penalty,
                "frequency_penalty": llm_interaction.llm_parameters.frequency_penalty,
                "repitition_penalty": llm_interaction.llm_parameters.repitition_penalty,
                "seed": llm_interaction.llm_parameters.seed,
            }

            for param_name, param_value in param_mapping.items():
                if param_value is not None:
                    conditions.append(getattr(LLMInteractionModel, param_name) == param_value)

        if not conditions:
            return None
        return select(LLMInteractionModel).where(and_(*conditions))

    async def create_many(self, llm_interactions: Iterable[LLMInteractionEntity]) -> bool:
        """
        Create multiple llm_interactions with their responses in a single transaction.
        If any llm_interactions or response fails to save, the entire transaction is rolled back.

        Args:
            llm_interactions: List of llm_interactions entities to create

        Returns:
            bool: True if all entities were saved successfully, False otherwise
        """
        if not llm_interactions:
            return True
        try:
            async with self.session() as session:
                for interaction in llm_interactions:
                    stmt = self.check_existance_statement(interaction)
                    if stmt is not None:
                        result = await session.execute(stmt)
                        existing = result.scalar_one_or_none()

                        if existing:
                            interaction.id = existing.id
                            if interaction.responses:
                                for response in interaction.responses:
                                    session.add(response.to_model())
                            continue

                    # Create interaction model
                    interaction_model = interaction.to_model()
                    session.add(interaction_model)
                    await session.flush()

                    # Create response models if they exist
                    if interaction.responses:
                        for response in interaction.responses:
                            session.add(response.to_model(llm_interaction=interaction))
            return True

        except SQLAlchemyError as e:
            logger.error(f"Database error while creating interactions: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error while creating interaction: {e}")
            return False


class LLMInteractionGroupRepository(Repository):
    def check_existance_statement(self, llm_interaction_group: LLMInteractionGroupEntity):
        conditions = []
        if llm_interaction_group.task and llm_interaction_group.task.id:
            conditions.append(LLMInteractionGroupModel.task_id == llm_interaction_group.task.id)
        if llm_interaction_group.name:
            conditions.append(LLMInteractionGroupModel.name == llm_interaction_group.name)

        if not conditions:
            return None
        return select(LLMInteractionGroupModel).where(and_(*conditions))


class EvaluationRepository(Repository):
    def check_existance_statement(self, entity: ToModelEntityType):
        """Evaluations can't be identified by unique combination, so this method is not implemented"""
        return None

    async def find_by_llm_response(self, llm_response: LLMResponseEntity) -> list[EvaluationEntity]:
        """Find all evaluations for a given llm_response"""
        async with self.session() as session:
            stmt = select(EvaluationModel).where(EvaluationModel.llm_response_id == llm_response.id)
            result = await session.execute(stmt)
            return [model.to_entity() for model in result.scalars().all()]


class EvaluationGroupRepository(Repository):
    def check_existance_statement(self, entity: ToModelEntityType):
        return None


# Create a single repository instance
repository = Repository()
repository.init_db_sync()

llm_service_repository = LLMServiceRepository()
dataset_repository = DatasetRepository()
prompt_template_repository = PromptTemplateRepository()
prompt_repository = PromptRepository()
task_repository = TaskRepository()
llm_interaction_group_repository = LLMInteractionGroupRepository()
llm_interaction_repository = LLMInteractionRepository()
evaluation_repository = EvaluationRepository()
evaluation_group_repository = EvaluationGroupRepository()
