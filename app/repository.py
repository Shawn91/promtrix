import asyncio
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Iterable

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import or_, select, and_
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
    LLMInteractionGroupEntity,
    LLMInteractionEntity,
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
                entity.id = db_model.id
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
        return select(LLMServiceModel).where(
            LLMServiceModel.llm == llm_service.llm,
            LLMServiceModel.llm_version == llm_service.llm_version,
            LLMServiceModel.quantization == llm_service.quantization,
        )


class DatasetRepository(Repository):
    def check_existance_statement(self, dataset: DatasetEntity):
        return select(DatasetModel).where(
            or_(
                DatasetModel.raw_dataset_dir == dataset.raw_dataset_dir,
                and_(DatasetModel.name == dataset.name, DatasetModel.version == dataset.version),
            )
        )


class PromptTemplateRepository(Repository):
    def check_existance_statement(self, prompt_template: PromptTemplateEntity):
        return select(PromptTemplateModel).where(
            and_(
                PromptTemplateModel.user == prompt_template.user, PromptTemplateModel.system == prompt_template.system
            )
        )


class PromptRepository(Repository):
    def check_existance_statement(self, prompt: PromptEntity):
        return select(PromptModel).where(PromptModel.user == prompt.user, PromptModel.system == prompt.system)


class TaskRepository(Repository):
    def check_existance_statement(self, task: TaskEntity):
        return select(TaskModel).where(TaskModel.name == task.name)


class LLMInteractionGroupRepository(Repository):
    def check_existance_statement(self, llm_interaction_group: LLMInteractionGroupEntity):
        return select(LLMInteractionGroupModel).where(
            LLMInteractionGroupModel.task_id == llm_interaction_group.task.id,
            LLMInteractionGroupModel.name == llm_interaction_group.name,
        )


class LLMInteractionRepository(Repository):
    def check_existance_statement(self, llm_interaction: LLMInteractionEntity):
        conditions = [
            LLMInteractionModel.group_id == llm_interaction.group.id,
            LLMInteractionModel.prompt_id == llm_interaction.prompt.id,
            LLMInteractionModel.llm_service_id == llm_interaction.llm_service.id,
        ]

        # Add LLM parameters that affect the response
        if llm_interaction.llm_parameters:
            param_conditions = [
                LLMInteractionModel.temperature == llm_interaction.llm_parameters.temperature,
                LLMInteractionModel.max_completion_tokens == llm_interaction.llm_parameters.max_completion_tokens,
                LLMInteractionModel.top_k == llm_interaction.llm_parameters.top_k,
                LLMInteractionModel.top_p == llm_interaction.llm_parameters.top_p,
                LLMInteractionModel.min_p == llm_interaction.llm_parameters.min_p,
                LLMInteractionModel.top_a == llm_interaction.llm_parameters.top_a,
                LLMInteractionModel.stop == llm_interaction.llm_parameters.stop,
                LLMInteractionModel.n == llm_interaction.llm_parameters.n,
                LLMInteractionModel.presence_penalty == llm_interaction.llm_parameters.presence_penalty,
                LLMInteractionModel.frequency_penalty == llm_interaction.llm_parameters.frequency_penalty,
                LLMInteractionModel.repitition_penalty == llm_interaction.llm_parameters.repitition_penalty,
                LLMInteractionModel.seed == llm_interaction.llm_parameters.seed,
            ]
            # Only add non-None parameters to the conditions
            conditions.extend([cond for cond in param_conditions if cond.right is not None])

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
                    result = await session.execute(stmt)
                    existing = result.scalar_one_or_none()

                    if existing:
                        interaction.id = existing.id
                        if interaction.responses:
                            for response in interaction.responses:
                                session.add(response.to_model(llm_interaction=interaction))
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
        return select(LLMInteractionGroupModel).where(
            LLMInteractionGroupModel.task_id == llm_interaction_group.task.id,
            LLMInteractionGroupModel.name == llm_interaction_group.name,
        )


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
