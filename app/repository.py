import asyncio
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Iterable

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import or_, and_, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import Config
from app.entities_models.db_models import *
from app.entities_models.entities import LLMServiceEntity, ToModelEntityType, DatasetEntity, PromptTemplateEntity
from app.shared.utils import logger


class Repository:
    def __init__(self):
        self._engine = create_async_engine(
            f"sqlite+aiosqlite:///{Config.DATABASE_PATH}", echo=Config.ENV == "development"
        )

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


# Create a single repository instance
repository = Repository()
repository.init_db_sync()

llm_service_repository = LLMServiceRepository()
dataset_repository = DatasetRepository()
prompt_template_repository = PromptTemplateRepository()
