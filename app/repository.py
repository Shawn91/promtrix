import asyncio
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import Config
from app.entities_models.db_models import *
from app.shared.utils import logger


class Repository:
    def __init__(self):
        print(Config.DATABASE_PATH)
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


class LLMServiceRepository(Repository):
    async def get(self, llm_service: LLMServiceEntity) -> LLMServiceEntity | None:
        async with self.session() as session:
            # First try to find by unique combination
            stmt = select(LLMServiceModel).where(
                LLMServiceModel.llm == llm_service.llm,
                LLMServiceModel.llm_version == llm_service.llm_version,
                LLMServiceModel.quantization == llm_service.quantization,
            )
            result = await session.execute(stmt)
            db_service = result.scalar_one_or_none()
            if not db_service and llm_service.id:
                # If not found by unique combination, try to find by ID
                db_service = await session.get(LLMServiceModel, llm_service.id)

            if db_service:
                return db_service.to_entity()
            return None

    async def create(self, llm_service: LLMServiceEntity) -> bool:
        """
        Create a new LLM service in the database.

        Args:
            llm_service: The LLM service entity to create

        Returns:
            bool: True if successfully created, False if service already exists
        """
        existing_service = await self.get(llm_service)
        if existing_service:
            return False
        try:
            async with self.session() as session:
                db_service = llm_service.to_model()
                session.add(db_service)
                return True
        except SQLAlchemyError as e:
            # Handle specific database errors
            logger.error(f"Database error while creating LLM service: {e}")
            return False
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error while creating LLM service: {e}")
            raise


# Create a single repository instance
repository = Repository()
repository.init_db_sync()

llm_service_repository = LLMServiceRepository()
