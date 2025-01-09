import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Type, Tuple, Dict, Optional, TypeVar, Iterable
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlmodel import select, and_, SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import Config
from app.models.db_models import (
    MyModel,
    LLMServiceModel,
    DatasetModel,
    PromptTemplateModel,
    PromptModel,
    TaskModel,
    LLMInteractionModel,
    LLMInteractionGroupModel,
    EvaluationModel,
    EvaluationGroupModel,
)
from app.shared.utils import logger

ModelType = TypeVar("ModelType", bound=MyModel)


class RepositoryManager:
    """Manages database engine and repositories"""

    _instance: Optional["RepositoryManager"] = None
    _engine: Optional[AsyncEngine] = None
    _repositories: Dict[Type[MyModel], "Repository"] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "RepositoryManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_engine(self, db_path: Path | str):
        """Set a new engine (useful for testing with different databases)"""
        self._engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("Database engine not initialized")
        return self._engine

    async def init_db(self):
        """Initialize the database schema"""
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    def init_db_sync(self, db_path: Path | None = None):
        """Synchronous wrapper to initialize database"""
        if not db_path:
            db_path = Config.DATABASE_PATH
        self.set_engine(db_path=db_path)
        try:
            if sys.platform == "win32":
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            else:
                loop = asyncio.get_event_loop()
            loop.run_until_complete(self.init_db())
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """Provides a transactional context"""
        async with AsyncSession(self.engine) as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    @classmethod
    def register_repository(cls, repository: "Repository"):
        """Register a repository for a specific model type"""
        cls._repositories[repository.model] = repository

    def get_repository(self, model_type: Type[ModelType]) -> "Repository":
        """Get the repository for a specific model type"""
        if model_type not in self._repositories:
            raise ValueError(f"No repository registered for model type {model_type}")
        return self._repositories[model_type]

    async def get(self, model: ModelType, session: Optional[AsyncSession] = None) -> ModelType | None:
        """Get a model using its appropriate repository"""
        repository = self.get_repository(type(model))
        if session:
            return await repository.get(model=model, session=session)
        else:
            async with self.transaction() as session:
                return await repository.get(model=model, session=session)

    async def save(self, model: ModelType, session: Optional[AsyncSession] = None) -> Tuple[ModelType, bool]:
        """Save a model using its appropriate repository"""
        repository = self.get_repository(type(model))
        if session:
            result = await repository.save(model=model, session=session)
            if result[1]:
                await session.commit()
                await session.refresh(model)
            return result
        else:
            async with self.transaction() as session:
                result = await repository.save(model=model, session=session)
                if result[1]:
                    await session.commit()
                    await session.refresh(model)
                return result

    async def save_many(
        self, models: Iterable[ModelType], session: Optional[AsyncSession] = None
    ) -> list[Tuple[ModelType, bool]]:
        models = list(models)
        assert all(isinstance(model, type(models[0])) for model in models)
        repository = self.get_repository(type(models[0]))
        if session:
            result = await repository.save_many(models=models, session=session)
            await session.commit()
            await asyncio.gather(
                *[session.refresh(model) for model, single_result in zip(models, result) if single_result[1]]
            )
            return result
        async with self.transaction() as session:
            result = await repository.save_many(models=models, session=session)
            await session.commit()
            await asyncio.gather(
                *[session.refresh(model) for model, single_result in zip(models, result) if single_result[1]]
            )
            return result


class Repository:
    @property
    def model(self) -> Type[MyModel]:
        raise NotImplementedError()

    async def get(self, model: ModelType, session: AsyncSession) -> ModelType | None:
        """get the corresponding model in database and update it with the given model parameter."""
        db_model = None
        if model.id:
            db_model = await session.get(type(model), model.id)
        if not db_model:
            stmt = self.check_existence_statement(model)
            if stmt is not None:
                result = await session.execute(stmt)
                db_model = result.scalar_one_or_none()
        if db_model:
            model.id = db_model.id
            # update model data in the database
            try:
                db_model = await session.merge(model)
                await session.flush()
                await session.commit()
            except SQLAlchemyError as e:
                logger.error(f"Database error while updating model of type {type(model)}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error while updating model of type {type(model)}: {e}")
                raise
        return db_model

    async def save(self, model: ModelType, session: AsyncSession) -> Tuple[ModelType, bool]:
        """save a model into database with session passed as parameter
        The second returned value of type bool indicates whether the model is new or not
        """
        existed_model = await self.get(model=model, session=session)
        if existed_model:
            return existed_model, False
        else:
            try:
                session.add(model)
                return model, True
            except SQLAlchemyError as e:
                logger.error(f"Database error while saving new model of type {type(model)}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error while saving new model of type {type(model)}: {e}")
                raise

    async def save_many(
        self, models: Iterable[ModelType], session: AsyncSession | None = None
    ) -> list[Tuple[ModelType, bool]]:
        return await asyncio.gather(*[self.save(model=model, session=session) for model in models])

    def check_existence_statement(self, model: ModelType):
        raise NotImplementedError()


class LLMServiceRepository(Repository):
    @property
    def model(self) -> Type[LLMServiceModel]:
        return LLMServiceModel

    def check_existence_statement(self, model: LLMServiceModel):
        assert model.llm is not None
        conditions = [LLMServiceModel.llm == model.llm]
        if model.llm_version is not None:
            conditions.append(LLMServiceModel.llm_version == model.llm_version)
        if model.quantization is not None:
            conditions.append(LLMServiceModel.quantization == model.quantization)
        return select(LLMServiceModel).where(and_(*conditions))


class DatasetRepository(Repository):
    @property
    def model(self) -> Type[DatasetModel]:
        return DatasetModel

    def check_existence_statement(self, model: DatasetModel):
        assert model.raw_dataset_dir is not None
        return select(DatasetModel).where(DatasetModel.raw_dataset_dir == model.raw_dataset_dir)


class PromptTemplateRepository(Repository):
    @property
    def model(self) -> Type[PromptTemplateModel]:
        return PromptTemplateModel

    def check_existence_statement(self, model: PromptTemplateModel):
        assert model.user is not None
        conditions = [PromptTemplateModel.user == model.user]
        if model.system is not None:
            conditions.append(PromptTemplateModel.system == model.system)
        return select(PromptTemplateModel).where(and_(*conditions))


class PromptRepository(Repository):
    @property
    def model(self) -> Type[PromptModel]:
        return PromptModel

    def check_existence_statement(self, model: PromptModel):
        assert model.user is not None
        conditions = [PromptModel.user == model.user]
        if model.system is not None:
            conditions.append(PromptModel.system == model.system)
        return select(PromptModel).where(and_(*conditions))


class TaskRepository(Repository):
    @property
    def model(self) -> Type[TaskModel]:
        return TaskModel

    def check_existence_statement(self, model: TaskModel):
        assert model.name is not None
        return select(TaskModel).where(TaskModel.name == model.name)


class LLMInteractionRepository(Repository):
    @property
    def model(self) -> Type[LLMInteractionModel]:
        return LLMInteractionModel

    def check_existence_statement(self, model: LLMInteractionModel):
        assert model.group_id is not None
        assert model.prompt_id is not None
        assert model.llm_service_id is not None
        assert model.llm_params_id is not None
        conditions = [
            LLMInteractionModel.group_id == model.group_id,
            LLMInteractionModel.prompt_id == model.prompt_id,
            LLMInteractionModel.llm_service_id == model.llm_service_id,
            LLMInteractionModel.llm_params_id == model.llm_params_id,
        ]
        return select(LLMInteractionModel).where(and_(*conditions))


class LLMInteractionGroupRepository(Repository):
    @property
    def model(self) -> Type[LLMInteractionGroupModel]:
        return LLMInteractionGroupModel

    def check_existence_statement(self, model: LLMInteractionGroupModel):
        assert model.task_id is not None and model.name is not None
        return select(LLMInteractionGroupModel).where(
            and_(LLMInteractionGroupModel.task_id == model.task_id, LLMInteractionGroupModel.name == model.name)
        )


class EvaluationRepository(Repository):
    @property
    def model(self) -> Type[EvaluationModel]:
        return EvaluationModel

    def check_existence_statement(self, model: EvaluationModel):
        return None

    async def find_by_llm_response(self, llm_response_id: UUID) -> list[EvaluationModel]:
        async with self.session() as session:
            stmt = select(EvaluationModel).where(EvaluationModel.llm_response_id == llm_response_id)
            result = await session.execute(stmt)
            return list(result.scalars().all())


class EvaluationGroupRepository(Repository):
    @property
    def model(self) -> Type[EvaluationGroupModel]:
        return EvaluationGroupModel

    def check_existence_statement(self, model: EvaluationGroupModel):
        assert model.name is not None and model.llm_interaction_group_id is not None
        return select(EvaluationGroupModel).where(
            and_(
                EvaluationGroupModel.name == model.name,
                EvaluationGroupModel.llm_interaction_group_id == model.llm_interaction_group_id,
            )
        )


# Create a single repository instance
# repository = Repository()
# repository.init_db_sync()
#
RepositoryManager.register_repository(repository=LLMServiceRepository())
RepositoryManager.register_repository(repository=DatasetRepository())
RepositoryManager.register_repository(repository=PromptTemplateRepository())
RepositoryManager.register_repository(repository=PromptRepository())
RepositoryManager.register_repository(repository=TaskRepository())
RepositoryManager.register_repository(repository=LLMInteractionRepository())
RepositoryManager.register_repository(repository=LLMInteractionGroupRepository())
RepositoryManager.register_repository(repository=EvaluationRepository())
RepositoryManager.register_repository(repository=EvaluationGroupRepository())
