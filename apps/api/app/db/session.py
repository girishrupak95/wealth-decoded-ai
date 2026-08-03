from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config.settings import Settings


class Database:
    """Own the application's asynchronous database engine and sessions."""

    def __init__(self, settings: Settings) -> None:
        self._engine = create_async_engine(
            str(settings.database_url),
            pool_pre_ping=True,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
        )
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def dispose(self) -> None:
        """Release database resources during application shutdown."""
        await self._engine.dispose()

    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Yield a transaction-ready asynchronous database session."""
        async with self._session_factory() as session:
            yield session
