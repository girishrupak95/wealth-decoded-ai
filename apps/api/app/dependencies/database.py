from collections.abc import AsyncGenerator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import Database


async def get_db_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Provide an asynchronous database session from application state."""
    database: Database = request.app.state.database
    async for session in database.session():
        yield session
