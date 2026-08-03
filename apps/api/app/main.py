from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from app.api.routes.health import router as health_router
from app.config.settings import Settings, get_settings
from app.core.logging import configure_logging
from app.db.session import Database
from app.middleware.request_id import RequestIDMiddleware


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the configured FastAPI application."""
    application_settings = settings or get_settings()
    configure_logging(application_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.database = Database(application_settings)
        logger.info("application_started", service=application_settings.app_name)
        yield
        await app.state.database.dispose()
        logger.info("application_stopped", service=application_settings.app_name)

    app = FastAPI(title=application_settings.app_name, lifespan=lifespan)
    app.add_middleware(RequestIDMiddleware, header_name=application_settings.request_id_header)
    app.include_router(health_router)
    return app


app = create_app()
