from typing import Annotated

from fastapi import APIRouter, Depends

from app.config.settings import Settings, get_settings
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    """Return service liveness information."""
    return HealthResponse(status="ok", service=settings.app_name)
