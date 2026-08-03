from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Public health-check response."""

    status: str
    service: str
