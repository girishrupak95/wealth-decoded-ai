"""Structured output validation."""

import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from shared.exceptions.ai import OutputValidationError


class OutputValidator:
    def validate(self, value: str, schema: type[BaseModel]) -> dict[str, Any]:
        try:
            payload = json.loads(value)
            if not isinstance(payload, dict):
                raise ValueError("Output must be a JSON object.")
            return schema.model_validate(payload).model_dump(mode="json")
        except (json.JSONDecodeError, ValidationError, ValueError) as error:
            raise OutputValidationError(str(error)) from error

    def validate_model(
        self, value: Mapping[str, Any] | BaseModel, model: type[BaseModel]
    ) -> BaseModel:
        payload = value.model_dump() if isinstance(value, BaseModel) else value
        try:
            return model.model_validate(payload)
        except ValidationError as error:
            raise OutputValidationError(str(error)) from error
