"""Structured output validation."""

import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from shared.exceptions.ai import OutputValidationError, OutputValidationIssue


class OutputValidator:
    def validate(self, value: str, schema: type[BaseModel]) -> dict[str, Any]:
        try:
            payload = json.loads(value)
            if not isinstance(payload, dict):
                raise ValueError("Output must be a JSON object.")
            return schema.model_validate(payload, extra="forbid").model_dump(mode="json")
        except ValidationError as error:
            raise OutputValidationError(
                "Structured output does not match the required schema.",
                error_count=error.error_count(),
                validation_issues=self._validation_issues(error, payload),
                invalid_output=payload,
            ) from error
        except (json.JSONDecodeError, ValueError) as error:
            raise OutputValidationError(str(error)) from error

    def validate_model(
        self, value: Mapping[str, Any] | BaseModel, model: type[BaseModel]
    ) -> BaseModel:
        payload = value.model_dump() if isinstance(value, BaseModel) else value
        try:
            return model.model_validate(payload, extra="forbid")
        except ValidationError as error:
            raise OutputValidationError(
                "Structured output does not match the required schema.",
                error_count=error.error_count(),
                validation_issues=self._validation_issues(error, payload),
                invalid_output=dict(payload),
            ) from error

    @staticmethod
    def _validation_issues(
        error: ValidationError, payload: Mapping[str, Any]
    ) -> tuple[OutputValidationIssue, ...]:
        issues: list[OutputValidationIssue] = []
        for item in error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        ):
            error_type = str(item.get("type", "schema_error"))
            location = tuple(item.get("loc", ()))
            field_path = ".".join(str(part) for part in location)
            message = str(item.get("msg", "Value does not satisfy schema"))[:500]
            issues.append(
                OutputValidationIssue(
                    field_path=field_path or "root",
                    error_type=error_type,
                    message=message,
                    location=location,
                    scene_id=OutputValidator._scene_id(payload, location),
                    context=OutputValidator._safe_context(item.get("ctx")),
                )
            )
        return tuple(issues)

    @staticmethod
    def _scene_id(payload: Mapping[str, Any], location: tuple[object, ...]) -> str | None:
        if len(location) < 2 or location[0] != "scenes" or not isinstance(location[1], int):
            return None
        scenes = payload.get("scenes")
        if not isinstance(scenes, list) or not 0 <= location[1] < len(scenes):
            return None
        scene = scenes[location[1]]
        if not isinstance(scene, dict):
            return None
        scene_id = scene.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id.strip():
            return None
        return scene_id.strip()[:100]

    @staticmethod
    def _safe_context(value: object) -> dict[str, str | int | float | bool | None] | None:
        if not isinstance(value, dict):
            return None
        safe: dict[str, str | int | float | bool | None] = {}
        for key, item in value.items():
            if isinstance(item, (str, int, float, bool)) or item is None:
                safe[str(key)[:100]] = item[:200] if isinstance(item, str) else item
        return safe or None
