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
                validation_issues=self._validation_issues(error),
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
                validation_issues=self._validation_issues(error),
            ) from error

    @staticmethod
    def _validation_issues(error: ValidationError) -> tuple[OutputValidationIssue, ...]:
        messages = {
            "missing": "Field required",
            "list_type": "Value must be an array",
            "string_type": "Value must be a string",
            "bool_type": "Value must be a boolean",
            "int_type": "Value must be an integer",
            "enum": "Value must use an allowed enum literal",
            "extra_forbidden": "Unexpected field is not allowed",
        }
        issues: list[OutputValidationIssue] = []
        for item in error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        ):
            error_type = str(item.get("type", "schema_error"))
            location = item.get("loc", ())
            field_path = ".".join(str(part) for part in location)
            issues.append(
                OutputValidationIssue(
                    field_path=field_path or "root",
                    error_type=error_type,
                    message=messages.get(error_type, "Value does not satisfy schema"),
                )
            )
        return tuple(issues)
