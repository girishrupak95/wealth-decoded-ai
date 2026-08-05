"""OpenAI implementation of the LLM client contract."""

from typing import Any

from openai import AsyncOpenAI

from app.config.settings import OpenAISettings
from shared.ai.llm_client import LLMClient, LLMRequest
from shared.exceptions.ai import OpenAIRequestError

_LEGACY_OPTIONAL_PARAMETER_MODEL_PREFIXES = ("gpt-3.5", "gpt-4", "o1-")


class OpenAIClient(LLMClient):
    def __init__(self, settings: OpenAISettings) -> None:
        super().__init__()
        self._settings = settings
        self._client = AsyncOpenAI(api_key=settings.api_key.get_secret_value())

    async def generate(self, request: LLMRequest) -> str:
        try:
            response = await self._client.responses.create(**self._request_parameters(request))
        except Exception as error:
            category, message = self._request_error_details(error)
            self._logger.error(
                "openai_request_failed",
                error_type=type(error).__name__,
                provider_category=category,
            )
            raise OpenAIRequestError(message) from error
        output_text: object = response.output_text
        if not isinstance(output_text, str) or not output_text:
            self._logger.error("openai_request_failed", provider_category="empty_response")
            raise OpenAIRequestError("OpenAI returned an empty response.")
        return output_text

    async def health(self) -> bool:
        try:
            await self._client.models.list()
        except Exception:
            return False
        return True

    async def close(self) -> None:
        await self._client.close()

    def _request_parameters(self, request: LLMRequest) -> dict[str, Any]:
        parameters: dict[str, Any] = {
            "model": self._settings.model,
            "input": request.template,
        }
        if request.system_template is not None:
            parameters["instructions"] = request.system_template
        if self._supports_optional_parameters():
            if self._settings.temperature is not None:
                parameters["temperature"] = self._settings.temperature
            parameters["max_output_tokens"] = self._settings.max_tokens
        return parameters

    def _supports_optional_parameters(self) -> bool:
        model = self._settings.model.casefold()
        return model.startswith(_LEGACY_OPTIONAL_PARAMETER_MODEL_PREFIXES)

    @staticmethod
    def _request_error_details(error: Exception) -> tuple[str, str]:
        status_code = getattr(error, "status_code", None)
        if status_code == 400:
            return (
                "unsupported_request_parameters",
                "OpenAI request failed because the configured request parameters are not supported "
                "by the selected model.",
            )
        if status_code == 401:
            return "authentication_failed", "OpenAI authentication failed."
        if status_code == 403:
            return "permission_denied", "OpenAI permission was denied."
        return "provider_request_failed", "OpenAI request failed."
