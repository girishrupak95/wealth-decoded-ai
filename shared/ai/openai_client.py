"""OpenAI implementation of the LLM client contract."""

from openai import AsyncOpenAI

from app.config.settings import OpenAISettings
from shared.ai.llm_client import LLMClient, LLMRequest


class OpenAIClient(LLMClient):
    def __init__(self, settings: OpenAISettings) -> None:
        super().__init__()
        self._settings = settings
        self._client = AsyncOpenAI(api_key=settings.api_key.get_secret_value())

    async def generate(self, request: LLMRequest) -> str:
        response = await self._client.responses.create(
            model=self._settings.model,
            input=request.template,
            instructions=request.system_template,
            temperature=self._settings.temperature,
            max_output_tokens=self._settings.max_tokens,
        )
        if not response.output_text:
            raise RuntimeError("OpenAI returned an empty response.")
        return response.output_text

    async def health(self) -> bool:
        try:
            await self._client.models.list()
        except Exception:
            return False
        return True

    async def close(self) -> None:
        await self._client.close()
