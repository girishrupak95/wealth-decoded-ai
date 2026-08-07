"""Reusable async execution lifecycle for agents."""

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from string import Template
from time import perf_counter
from typing import Any
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, Field

from shared.ai.knowledge_loader import KnowledgeLoader
from shared.ai.llm_client import LLMClient, LLMRequest
from shared.ai.output_validator import OutputValidator
from shared.ai.prompt_loader import PromptLoader
from shared.constants import (
    DEBUG_DIRECTORY_NAME,
    GENERATED_DIRECTORY_NAME,
    RAW_LLM_DEBUG_FILENAME_SUFFIX,
)
from shared.exceptions.ai import OutputValidationError


class AgentRequest(BaseModel):
    prompt_name: str
    system_prompt_name: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class AgentExecution(BaseModel):
    output: dict[str, Any]
    execution_id: str
    duration_ms: float


class BaseAgent(ABC):
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        prompt_loader: PromptLoader,
        knowledge_loader: KnowledgeLoader,
        output_validator: OutputValidator,
        logger_instance: Any = logger,
    ) -> None:
        self._llm_client = llm_client
        self._prompt_loader = prompt_loader
        self._knowledge_loader = knowledge_loader
        self._output_validator = output_validator
        self._logger = logger_instance.bind(component=self.__class__.__name__)

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable agent name."""

    @property
    @abstractmethod
    def output_schema(self) -> type[BaseModel]:
        """Return the model response schema."""

    async def execute(self, request: AgentRequest) -> AgentExecution:
        started_at = perf_counter()
        execution_id = str(uuid4())
        event_logger = self._logger.bind(execution_id=execution_id, agent_name=self.name)
        event_logger.info("agent_execution_started", duration=0.0, status="started")
        try:
            knowledge = self._knowledge_loader.load_all()
            context = {
                key: json.dumps(value) if not isinstance(value, str) else value
                for key, value in request.context.items()
            }
            context["knowledge"] = json.dumps(knowledge, sort_keys=True)
            user_prompt = Template(self._prompt_loader.load(request.prompt_name)).substitute(
                context
            )
            system_prompt = (
                Template(self._prompt_loader.load(request.system_prompt_name)).substitute(context)
                if request.system_prompt_name
                else None
            )
            schema_instruction = self._build_output_schema_instruction()
            system_prompt = (
                f"{system_prompt}\n\n{schema_instruction}" if system_prompt else schema_instruction
            )
            raw_output = await self._llm_client.generate(
                LLMRequest(
                    template=user_prompt,
                    system_template=system_prompt,
                    context=request.context,
                    knowledge=knowledge,
                )
            )
            self._save_debug_raw_output(raw_output, event_logger)
            output = self._output_validator.validate(raw_output, self.output_schema)
        except OutputValidationError as error:
            event_logger.warning(
                "agent_output_validation_failed",
                duration=round((perf_counter() - started_at) * 1000, 3),
                status="failed",
                exception_type=type(error).__name__,
                error_count=getattr(error, "error_count", None),
            )
            raise
        except Exception:
            event_logger.exception(
                "agent_execution_failed",
                duration=round((perf_counter() - started_at) * 1000, 3),
                status="failed",
            )
            raise
        duration = round((perf_counter() - started_at) * 1000, 3)
        event_logger.info("agent_execution_finished", duration=duration, status="succeeded")
        return AgentExecution(output=output, execution_id=execution_id, duration_ms=duration)

    def _build_output_schema_instruction(self) -> str:
        schema = json.dumps(self.output_schema.model_json_schema(), sort_keys=True)
        return (
            "Return exactly one valid JSON object. The JSON must conform exactly to the "
            "supplied JSON Schema. Use every required field. Do not rename fields. Do not "
            "include fields not present in the schema. Do not include Markdown fences. Do not "
            "include commentary before or after the JSON. Numeric values must be JSON numbers. "
            "Arrays must use the expected item structure.\n\nJSON Schema:\n"
            f"{schema}"
        )

    def _save_debug_raw_output(self, raw_output: str, event_logger: Any) -> None:
        if os.environ.get("WEALTH_DEBUG_SAVE_RAW_LLM") != "1":
            return
        try:
            path = self._debug_output_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self._format_debug_raw_output(raw_output), encoding="utf-8")
            event_logger.info("debug_output_saved", debug_output_saved=True)
        except Exception:
            event_logger.warning("debug_output_save_failed", debug_output_saved=False)

    def _debug_output_path(self) -> Path:
        agent_name = self.name.replace("-", "_")
        return (
            Path.cwd()
            / GENERATED_DIRECTORY_NAME
            / DEBUG_DIRECTORY_NAME
            / f"{agent_name}{RAW_LLM_DEBUG_FILENAME_SUFFIX}"
        )

    @staticmethod
    def _format_debug_raw_output(raw_output: str) -> str:
        try:
            return json.dumps(json.loads(raw_output), indent=2, sort_keys=True)
        except json.JSONDecodeError:
            return raw_output
